from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.schedule import get_match_schedule
from jupr_app.services import ServiceContext, submit_match_batch
from jupr_app.services.admin_live_ladder_operation_service import (
    build_match_log_recovery_url,
    deterministic_match_context_id,
    is_staging_write_gate_enabled,
    stable_request_fingerprint,
)
from jupr_app.data.load import load_data

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM = "SAVE MONEYBALL"
MONEYBALL_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES"


def is_admin_moneyball_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "").strip().lower() in TRUTHY


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def expected_share(t1_avg: float, t2_avg: float) -> float:
    try:
        return 1.0 / (1.0 + 10 ** ((float(t2_avg) - float(t1_avg)) / 400.0))
    except Exception:
        return 0.5


def expected_scoreline_from_share(p: float, goal_points: int = 11) -> tuple[int, int, int]:
    p = max(0.0001, min(0.9999, float(p if p is not None else 0.5)))
    if abs(p - 0.5) < 1e-12:
        return goal_points, goal_points, 0
    if p > 0.5:
        opp = max(0, min(goal_points, int(round(goal_points * (1.0 - p) / p))))
        return goal_points, opp, goal_points - opp
    me = max(0, min(goal_points, int(round(goal_points * p / (1.0 - p)))))
    return me, goal_points, me - goal_points


def _players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name,rating,active,is_active").eq("club_id", str(club_id)).order("name", desc=False).execute())
    except Exception:
        rows = _safe_rows(supabase.table("players").select("id,name,rating").eq("club_id", str(club_id)).execute())
    result = []
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None:
            continue
        result.append({"id": pid, "name": _clean_text(row.get("name"), limit=160) or f"Player {pid}", "rating": _safe_float(row.get("rating"), 1200.0), "is_active": bool(row.get("active", row.get("is_active", True)))})
    return result


def _league_options(supabase: Any, *, club_id: str) -> list[str]:
    try:
        rows = _safe_rows(supabase.table("leagues_metadata").select("league_name,is_active,status").eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    options = ["OVERALL"]
    for row in rows:
        name = _clean_text(row.get("league_name"), limit=120)
        if name and (bool(row.get("is_active", False)) or str(row.get("status") or "").lower() == "active"):
            options.append(name)
    return sorted(set(options), key=lambda value: (value != "OVERALL", value.lower()))


def _league_ratings(supabase: Any, *, club_id: str, league_name: str) -> dict[int, float]:
    if str(league_name or "") == "OVERALL":
        return {}
    try:
        rows = _safe_rows(supabase.table("league_ratings").select("player_id,league_name,rating").eq("club_id", str(club_id)).eq("league_name", str(league_name)).execute())
    except Exception:
        rows = []
    result: dict[int, float] = {}
    for row in rows:
        pid = _safe_int(row.get("player_id"))
        if pid is not None:
            result[pid] = _safe_float(row.get("rating"), 1200.0)
    return result


def build_admin_moneyball_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_moneyball_enabled():
        return {"enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_MONEYBALL to use Moneyball in Next."]}
    players = _players(supabase, club_id=str(club_id)) if supabase is not None else []
    writes_enabled = is_staging_write_gate_enabled(MONEYBALL_WRITE_FLAG)
    return {
        "enabled": True,
        "authority": "python_fastapi",
        "writes_enabled": writes_enabled,
        "status": "ready_for_moneyball" if writes_enabled else "read_only_streamlit_fallback",
        "players": players,
        "league_options": _league_options(supabase, club_id=str(club_id)) if supabase is not None else ["OVERALL"],
        "warnings": [] if writes_enabled else [
            f"Next Moneyball writes require JUPR_ENV=staging and {MONEYBALL_WRITE_FLAG}=1 on FastAPI. Use Streamlit Moneyball otherwise."
        ],
        "confirmation_text": {"publish": CONFIRM},
        "streamlit_fallback": "moneyball",
        "recovery": {"match_log_url": "/admin/match-log", "replay_history_url": "/admin/replay-history"},
    }


def build_moneyball_preview(supabase: Any, *, club_id: str, player_ids: list[int], rating_context: str = "OVERALL", win_rate: float = 5.0, point_rate: float = 2.0) -> dict[str, Any]:
    if not is_admin_moneyball_enabled():
        raise PermissionError("Next Moneyball is disabled.")
    ids = [int(pid) for pid in player_ids or []]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Moneyball requires exactly 8 unique players.")
    players = {row["id"]: row for row in _players(supabase, club_id=str(club_id))}
    missing = [pid for pid in ids if pid not in players]
    if missing:
        raise ValueError(f"unknown player ids: {missing}")
    league_ratings = _league_ratings(supabase, club_id=str(club_id), league_name=rating_context)
    ratings = {pid: float(league_ratings.get(pid, players[pid].get("rating") or 1200.0)) for pid in ids}
    raw_schedule = get_match_schedule("8-Player", ids)
    matches: list[dict[str, Any]] = []
    for idx, match in enumerate(raw_schedule, start=1):
        desc = str(match.get("desc", ""))
        rnd_match = re.search(r"Rnd\s*(\d+)", desc)
        court_match = re.search(r"Ct\s*(\d+)", desc)
        t1 = [int(match.get("t1", [])[0]), int(match.get("t1", [])[1])]
        t2 = [int(match.get("t2", [])[0]), int(match.get("t2", [])[1])]
        t1_avg = (ratings[t1[0]] + ratings[t1[1]]) / 2.0
        t2_avg = (ratings[t2[0]] + ratings[t2[1]]) / 2.0
        p = expected_share(t1_avg, t2_avg)
        s1, s2, margin = expected_scoreline_from_share(p, goal_points=11)
        matches.append({
            "row_id": f"moneyball-{idx}",
            "match_index": idx,
            "round": int(rnd_match.group(1)) if rnd_match else None,
            "court": int(court_match.group(1)) if court_match else None,
            "team_1": [{"id": t1[0], "name": players[t1[0]]["name"]}, {"id": t1[1], "name": players[t1[1]]["name"]}],
            "team_2": [{"id": t2[0], "name": players[t2[0]]["name"]}, {"id": t2[1], "name": players[t2[1]]["name"]}],
            "t1_p1": t1[0], "t1_p2": t1[1], "t2_p1": t2[0], "t2_p2": t2[1],
            "exp_p_t1": p,
            "expected_win_pct_t1": round(p * 100.0, 1),
            "expected_score": f"{s1}–{s2}",
            "expected_margin": margin,
        })
    payload = {"ok": True, "mode": "moneyball_preview", "rating_context": rating_context, "win_rate": float(win_rate), "point_rate": float(point_rate), "players": [players[pid] for pid in ids], "ratings": {str(pid): ratings[pid] for pid in ids}, "matches": matches}
    payload["preview_fingerprint"] = stable_request_fingerprint(
        {
            "club_id": str(club_id),
            "rating_context": str(rating_context),
            "win_rate": float(win_rate),
            "point_rate": float(point_rate),
            "player_ids_in_order": ids,
            "ratings": payload["ratings"],
            "matches": matches,
        }
    )
    payload["authority"] = "python_fastapi"
    return payload


def compute_moneyball_settlement(*, matches: list[dict[str, Any]], scores: list[dict[str, Any]], win_rate: float, point_rate: float) -> dict[str, Any]:
    by_id = {str(score.get("row_id")): score for score in scores or []}
    stats: dict[int, dict[str, Any]] = {}
    tie_matches: list[int] = []
    for match in matches:
        for pid in (match["t1_p1"], match["t1_p2"], match["t2_p1"], match["t2_p2"]):
            stats.setdefault(int(pid), {"player_id": int(pid), "gp": 0, "wins": 0, "losses": 0, "pd": 0, "exp_wins": 0.0, "exp_pd": 0.0})
        p_t1 = float(match.get("exp_p_t1") or 0.5)
        margin = int(match.get("expected_margin") or 0)
        for pid in (match["t1_p1"], match["t1_p2"]):
            stats[int(pid)]["exp_wins"] += p_t1
            stats[int(pid)]["exp_pd"] += margin
        for pid in (match["t2_p1"], match["t2_p2"]):
            stats[int(pid)]["exp_wins"] += 1.0 - p_t1
            stats[int(pid)]["exp_pd"] -= margin
        score = by_id.get(str(match.get("row_id"))) or {}
        s1 = _safe_int(score.get("score_t1")) or 0
        s2 = _safe_int(score.get("score_t2")) or 0
        if (s1 + s2) <= 0:
            continue
        if s1 == s2:
            tie_matches.append(int(match.get("match_index") or 0))
            continue
        t1_win = s1 > s2
        pd = s1 - s2
        for pid in (match["t1_p1"], match["t1_p2"]):
            stats[int(pid)]["gp"] += 1
            stats[int(pid)]["pd"] += pd
            stats[int(pid)]["wins" if t1_win else "losses"] += 1
        for pid in (match["t2_p1"], match["t2_p2"]):
            stats[int(pid)]["gp"] += 1
            stats[int(pid)]["pd"] -= pd
            stats[int(pid)]["losses" if t1_win else "wins"] += 1
    rows = []
    for row in stats.values():
        row = dict(row)
        row["win_delta"] = float(row["wins"]) - float(row["exp_wins"])
        row["pd_delta"] = float(row["pd"]) - float(row["exp_pd"])
        row["net"] = round((row["win_delta"] * float(win_rate)) + (row["pd_delta"] * float(point_rate)), 2)
        rows.append(row)
    drift = round(-sum(float(row["net"]) for row in rows), 2)
    if rows and abs(drift) >= 0.01:
        idx = max(range(len(rows)), key=lambda i: abs(float(rows[i]["net"])))
        rows[idx]["net"] = round(float(rows[idx]["net"]) + drift, 2)
    rows.sort(key=lambda row: float(row["net"]), reverse=True)
    for row in rows:
        net = float(row.get("net") or 0.0)
        row["settlement_direction"] = "receives" if net > 0 else ("owes" if net < 0 else "even")
        row["settlement_amount"] = abs(net)
    return {"standings": rows, "tie_matches": tie_matches, "net_total": round(sum(float(row["net"]) for row in rows), 2)}


def build_moneyball_settlement_preview(
    supabase: Any,
    *,
    club_id: str,
    player_ids: list[int],
    scores: list[dict[str, Any]],
    rating_context: str = "OVERALL",
    win_rate: float = 5.0,
    point_rate: float = 2.0,
) -> dict[str, Any]:
    preview = build_moneyball_preview(
        supabase,
        club_id=str(club_id),
        player_ids=player_ids,
        rating_context=rating_context,
        win_rate=win_rate,
        point_rate=point_rate,
    )
    settlement = compute_moneyball_settlement(
        matches=preview["matches"],
        scores=scores,
        win_rate=win_rate,
        point_rate=point_rate,
    )
    names = {int(row["id"]): str(row["name"]) for row in preview["players"]}
    for row in settlement["standings"]:
        row["player_name"] = names.get(int(row["player_id"]), f"Player {row['player_id']}")
    scored_rows = [
        {
            "row_id": str(row.get("row_id") or ""),
            "score_t1": _safe_int(row.get("score_t1")),
            "score_t2": _safe_int(row.get("score_t2")),
        }
        for row in scores or []
    ]
    fingerprint = stable_request_fingerprint(
        {"preview_fingerprint": preview["preview_fingerprint"], "scores": scored_rows, "settlement": settlement}
    )
    scores_by_row_id = {str(row.get("row_id") or ""): row for row in scored_rows}
    would_publish_count = sum(
        1
        for match in preview["matches"]
        if (
            (score := scores_by_row_id.get(str(match.get("row_id") or ""))) is not None
            and (int(score.get("score_t1") or 0) + int(score.get("score_t2") or 0)) > 0
            and int(score.get("score_t1") or 0) != int(score.get("score_t2") or 0)
        )
    )
    return {
        "ok": True,
        "mode": "moneyball_settlement_preview",
        "authority": "python_fastapi",
        "preview_fingerprint": preview["preview_fingerprint"],
        "settlement_fingerprint": fingerprint,
        "preview": preview,
        "settlement": settlement,
        "would_publish_count": would_publish_count,
    }


def submit_admin_moneyball(
    supabase: Any,
    *,
    club_id: str,
    player_ids: list[int],
    scores: list[dict[str, Any]],
    rating_context: str,
    league_name: str,
    week_tag: str,
    match_type: str,
    win_rate: float,
    point_rate: float,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_settlement_fingerprint: str | None = None,
    publish_context_prefix: str | None = None,
    source: str = "next_moneyball_admin",
) -> dict[str, Any]:
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM:
        raise ValueError(f"Type {CONFIRM} to save Moneyball matches.")
    settlement_preview = build_moneyball_settlement_preview(
        supabase,
        club_id=str(club_id),
        player_ids=player_ids,
        scores=scores,
        rating_context=rating_context,
        win_rate=win_rate,
        point_rate=point_rate,
    )
    if expected_settlement_fingerprint and str(expected_settlement_fingerprint) != str(settlement_preview["settlement_fingerprint"]):
        raise ValueError("Moneyball settlement preview is stale. Review the Python settlement again before official publish.")
    preview = settlement_preview["preview"]
    settlement = settlement_preview["settlement"]
    by_id = {str(score.get("row_id")): score for score in scores or []}
    match_payloads: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for match in preview["matches"]:
        score = by_id.get(str(match.get("row_id"))) or {}
        s1 = _safe_int(score.get("score_t1")) or 0
        s2 = _safe_int(score.get("score_t2")) or 0
        if (s1 + s2) <= 0 or s1 == s2:
            continue
        match_payloads.append({
            "t1_p1": match["t1_p1"], "t1_p2": match["t1_p2"], "t2_p1": match["t2_p1"], "t2_p2": match["t2_p2"],
            "s1": s1, "s2": s2, "score_t1": s1, "score_t2": s2,
            "date": now,
            "league": _clean_text(league_name, limit=120) or "Moneyball",
            "week_tag": _clean_text(week_tag, limit=120) or f"Moneyball {now[:10]}",
            "match_type": _clean_text(match_type, limit=120) or "Moneyball RR",
            "context_type": "moneyball",
            "context_id": (
                deterministic_match_context_id(
                    operation_key=_clean_text(publish_context_prefix, limit=80),
                    slot=int(match["match_index"]),
                )
                if _clean_text(publish_context_prefix, limit=80)
                else f"{_clean_text(week_tag, limit=120) or now[:10]}:match-{match['match_index']}"
            ),
            "is_popup": False,
        })
    if not match_payloads:
        raise ValueError("No valid non-tied scored Moneyball matches to save.")
    df_players_all, _active, df_leagues, _matches, df_meta, _badges, _player_badges, name_to_id, _id_to_name, _schema_degraded, _schema_reason = load_data(supabase, str(club_id), match_limit=5000)
    service_ctx = ServiceContext(supabase=supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, source=source)
    result = submit_match_batch(service_ctx, match_payloads, name_to_id=name_to_id, df_players_all=df_players_all, df_leagues=df_leagues, df_meta=df_meta)
    if not result.ok:
        raise ValueError("; ".join(result.errors) or "Unable to save Moneyball matches.")
    audit = build_activity_payload(
        club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="submit_moneyball_admin", entity_type="matches", entity_id="moneyball", before_json={},
        after_json={"source_client": "fastapi/nextjs", "match_count": len(match_payloads), "settlement": settlement, "result": result.data if isinstance(result.data, dict) else {}}, source_page=source, flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, audit)
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "moneyball_submit",
        "official_publish": True,
        "submitted_count": len(match_payloads),
        "settlement": settlement,
        "settlement_fingerprint": settlement_preview["settlement_fingerprint"],
        "preview": preview,
        "result": result.data,
        "match_context_ids": [str(row["context_id"]) for row in match_payloads],
        "correction": {
            "match_log_url": build_match_log_recovery_url(
                context_type="moneyball",
                context_ids=[
                    str(row.get("context_id") or "") for row in match_payloads
                ],
            ),
            "replay_history_url": "/admin/replay-history",
            "instructions": "Correct official Moneyball rows in Match Log, then run and verify Replay History. Never resubmit the night as a correction.",
        },
        "warnings": [write.warning] if write.warning else [],
    }
