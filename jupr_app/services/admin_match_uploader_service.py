from __future__ import annotations

import os
from typing import Any

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.context import ServiceContext
from jupr_app.services.match_service import submit_match_batch

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
MAX_MATCH_UPLOADER_BATCH_ROWS = 200


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_match_uploader_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _score_entry_player_ids(matches: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for match in matches or []:
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _safe_int(match.get(key))
            if pid is not None and int(pid) not in ids:
                ids.append(int(pid))
    return ids


def _fetch_players(supabase: Any, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = _safe_rows(supabase.table("players").select("id,name,rating,wins,losses,matches_played").eq("club_id", str(club_id)).execute())
    except Exception:
        return {}
    allowed = {int(pid) for pid in player_ids}
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None and int(pid) in allowed:
            result[int(pid)] = dict(row)
    return result


def _latest_match_id(supabase: Any, *, club_id: str) -> Any:
    try:
        rows = _safe_rows(supabase.table("matches").select("id").eq("club_id", str(club_id)).order("date", desc=True).limit(1).execute())
        return rows[0].get("id") if rows else None
    except Exception:
        return None


def _score_feedback(*, before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]], player_ids: list[int], latest_match_id: Any = None) -> dict[str, Any]:
    affected = []
    ratings_updated = False
    for pid in player_ids:
        b = before.get(int(pid), {})
        a = after.get(int(pid), {})
        rb = b.get("rating")
        ra = a.get("rating")
        try:
            delta = None if rb is None or ra is None else float(ra) - float(rb)
        except Exception:
            delta = None
        if delta not in (None, 0):
            ratings_updated = True
        affected.append(
            {
                "id": int(pid),
                "name": a.get("name") or b.get("name") or f"Player {int(pid)}",
                "rating_before": rb,
                "rating_after": ra,
                "rating_delta": delta,
                "matches_played_before": b.get("matches_played"),
                "matches_played_after": a.get("matches_played"),
            }
        )
    return {"ratings_updated": ratings_updated, "affected_players": affected, "latest_match_id": latest_match_id}


def _normalize_match(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    pids = {key: _safe_int(row.get(key)) for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")}
    if any(value is None for value in pids.values()):
        return None
    score_t1 = _safe_int(row.get("score_t1", row.get("s1"))) or 0
    score_t2 = _safe_int(row.get("score_t2", row.get("s2"))) or 0
    if (score_t1 + score_t2) <= 0:
        return None
    match_type = _clean_text(row.get("match_type") or "Live Match", limit=80)
    league = _clean_text(row.get("league") or ("POPUP" if match_type == "PopUp" else "Open"), limit=120)
    payload = {
        "date": _clean_text(row.get("date"), limit=80) or None,
        "league": league,
        "match_type": match_type,
        "week_tag": _clean_text(row.get("week_tag"), limit=80),
        "t1_p1": int(pids["t1_p1"] or 0),
        "t1_p2": int(pids["t1_p2"] or 0),
        "t2_p1": int(pids["t2_p1"] or 0),
        "t2_p2": int(pids["t2_p2"] or 0),
        "score_t1": int(score_t1),
        "score_t2": int(score_t2),
        "is_popup": bool(row.get("is_popup") or match_type == "PopUp"),
        "context_type": _clean_text(row.get("context_type"), limit=80) or None,
        "context_id": row.get("context_id"),
    }
    rating_scope = _clean_text(row.get("rating_scope"), limit=40)
    if rating_scope:
        payload["rating_scope"] = rating_scope
    return payload


def _normalize_batch(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [_normalize_match(row) for row in (matches or [])]
    clean_rows = [row for row in clean if row]
    if not clean_rows:
        raise ValueError("No valid match rows were provided.")
    if len(clean_rows) > MAX_MATCH_UPLOADER_BATCH_ROWS:
        raise ValueError(f"No more than {MAX_MATCH_UPLOADER_BATCH_ROWS} matches can be submitted at once.")
    return clean_rows


def build_admin_match_uploader_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "submit_endpoint": None,
            "max_batch_rows": MAX_MATCH_UPLOADER_BATCH_ROWS,
            "league_options": ["Open", "POPUP"],
            "week_tag_options": [f"Week {idx}" for idx in range(1, 13)] + ["Playoffs", "Finals", "Event"],
            "warnings": ["Next Match Uploader is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER on FastAPI for the closed-club pilot."],
        }
    league_options = ["Open", "POPUP"]
    try:
        rows = _safe_rows(supabase.table("leagues_metadata").select("league_name,is_active,status").eq("club_id", str(club_id)).execute())
        names = sorted({_clean_text(row.get("league_name"), limit=120) for row in rows if _clean_text(row.get("league_name"), limit=120)})
        active_names = [name for name in names if name.upper() != "OVERALL"]
        if active_names:
            league_options = active_names + (["POPUP"] if "POPUP" not in active_names else [])
    except Exception:
        pass
    return {
        "enabled": True,
        "status": "ready_for_manual_batch",
        "submit_endpoint": "/admin/clubs/{club_id}/match-uploader/batch",
        "max_batch_rows": MAX_MATCH_UPLOADER_BATCH_ROWS,
        "league_options": league_options,
        "week_tag_options": [f"Week {idx}" for idx in range(1, 21)] + ["Playoffs", "Finals", "Event"],
        "warnings": [],
    }


def submit_admin_match_uploader_batch(
    supabase: Any,
    *,
    club_id: str,
    matches: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    source: str = "next_match_uploader",
) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        raise PermissionError("Next Match Uploader is disabled.")
    clean_matches = _normalize_batch(matches)
    player_ids = _score_entry_player_ids(clean_matches)
    before_players = _fetch_players(supabase, club_id=str(club_id), player_ids=player_ids)
    (
        df_players_all,
        _df_players_active,
        df_leagues,
        _df_matches,
        df_meta,
        _df_badges,
        _df_player_badges,
        name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, str(club_id))
    service_ctx = ServiceContext(
        supabase=supabase,
        club_id=str(club_id),
        source=source,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
    )
    result = submit_match_batch(
        service_ctx,
        clean_matches,
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    if not result.ok:
        raise ValueError("; ".join(result.errors) or "Unable to submit match batch")
    after_players = _fetch_players(supabase, club_id=str(club_id), player_ids=player_ids)
    latest_match_id = _latest_match_id(supabase, club_id=str(club_id))
    feedback = _score_feedback(before=before_players, after=after_players, player_ids=player_ids, latest_match_id=latest_match_id)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="submit_match_uploader_batch",
        entity_type="matches",
        entity_id="batch",
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "match_count": len(clean_matches),
            "result_summary": result.data if isinstance(result.data, dict) else {"ok": True},
            "feedback": feedback,
        },
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "match_uploader_batch",
        "submitted_count": len(clean_matches),
        "result": result.data,
        "feedback": feedback,
        "warnings": warnings,
    }
