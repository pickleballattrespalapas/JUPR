from __future__ import annotations

from typing import Any

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.singles_match_processing import process_singles_matches
from jupr_app.services.admin_match_uploader_service import is_admin_match_uploader_enabled, is_api_audit_log_required


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


def _fetch_players(supabase: Any, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,rating,wins,losses,matches_played,singles_rating,singles_wins,singles_losses,singles_matches_played")
            .eq("club_id", str(club_id))
            .in_("id", sorted({int(pid) for pid in player_ids}))
            .execute()
        )
    except Exception:
        rows = []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None:
            result[int(pid)] = dict(row)
    return result


def _latest_match_id(supabase: Any, *, club_id: str) -> Any:
    try:
        rows = _safe_rows(supabase.table("matches").select("id").eq("club_id", str(club_id)).order("date", desc=True).limit(1).execute())
        return rows[0].get("id") if rows else None
    except Exception:
        return None


def _singles_feedback(*, before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]], player_ids: list[int], latest_match_id: Any = None) -> dict[str, Any]:
    affected = []
    ratings_updated = False
    for pid in player_ids:
        b = before.get(int(pid), {})
        a = after.get(int(pid), {})
        rb = b.get("singles_rating", b.get("rating"))
        ra = a.get("singles_rating", a.get("rating"))
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
                "matches_played_before": b.get("singles_matches_played"),
                "matches_played_after": a.get("singles_matches_played"),
            }
        )
    return {"ratings_updated": ratings_updated, "rating_type": "singles", "affected_players": affected, "latest_match_id": latest_match_id}


def _normalize_single_match(match: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(match, dict):
        raise ValueError("Singles match payload is required.")
    p1 = _safe_int(match.get("t1_p1") or match.get("player1_id") or match.get("player_a_id"))
    p2 = _safe_int(match.get("t2_p1") or match.get("player2_id") or match.get("player_b_id"))
    if p1 is None or p2 is None:
        raise ValueError("Select two singles players.")
    if int(p1) == int(p2):
        raise ValueError("A singles match requires two different players.")
    score_t1 = _safe_int(match.get("score_t1", match.get("s1")))
    score_t2 = _safe_int(match.get("score_t2", match.get("s2")))
    if score_t1 is None or score_t2 is None or score_t1 < 0 or score_t2 < 0:
        raise ValueError("Singles scores must be non-negative numbers.")
    if int(score_t1) == int(score_t2):
        raise ValueError("Singles matches cannot be tied.")
    if int(score_t1) + int(score_t2) <= 0:
        raise ValueError("Enter a non-zero singles score.")
    rating_scope = _clean_text(match.get("rating_scope"), limit=40)
    return {
        "date": _clean_text(match.get("date"), limit=80) or None,
        "league": _clean_text(match.get("league") or "Singles", limit=120) or "Singles",
        "match_type": _clean_text(match.get("match_type") or "Singles", limit=80) or "Singles",
        "week_tag": _clean_text(match.get("week_tag") or "Singles", limit=80),
        "t1_p1": int(p1),
        "t2_p1": int(p2),
        "score_t1": int(score_t1),
        "score_t2": int(score_t2),
        "match_format": "singles",
        "rating_scope": rating_scope or "",
        "context_type": _clean_text(match.get("context_type"), limit=80) or None,
        "context_id": match.get("context_id"),
    }


def submit_admin_singles_match(
    supabase: Any,
    *,
    club_id: str,
    match: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str = "next_match_uploader_singles",
) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        raise PermissionError("Next Match Uploader is disabled.")
    clean_match = _normalize_single_match(match)
    player_ids = [int(clean_match["t1_p1"]), int(clean_match["t2_p1"])]
    before_players = _fetch_players(supabase, club_id=str(club_id), player_ids=player_ids)
    (
        df_players_all,
        _df_players_active,
        _df_leagues,
        _df_matches,
        _df_meta,
        _df_badges,
        _df_player_badges,
        name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, str(club_id))
    result = process_singles_matches(
        [clean_match],
        supabase=supabase,
        club_id=str(club_id),
        name_to_id=name_to_id,
        df_players_all=df_players_all,
    )
    if int(result.get("inserted") or 0) != 1:
        raise RuntimeError("Singles match submission did not insert one official match row.")
    after_players = _fetch_players(supabase, club_id=str(club_id), player_ids=player_ids)
    feedback = _singles_feedback(before=before_players, after=after_players, player_ids=player_ids, latest_match_id=_latest_match_id(supabase, club_id=str(club_id)))
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="submit_singles_match_uploader",
        entity_type="matches",
        entity_id="singles",
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "match_format": "singles",
            "result_summary": result,
            "feedback": feedback,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "singles_match_uploader",
        "submitted_count": 1,
        "result": result,
        "feedback": feedback,
        "warnings": warnings,
    }
