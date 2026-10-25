"""Read current official scores without rewriting the retained publish request."""

from __future__ import annotations

from typing import Any


SCORE_FIELDS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2")


def read_league_round_scores(supabase: Any, *, club_id: str, round_row: dict) -> dict:
    result = dict(round_row)
    saved = [dict(row) for row in round_row.get("matches_json") or []]
    result["matches_json"] = saved
    result["score_readback_status"] = "snapshot"
    contexts = [str(row.get("context_id") or "") for row in saved]
    if not saved or round_row.get("status") != "submitted":
        return result
    if not all(contexts) or len(set(contexts)) != len(contexts):
        result["score_readback_message"] = "Saved scores shown. This older round has no complete official match references."
        return result
    try:
        rows = []
        for start in range(0, len(contexts), 100):
            response = (
                supabase.table("matches").select("*")
                .eq("club_id", str(club_id)).eq("context_type", "league_live_session")
                .in_("context_id", contexts[start:start + 100]).execute()
            )
            rows.extend(response.data or [])
        by_context = {str(row.get("context_id")): row for row in rows}
        if len(rows) != len(contexts) or len(by_context) != len(contexts):
            raise ValueError("Some official results are missing or duplicated.")
        saved_ids = {str(value) for value in round_row.get("submitted_match_ids") or []}
        if saved_ids and saved_ids != {str(row.get("id")) for row in rows}:
            raise ValueError("Official match references no longer match this round.")
        current = []
        corrected_count = 0
        blocked = False
        for original, context in zip(saved, contexts):
            official = by_context[context]
            corrected = any(official.get(key) != original.get(key) for key in SCORE_FIELDS)
            corrected_count += int(corrected)
            excluded = official.get("is_active") is False or bool(official.get("deleted_at"))
            scores = [official.get(key) for key in ("score_t1", "score_t2")]
            valid = all(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and int(value) == value for value in scores)
            valid = valid and scores[0] != scores[1]
            blocked = blocked or excluded or not valid
            current.append({
                **original,
                **{key: official.get(key) for key in SCORE_FIELDS},
                "official_match_id": official.get("id"),
                "official_row_version": official.get("row_version"),
                "score_corrected": corrected,
                "score_excluded": excluded or not valid,
            })
        result.update({
            "matches_json": current,
            "score_readback_status": "blocked" if blocked else "current",
            "score_correction_count": corrected_count,
            "score_readback_message": "An official result was excluded or is incomplete. Review it in Match Log before applying movement." if blocked else None,
        })
    except Exception:
        # Keep the round visible during a read failure, but never treat the old
        # snapshot as verified or allow it to drive a new movement write.
        result["score_readback_status"] = "unavailable"
        result["score_readback_message"] = "Current official scores could not be verified. Saved scores are shown; refresh before correcting scores or applying movement."
    return result
