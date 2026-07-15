from __future__ import annotations

from typing import Any, Callable


OPTIONAL_MATCH_INSERT_COLUMNS = ("rating_scope", "rating_bonus_elo", "rating_bonus_reason", "match_format")


def _safe_positive_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 0.0


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_match_format(value: Any) -> str:
    clean = str(value or "").strip().lower()
    if clean == "singles":
        return "singles"
    return "doubles"


def build_match_row(
    *,
    club_id: str,
    dt_val: str,
    league_name: str,
    pids: tuple[int | None, int | None, int | None, int | None],
    scores: tuple[int, int],
    stored_elo_delta: float,
    match_type: str,
    week_tag: str,
    start_ratings: tuple[float | None, float | None, float | None, float | None],
    end_ratings: tuple[float | None, float | None, float | None, float | None],
    context: dict[str, Any],
    rating_scope: str,
    match_format: str | None = None,
) -> dict[str, Any]:
    p1, p2, p3, p4 = pids
    s1, s2 = scores
    ro1, ro2, ro3, ro4 = start_ratings
    end_r1, end_r2, end_r3, end_r4 = end_ratings
    row = {
        "club_id": club_id,
        "date": dt_val,
        "league": league_name,
        "t1_p1": p1,
        "t1_p2": p2,
        "t2_p1": p3,
        "t2_p2": p4,
        "score_t1": s1,
        "score_t2": s2,
        "elo_delta": float(stored_elo_delta),
        "match_type": match_type,
        "week_tag": week_tag,
        "t1_p1_r": _maybe_float(ro1),
        "t1_p2_r": _maybe_float(ro2),
        "t2_p1_r": _maybe_float(ro3),
        "t2_p2_r": _maybe_float(ro4),
        "t1_p1_r_end": _maybe_float(end_r1),
        "t1_p2_r_end": _maybe_float(end_r2),
        "t2_p1_r_end": _maybe_float(end_r3),
        "t2_p2_r_end": _maybe_float(end_r4),
        "context_type": context.get("context_type"),
        "context_id": context.get("context_id"),
        "tournament_id": context.get("tournament_id"),
        "tournament_game_id": context.get("tournament_game_id"),
        "rating_scope": rating_scope,
        "match_format": _normalize_match_format(match_format or context.get("match_format")),
    }
    bonus_elo = _safe_positive_float(context.get("rating_bonus_elo", context.get("winner_bonus_elo")))
    if bonus_elo > 0:
        row["rating_bonus_elo"] = float(bonus_elo)
        row["rating_bonus_reason"] = str(
            context.get("rating_bonus_reason") or context.get("winner_bonus_reason") or "winner_bonus"
        ).strip()[:200]
    return row


def insert_match_chunks_with_rating_scope_fallback(*, db_matches: list[dict[str, Any]], supabase, sb_retry: Callable):
    unsupported_optional_columns: set[str] = set()

    def _strip_unsupported(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not unsupported_optional_columns:
            return chunk
        return [
            {key: value for key, value in row.items() if key not in unsupported_optional_columns}
            for row in chunk
        ]

    def _mark_unsupported_from_text(text: str) -> bool:
        lower = str(text or "").lower()
        changed = False
        if "matches_rating_scope_check" in lower and "rating_scope" not in unsupported_optional_columns:
            unsupported_optional_columns.add("rating_scope")
            changed = True
        if "column" not in lower and "schema cache" not in lower:
            return changed
        for column in OPTIONAL_MATCH_INSERT_COLUMNS:
            if column in lower and column not in unsupported_optional_columns:
                unsupported_optional_columns.add(column)
                changed = True
        return changed

    def _insert_match_chunk(chunk: list[dict[str, Any]]):
        for _attempt in range(len(OPTIONAL_MATCH_INSERT_COLUMNS) + 1):
            payload = _strip_unsupported(chunk)
            try:
                result = sb_retry(lambda payload=payload: supabase.table("matches").insert(payload).execute())
            except Exception as exc:
                if _mark_unsupported_from_text(str(exc)):
                    continue
                raise
            error_obj = getattr(result, "error", None) if result is not None else None
            if error_obj and _mark_unsupported_from_text(str(error_obj)):
                continue
            return result
        payload = _strip_unsupported(chunk)
        return sb_retry(lambda payload=payload: supabase.table("matches").insert(payload).execute())

    for i in range(0, len(db_matches), 300):
        chunk = db_matches[i : i + 300]
        insert_result = _insert_match_chunk(chunk)
        inserted_rows = getattr(insert_result, "data", None) if insert_result is not None else None
        if not inserted_rows:
            error_obj = getattr(insert_result, "error", None) if insert_result is not None else None
            sample = chunk[0] if chunk else {}
            debug_payload = {
                "club_id": sample.get("club_id"),
                "league": sample.get("league"),
                "match_format": sample.get("match_format"),
                "t1_p1": sample.get("t1_p1"),
                "t1_p2": sample.get("t1_p2"),
                "t2_p1": sample.get("t2_p1"),
                "t2_p2": sample.get("t2_p2"),
                "score_t1": sample.get("score_t1"),
                "score_t2": sample.get("score_t2"),
                "tournament_game_id": sample.get("tournament_game_id"),
            }
            raise RuntimeError("Failed to insert match rows into matches table; " f"error={error_obj!r}; sample_row={debug_payload}")
