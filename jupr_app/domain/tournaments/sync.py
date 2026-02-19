from __future__ import annotations
from typing import Any

from jupr_app.domain.match_pipeline import record_match, update_match


def sync_tournament_game_to_match(
    *,
    supabase: Any,
    club_id: str,
    game: dict,
    match_payload: dict,
    name_to_id,
    df_players_all,
    df_leagues,
    df_meta,
) -> None:
    """Persist tournament game scores through the canonical match pipeline."""

    game_id = game.get("id")
    if not game_id:
        return

    _ = (name_to_id, df_players_all, df_leagues, df_meta)

    team_a_player_ids = [pid for pid in [match_payload.get("t1_p1"), match_payload.get("t1_p2")] if pid is not None]
    team_b_player_ids = [pid for pid in [match_payload.get("t2_p1"), match_payload.get("t2_p2")] if pid is not None]

    existing_resp = (
        supabase.table("matches")
        .select("id,score_t1,score_t2")
        .eq("club_id", str(club_id))
        .eq("tournament_game_id", str(game_id))
        .limit(1)
        .execute()
    )
    existing_rows = getattr(existing_resp, "data", None) or []

    common_fields = {
        "score_a": int(match_payload.get("s1") or 0),
        "score_b": int(match_payload.get("s2") or 0),
        "played_at": match_payload.get("date"),
        "league": match_payload.get("league"),
        "week_tag": match_payload.get("week_tag"),
        "tournament_id": match_payload.get("tournament_id"),
        "tournament_game_id": match_payload.get("tournament_game_id"),
        "is_popup": bool(match_payload.get("is_popup", True)),
    }

    if not existing_rows:
        record_match(
            supabase,
            club_id=str(club_id),
            team_a_player_ids=team_a_player_ids,
            team_b_player_ids=team_b_player_ids,
            context_type="tournament",
            context_id=str(match_payload.get("tournament_id") or game.get("tournament_id") or ""),
            source="tournament",
            **common_fields,
        )
        return

    existing = existing_rows[0]
    if int(existing.get("score_t1") or 0) == common_fields["score_a"] and int(existing.get("score_t2") or 0) == common_fields["score_b"]:
        return

    update_match(
        supabase,
        match_id=str(existing.get("id")),
        **common_fields,
    )
