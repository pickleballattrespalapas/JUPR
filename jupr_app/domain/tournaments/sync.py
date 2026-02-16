from __future__ import annotations
from typing import Any
from jupr_app.domain.match_processing import process_matches


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
    """
    Idempotent sync:
    - Deletes any existing match row for this tournament game
    - Rebuilds canonical match entry
    """

    game_id = game.get("id")
    if not game_id:
        return

    # Delete existing canonical match
    supabase.table("matches") \
        .delete() \
        .eq("tournament_game_id", game_id) \
        .execute()

    # Insert fresh canonical match
    process_matches(
        [match_payload],
        supabase=supabase,
        club_id=str(club_id),
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
