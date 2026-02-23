from __future__ import annotations
# Tournament match sync writes are idempotent via direct upsert.
from typing import Any

from jupr_app.data.sb_write import sb_upsert


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
    """Persist tournament game scores idempotently to matches."""

    game_id = game.get("id")
    if not game_id:
        return

    _ = (name_to_id, df_players_all, df_leagues, df_meta)

    team_a_player_ids = [pid for pid in [match_payload.get("t1_p1"), match_payload.get("t1_p2")] if pid is not None]
    team_b_player_ids = [pid for pid in [match_payload.get("t2_p1"), match_payload.get("t2_p2")] if pid is not None]

    sb_upsert(
        supabase,
        "matches",
        {
            "club_id": str(club_id),
            "score_t1": int(match_payload.get("s1") or 0),
            "score_t2": int(match_payload.get("s2") or 0),
            "date": match_payload.get("date"),
            "league": match_payload.get("league"),
            "week_tag": match_payload.get("week_tag"),
            "tournament_id": match_payload.get("tournament_id"),
            "tournament_game_id": str(match_payload.get("tournament_game_id") or game_id),
            "match_type": "PopUp" if bool(match_payload.get("is_popup", True)) else match_payload.get("match_type"),
            "t1_p1": team_a_player_ids[0] if len(team_a_player_ids) > 0 else None,
            "t1_p2": team_a_player_ids[1] if len(team_a_player_ids) > 1 else None,
            "t2_p1": team_b_player_ids[0] if len(team_b_player_ids) > 0 else None,
            "t2_p2": team_b_player_ids[1] if len(team_b_player_ids) > 1 else None,
            "context_type": "tournament",
            "context_id": str(match_payload.get("tournament_id") or game.get("tournament_id") or ""),
        },
        conflict="club_id,tournament_game_id",
    )


def cleanup_duplicate_tournament_games(supabase: Any, tournament_id: str) -> Any:
    """Delete duplicate tournament games for a single tournament via SQL RPC."""
    return supabase.rpc("dedupe_tournament_games", {"t_id": str(tournament_id)}).execute()
