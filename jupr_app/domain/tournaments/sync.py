from __future__ import annotations
# Match writes must go through match_pipeline.
import hashlib
from typing import Any

from jupr_app.domain.match_pipeline import record_match, update_match


def _build_tournament_match_idempotency_key(*, club_id: str, game_id: str, common_fields: dict[str, Any]) -> str:
    seed = "|".join(
        [
            str(club_id).strip(),
            str(game_id).strip(),
            str(common_fields.get("tournament_id") or "").strip(),
            str(common_fields.get("tournament_game_id") or "").strip(),
            str(common_fields.get("t1_p1") or ""),
            str(common_fields.get("t1_p2") or ""),
            str(common_fields.get("t2_p1") or ""),
            str(common_fields.get("t2_p2") or ""),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"tournament:{club_id}:{game_id}:{digest}"


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
        "score_t1": int(match_payload.get("s1") or 0),
        "score_t2": int(match_payload.get("s2") or 0),
        "date": match_payload.get("date"),
        "league": match_payload.get("league"),
        "week_tag": match_payload.get("week_tag"),
        "tournament_id": match_payload.get("tournament_id"),
        "tournament_game_id": match_payload.get("tournament_game_id"),
        "match_type": "PopUp" if bool(match_payload.get("is_popup", True)) else match_payload.get("match_type"),
        "t1_p1": team_a_player_ids[0] if len(team_a_player_ids) > 0 else None,
        "t1_p2": team_a_player_ids[1] if len(team_a_player_ids) > 1 else None,
        "t2_p1": team_b_player_ids[0] if len(team_b_player_ids) > 0 else None,
        "t2_p2": team_b_player_ids[1] if len(team_b_player_ids) > 1 else None,
    }

    if not existing_rows:
        idempotency_key = _build_tournament_match_idempotency_key(
            club_id=str(club_id),
            game_id=str(game_id),
            common_fields=common_fields,
        )
        record_match(
            supabase=supabase,
            club_id=str(club_id),
            match_payload={
                **common_fields,
                "context_type": "tournament",
                "context_id": str(match_payload.get("tournament_id") or game.get("tournament_id") or ""),
                "idempotency_key": idempotency_key,
            },
        )
        return

    existing = existing_rows[0]
    if int(existing.get("score_t1") or 0) == common_fields["score_t1"] and int(existing.get("score_t2") or 0) == common_fields["score_t2"]:
        return

    update_match(
        supabase=supabase,
        club_id=str(club_id),
        match_id=int(existing.get("id")),
        patch=common_fields,
    )
