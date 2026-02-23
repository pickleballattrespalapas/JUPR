from __future__ import annotations

import logging
from typing import Any

from jupr_app.data.sb_write import sb_upsert
from jupr_app.domain.gamification.badge_types import BadgeCandidate


logger = logging.getLogger(__name__)

PODIUM_BADGE_MAP = {
    1: "tournament_champion",
    2: "tournament_runner_up",
    3: "tournament_third_place",
}


def fetch_tournament_podium(supabase: Any, tournament_id: str) -> list[dict[str, Any]]:
    if supabase is None or not tournament_id:
        return []
    try:
        resp = (
            supabase.table("tournament_podium")
            .select("placement,team_id,source")
            .eq("tournament_id", tournament_id)
            .order("placement", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        logger.exception("Failed to fetch tournament podium", extra={"tournament_id": tournament_id})
        return []


def upsert_tournament_podium(
    supabase: Any,
    club_id: str,
    tournament_id: str,
    payload: list[dict[str, Any]],
) -> None:
    if supabase is None or not tournament_id or not payload:
        return
    assert club_id, "club_id must be present for tournament writes"
    for row in payload:
        if "club_id" not in row:
            # Explicit club_id for tenant isolation (RLS + multi-club safety)
            row["club_id"] = str(club_id)

    try:
        sb_upsert(
            supabase,
            "tournament_podium",
            payload,
            conflict="club_id,tournament_id,placement",
        )
    except Exception:
        logger.exception("Failed to upsert tournament podium", extra={"tournament_id": tournament_id})


def build_tournament_podium_candidates(
    ctx: Any, tournament_id: str, tournament_name: str | None
) -> list[BadgeCandidate]:
    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id or not tournament_id:
        return []

    podium_rows = fetch_tournament_podium(supabase, tournament_id)
    if not podium_rows:
        return []

    team_ids = sorted({row.get("team_id") for row in podium_rows if row.get("team_id")})
    if not team_ids:
        return []

    try:
        teams_resp = (
            supabase.table("tournament_teams")
            .select("id,team_number,player1_id,player2_id")
            .in_("id", team_ids)
            .execute()
        )
        teams = teams_resp.data or []
    except Exception:
        logger.exception("Failed to fetch tournament teams", extra={"tournament_id": tournament_id})
        return []

    teams_by_id = {t["id"]: t for t in teams}
    candidates: list[BadgeCandidate] = []
    for row in podium_rows:
        placement = int(row.get("placement", 0) or 0)
        badge_id = PODIUM_BADGE_MAP.get(placement)
        team_id = row.get("team_id")
        if not badge_id or not team_id:
            continue
        team = teams_by_id.get(team_id)
        if not team:
            continue
        for player_id in [team.get("player1_id"), team.get("player2_id")]:
            if not player_id:
                continue
            candidates.append(
                BadgeCandidate(
                    badge_id=badge_id,
                    player_id=int(player_id),
                    club_id=club_id,
                    context_type="tournament",
                    context_id=f"{tournament_id}:podium:{placement}",
                    match_id=None,
                    value_json={
                        "tournament_id": tournament_id,
                        "tournament_name": tournament_name,
                        "placement": placement,
                        "team_id": team_id,
                        "team_number": team.get("team_number"),
                    },
                    value_num=float(placement),
                )
            )

    return candidates


def award_tournament_trophies_from_podium(ctx: Any, tournament_id: str, tournament_name: str | None) -> list[BadgeCandidate]:
    from jupr_app.domain.gamification.ensure_badges import ensure_badges

    return ensure_badges(
        ctx,
        tournament_id=tournament_id,
        tournament_name=tournament_name,
        award_timing="manual",
        status="live",
    )
