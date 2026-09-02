from __future__ import annotations

from typing import Any

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.match_explorer import build_match_explorer_projection
from jupr_app.services.public_league_visibility import (
    ACTIVE_LEAGUE_VIEW,
    public_league_view,
)
from jupr_app.services.public_player_service import get_public_players

LEAGUE_RATING_SELECT = "player_id,league_name,rating,is_active"
LEAGUE_META_SELECT = "league_name,is_active,status,k_factor"


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _jupr(elo: float | None) -> float | None:
    if elo is None:
        return None
    return float(elo) / 400.0


def _active_league_name(row: dict[str, Any]) -> str | None:
    league_name = str(row.get("league_name") or "").strip()
    return league_name if public_league_view(row) == ACTIVE_LEAGUE_VIEW else None


def get_public_match_explorer_contexts(supabase: Any, *, club_id: str) -> list[str]:
    """Return public rating contexts available to the Match Explorer."""

    contexts = {"OVERALL"}
    cid = str(club_id).strip()

    try:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select(LEAGUE_META_SELECT)
            .eq("club_id", cid)
            .execute()
        )
    except Exception:
        rows = []

    for row in rows:
        league_name = _active_league_name(row)
        if league_name:
            contexts.add(league_name)

    return ["OVERALL"] + sorted(name for name in contexts if name != "OVERALL")


def _k_factor_for_context(supabase: Any, *, club_id: str, context_name: str) -> int:
    if str(context_name or "").strip().upper() == "OVERALL":
        return int(DEFAULT_K_FACTOR)
    try:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select(LEAGUE_META_SELECT)
            .eq("club_id", str(club_id))
            .eq("league_name", str(context_name).strip())
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    if rows:
        k_value = _safe_int(rows[0].get("k_factor"), int(DEFAULT_K_FACTOR))
        return int(k_value or DEFAULT_K_FACTOR)
    return int(DEFAULT_K_FACTOR)


def _league_rating_map(supabase: Any, *, club_id: str, context_name: str, player_ids: list[int]) -> dict[int, float]:
    context = str(context_name or "").strip()
    if not context or context.upper() == "OVERALL":
        return {}
    try:
        query = (
            supabase.table("league_ratings")
            .select(LEAGUE_RATING_SELECT)
            .eq("club_id", str(club_id))
            .eq("league_name", context)
        )
        if hasattr(query, "in_"):
            query = query.in_("player_id", player_ids)
        rows = _safe_rows(query.execute())
    except Exception:
        rows = []

    ratings: dict[int, float] = {}
    for row in rows:
        if row.get("is_active") is False:
            continue
        pid = _safe_int(row.get("player_id"))
        rating = _safe_float(row.get("rating"))
        if pid is not None and rating is not None:
            ratings[int(pid)] = float(rating)
    return ratings


def _public_player_payload(row: dict[str, Any], context_rating: float) -> dict[str, Any]:
    overall_rating = _safe_float(row.get("rating"), 1200.0) or 1200.0
    return {
        "id": _safe_int(row.get("id")) or row.get("id"),
        "name": str(row.get("name") or "Player"),
        "overall_rating": float(overall_rating),
        "overall_jupr": _jupr(float(overall_rating)),
        "context_rating": float(context_rating),
        "context_jupr": _jupr(float(context_rating)),
    }


def build_public_match_explorer_preview(
    supabase: Any,
    *,
    club_id: str,
    me: int,
    partner: int,
    opp1: int,
    opp2: int,
    context_name: str = "OVERALL",
    score_you: int = 11,
    score_opp: int = 9,
) -> dict[str, Any]:
    """Build a public-safe matchup preview without writing data."""

    player_ids = [int(me), int(partner), int(opp1), int(opp2)]
    if len(set(player_ids)) != 4:
        raise ValueError("Select four different players.")

    score_you = max(0, int(score_you))
    score_opp = max(0, int(score_opp))
    context = str(context_name or "OVERALL").strip() or "OVERALL"
    available_contexts = get_public_match_explorer_contexts(supabase, club_id=str(club_id))
    if context.upper() == "OVERALL":
        context = "OVERALL"
    elif context not in available_contexts:
        raise ValueError("Selected rating context is unavailable.")

    public_players = get_public_players(supabase, club_id=str(club_id), limit=1000)
    players_by_id: dict[int, dict[str, Any]] = {}
    for player in public_players:
        pid = _safe_int(player.get("id"))
        if pid is not None:
            players_by_id[int(pid)] = player

    missing = [pid for pid in player_ids if pid not in players_by_id]
    if missing:
        raise ValueError("One or more selected players are unavailable.")

    league_ratings = _league_rating_map(
        supabase,
        club_id=str(club_id),
        context_name=context,
        player_ids=player_ids,
    )

    def context_rating(pid: int) -> float:
        player = players_by_id[int(pid)]
        overall = _safe_float(player.get("rating"), 1200.0) or 1200.0
        return float(league_ratings.get(int(pid), overall))

    ratings = {pid: context_rating(pid) for pid in player_ids}
    you_avg = (ratings[int(me)] + ratings[int(partner)]) / 2.0
    opp_avg = (ratings[int(opp1)] + ratings[int(opp2)]) / 2.0
    k_factor = _k_factor_for_context(supabase, club_id=str(club_id), context_name=context)
    projection = build_match_explorer_projection(
        team_you_avg=float(you_avg),
        team_opponents_avg=float(opp_avg),
        score_you=int(score_you),
        score_opponents=int(score_opp),
        k_factor=int(k_factor),
    )

    you_players = [
        _public_player_payload(players_by_id[int(me)], ratings[int(me)]),
        _public_player_payload(players_by_id[int(partner)], ratings[int(partner)]),
    ]
    opponent_players = [
        _public_player_payload(players_by_id[int(opp1)], ratings[int(opp1)]),
        _public_player_payload(players_by_id[int(opp2)], ratings[int(opp2)]),
    ]

    player_impacts = []
    for role, player, delta_elo in (
        ("You", you_players[0], projection["rating_delta"]["you_team_elo"]),
        ("Partner", you_players[1], projection["rating_delta"]["you_team_elo"]),
        ("Opponent 1", opponent_players[0], projection["rating_delta"]["opponent_team_elo"]),
        ("Opponent 2", opponent_players[1], projection["rating_delta"]["opponent_team_elo"]),
    ):
        current_elo = float(player["context_rating"])
        player_impacts.append(
            {
                "role": role,
                "player": player,
                "current_rating": current_elo,
                "current_jupr": _jupr(current_elo),
                "projected_rating": current_elo + float(delta_elo),
                "projected_jupr": _jupr(current_elo + float(delta_elo)),
                "delta_elo": float(delta_elo),
                "delta_jupr": _jupr(float(delta_elo)),
            }
        )

    return {
        "context": {"name": context, "k_factor": int(k_factor)},
        "teams": {
            "you": {
                "average_rating": float(you_avg),
                "average_jupr": _jupr(float(you_avg)),
                "players": you_players,
            },
            "opponents": {
                "average_rating": float(opp_avg),
                "average_jupr": _jupr(float(opp_avg)),
                "players": opponent_players,
            },
        },
        **projection,
        "player_impacts": player_impacts,
    }
