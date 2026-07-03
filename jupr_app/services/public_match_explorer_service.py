from __future__ import annotations

from typing import Any

from jupr_app.domain.constants import CAP_LOSER_GAIN_ELO, DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO
from jupr_app.domain.ratings import calculate_hybrid_elo
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


def _win_label(probability: float) -> str:
    if probability >= 0.70:
        return "Heavy Favorite"
    if probability >= 0.55:
        return "Favored"
    if probability >= 0.45:
        return "Toss-up"
    if probability >= 0.30:
        return "Underdog"
    return "Heavy Underdog"


def _active_league_name(row: dict[str, Any]) -> str | None:
    league_name = str(row.get("league_name") or "").strip()
    if not league_name or league_name.upper() == "OVERALL":
        return None
    status = str(row.get("status") or "").strip().lower()
    is_active = row.get("is_active")
    if is_active is False:
        return None
    if status and status not in {"active", "published", "live"}:
        return None
    return league_name


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

    if len(contexts) == 1:
        try:
            lr_rows = _safe_rows(
                supabase.table("league_ratings")
                .select("league_name,is_active")
                .eq("club_id", cid)
                .execute()
            )
        except Exception:
            lr_rows = []
        for row in lr_rows:
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
    expected_you = 1.0 / (1.0 + 10 ** ((opp_avg - you_avg) / 400.0))
    k_factor = _k_factor_for_context(supabase, club_id=str(club_id), context_name=context)

    delta_you_elo, delta_opp_elo = calculate_hybrid_elo(
        you_avg,
        opp_avg,
        int(score_you),
        int(score_opp),
        k_factor=int(k_factor),
        min_win_delta=float(MIN_WIN_DELTA_ELO),
        cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
    )

    you_players = [
        _public_player_payload(players_by_id[int(me)], ratings[int(me)]),
        _public_player_payload(players_by_id[int(partner)], ratings[int(partner)]),
    ]
    opponent_players = [
        _public_player_payload(players_by_id[int(opp1)], ratings[int(opp1)]),
        _public_player_payload(players_by_id[int(opp2)], ratings[int(opp2)]),
    ]

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
        "expected": {
            "you": float(expected_you),
            "opponents": float(1.0 - expected_you),
            "label": _win_label(float(expected_you)),
        },
        "score": {"you": int(score_you), "opponents": int(score_opp)},
        "rating_delta": {
            "you_team_elo": float(delta_you_elo),
            "opponent_team_elo": float(delta_opp_elo),
            "you_team_jupr": _jupr(float(delta_you_elo)),
            "opponent_team_jupr": _jupr(float(delta_opp_elo)),
        },
    }
