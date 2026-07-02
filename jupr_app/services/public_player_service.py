from __future__ import annotations

from datetime import date, datetime
from typing import Any

PLAYER_SELECT = "id,club_id,name,rating,wins,losses,matches_played,active,last_game_at,inactive_at"
PLAYER_MINIMAL_SELECT = "id,club_id,name,rating,wins,losses,matches_played"
LEAGUE_RATINGS_SELECT = "id,club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active"
MATCH_SELECT = "*"


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_active_player(row: dict[str, Any]) -> bool:
    if row.get("inactive_at"):
        return False
    if "active" in row and row.get("active") is False:
        return False
    return True


def _player_base(row: dict[str, Any]) -> dict[str, Any]:
    wins = _int_or_none(row.get("wins")) or 0
    losses = _int_or_none(row.get("losses")) or 0
    matches_played = _int_or_none(row.get("matches_played"))
    if matches_played is None:
        matches_played = wins + losses
    return {
        "id": _int_or_none(row.get("id")) or row.get("id"),
        "club_id": str(row.get("club_id") or ""),
        "name": str(row.get("name") or "Player"),
        "rating": _float_or_none(row.get("rating")),
        "wins": wins,
        "losses": losses,
        "matches_played": matches_played,
        "is_active": _is_active_player(row),
        "last_game_at": _json_safe(row.get("last_game_at")),
    }


def _public_league_rating(row: dict[str, Any]) -> dict[str, Any]:
    wins = _int_or_none(row.get("wins")) or 0
    losses = _int_or_none(row.get("losses")) or 0
    matches_played = _int_or_none(row.get("matches_played"))
    if matches_played is None:
        matches_played = wins + losses
    return {
        "id": row.get("id"),
        "league_name": row.get("league_name"),
        "rating": _float_or_none(row.get("rating")),
        "starting_rating": _float_or_none(row.get("starting_rating")),
        "wins": wins,
        "losses": losses,
        "matches_played": matches_played,
        "is_active": row.get("is_active", True),
    }


def _fetch_players(supabase: Any, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("players")
            .select(PLAYER_SELECT)
            .eq("club_id", club_id)
            .execute()
        )
    except Exception:
        return _safe_rows(
            supabase.table("players")
            .select(PLAYER_MINIMAL_SELECT)
            .eq("club_id", club_id)
            .execute()
        )


def _fetch_player(supabase: Any, club_id: str, player_id: int | str) -> dict[str, Any] | None:
    try:
        row = _safe_first(
            supabase.table("players")
            .select(PLAYER_SELECT)
            .eq("club_id", club_id)
            .eq("id", player_id)
            .limit(1)
            .execute()
        )
    except Exception:
        row = _safe_first(
            supabase.table("players")
            .select(PLAYER_MINIMAL_SELECT)
            .eq("club_id", club_id)
            .eq("id", player_id)
            .limit(1)
            .execute()
        )
    return row


def _fetch_league_ratings(supabase: Any, club_id: str, player_id: int | str | None = None) -> list[dict[str, Any]]:
    query = supabase.table("league_ratings").select(LEAGUE_RATINGS_SELECT).eq("club_id", club_id)
    if player_id is not None:
        query = query.eq("player_id", player_id)
    try:
        return _safe_rows(query.execute())
    except Exception:
        return []


def _fetch_recent_matches(supabase: Any, club_id: str, *, limit: int = 300) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("matches")
            .select(MATCH_SELECT)
            .eq("club_id", club_id)
            .order("date", desc=True)
            .limit(int(limit))
            .execute()
        )
    except Exception:
        try:
            return _safe_rows(
                supabase.table("matches")
                .select(MATCH_SELECT)
                .eq("club_id", club_id)
                .order("id", desc=True)
                .limit(int(limit))
                .execute()
            )
        except Exception:
            return []


def _match_includes_player(row: dict[str, Any], player_id: int | str) -> bool:
    pid = str(player_id)
    return any(str(row.get(col)) == pid for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))


def _match_player_ids(row: dict[str, Any]) -> list[Any]:
    return [row.get(col) for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if row.get(col) is not None]


def _public_match(row: dict[str, Any], name_by_id: dict[str, str]) -> dict[str, Any]:
    t1_ids = [row.get("t1_p1"), row.get("t1_p2")]
    t2_ids = [row.get("t2_p1"), row.get("t2_p2")]
    score_t1 = _int_or_none(row.get("score_t1"))
    score_t2 = _int_or_none(row.get("score_t2"))
    winner = None
    if score_t1 is not None and score_t2 is not None and score_t1 != score_t2:
        winner = "team_1" if score_t1 > score_t2 else "team_2"
    return {
        "id": row.get("id"),
        "club_id": row.get("club_id"),
        "date": _json_safe(row.get("date")),
        "league": row.get("league"),
        "week_tag": row.get("week_tag"),
        "match_type": row.get("match_type"),
        "rating_scope": row.get("rating_scope"),
        "context_type": row.get("context_type"),
        "context_id": row.get("context_id"),
        "team_1": [
            {"id": pid, "name": name_by_id.get(str(pid), "Player")} for pid in t1_ids if pid is not None
        ],
        "team_2": [
            {"id": pid, "name": name_by_id.get(str(pid), "Player")} for pid in t2_ids if pid is not None
        ],
        "score_t1": score_t1,
        "score_t2": score_t2,
        "winner": winner,
        "elo_delta": _float_or_none(row.get("elo_delta")),
    }


def get_public_players(
    supabase: Any,
    *,
    club_id: str,
    search: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    cid = str(club_id).strip()
    rows = [_player_base(row) for row in _fetch_players(supabase, cid)]
    query = str(search or "").strip().casefold()
    if query:
        rows = [row for row in rows if query in str(row.get("name") or "").casefold()]
    rows.sort(key=lambda row: (not bool(row.get("is_active")), str(row.get("name") or "").casefold()))
    return rows[: max(1, min(int(limit or 500), 1000))]


def get_public_player_profile(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | str,
    recent_match_limit: int = 12,
) -> dict[str, Any] | None:
    cid = str(club_id).strip()
    row = _fetch_player(supabase, cid, player_id)
    if not row:
        return None
    player = _player_base(row)
    league_ratings = [_public_league_rating(r) for r in _fetch_league_ratings(supabase, cid, player_id)]
    league_ratings.sort(key=lambda r: str(r.get("league_name") or "").casefold())

    players = _fetch_players(supabase, cid)
    name_by_id = {str(p.get("id")): str(p.get("name") or "Player") for p in players}
    matches = [m for m in _fetch_recent_matches(supabase, cid, limit=400) if _match_includes_player(m, player_id)]
    public_matches = [_public_match(m, name_by_id) for m in matches[: max(1, min(int(recent_match_limit), 50))]]
    return {
        "player": player,
        "league_ratings": league_ratings,
        "recent_matches": public_matches,
    }


def get_public_matches(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    cid = str(club_id).strip()
    players = _fetch_players(supabase, cid)
    name_by_id = {str(p.get("id")): str(p.get("name") or "Player") for p in players}
    rows = _fetch_recent_matches(supabase, cid, limit=max(int(limit or 100), 300 if player_id is not None else int(limit or 100)))
    if player_id is not None:
        rows = [row for row in rows if _match_includes_player(row, player_id)]
    public_rows = [_public_match(row, name_by_id) for row in rows]
    return public_rows[: max(1, min(int(limit or 100), 500))]
