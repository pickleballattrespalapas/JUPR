from __future__ import annotations

from typing import Any


PUBLIC_LEADERBOARD_FIELDS = {
    "club_id",
    "league_name",
    "player_id",
    "player_name",
    "rating",
    "rating_jupr",
    "wins",
    "losses",
    "matches_played",
    "is_active",
    "rank_position",
    "updated_at",
}


def _normalize_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        clean = {k: row.get(k) for k in PUBLIC_LEADERBOARD_FIELDS if k in row}
        clean.setdefault("rating_jupr", clean.get("rating"))
        clean.setdefault("matches_played", (clean.get("wins") or 0) + (clean.get("losses") or 0))
        normalized.append(clean)
    return normalized


def _fetch_from_view(supabase: Any, club_id: str, league_name: str | None) -> list[dict[str, Any]]:
    query = (
        supabase.table("public_leaderboards")
        .select(
            "club_id,league_name,player_id,player_name,rating,rating_jupr,wins,losses,matches_played,is_active,rank_position,updated_at"
        )
        .eq("club_id", club_id)
    )
    if league_name:
        query = query.eq("league_name", league_name)
    return query.order("rank_position", desc=False).execute().data or []


def _fetch_fallback(supabase: Any, club_id: str, league_name: str | None) -> list[dict[str, Any]]:
    query = (
        supabase.table("league_ratings")
        .select("club_id,league_name,player_id,rating,wins,losses,matches_played,is_active")
        .eq("club_id", club_id)
    )
    if league_name:
        query = query.eq("league_name", league_name)

    ratings = query.execute().data or []
    players = (
        supabase.table("players")
        .select("id,name")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    name_by_id = {p.get("id"): p.get("name") for p in players}

    enriched = []
    for row in ratings:
        pid = row.get("player_id")
        enriched.append(
            {
                "club_id": row.get("club_id", club_id),
                "league_name": row.get("league_name"),
                "player_id": pid,
                "player_name": name_by_id.get(pid, "Player"),
                "rating": row.get("rating"),
                "rating_jupr": row.get("rating"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "matches_played": row.get("matches_played"),
                "is_active": row.get("is_active"),
                "updated_at": None,
            }
        )

    sorted_rows = sorted(enriched, key=lambda r: (-(float(r.get("rating") or 0.0)), str(r.get("player_name") or "")))
    for idx, row in enumerate(sorted_rows, start=1):
        row["rank_position"] = idx
    return sorted_rows


def get_public_leaderboard(supabase: Any, club_id: str, league_name: str | None = None) -> list[dict[str, Any]]:
    """Read public leaderboard rows from view with a table fallback.

    This is intentionally safe for public APIs and excludes private player data.
    """
    cid = str(club_id).strip()
    if not cid:
        return []

    try:
        return _normalize_rows(_fetch_from_view(supabase, cid, league_name))
    except Exception:
        return _normalize_rows(_fetch_fallback(supabase, cid, league_name))
