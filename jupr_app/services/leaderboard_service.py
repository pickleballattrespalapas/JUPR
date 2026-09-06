from __future__ import annotations

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.gamification.presentation import badge_category

import re
from typing import Any

from jupr_app.services.public_league_visibility import (
    ACTIVE_LEAGUE_VIEW,
    normalize_public_league_view,
    public_league_view,
)


OVERALL_SCOPE = "OVERALL"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_BADGES_PER_PLAYER = 3

PUBLIC_LEADERBOARD_FIELDS = {
    "rank",
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

PLAYER_SELECT = (
    "id,club_id,name,rating,starting_rating,wins,losses,matches_played,"
    "active,inactive_at"
)
PLAYER_SELECT_FALLBACK = "id,club_id,name,rating,wins,losses,matches_played,active,inactive_at"
LEAGUE_RATING_SELECT = (
    "club_id,league_name,player_id,rating,starting_rating,wins,losses,"
    "matches_played,is_active"
)
LEAGUE_RATING_SELECT_FALLBACK = (
    "club_id,league_name,player_id,rating,wins,losses,matches_played,is_active"
)
LEAGUE_META_SELECT = "club_id,league_name,is_active,status,min_games"
LEAGUE_META_SELECT_FALLBACK = "club_id,league_name,is_active,min_games"
PLAYER_BADGE_SELECT = "club_id,player_id,badge_id,earned_at,revoked_at"
BADGE_SELECT = "badge_id,name,prestige,category,icon_key,rarity,state,is_active"

_HTML_TAG_RE = re.compile(r"<[^>]*>")


class LeaderboardDataUnavailable(RuntimeError):
    """Raised when the server-only leaderboard source cannot be read."""


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _query_rows(
    supabase: Any,
    table_name: str,
    select_options: tuple[str, ...],
    *,
    club_id: str | None = None,
    required: bool = False,
) -> list[dict[str, Any]]:
    """Read a server-side table using a narrow projection and schema fallback."""

    last_error: Exception | None = None
    for columns in select_options:
        try:
            def query_factory():
                query = supabase.table(table_name).select(columns)
                return query.eq("club_id", str(club_id)) if club_id is not None else query
            return read_all_rows(query_factory, order="badge_id" if table_name == "badges" else "id")
        except Exception as exc:
            last_error = exc
            continue
    if required and last_error is not None:
        raise LeaderboardDataUnavailable(f"Unable to read the {table_name} leaderboard projection.") from last_error
    return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _jupr(value: Any) -> float | None:
    rating = _safe_float(value)
    if rating is None:
        return None
    # Canonical persisted ratings are Elo-scaled. The <=20 branch keeps older
    # public views that already projected JUPR from being divided twice.
    return float(rating) / 400.0 if abs(float(rating)) > 20.0 else float(rating)


def _plain_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", text).replace("<", "").replace(">", "")).strip() or None


def _player_is_active(row: dict[str, Any]) -> bool:
    if row.get("inactive_at"):
        return False
    if row.get("active") is False or row.get("is_active") is False:
        return False
    return True


def _fetch_players(supabase: Any, club_id: str) -> list[dict[str, Any]]:
    return _query_rows(
        supabase,
        "players",
        (PLAYER_SELECT, PLAYER_SELECT_FALLBACK, "id,club_id,name,rating,active"),
        club_id=club_id,
        required=True,
    )


def _fetch_league_ratings(supabase: Any, club_id: str) -> list[dict[str, Any]]:
    return _query_rows(
        supabase,
        "league_ratings",
        (LEAGUE_RATING_SELECT, LEAGUE_RATING_SELECT_FALLBACK),
        club_id=club_id,
        required=True,
    )


def _scope_options(
    supabase: Any,
    *,
    club_id: str,
    league_view: str,
) -> list[dict[str, Any]]:
    meta_rows = _query_rows(
        supabase,
        "leagues_metadata",
        (LEAGUE_META_SELECT, LEAGUE_META_SELECT_FALLBACK),
        club_id=club_id,
    )
    metadata: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        if public_league_view(row) != league_view:
            continue
        name = str(row.get("league_name") or "").strip()
        metadata[name] = {
            "name": name,
            "label": name,
            "min_games": max(0, _safe_int(row.get("min_games"), 0) or 0),
        }

    league_scopes = [metadata[name] for name in sorted(metadata, key=str.casefold)]
    if league_view == ACTIVE_LEAGUE_VIEW:
        return [
            {"name": OVERALL_SCOPE, "label": "Overall", "min_games": 0},
            *league_scopes,
        ]
    return league_scopes


def _badge_map(supabase: Any, *, club_id: str) -> dict[str, list[dict[str, Any]]]:
    earned_rows = _query_rows(
        supabase,
        "player_badges",
        (PLAYER_BADGE_SELECT,),
        club_id=club_id,
        required=True,
    )
    if not earned_rows:
        return {}
    badge_rows = _query_rows(
        supabase,
        "badges",
        (BADGE_SELECT, "badge_id,name,prestige,category,icon_key,rarity"),
        required=True,
    )
    definitions: dict[str, dict[str, Any]] = {}
    for row in badge_rows:
        badge_id = str(row.get("badge_id") or "").strip()
        name = _plain_text(row.get("name"))
        if not badge_id or not name:
            continue
        definitions[badge_id] = {
            "badge_id": badge_id,
            "name": name,
            "prestige": _safe_int(row.get("prestige"), 0) or 0,
            "category": badge_category(badge_id),
            "icon_key": _plain_text(row.get("icon_key")),
            "rarity": _plain_text(row.get("rarity")),
        }

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in earned_rows:
        if row.get("revoked_at"):
            continue
        player_id = str(row.get("player_id") or "").strip()
        badge_id = str(row.get("badge_id") or "").strip()
        definition = definitions.get(badge_id)
        if not player_id or definition is None:
            continue
        clean = dict(definition)
        clean["earned_at"] = row.get("earned_at") or row.get("created_at")
        current = grouped.setdefault(player_id, {}).get(badge_id)
        if current is None or str(clean.get("earned_at") or "") > str(current.get("earned_at") or ""):
            grouped[player_id][badge_id] = clean

    result: dict[str, list[dict[str, Any]]] = {}
    for player_id, by_badge in grouped.items():
        result[player_id] = sorted(
            by_badge.values(),
            key=lambda item: (
                -int(item.get("prestige") or 0),
                str(item.get("name") or "").casefold(),
                str(item.get("badge_id") or ""),
            ),
        )
    return result


def _overall_rows(players: list[dict[str, Any]], *, club_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in players:
        player_id = row.get("id")
        if player_id is None:
            continue
        wins = _safe_int(row.get("wins"), 0) or 0
        losses = _safe_int(row.get("losses"), 0) or 0
        rows.append(
            {
                "club_id": str(row.get("club_id") or club_id),
                "league_name": OVERALL_SCOPE,
                "player_id": player_id,
                "player_name": str(row.get("name") or f"Player {player_id}"),
                "rating": _safe_float(row.get("rating")),
                "starting_rating": _safe_float(row.get("starting_rating"), _safe_float(row.get("rating"))),
                "wins": wins,
                "losses": losses,
                "matches_played": _safe_int(row.get("matches_played"), wins + losses) or 0,
                "is_active": _player_is_active(row),
                "updated_at": row.get("updated_at"),
            }
        )
    return rows


def _league_rows(
    rating_rows: list[dict[str, Any]],
    players: list[dict[str, Any]],
    *,
    club_id: str,
    league_name: str,
) -> list[dict[str, Any]]:
    players_by_id = {str(row.get("id")): row for row in players if row.get("id") is not None}
    rows: list[dict[str, Any]] = []
    for row in rating_rows:
        if str(row.get("league_name") or "").strip() != str(league_name).strip():
            continue
        player_id = row.get("player_id")
        if player_id is None:
            continue
        player = players_by_id.get(str(player_id), {})
        wins = _safe_int(row.get("wins"), 0) or 0
        losses = _safe_int(row.get("losses"), 0) or 0
        rows.append(
            {
                "club_id": str(row.get("club_id") or club_id),
                "league_name": str(league_name),
                "player_id": player_id,
                "player_name": str(player.get("name") or row.get("player_name") or f"Player {player_id}"),
                "rating": _safe_float(row.get("rating")),
                "starting_rating": _safe_float(row.get("starting_rating"), _safe_float(row.get("rating"))),
                "wins": wins,
                "losses": losses,
                "matches_played": _safe_int(row.get("matches_played"), wins + losses) or 0,
                "is_active": row.get("is_active") is not False and _player_is_active(player),
                "updated_at": row.get("updated_at"),
            }
        )
    return rows


def _decorate_rows(
    rows: list[dict[str, Any]],
    *,
    min_games: int,
    league_name: str,
    badges_by_player: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -(_safe_float(row.get("rating"), float("-inf")) or float("-inf")),
            str(row.get("player_name") or "").casefold(),
            str(row.get("player_id") or ""),
        ),
    )
    output: list[dict[str, Any]] = []
    previous_rating: float | None = None
    for rank, row in enumerate(ordered, start=1):
        rating = _safe_float(row.get("rating"))
        starting_rating = _safe_float(row.get("starting_rating"), rating)
        rating_jupr = _jupr(rating)
        starting_jupr = _jupr(starting_rating)
        wins = _safe_int(row.get("wins"), 0) or 0
        losses = _safe_int(row.get("losses"), 0) or 0
        matches = _safe_int(row.get("matches_played"), wins + losses) or 0
        player_badges = badges_by_player.get(str(row.get("player_id") or ""), [])
        clean = {
            "rank": rank,
            "rank_position": rank,
            "club_id": str(row.get("club_id") or ""),
            "league_name": str(league_name),
            "player_id": row.get("player_id"),
            "player_name": str(row.get("player_name") or "Player"),
            "rating": rating,
            "rating_jupr": rating_jupr,
            "starting_rating": starting_rating,
            "starting_rating_jupr": starting_jupr,
            "rating_gain_jupr": (
                None if rating_jupr is None or starting_jupr is None else rating_jupr - starting_jupr
            ),
            "gap_jupr": (
                None if previous_rating is None or rating is None else (previous_rating - rating) / 400.0
            ),
            "wins": wins,
            "losses": losses,
            "matches_played": matches,
            "win_pct": None if matches <= 0 else (float(wins) / float(matches)) * 100.0,
            "is_active": bool(row.get("is_active")),
            "qualified": None if league_name == OVERALL_SCOPE else matches >= int(min_games),
            "min_games": int(min_games),
            "badges": player_badges[:MAX_BADGES_PER_PLAYER],
            "badge_count": len(player_badges),
            "updated_at": row.get("updated_at"),
        }
        output.append(clean)
        previous_rating = rating
    return output


def _sort_rows(rows: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    clean_sort = str(sort or "rank").strip().lower()
    if clean_sort == "name":
        key = lambda row: (str(row.get("player_name") or "").casefold(), int(row.get("rank") or 0))
    elif clean_sort == "matches":
        key = lambda row: (-int(row.get("matches_played") or 0), int(row.get("rank") or 0))
    elif clean_sort == "win_pct":
        key = lambda row: (-(float(row["win_pct"]) if row.get("win_pct") is not None else -1.0), int(row.get("rank") or 0))
    elif clean_sort == "gain":
        key = lambda row: (-(float(row["rating_gain_jupr"]) if row.get("rating_gain_jupr") is not None else float("-inf")), int(row.get("rank") or 0))
    elif clean_sort == "rating":
        key = lambda row: (-(float(row["rating_jupr"]) if row.get("rating_jupr") is not None else float("-inf")), int(row.get("rank") or 0))
    else:
        key = lambda row: int(row.get("rank") or 0)
    return sorted(rows, key=key)


def _highlight_rows(rows: list[dict[str, Any]], *, min_games: int, league_name: str) -> dict[str, list[dict[str, Any]]]:
    source = rows
    if league_name != OVERALL_SCOPE and min_games > 0:
        source = [row for row in rows if row.get("qualified") is True]

    def top(key: str, *, null_last: bool = False) -> list[dict[str, Any]]:
        def value(row: dict[str, Any]) -> float:
            raw = row.get(key)
            if raw is None and null_last:
                return float("-inf")
            return _safe_float(raw, 0.0) or 0.0

        return sorted(source, key=lambda row: (-value(row), int(row.get("rank") or 0)))[:5]

    return {
        "highest_rating": top("rating_jupr", null_last=True),
        "most_improved": top("rating_gain_jupr", null_last=True),
        "best_win_pct": top("win_pct", null_last=True),
        "most_wins": top("wins"),
    }


def build_public_leaderboard(
    supabase: Any,
    *,
    club_id: str,
    league_name: str | None = None,
    league_view: str = ACTIVE_LEAGUE_VIEW,
    status: str = "active",
    search: str | None = None,
    sort: str = "rank",
    player_id: str | int | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> dict[str, Any]:
    """Build the complete, public-safe leaderboard projection for one scope.

    Supabase is only accessed by FastAPI. The returned contract is an explicit
    allowlist and never forwards raw player, badge, or metadata rows.
    """

    cid = str(club_id or "").strip()
    safe_limit = max(1, min(int(limit or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
    safe_offset = max(0, int(offset or 0))
    clean_status = str(status or "active").strip().lower()
    if clean_status not in {"active", "inactive", "all"}:
        clean_status = "active"
    clean_sort = str(sort or "rank").strip().lower()
    if clean_sort not in {"rank", "rating", "matches", "win_pct", "gain", "name"}:
        clean_sort = "rank"
    clean_search = str(search or "").strip()
    clean_league_view = normalize_public_league_view(league_view)

    players = _fetch_players(supabase, cid) if cid else []
    rating_rows = _fetch_league_ratings(supabase, cid) if cid else []
    scopes = (
        _scope_options(
            supabase,
            club_id=cid,
            league_view=clean_league_view,
        )
        if cid
        else (
            [{"name": OVERALL_SCOPE, "label": "Overall", "min_games": 0}]
            if clean_league_view == ACTIVE_LEAGUE_VIEW
            else []
        )
    )
    names = [str(scope.get("name") or "") for scope in scopes]
    requested = str(
        league_name
        or (OVERALL_SCOPE if clean_league_view == ACTIVE_LEAGUE_VIEW else "")
    ).strip()
    if clean_league_view == ACTIVE_LEAGUE_VIEW and requested.upper() == OVERALL_SCOPE:
        selected = OVERALL_SCOPE
    else:
        default_selected = (
            OVERALL_SCOPE
            if clean_league_view == ACTIVE_LEAGUE_VIEW
            else (names[0] if names else "")
        )
        selected = next(
            (name for name in names if name.casefold() == requested.casefold()),
            default_selected,
        )
    selected_meta = next(
        (dict(scope) for scope in scopes if scope.get("name") == selected),
        {
            "name": selected,
            "label": selected or "Past leagues",
            "min_games": 0,
        },
    )
    min_games = max(0, _safe_int(selected_meta.get("min_games"), 0) or 0)

    if selected == OVERALL_SCOPE:
        base_rows = _overall_rows(players, club_id=cid)
    elif selected:
        base_rows = _league_rows(rating_rows, players, club_id=cid, league_name=selected)
    else:
        base_rows = []

    badges = _badge_map(supabase, club_id=cid) if cid and base_rows else {}
    all_ranked = _decorate_rows(
        base_rows,
        min_games=min_games,
        league_name=selected,
        badges_by_player=badges,
    )
    if clean_status == "active":
        status_rows = [row for row in base_rows if row.get("is_active") is True]
    elif clean_status == "inactive":
        status_rows = [row for row in base_rows if row.get("is_active") is not True]
    else:
        status_rows = list(base_rows)
    ranked = _decorate_rows(
        status_rows,
        min_games=min_games,
        league_name=selected,
        badges_by_player=badges,
    )

    filtered = ranked
    if clean_search:
        needle = clean_search.casefold()
        filtered = [row for row in ranked if needle in str(row.get("player_name") or "").casefold()]

    snapshot = None
    if player_id is not None and str(player_id).strip():
        snapshot = next((row for row in ranked if str(row.get("player_id")) == str(player_id)), None)
        if snapshot is None:
            snapshot = next((row for row in all_ranked if str(row.get("player_id")) == str(player_id)), None)
    elif clean_search and len(filtered) == 1:
        snapshot = filtered[0]

    displayed = _sort_rows(filtered, clean_sort)
    total = len(displayed)
    page_rows = displayed[safe_offset : safe_offset + safe_limit]
    active_count = sum(1 for row in base_rows if row.get("is_active") is True)

    return {
        "scopes": scopes,
        "selected_scope": selected,
        "scope": selected_meta,
        "filters": {
            "league_view": clean_league_view,
            "status": clean_status,
            "search": clean_search,
            "sort": clean_sort,
        },
        "summary": {
            "ranked_players": len(base_rows),
            "active_players": active_count,
            "inactive_players": max(0, len(base_rows) - active_count),
            "leaderboard_scopes": len(scopes),
            "filtered_players": total,
        },
        "leaderboard": page_rows,
        "snapshot": snapshot,
        "highlights": _highlight_rows(filtered, min_games=min_games, league_name=selected),
        "pagination": {
            "total": total,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": safe_offset + len(page_rows) < total,
        },
    }


# Compatibility read model retained for older callers while the richer page
# contract above is rolled out. It keeps the original view-first behavior.
def _normalize_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        clean = {key: row.get(key) for key in PUBLIC_LEADERBOARD_FIELDS if key in row}
        clean.setdefault("rating_jupr", clean.get("rating"))
        clean.setdefault("matches_played", (clean.get("wins") or 0) + (clean.get("losses") or 0))
        if clean.get("rank") is None and clean.get("rank_position") is not None:
            clean["rank"] = clean.get("rank_position")
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
    players = supabase.table("players").select("id,name").eq("club_id", club_id).execute().data or []
    name_by_id = {player.get("id"): player.get("name") for player in players}

    enriched = []
    for row in ratings:
        player_id = row.get("player_id")
        enriched.append(
            {
                "club_id": row.get("club_id", club_id),
                "league_name": row.get("league_name"),
                "player_id": player_id,
                "player_name": name_by_id.get(player_id, "Player"),
                "rating": row.get("rating"),
                "rating_jupr": row.get("rating"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "matches_played": row.get("matches_played"),
                "is_active": row.get("is_active"),
                "updated_at": None,
            }
        )

    sorted_rows = sorted(enriched, key=lambda row: (-(float(row.get("rating") or 0.0)), str(row.get("player_name") or "")))
    for index, row in enumerate(sorted_rows, start=1):
        row["rank_position"] = index
    return sorted_rows


def get_public_leaderboard(supabase: Any, club_id: str, league_name: str | None = None) -> list[dict[str, Any]]:
    """Read the legacy public leaderboard list from its view/table fallback."""

    cid = str(club_id).strip()
    if not cid:
        return []
    try:
        return _normalize_rows(_fetch_from_view(supabase, cid, league_name))
    except Exception:
        return _normalize_rows(_fetch_fallback(supabase, cid, league_name))
