from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Mapping

LEAGUE_META_SELECT = (
    "club_id,league_name,league_type,match_format,is_active,status,min_games,k_factor,"
    "schedule_config,awards_config"
)
LEAGUE_RATINGS_SELECT = "club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at"
PLAYER_SELECT = "id,club_id,name,rating,active,inactive_at"
MATCH_SELECT = (
    "id,club_id,date,league,match_type,week_tag,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,"
    "t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end,"
    "deleted_at"
)
DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES = 4


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


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


def _jupr(elo: float | None) -> float | None:
    if elo is None:
        return None
    return float(elo) / 400.0


def _win_pct(wins: int, losses: int, games: int | None = None) -> float | None:
    played = int(games if games is not None else wins + losses)
    if played <= 0:
        return None
    return float(wins) / float(played) * 100.0


def _parse_week_num(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\bweek\s*#?\s*(\d+)\b", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _week_label(week_num: int | None) -> str:
    return f"Week {int(week_num)}" if week_num is not None else "Unspecified"


def _active_league_name(row: dict[str, Any]) -> str | None:
    league_name = str(row.get("league_name") or row.get("league") or "").strip()
    if not league_name or league_name.upper() == "OVERALL" or league_name.upper() == "POPUP":
        return None
    if row.get("is_active") is False:
        return None
    status = str(row.get("status") or "").strip().lower()
    if status and status not in {"active", "published", "live"}:
        return None
    return league_name


def _is_active_player(row: dict[str, Any]) -> bool:
    if row.get("inactive_at"):
        return False
    if row.get("active") is False:
        return False
    return True


def _fetch_table(
    supabase: Any,
    table_name: str,
    select_cols: str,
    *,
    club_id: str,
    filters: dict[str, Any] | None = None,
    null_filters: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        query = supabase.table(table_name).select(select_cols).eq("club_id", str(club_id))
        for field, value in (filters or {}).items():
            query = query.eq(str(field), value)
        for field in null_filters:
            query = query.is_(str(field), None)
        if limit is not None:
            query = query.limit(int(limit))
        return _safe_rows(query.execute())
    except Exception:
        return []


def _fetch_players(
    supabase: Any,
    club_id: str,
    *,
    include_inactive: bool = False,
) -> dict[int, dict[str, Any]]:
    rows = _fetch_table(supabase, "players", PLAYER_SELECT, club_id=club_id)
    players: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None or (not include_inactive and not _is_active_player(row)):
            continue
        players[int(pid)] = {
            "id": int(pid),
            "name": str(row.get("name") or f"Player {pid}"),
            "rating": _safe_float(row.get("rating"), 1200.0) or 1200.0,
        }
    return players


def _league_sort_key(name: str) -> tuple[int, str]:
    lowered = str(name).lower()
    if "ladder" in lowered:
        return (0, lowered)
    if "open" in lowered:
        return (1, lowered)
    return (2, lowered)


def _public_league_meta(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    schedule_config = meta.get("schedule_config")
    configured_weeks = (
        _safe_int(schedule_config.get("weeks"), None)
        if isinstance(schedule_config, dict)
        else None
    )
    return {
        "name": str(name),
        "min_games": _safe_int(meta.get("min_games"), 0) or 0,
        "k_factor": _safe_int(meta.get("k_factor"), None),
        "league_type": str(meta.get("league_type") or "Individual"),
        "match_format": (
            "singles"
            if str(meta.get("match_format") or "").strip().casefold()
            == "singles"
            else "doubles"
        ),
        "start_week": _safe_int(meta.get("start_week"), None),
        "end_week": _safe_int(meta.get("end_week"), None),
        "num_weeks": _safe_int(
            meta.get(
                "num_weeks",
                meta.get(
                    "total_weeks",
                    meta.get("weeks", configured_weeks),
                ),
            ),
            None,
        ),
    }


def _get_public_league_results_overview_data(
    supabase: Any, *, club_id: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return the public overview plus private metadata for request-local reuse."""

    cid = str(club_id).strip()
    names: set[str] = set()
    meta_rows = _fetch_table(supabase, "leagues_metadata", LEAGUE_META_SELECT, club_id=cid)
    meta_by_name: dict[str, dict[str, Any]] = {}
    known_meta_names: set[str] = set()
    for row in meta_rows:
        raw_name = str(row.get("league_name") or "").strip()
        if raw_name:
            known_meta_names.add(raw_name)
        league_name = _active_league_name(row)
        if league_name:
            names.add(league_name)
            meta_by_name[league_name] = row

    rating_rows = _fetch_table(supabase, "league_ratings", "league_name,is_active", club_id=cid, limit=5000)
    for row in rating_rows:
        league_name = _active_league_name(row)
        if league_name and league_name not in known_meta_names:
            names.add(league_name)

    match_rows = _fetch_table(
        supabase,
        "matches",
        "league,match_type,score_t1,score_t2,deleted_at",
        club_id=cid,
        null_filters=("deleted_at",),
        limit=5000,
    )
    for row in match_rows:
        if row.get("deleted_at") not in (None, ""):
            continue
        league_name = _active_league_name(row)
        match_type = str(row.get("match_type") or "").strip()
        score_t1 = _safe_int(row.get("score_t1"), 0) or 0
        score_t2 = _safe_int(row.get("score_t2"), 0) or 0
        if (
            league_name
            and league_name not in known_meta_names
            and match_type != "PopUp"
            and (score_t1 + score_t2) > 0
        ):
            names.add(league_name)

    leagues = []
    for name in sorted(names, key=_league_sort_key):
        meta = meta_by_name.get(name, {})
        leagues.append(_public_league_meta(name, meta))
    return {"leagues": leagues}, meta_by_name


def get_public_league_results_overview(supabase: Any, *, club_id: str) -> dict[str, Any]:
    """Return public-safe league options for League Results."""

    overview, _metadata_by_name = _get_public_league_results_overview_data(
        supabase, club_id=club_id
    )
    return overview


def _selected_league(overview: dict[str, Any], league_name: str | None) -> str | None:
    names = [str(item.get("name")) for item in overview.get("leagues", []) if item.get("name")]
    requested = str(league_name or "").strip()
    if requested:
        # An explicit deep link must never silently substitute a different
        # league.  Admin League Manager relies on this contract to preserve
        # the selected league while moving between its tools.
        return requested if requested in names else None
    return names[0] if names else None


def _league_meta(overview: dict[str, Any], league_name: str | None) -> dict[str, Any] | None:
    if not league_name:
        return None
    for item in overview.get("leagues", []):
        if str(item.get("name")) == str(league_name):
            return dict(item)
    return {
        "name": str(league_name),
        "min_games": 0,
        "k_factor": None,
        "start_week": None,
        "end_week": None,
        "num_weeks": None,
    }


def _league_matches(supabase: Any, *, club_id: str, league_name: str) -> list[dict[str, Any]]:
    rows = _fetch_table(
        supabase,
        "matches",
        MATCH_SELECT,
        club_id=club_id,
        filters={"league": str(league_name)},
        null_filters=("deleted_at",),
        limit=5000,
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("deleted_at") not in (None, ""):
            continue
        if str(row.get("league") or "").strip() != str(league_name).strip():
            continue
        if str(row.get("match_type") or "").strip() == "PopUp":
            continue
        score_t1 = _safe_int(row.get("score_t1"), 0) or 0
        score_t2 = _safe_int(row.get("score_t2"), 0) or 0
        if (score_t1 + score_t2) <= 0:
            continue
        clean = dict(row)
        clean["score_t1"] = score_t1
        clean["score_t2"] = score_t2
        clean["week_num"] = _parse_week_num(row.get("week_tag"))
        result.append(clean)
    result.sort(key=lambda row: (str(_json_safe(row.get("date")) or ""), _safe_int(row.get("id"), 0) or 0))
    return result


def _rating_snapshot(row: dict[str, Any], player_id: int) -> tuple[float | None, float | None]:
    pid = int(player_id)
    columns = {
        "t1_p1": ("t1_p1_r", "t1_p1_r_end"),
        "t1_p2": ("t1_p2_r", "t1_p2_r_end"),
        "t2_p1": ("t2_p1_r", "t2_p1_r_end"),
        "t2_p2": ("t2_p2_r", "t2_p2_r_end"),
    }
    for player_col, (start_col, end_col) in columns.items():
        if _safe_int(row.get(player_col)) != pid:
            continue
        start = _safe_float(row.get(start_col))
        end = _safe_float(row.get(end_col))
        return start, end
    return None, None


def _rating_snapshot_delta(row: dict[str, Any], player_id: int) -> float | None:
    start, end = _rating_snapshot(row, player_id)
    if start is None or end is None:
        return None
    return float(end) - float(start)


def _expand_matches(matches: list[dict[str, Any]], players_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for match in matches:
        score_t1 = _safe_int(match.get("score_t1"), 0) or 0
        score_t2 = _safe_int(match.get("score_t2"), 0) or 0
        if score_t1 == score_t2:
            continue
        team_1_win = score_t1 > score_t2
        for player_id, team in [
            (_safe_int(match.get("t1_p1")), 1),
            (_safe_int(match.get("t1_p2")), 1),
            (_safe_int(match.get("t2_p1")), 2),
            (_safe_int(match.get("t2_p2")), 2),
        ]:
            if player_id is None or int(player_id) not in players_by_id:
                continue
            won = bool(team_1_win and team == 1) or bool((not team_1_win) and team == 2)
            rating_start, rating_end = _rating_snapshot(match, int(player_id))
            expanded.append(
                {
                    "match_id": match.get("id"),
                    "match_date": _json_safe(match.get("date")),
                    "week_num": match.get("week_num"),
                    "player_id": int(player_id),
                    "player_name": players_by_id[int(player_id)]["name"],
                    "games": 1,
                    "wins": 1 if won else 0,
                    "losses": 0 if won else 1,
                    "rating_start_elo": rating_start,
                    "rating_end_elo": rating_end,
                    "rating_delta_elo": (
                        float(rating_end) - float(rating_start)
                        if rating_start is not None and rating_end is not None
                        else None
                    ),
                }
            )
    return expanded


def _summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_keys)
        item = grouped.setdefault(
            key,
            {field: row.get(field) for field in group_keys} | {"games": 0, "wins": 0, "losses": 0, "rating_delta_elo": 0.0, "rating_delta_count": 0},
        )
        item["games"] += int(row.get("games") or 0)
        item["wins"] += int(row.get("wins") or 0)
        item["losses"] += int(row.get("losses") or 0)
        delta = _safe_float(row.get("rating_delta_elo"))
        if delta is not None:
            item["rating_delta_elo"] += float(delta)
            item["rating_delta_count"] += 1

    output: list[dict[str, Any]] = []
    for item in grouped.values():
        games = int(item.get("games") or 0)
        wins = int(item.get("wins") or 0)
        losses = int(item.get("losses") or 0)
        delta_count = int(item.pop("rating_delta_count") or 0)
        delta_elo = float(item.pop("rating_delta_elo") or 0.0)
        item["win_pct"] = _win_pct(wins, losses, games)
        item["rating_delta_jupr"] = _jupr(delta_elo) if delta_count > 0 else None
        output.append(item)
    return output


def _league_rating_rows(
    supabase: Any, *, club_id: str, league_name: str
) -> list[dict[str, Any]]:
    return _fetch_table(
        supabase,
        "league_ratings",
        LEAGUE_RATINGS_SELECT,
        club_id=club_id,
        filters={"league_name": str(league_name)},
        limit=5000,
    )


def _standing_rows(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    players_by_id: dict[int, dict[str, Any]],
    include_inactive: bool = False,
    rating_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = (
        [dict(row) for row in rating_rows]
        if rating_rows is not None
        else _league_rating_rows(
            supabase, club_id=club_id, league_name=league_name
        )
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("league_name") or "").strip() != str(league_name).strip():
            continue
        if (
            row.get("is_active") is False or bool(row.get("inactive_at"))
        ) and not include_inactive:
            continue
        pid = _safe_int(row.get("player_id"))
        if pid is None or int(pid) not in players_by_id:
            continue
        wins = _safe_int(row.get("wins"), 0) or 0
        losses = _safe_int(row.get("losses"), 0) or 0
        matches_played = _safe_int(row.get("matches_played"), wins + losses) or (wins + losses)
        rating = _safe_float(row.get("rating"), players_by_id[int(pid)].get("rating"))
        starting_rating = _safe_float(row.get("starting_rating"), rating)
        rating_delta = None if rating is None or starting_rating is None else float(rating) - float(starting_rating)
        output.append(
            {
                "player_id": int(pid),
                "player_name": players_by_id[int(pid)]["name"],
                "rating": rating,
                "rating_jupr": _jupr(rating),
                "wins": wins,
                "losses": losses,
                "matches_played": matches_played,
                "win_pct": _win_pct(wins, losses, matches_played),
                "rating_delta_jupr": _jupr(rating_delta),
            }
        )
    output.sort(key=lambda row: (-(row.get("rating") or 0), -(row.get("matches_played") or 0), str(row.get("player_name") or "").lower()))
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def _canonical_season_rows(
    standings: list[dict[str, Any]],
    match_cumulative: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one public season record per player.

    ``league_ratings`` is the materialized authority for a rated league's
    season record, current rating, and rating rank. Match-derived totals remain
    useful for players who do not yet have a league-ratings row, but they must
    not replace the official record for a rated player and create contradictory
    public summaries.
    """

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for standing in standings:
        player_id = _safe_int(standing.get("player_id"))
        if player_id is None:
            continue
        seen.add(player_id)
        rows.append(
            {
                "week_num": None,
                "player_id": player_id,
                "player_name": standing.get("player_name"),
                "games": standing.get("matches_played", 0),
                "wins": standing.get("wins", 0),
                "losses": standing.get("losses", 0),
                "win_pct": standing.get("win_pct"),
                "rating_delta_jupr": standing.get("rating_delta_jupr"),
                "rating_jupr": standing.get("rating_jupr"),
                "rank": standing.get("rank"),
                "prev_rank": None,
                "rank_delta": None,
            }
        )

    for match_row in match_cumulative:
        player_id = _safe_int(match_row.get("player_id"))
        if player_id is None or player_id in seen:
            continue
        rows.append(
            {
                **match_row,
                "week_num": None,
                "rating_jupr": None,
                "rank": None,
                "prev_rank": None,
                "rank_delta": None,
            }
        )

    return rows


def _standings_with_fallback_players(
    standings: list[dict[str, Any]],
    season_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep match-only participants visible without duplicating season tables."""

    rows = [dict(row) for row in standings]
    seen = {
        int(player_id)
        for row in standings
        if (player_id := _safe_int(row.get("player_id"))) is not None
    }
    for season_row in season_rows:
        player_id = _safe_int(season_row.get("player_id"))
        if player_id is None or player_id in seen:
            continue
        seen.add(player_id)
        rows.append(
            {
                "player_id": player_id,
                "player_name": season_row.get("player_name"),
                "rating": None,
                "rating_jupr": None,
                "wins": season_row.get("wins", 0),
                "losses": season_row.get("losses", 0),
                "matches_played": season_row.get("games", 0),
                "win_pct": season_row.get("win_pct"),
                "rating_delta_jupr": None,
                "rank": None,
            }
        )
    return rows


def _week_list(matches: list[dict[str, Any]], league: dict[str, Any] | None) -> list[dict[str, Any]]:
    league = league or {}
    result_week_nums = sorted(
        {int(row["week_num"]) for row in matches if row.get("week_num") is not None}
    )
    start_week = _safe_int(league.get("start_week"))
    end_week = _safe_int(league.get("end_week"))
    num_weeks = _safe_int(league.get("num_weeks"))
    if start_week is not None and end_week is not None and start_week > 0 and end_week >= start_week:
        week_nums = list(range(start_week, end_week + 1))
    elif num_weeks is not None and num_weeks > 0:
        week_nums = list(range(1, num_weeks + 1))
    elif result_week_nums:
        week_nums = list(range(min(result_week_nums), max(result_week_nums) + 1))
    else:
        week_nums = []
    result_set = set(result_week_nums)
    return [
        {
            "week_num": week_num,
            "week_label": _week_label(week_num),
            "has_results": week_num in result_set,
        }
        for week_num in week_nums
    ]


def _weekly_rating_rankings(expanded: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    ordered = sorted(
        [row for row in expanded if row.get("week_num") is not None],
        key=lambda row: (
            int(row.get("week_num") or 0),
            str(row.get("match_date") or ""),
            _safe_int(row.get("match_id"), 0) or 0,
        ),
    )
    snapshots: dict[tuple[int, int], dict[str, Any]] = {}
    for row in ordered:
        week_num = int(row.get("week_num"))
        player_id = int(row.get("player_id"))
        start = _safe_float(row.get("rating_start_elo"))
        end = _safe_float(row.get("rating_end_elo"))
        if start is None or end is None:
            continue
        item = snapshots.setdefault(
            (week_num, player_id),
            {"rating_start_elo": start, "rating_end_elo": end},
        )
        item["rating_end_elo"] = end

    by_week: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (week_num, player_id), item in snapshots.items():
        by_week[week_num].append((player_id, item))
    for rows in by_week.values():
        previous_rating: float | None = None
        dense_rank = 0
        for player_id, item in sorted(
            rows,
            key=lambda entry: (-float(entry[1]["rating_end_elo"]), entry[0]),
        ):
            rating = float(item["rating_end_elo"])
            if previous_rating is None or rating != previous_rating:
                dense_rank += 1
                previous_rating = rating
            item["rank"] = dense_rank

    by_player: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (week_num, player_id), item in snapshots.items():
        by_player[player_id].append((week_num, item))
    for rows in by_player.values():
        previous_rank: int | None = None
        for _week_num, item in sorted(rows, key=lambda entry: entry[0]):
            current_rank = _safe_int(item.get("rank"))
            item["prev_rank"] = previous_rank
            item["rank_delta"] = (
                previous_rank - current_rank
                if previous_rank is not None and current_rank is not None
                else None
            )
            previous_rank = current_rank

    output: dict[tuple[int, int], dict[str, Any]] = {}
    for key, item in snapshots.items():
        start = float(item["rating_start_elo"])
        end = float(item["rating_end_elo"])
        output[key] = {
            "rating_start_jupr": _jupr(start),
            "rating_jupr": _jupr(end),
            "rating_delta_jupr": _jupr(end - start),
            "rank": item.get("rank"),
            "prev_rank": item.get("prev_rank"),
            "rank_delta": item.get("rank_delta"),
        }
    return output


def _public_stat_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "week_num": row.get("week_num"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "games": row.get("games"),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "win_pct": row.get("win_pct"),
        "rating_delta_jupr": row.get("rating_delta_jupr"),
        "rating_jupr": row.get("rating_jupr"),
        "rank": row.get("rank"),
        "prev_rank": row.get("prev_rank"),
        "rank_delta": row.get("rank_delta"),
    }


def _highlights(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    min_games: int,
    week_num: int | None = None,
) -> dict[str, Any]:
    current = list(rows)
    climbers = sorted(
        [
            row
            for row in current
            if (row.get("rating_delta_jupr") or 0) > 0
        ],
        key=lambda row: (row.get("rating_delta_jupr") or 0, row.get("games") or 0),
        reverse=True,
    )[:3]
    qualified = [
        row
        for row in current
        if int(row.get("games") or 0) >= int(min_games) and row.get("win_pct") is not None
    ]
    best = sorted(
        qualified,
        key=lambda row: (row.get("win_pct") or 0, row.get("games") or 0),
        reverse=True,
    )[:3]
    active = sorted(
        current,
        key=lambda row: (row.get("games") or 0, row.get("wins") or 0),
        reverse=True,
    )[:3]
    return {
        "scope": scope,
        "week_num": week_num,
        "min_games": int(min_games),
        "biggest_climbers": [_public_stat_row(row) for row in climbers],
        "best_win_pct": [_public_stat_row(row) for row in best],
        "most_active": [_public_stat_row(row) for row in active],
    }


def _player_options(
    standings: list[dict[str, Any]],
    cumulative: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in [*standings, *cumulative]:
        player_id = _safe_int(row.get("player_id"))
        if player_id is None or player_id in seen:
            continue
        seen.add(player_id)
        options.append(
            {"player_id": player_id, "player_name": str(row.get("player_name") or f"Player {player_id}")}
        )
    return options


def _recent_player_matches(
    matches: list[dict[str, Any]],
    *,
    player_id: int | None,
    players_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if player_id is None:
        return []

    def public_player(pid: int | None) -> dict[str, Any] | None:
        if pid is None:
            return None
        player = players_by_id.get(int(pid))
        return {
            "player_id": int(pid),
            "player_name": str((player or {}).get("name") or f"Player {pid}"),
        }

    output: list[dict[str, Any]] = []
    ordered = sorted(
        matches,
        key=lambda row: (str(_json_safe(row.get("date")) or ""), _safe_int(row.get("id"), 0) or 0),
        reverse=True,
    )
    for match in ordered:
        team_1 = [_safe_int(match.get("t1_p1")), _safe_int(match.get("t1_p2"))]
        team_2 = [_safe_int(match.get("t2_p1")), _safe_int(match.get("t2_p2"))]
        if player_id in team_1:
            own, opponents = team_1, team_2
            score_for = _safe_int(match.get("score_t1"), 0) or 0
            score_against = _safe_int(match.get("score_t2"), 0) or 0
        elif player_id in team_2:
            own, opponents = team_2, team_1
            score_for = _safe_int(match.get("score_t2"), 0) or 0
            score_against = _safe_int(match.get("score_t1"), 0) or 0
        else:
            continue
        partner_id = next((pid for pid in own if pid is not None and pid != player_id), None)
        output.append(
            {
                "match_id": match.get("id"),
                "date": _json_safe(match.get("date")),
                "week_num": match.get("week_num"),
                "week_label": _week_label(match.get("week_num")),
                "partner": public_player(partner_id),
                "opponents": [player for player in (public_player(pid) for pid in opponents) if player],
                "result": "W" if score_for > score_against else "L" if score_for < score_against else "D",
                "score_for": score_for,
                "score_against": score_against,
                "rating_delta_jupr": _jupr(_rating_snapshot_delta(match, player_id)),
            }
        )
        if len(output) >= 15:
            break
    return output


def _build_resolved_league_results(
    supabase: Any,
    *,
    club_id: str,
    overview: dict[str, Any],
    selected: str | None,
    week_num: int | None = None,
    player_id: int | None = None,
    weekly_min_games: int = DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES,
    include_inactive_players: bool = False,
    include_inactive_ratings: bool = False,
    league_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build sanitized results after the caller has resolved league visibility."""

    cid = str(club_id).strip()
    if not selected:
        empty_highlights = _highlights([], scope="week", min_games=weekly_min_games)
        return {
            **overview,
            "selected_league": None,
            "league": None,
            "standings": [],
            "weeks": [],
            "selected_week": None,
            "weekly_results": [],
            "cumulative": [],
            "players": [],
            "selected_player_id": None,
            "player_summary": None,
            "player_weekly": [],
            "recent_matches": [],
            "weekly_highlights": empty_highlights,
            "season_highlights": _highlights([], scope="season", min_games=1),
            "highlights": empty_highlights,
            "award_progress": {"awards": [], "award_count": 0},
        }

    players_by_id = _fetch_players(
        supabase,
        cid,
        include_inactive=include_inactive_players,
    )
    league = _league_meta(overview, selected)
    rating_rows = _league_rating_rows(
        supabase, club_id=cid, league_name=selected
    )
    rating_standings = _standing_rows(
        supabase,
        club_id=cid,
        league_name=selected,
        players_by_id=players_by_id,
        include_inactive=include_inactive_ratings,
        rating_rows=rating_rows,
    )
    league_member_ids = {
        int(row["player_id"])
        for row in rating_standings
        if _safe_int(row.get("player_id")) is not None
    }
    award_rating_rows = [
        dict(row)
        for row in rating_rows
        if (rating_player_id := _safe_int(row.get("player_id"))) is not None
        and rating_player_id in league_member_ids
    ]
    scoped_players = (
        players_by_id
        if include_inactive_ratings
        else {
            player_id: player
            for player_id, player in players_by_id.items()
            if player_id in league_member_ids
        }
    )
    matches = _league_matches(supabase, club_id=cid, league_name=selected)
    expanded = _expand_matches(matches, scoped_players)
    weekly = _summarize(expanded, ("week_num", "player_id", "player_name"))
    weekly_ratings = _weekly_rating_rankings(expanded)
    for row in weekly:
        row.update(
            weekly_ratings.get(
                (int(row.get("week_num") or 0), int(row.get("player_id") or 0)),
                {
                    "rating_start_jupr": None,
                    "rating_jupr": None,
                    "rank": None,
                    "prev_rank": None,
                    "rank_delta": None,
                },
            )
        )
    weekly.sort(key=lambda row: (row.get("week_num") or 0, -(row.get("wins") or 0), -(row.get("games") or 0), str(row.get("player_name") or "").lower()))
    match_cumulative = _summarize(expanded, ("player_id", "player_name"))
    match_cumulative.sort(
        key=lambda row: (
            -(row.get("wins") or 0),
            -(row.get("games") or 0),
            str(row.get("player_name") or "").lower(),
        )
    )
    cumulative = _canonical_season_rows(rating_standings, match_cumulative)
    standings = _standings_with_fallback_players(rating_standings, cumulative)
    standing_by_player = {
        int(row["player_id"]): row for row in standings if _safe_int(row.get("player_id")) is not None
    }
    weeks = _week_list(matches, league)
    valid_weeks = [int(row["week_num"]) for row in weeks]
    selected_week = int(week_num) if week_num is not None and int(week_num) in valid_weeks else None
    if selected_week is None:
        selected_week = max(valid_weeks) if valid_weeks else None
    selected_week_rows = [row for row in weekly if row.get("week_num") == selected_week]
    weekly_min_games = max(1, min(20, int(weekly_min_games or DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES)))
    season_min_games = max(1, int((league or {}).get("min_games") or 1))
    weekly_highlights = _highlights(
        selected_week_rows,
        scope="week",
        min_games=weekly_min_games,
        week_num=selected_week,
    )
    season_highlight_rows = [
        {
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "games": row.get("matches_played"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "win_pct": row.get("win_pct"),
            "rating_jupr": row.get("rating_jupr"),
            "rating_delta_jupr": row.get("rating_delta_jupr"),
            "rank": row.get("rank"),
        }
        for row in standings
    ]
    season_highlights = _highlights(
        season_highlight_rows,
        scope="season",
        min_games=season_min_games,
    )
    player_options = _player_options(standings, cumulative)
    valid_player_ids = {int(row["player_id"]) for row in player_options}
    selected_player_id = int(player_id) if player_id is not None and int(player_id) in valid_player_ids else None
    if selected_player_id is None and player_options:
        selected_player_id = int(player_options[0]["player_id"])
    selected_standing = standing_by_player.get(int(selected_player_id or 0), {})
    selected_cumulative = next(
        (row for row in cumulative if int(row.get("player_id") or 0) == int(selected_player_id or 0)),
        {},
    )
    player_summary = (
        {
            "player_id": selected_player_id,
            "player_name": selected_standing.get("player_name")
            or selected_cumulative.get("player_name")
            or f"Player {selected_player_id}",
            "rank": selected_standing.get("rank"),
            "rating_jupr": selected_standing.get("rating_jupr"),
            "rating_delta_jupr": selected_standing.get("rating_delta_jupr"),
            "games": selected_standing.get("matches_played", selected_cumulative.get("games", 0)),
            "wins": selected_standing.get("wins", selected_cumulative.get("wins", 0)),
            "losses": selected_standing.get("losses", selected_cumulative.get("losses", 0)),
            "win_pct": selected_standing.get("win_pct", selected_cumulative.get("win_pct")),
        }
        if selected_player_id is not None
        else None
    )
    player_weekly = [
        row for row in weekly if int(row.get("player_id") or 0) == int(selected_player_id or 0)
    ]

    return {
        **overview,
        "selected_league": selected,
        "league": league,
        "standings": standings,
        "weeks": weeks,
        "selected_week": selected_week,
        "weekly_results": weekly,
        "cumulative": cumulative,
        "players": player_options,
        "selected_player_id": selected_player_id,
        "player_summary": player_summary,
        "player_weekly": player_weekly,
        "recent_matches": _recent_player_matches(
            matches,
            player_id=selected_player_id,
            players_by_id=players_by_id,
        ),
        "weekly_highlights": weekly_highlights,
        "season_highlights": season_highlights,
        # Compatibility alias for the former latest-week highlight object.
        "highlights": weekly_highlights,
        "award_progress": _public_award_progress(
            supabase,
            club_id=cid,
            league_name=selected,
            metadata=league_metadata,
            league_rows=award_rating_rows,
            match_rows=matches,
            player_rows=[
                {"club_id": cid, **dict(player)}
                for player in players_by_id.values()
            ],
        ),
    }


def _public_award_progress(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    metadata: Mapping[str, Any] | None = None,
    league_rows: list[dict[str, Any]] | None = None,
    match_rows: list[dict[str, Any]] | None = None,
    player_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from jupr_app.services.admin_league_awards_service import (
        get_public_league_award_progress,
    )

    return get_public_league_award_progress(
        supabase,
        club_id=str(club_id),
        league_name=str(league_name),
        metadata=metadata,
        league_rows=league_rows,
        match_rows=match_rows,
        player_rows=player_rows,
    )


def build_public_league_results(
    supabase: Any,
    *,
    club_id: str,
    league_name: str | None = None,
    week_num: int | None = None,
    player_id: int | None = None,
    weekly_min_games: int = DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES,
) -> dict[str, Any]:
    """Build active-only public League Results for one club/league."""

    cid = str(club_id).strip()
    overview, metadata_by_name = _get_public_league_results_overview_data(
        supabase, club_id=cid
    )
    selected = _selected_league(overview, league_name)
    return _build_resolved_league_results(
        supabase,
        club_id=cid,
        overview=overview,
        selected=selected,
        week_num=week_num,
        player_id=player_id,
        weekly_min_games=weekly_min_games,
        league_metadata=metadata_by_name.get(str(selected or "")),
    )
