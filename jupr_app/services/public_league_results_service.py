from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any

LEAGUE_META_SELECT = "club_id,league_name,is_active,status,min_games,k_factor,start_week,end_week,num_weeks,total_weeks,weeks"
LEAGUE_RATINGS_SELECT = "club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active"
PLAYER_SELECT = "id,club_id,name,rating,active,inactive_at"
MATCH_SELECT = (
    "id,club_id,date,league,match_type,week_tag,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,"
    "t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end"
)


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
    match = re.search(r"(\d+)", str(value))
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
    if status and status not in {"active", "published", "live", "draft"}:
        return None
    return league_name


def _is_active_player(row: dict[str, Any]) -> bool:
    if row.get("inactive_at"):
        return False
    if row.get("active") is False:
        return False
    return True


def _fetch_table(supabase: Any, table_name: str, select_cols: str, *, club_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        query = supabase.table(table_name).select(select_cols).eq("club_id", str(club_id))
        if limit is not None:
            query = query.limit(int(limit))
        return _safe_rows(query.execute())
    except Exception:
        return []


def _fetch_players(supabase: Any, club_id: str) -> dict[int, dict[str, Any]]:
    rows = _fetch_table(supabase, "players", PLAYER_SELECT, club_id=club_id)
    players: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None or not _is_active_player(row):
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


def get_public_league_results_overview(supabase: Any, *, club_id: str) -> dict[str, Any]:
    """Return public-safe league options for League Results."""

    cid = str(club_id).strip()
    names: set[str] = set()
    meta_rows = _fetch_table(supabase, "leagues_metadata", LEAGUE_META_SELECT, club_id=cid)
    meta_by_name: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        league_name = _active_league_name(row)
        if league_name:
            names.add(league_name)
            meta_by_name[league_name] = row

    rating_rows = _fetch_table(supabase, "league_ratings", "league_name,is_active", club_id=cid, limit=5000)
    for row in rating_rows:
        league_name = _active_league_name(row)
        if league_name:
            names.add(league_name)

    match_rows = _fetch_table(supabase, "matches", "league,match_type,score_t1,score_t2", club_id=cid, limit=5000)
    for row in match_rows:
        league_name = _active_league_name(row)
        match_type = str(row.get("match_type") or "").strip()
        score_t1 = _safe_int(row.get("score_t1"), 0) or 0
        score_t2 = _safe_int(row.get("score_t2"), 0) or 0
        if league_name and match_type != "PopUp" and (score_t1 + score_t2) > 0:
            names.add(league_name)

    leagues = []
    for name in sorted(names, key=_league_sort_key):
        meta = meta_by_name.get(name, {})
        leagues.append(
            {
                "name": name,
                "min_games": _safe_int(meta.get("min_games"), 0) or 0,
                "k_factor": _safe_int(meta.get("k_factor"), None),
            }
        )
    return {"leagues": leagues}


def _selected_league(overview: dict[str, Any], league_name: str | None) -> str | None:
    names = [str(item.get("name")) for item in overview.get("leagues", []) if item.get("name")]
    requested = str(league_name or "").strip()
    if requested and requested in names:
        return requested
    return names[0] if names else None


def _league_meta(overview: dict[str, Any], league_name: str | None) -> dict[str, Any] | None:
    if not league_name:
        return None
    for item in overview.get("leagues", []):
        if str(item.get("name")) == str(league_name):
            return dict(item)
    return {"name": str(league_name), "min_games": 0, "k_factor": None}


def _league_matches(supabase: Any, *, club_id: str, league_name: str) -> list[dict[str, Any]]:
    rows = _fetch_table(supabase, "matches", MATCH_SELECT, club_id=club_id, limit=5000)
    result: list[dict[str, Any]] = []
    for row in rows:
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


def _rating_snapshot_delta(row: dict[str, Any], player_id: int) -> float | None:
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
        if start is None or end is None:
            return None
        return float(end) - float(start)
    return None


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
            expanded.append(
                {
                    "match_id": match.get("id"),
                    "week_num": match.get("week_num"),
                    "player_id": int(player_id),
                    "player_name": players_by_id[int(player_id)]["name"],
                    "games": 1,
                    "wins": 1 if won else 0,
                    "losses": 0 if won else 1,
                    "rating_delta_elo": _rating_snapshot_delta(match, int(player_id)),
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


def _standing_rows(supabase: Any, *, club_id: str, league_name: str, players_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _fetch_table(supabase, "league_ratings", LEAGUE_RATINGS_SELECT, club_id=club_id, limit=5000)
    output: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("league_name") or "").strip() != str(league_name).strip():
            continue
        if row.get("is_active") is False:
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


def _week_list(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    week_nums = sorted({int(row["week_num"]) for row in matches if row.get("week_num") is not None})
    return [{"week_num": week_num, "week_label": _week_label(week_num)} for week_num in week_nums]


def _weekly_highlights(weekly_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not weekly_rows:
        return {"biggest_climbers": [], "best_win_pct": [], "most_active": []}
    recent_week = max(int(row["week_num"]) for row in weekly_rows if row.get("week_num") is not None)
    current = [row for row in weekly_rows if row.get("week_num") == recent_week]

    def public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "week_num": row.get("week_num"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "games": row.get("games"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "win_pct": row.get("win_pct"),
            "rating_delta_jupr": row.get("rating_delta_jupr"),
        }

    climbers_source = [row for row in current if row.get("rating_delta_jupr") is not None]
    if climbers_source:
        climbers = sorted(climbers_source, key=lambda row: row.get("rating_delta_jupr") or 0, reverse=True)[:3]
    else:
        climbers = sorted(current, key=lambda row: (row.get("wins") or 0, row.get("games") or 0), reverse=True)[:3]
    best = sorted([row for row in current if int(row.get("games") or 0) > 0], key=lambda row: (row.get("win_pct") or 0, row.get("games") or 0), reverse=True)[:3]
    active = sorted(current, key=lambda row: (row.get("games") or 0, row.get("wins") or 0), reverse=True)[:3]
    return {
        "biggest_climbers": [public_row(row) for row in climbers],
        "best_win_pct": [public_row(row) for row in best],
        "most_active": [public_row(row) for row in active],
    }


def build_public_league_results(supabase: Any, *, club_id: str, league_name: str | None = None) -> dict[str, Any]:
    """Build the public League Results payload for one club/league."""

    cid = str(club_id).strip()
    overview = get_public_league_results_overview(supabase, club_id=cid)
    selected = _selected_league(overview, league_name)
    if not selected:
        return {**overview, "selected_league": None, "league": None, "standings": [], "weeks": [], "weekly_results": [], "cumulative": [], "highlights": {"biggest_climbers": [], "best_win_pct": [], "most_active": []}}

    players_by_id = _fetch_players(supabase, cid)
    matches = _league_matches(supabase, club_id=cid, league_name=selected)
    expanded = _expand_matches(matches, players_by_id)
    weekly = _summarize(expanded, ("week_num", "player_id", "player_name"))
    weekly.sort(key=lambda row: (row.get("week_num") or 0, -(row.get("wins") or 0), -(row.get("games") or 0), str(row.get("player_name") or "").lower()))
    cumulative = _summarize(expanded, ("player_id", "player_name"))
    cumulative.sort(key=lambda row: (-(row.get("wins") or 0), -(row.get("games") or 0), str(row.get("player_name") or "").lower()))

    return {
        **overview,
        "selected_league": selected,
        "league": _league_meta(overview, selected),
        "standings": _standing_rows(supabase, club_id=cid, league_name=selected, players_by_id=players_by_id),
        "weeks": _week_list(matches),
        "weekly_results": weekly,
        "cumulative": cumulative,
        "highlights": _weekly_highlights(weekly),
    }
