from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.constants import CAP_LOSER_GAIN_ELO, DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO
from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.services.admin_league_awards_service import preview_admin_league_awards
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)
from jupr_app.services.team_league_service import get_admin_team_league

PREVIOUS_MONTH_MIN_GAMES = 10
DEFAULT_TOP_PLAYERS_LIMIT = 50
MAX_TOP_PLAYERS_LIMIT = 200
MAX_MATCH_ROWS = 20_000


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _previous_month_window(now_utc: datetime) -> tuple[datetime, datetime]:
    current = _utc_datetime(now_utc) or datetime.now(timezone.utc)
    end = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start = end.replace(year=end.year - 1, month=12) if end.month == 1 else end.replace(month=end.month - 1)
    return start, end


def _fetch_table(supabase: Any, table_name: str, *, club_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    query = supabase.table(table_name).select("*").eq("club_id", str(club_id))
    if limit is not None:
        query = query.limit(int(limit))
    return _safe_rows(query.execute())


def _active_player(row: dict[str, Any]) -> bool:
    active_flag = row.get("active", row.get("is_active", True))
    return active_flag is not False and not bool(row.get("inactive_at"))


def _participant_ids(match: dict[str, Any]) -> tuple[int, int, int, int] | None:
    values = tuple(_safe_int(match.get(field)) for field in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _league_participant_slots(match: dict[str, Any]) -> tuple[str, ...] | None:
    """Return canonical participant slots for a complete league match.

    The previous-month Top Players printable intentionally consumes doubles /
    overall matches only, so its four-player ``_participant_ids`` contract stays
    unchanged. League-night printouts also support canonical singles rows, where
    both partner slots are null.
    """

    primary_slots = ("t1_p1", "t2_p1")
    if any(_safe_int(match.get(slot)) is None for slot in primary_slots):
        return None

    partner_ids = (_safe_int(match.get("t1_p2")), _safe_int(match.get("t2_p2")))
    match_format = _clean_text(match.get("match_format"), limit=40).casefold()
    if match_format == "singles":
        # A row declared as singles must not quietly accept stray partners.
        return primary_slots if partner_ids == (None, None) else None
    if match_format == "doubles" and any(player_id is None for player_id in partner_ids):
        return None
    if partner_ids == (None, None):
        # Preserve legacy singles rows that predate the explicit format column.
        return primary_slots
    if any(player_id is None for player_id in partner_ids):
        return None
    return ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


def _scored_match(match: dict[str, Any]) -> bool:
    if match.get("is_active") is False or match.get("deleted_at"):
        return False
    score_1 = _safe_int(match.get("score_t1"), 0) or 0
    score_2 = _safe_int(match.get("score_t2"), 0) or 0
    return score_1 + score_2 > 0


def build_admin_top_players_printable(
    supabase: Any,
    *,
    club_id: str,
    now_utc: datetime | None = None,
    min_games: int = PREVIOUS_MONTH_MIN_GAMES,
    limit: int = DEFAULT_TOP_PLAYERS_LIMIT,
) -> dict[str, Any]:
    """Build the authenticated previous-calendar-month Top Players print model."""

    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")

    minimum = max(1, min(int(min_games), 1_000))
    row_limit = max(1, min(int(limit), MAX_TOP_PLAYERS_LIMIT))
    current = _utc_datetime(now_utc) if now_utc is not None else datetime.now(timezone.utc)
    assert current is not None
    start, end = _previous_month_window(current)

    player_rows = _fetch_table(supabase, "players", club_id=str(club_id), limit=10_000)
    players: dict[int, dict[str, Any]] = {}
    for row in player_rows:
        player_id = _safe_int(row.get("id"))
        if player_id is None or not _active_player(row):
            continue
        players[int(player_id)] = {
            "player_id": int(player_id),
            "player_name": _clean_text(row.get("name"), limit=160) or f"Player {player_id}",
            "rating": _safe_float(row.get("rating"), _safe_float(row.get("elo"), 1200.0)) or 1200.0,
        }

    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
    for match in _fetch_table(supabase, "matches", club_id=str(club_id), limit=MAX_MATCH_ROWS):
        played_at = _utc_datetime(match.get("date_dt") or match.get("date"))
        participants = _participant_ids(match)
        if played_at is None or not (start <= played_at < end) or participants is None or not _scored_match(match):
            continue
        score_1 = _safe_int(match.get("score_t1"), 0) or 0
        score_2 = _safe_int(match.get("score_t2"), 0) or 0
        for player_id in participants:
            if player_id in players:
                stats[player_id]["games"] += 1
        if score_1 == score_2:
            continue
        winners = participants[:2] if score_1 > score_2 else participants[2:]
        losers = participants[2:] if score_1 > score_2 else participants[:2]
        for player_id in winners:
            if player_id in players:
                stats[player_id]["wins"] += 1
        for player_id in losers:
            if player_id in players:
                stats[player_id]["losses"] += 1

    rankings: list[dict[str, Any]] = []
    for player_id, player in players.items():
        player_stats = stats.get(player_id, {"wins": 0, "losses": 0, "games": 0})
        if int(player_stats["games"]) < minimum:
            continue
        rating = float(player["rating"])
        rankings.append(
            {
                **player,
                "rating_jupr": rating / 400.0,
                "wins": int(player_stats["wins"]),
                "losses": int(player_stats["losses"]),
                "games": int(player_stats["games"]),
                "record": f"{int(player_stats['wins'])}-{int(player_stats['losses'])}",
            }
        )
    rankings.sort(
        key=lambda row: (
            -float(row.get("rating") or 0),
            -int(row.get("games") or 0),
            -int(row.get("wins") or 0),
            str(row.get("player_name") or "").lower(),
        )
    )
    rankings = rankings[:row_limit]
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank

    return {
        "ok": True,
        "mode": "league_top_players_printable",
        "period": {
            "label": start.strftime("%B %Y"),
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "timezone": "UTC",
        },
        "minimum_games": minimum,
        "limit": row_limit,
        "rankings": rankings,
        "ranking_count": len(rankings),
        "empty_message": None
        if rankings
        else f"No eligible active players with at least {minimum} games in {start.strftime('%B %Y')}.",
    }


def _parse_week_num(value: Any) -> int | None:
    match = re.search(
        r"\bweek\s*#?\s*(\d+)\b",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _snapshot_pair(match: dict[str, Any], slot: str) -> tuple[float, float] | None:
    start = _safe_float(match.get(f"{slot}_r"))
    end = _safe_float(match.get(f"{slot}_r_end"))
    if start is None or end is None:
        return None
    return float(start), float(end)


def _league_matches(supabase: Any, *, club_id: str, league_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _fetch_table(supabase, "matches", club_id=str(club_id), limit=MAX_MATCH_ROWS):
        if _clean_text(match.get("league"), limit=120) != league_name:
            continue
        if _clean_text(match.get("match_type"), limit=80).lower() == "popup":
            continue
        participant_slots = _league_participant_slots(match)
        if participant_slots is None or not _scored_match(match):
            continue
        participant_ids = tuple(_safe_int(match.get(slot)) for slot in participant_slots)
        if any(player_id is None for player_id in participant_ids):
            continue
        participants = tuple(int(player_id) for player_id in participant_ids if player_id is not None)
        clean = dict(match)
        clean["week_num"] = _parse_week_num(match.get("week_tag"))
        clean["_played_at"] = _utc_datetime(match.get("date_dt") or match.get("date"))
        clean["_participants"] = participants
        clean["_participant_slots"] = participant_slots
        rows.append(clean)
    rows.sort(
        key=lambda row: (
            row.get("_played_at") or datetime.min.replace(tzinfo=timezone.utc),
            _safe_int(row.get("id"), 0) or 0,
        )
    )
    return rows


def _rating_seeds(detail: dict[str, Any], players: dict[int, str]) -> dict[int, float]:
    seeds: dict[int, float] = {player_id: 1200.0 for player_id in players}
    for row in detail.get("standings") or []:
        player_id = _safe_int(row.get("player_id"))
        if player_id is None:
            continue
        seed = _safe_float(row.get("starting_rating"), _safe_float(row.get("rating"), 1200.0)) or 1200.0
        seeds[int(player_id)] = float(seed)
    return seeds


def _weekly_player_rows(
    matches: list[dict[str, Any]],
    *,
    selected_week: int,
    player_names: dict[int, str],
    initial_ratings: dict[int, float],
    k_factor: int,
) -> tuple[list[dict[str, Any]], int]:
    ratings = dict(initial_ratings)
    weekly: dict[int, dict[str, Any]] = {}
    replayed_match_count = 0
    selected_positions = [index for index, match in enumerate(matches) if match.get("week_num") == selected_week]
    relevant_matches = matches[: max(selected_positions) + 1] if selected_positions else []

    for match in relevant_matches:
        participants = match["_participants"]
        slots = match["_participant_slots"]
        team_size = len(participants) // 2
        score_1 = _safe_int(match.get("score_t1"), 0) or 0
        score_2 = _safe_int(match.get("score_t2"), 0) or 0
        snapshots = [_snapshot_pair(match, slot) for slot in slots]
        deltas: list[float]
        if all(pair is not None for pair in snapshots):
            complete = [pair for pair in snapshots if pair is not None]
            deltas = [end - start for start, end in complete]
            for player_id, (_start, end) in zip(participants, complete):
                ratings[player_id] = float(end)
        else:
            starts: list[float] = []
            for player_id, slot in zip(participants, slots):
                snapshot = _snapshot_pair(match, slot)
                start_only = _safe_float(match.get(f"{slot}_r"))
                start = float(start_only if start_only is not None else ratings.get(player_id, 1200.0))
                starts.append(start)
                if snapshot is not None:
                    ratings[player_id] = snapshot[1]
            delta_1, delta_2 = calculate_hybrid_elo(
                sum(starts[:team_size]) / float(team_size),
                sum(starts[team_size:]) / float(team_size),
                score_1,
                score_2,
                k_factor=float(k_factor),
                min_win_delta=float(MIN_WIN_DELTA_ELO),
                cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
            )
            deltas = ([delta_1] * team_size) + ([delta_2] * team_size)
            for player_id, start, delta in zip(participants, starts, deltas):
                ratings[player_id] = start + float(delta)
            replayed_match_count += 1

        if match.get("week_num") != selected_week:
            continue
        for index, player_id in enumerate(participants):
            item = weekly.setdefault(
                int(player_id),
                {
                    "player_id": int(player_id),
                    "player_name": player_names.get(int(player_id), f"Player {int(player_id)}"),
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "rating_delta_elo": 0.0,
                },
            )
            item["games"] += 1
            item["rating_delta_elo"] += float(deltas[index])
            if score_1 != score_2:
                won = (index < team_size and score_1 > score_2) or (
                    index >= team_size and score_2 > score_1
                )
                item["wins" if won else "losses"] += 1

    output: list[dict[str, Any]] = []
    for item in weekly.values():
        games = int(item["games"])
        wins = int(item["wins"])
        output.append(
            {
                **item,
                "rating_delta_jupr": float(item["rating_delta_elo"]) / 400.0,
                "win_pct": (float(wins) / float(games) * 100.0) if games else None,
            }
        )
    return output, replayed_match_count


def build_admin_league_printout(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    week_num: int | None = None,
) -> dict[str, Any]:
    """Build a read-only league-night print model with Python-authoritative leaders."""

    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")

    detail = get_admin_league_manager_detail(supabase, club_id=str(club_id), league_name=clean_league)
    matches = _league_matches(supabase, club_id=str(club_id), league_name=clean_league)
    available_weeks = sorted({int(row["week_num"]) for row in matches if row.get("week_num") is not None})
    requested_week = int(week_num) if week_num is not None else None
    if requested_week is not None and requested_week not in available_weeks:
        raise ValueError("week_num is not a scored week for this league")
    selected_week = (
        requested_week
        if requested_week is not None
        else (available_weeks[-1] if available_weeks else None)
    )

    players = {
        int(row["player_id"]): str(row.get("player_name") or f"Player {row['player_id']}")
        for row in detail.get("roster") or []
        if row.get("player_id") is not None
    }
    weekly_rows: list[dict[str, Any]] = []
    replayed_match_count = 0
    if selected_week is not None:
        weekly_rows, replayed_match_count = _weekly_player_rows(
            matches,
            selected_week=int(selected_week),
            player_names=players,
            initial_ratings=_rating_seeds(detail, players),
            k_factor=int(detail.get("league", {}).get("k_factor") or DEFAULT_K_FACTOR),
        )

    rating_leaders = sorted(
        weekly_rows,
        key=lambda row: (
            -float(row.get("rating_delta_jupr") or 0),
            -int(row.get("games") or 0),
            -int(row.get("wins") or 0),
            str(row.get("player_name") or "").lower(),
        ),
    )[:5]
    win_leaders = sorted(
        weekly_rows,
        key=lambda row: (
            -int(row.get("wins") or 0),
            -int(row.get("games") or 0),
            -float(row.get("win_pct") or 0),
            str(row.get("player_name") or "").lower(),
        ),
    )[:5]
    awards = preview_admin_league_awards(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
    ).get("awards", [])
    warnings: list[str] = []
    if replayed_match_count:
        warnings.append(
            f"Replayed {replayed_match_count} match(es) in Python because complete "
            "stored rating snapshots were unavailable."
        )

    league = detail.get("league") or {}
    is_team_league = _clean_text(league.get("league_type"), limit=80).casefold() == "team"
    team_print = {
        "standings": [],
        "teams": [],
        "substitute_pool": [],
    }
    if is_team_league:
        try:
            team_detail = get_admin_team_league(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
            )
        except ValueError:
            # A metadata-only Team draft may not have its Team League settings
            # row yet. Keep the print model empty instead of mislabeling generic
            # player ratings as team standings or assigned rosters.
            team_detail = None
        if team_detail is not None:
            player_names = {
                int(row["id"]): _clean_text(row.get("name"), limit=160)
                or f"Player {int(row['id'])}"
                for row in team_detail.get("players") or []
                if _safe_int(row.get("id")) is not None
            }
            team_print["standings"] = list(team_detail.get("standings") or [])
            team_print["teams"] = [
                {
                    "team_id": str(team.get("id") or ""),
                    "team_name": _clean_text(team.get("team_name"), limit=160)
                    or "Team",
                    "status": _clean_text(team.get("status"), limit=40),
                    "roster_complete": bool(team.get("roster_complete")),
                    "members": [
                        {
                            "player_id": int(member["player_id"]),
                            "player_name": _clean_text(
                                member.get("player_name")
                                or player_names.get(int(member["player_id"])),
                                limit=160,
                            )
                            or f"Player {int(member['player_id'])}",
                            "role": _clean_text(member.get("role"), limit=24),
                            "status": _clean_text(member.get("status"), limit=24),
                        }
                        for member in team.get("members") or []
                        if _safe_int(member.get("player_id")) is not None
                        and _clean_text(member.get("status"), limit=24)
                        in {"active", "invited"}
                    ],
                }
                for team in team_detail.get("teams") or []
                if _clean_text(team.get("status"), limit=40)
                in {"confirmed", "pending_partner"}
            ]
            team_print["substitute_pool"] = [
                {
                    "player_id": int(row["player_id"]),
                    "player_name": player_names.get(
                        int(row["player_id"]), f"Player {int(row['player_id'])}"
                    ),
                    "status": _clean_text(row.get("status"), limit=24),
                    "note": _clean_text(row.get("note"), limit=240),
                }
                for row in team_detail.get("substitute_pool") or []
                if _safe_int(row.get("player_id")) is not None
                and _clean_text(row.get("status"), limit=24)
                in {"available", "unavailable"}
            ]
    active_roster = [row for row in detail.get("roster") or [] if row.get("in_league")]
    printable_sections = {
        "schedule": bool(detail.get("schedule_preview")),
        "weekly_leaders": bool(weekly_rows),
        "season_leaders": bool(awards),
        # League-rating rows are player standings. They must not be advertised as
        # team standings until the print model carries the actual team table.
        "standings": bool(detail.get("standings")) and not is_team_league,
        "roster": bool(active_roster) and not is_team_league,
        "team_standings": bool(team_print["standings"]),
        "team_rosters": bool(team_print["teams"]),
        "substitute_pool": bool(team_print["substitute_pool"]),
    }
    has_printable_data = any(printable_sections.values())
    if not has_printable_data:
        warnings.append(
            "No printable league-night data is available yet; add a schedule, "
            "league roster, or scored results before printing."
        )

    return {
        "ok": True,
        "mode": "league_manager_printout",
        "league_name": clean_league,
        "available_weeks": available_weeks,
        "selected_week": selected_week,
        "detail": detail,
        "weekly_rating_leaders": rating_leaders,
        "weekly_win_leaders": win_leaders,
        "season_top_performers": awards,
        "season_top_performer_count": len(awards),
        "team_print": team_print,
        "has_printable_data": has_printable_data,
        "printable_sections": printable_sections,
        "rating_source": "stored_snapshots" if replayed_match_count == 0 else "stored_snapshots_with_python_replay",
        "warnings": warnings,
    }
