from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import uuid
from typing import Any

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_TOURNAMENTS,
    has_permission,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.domain.tournaments import (
    build_playoff_games,
    compute_round_robin_standings,
    finalize_game,
    resolve_playoff_dependencies,
)
from jupr_app.domain.tournaments.score_policy import (
    GAME_TARGETS,
    require_tournament_score,
    resolve_tournament_scoring_format,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
    get_tournament_admin_operation_record,
    get_tournament_admin_operation_record_by_idempotency_key,
    reconcile_tournament_admin_guarded_operation,
    require_tournament_admin_mutation_runtime,
    run_tournament_admin_guarded_operation,
    tournament_admin_mutation_status,
)
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from jupr_app.services.admin_tournament_podium_review_service import (
    build_admin_tournament_podium_review_fingerprint,
    find_current_admin_tournament_podium_review,
)


TOURNAMENT_DAY_LIVE_SURFACE = "tournament_live"
TOURNAMENT_DAY_LIVE_ENTITY = "tournament_registration_day"
TOURNAMENT_DAY_LIVE_RECONCILE_CONFIRMATION = "RECONCILE DAY OPERATIONS"
ACTIVE_OPERATION_STATUSES = {"intent", "mutated", "recovery_required"}
ACTIVE_QUEUE_STATES = {"HELD", "CALLED", "ON_COURT"}
NON_PLAYED_RESULT_TYPES = {"FORFEIT", "NO_SHOW", "RETIREMENT"}
SUPPORTED_ADVANCE_COUNTS = (4, 5, 6)
COMMAND_CONFIRMATIONS = {
    "activate_day": "ACTIVATE DAY",
    "activate_draw": "ACTIVATE DRAW",
    "pause_draw": "PAUSE DRAW",
    "resume_draw": "RESUME DRAW",
    "auto_fill_courts": "AUTO FILL COURTS",
    "assign_next_court": "ASSIGN NEXT OPEN COURT",
    "assign_game_to_court": "ASSIGN GAME TO COURT",
    "requeue_game": "RETURN GAME TO QUEUE",
    "move_game_to_court": "MOVE GAME TO COURT",
    "score_and_release": "SAVE SCORE AND RELEASE COURT",
    "correct_completed_score": "CORRECT COMPLETED SCORE",
    "record_non_played_result": "RECORD NON-PLAYED RESULT",
    "generate_playoffs": "GENERATE PLAYOFFS",
    "close_day": "CLOSE TOURNAMENT DAY",
}
COMMAND_ACTIONS = {
    action: f"tournament_day_live_{action}" for action in COMMAND_CONFIRMATIONS
}
ACTION_COMMANDS = {value: key for key, value in COMMAND_ACTIONS.items()}
STATE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or []) if isinstance(row, dict)]
    except Exception:
        return []


def _rows(
    supabase: Any,
    table: str,
    *,
    filters: tuple[tuple[str, str, Any], ...],
    required: bool = True,
) -> list[dict[str, Any]]:
    if not filters:
        raise RuntimeError(f"Refusing unscoped service-role read from {table}.")
    try:
        query = supabase.table(table).select("*")
        for operation, column, value in filters:
            if operation == "eq":
                query = query.eq(column, value)
            elif operation == "in":
                query = query.in_(column, list(value))
            elif operation == "is":
                query = query.is_(column, value)
            else:
                raise RuntimeError(f"Unsupported scoped read operation {operation}.")
        return _safe_rows(query.execute())
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"Tournament day authority {table} is unavailable; keep day-live writes closed."
            ) from exc
        return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_canonical(item) for item in value]
        if all(isinstance(item, dict) and item.get("id") not in (None, "") for item in normalized):
            return sorted(normalized, key=lambda item: str(item.get("id")))
        return normalized
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _game_scoring(event: dict[str, Any] | None) -> dict[str, Any]:
    try:
        format_code = resolve_tournament_scoring_format(event)
    except ValueError as exc:
        return {
            "format": None,
            "target": None,
            "win_by_two": None,
            "best_of_three_score_semantics": None,
            "blocker": str(exc),
        }
    return {
        "format": format_code,
        "target": GAME_TARGETS.get(format_code, 2),
        "win_by_two": format_code != "BEST_2_OF_3",
        "best_of_three_score_semantics": (
            "games_won_2_0_or_2_1" if format_code == "BEST_2_OF_3" else None
        ),
        "blocker": None,
    }


def _score_review(
    game: dict[str, Any],
    score_a: int,
    score_b: int,
    *,
    acknowledged: bool,
) -> dict[str, Any]:
    scoring = dict(game.get("scoring") or {})
    # Compatibility for retained commands/snapshots created before scoring
    # metadata was projected. Fresh authoritative snapshots always resolve the
    # configured event and therefore include the ``scoring`` key, even when
    # configuration is invalid. Never turn that explicit blocker into a legacy
    # GAME_TO_11 default.
    format_code = (
        _text(scoring.get("format"))
        if "scoring" in game
        else "GAME_TO_11"
    )
    return require_tournament_score(
        score_a,
        score_b,
        scoring_format=format_code,
        unusual_score_acknowledged=acknowledged,
    )


def _blocker(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **{k: v for k, v in extra.items() if v is not None}}


def _readiness(blockers: list[dict[str, Any]], confirmation: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ready": not blockers, "blockers": blockers}
    if confirmation:
        result["confirmation"] = confirmation
    return result


def _day_option(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _text(row.get("id")),
        "label": _text(row.get("label") or row.get("name") or row.get("event_date") or "Tournament day"),
        "event_date": row.get("event_date"),
        "sort_order": _safe_int(row.get("sort_order"), 0),
        "court_count": _safe_int(row.get("court_count"), 0),
        "court_labels": list(row.get("court_labels") or []),
        "available_court_ids": [str(value) for value in (row.get("available_court_ids") or [])],
    }


def _is_finalized_non_tied(game: dict[str, Any]) -> bool:
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    team_a = _text(game.get("team_a_id"))
    team_b = _text(game.get("team_b_id"))
    winner = _text(game.get("winner_team_id"))
    loser = _text(game.get("loser_team_id"))
    expected_winner = team_a if score_a is not None and score_b is not None and score_a > score_b else team_b
    expected_loser = team_b if expected_winner == team_a else team_a
    return bool(
        game.get("finalized_at")
        and score_a is not None
        and score_b is not None
        and score_a != score_b
        and team_a
        and team_b
        and team_a != team_b
        and winner == expected_winner
        and loser == expected_loser
    )


def _podium_matches_playoff_results(
    playoff_games: list[dict[str, Any]],
    podium_rows: list[dict[str, Any]],
) -> bool:
    final_games = [
        row
        for row in playoff_games
        if _text(row.get("playoff_round")).upper() == "FINAL"
    ]
    bronze_games = [
        row
        for row in playoff_games
        if _text(row.get("playoff_round")).upper() == "BRONZE"
    ]
    podium_by_placement = {
        _safe_int(row.get("placement")): _text(row.get("team_id"))
        for row in podium_rows
    }
    return bool(
        len(podium_rows) == 3
        and set(podium_by_placement) == {1, 2, 3}
        and len(set(podium_by_placement.values())) == 3
        and len(final_games) == 1
        and len(bronze_games) == 1
        and _is_finalized_non_tied(final_games[0])
        and _is_finalized_non_tied(bronze_games[0])
        and podium_by_placement.get(1)
        == _text(final_games[0].get("winner_team_id"))
        and podium_by_placement.get(2)
        == _text(final_games[0].get("loser_team_id"))
        and podium_by_placement.get(3)
        == _text(bronze_games[0].get("winner_team_id"))
    )


def _game_sort_key(game: dict[str, Any]) -> tuple[Any, ...]:
    stage = _text(game.get("stage")).upper()
    return (
        0 if stage == "ROUND_ROBIN" else 1,
        _safe_int(game.get("rr_round_number"), 1_000_000),
        _safe_int(game.get("rr_slot_number"), 1_000_000),
        _text(game.get("playoff_round")),
        _text(game.get("playoff_game_code")),
        _text(game.get("id")),
    )


def _round_label(game: dict[str, Any]) -> str:
    if _text(game.get("stage")).upper() == "ROUND_ROBIN":
        number = _safe_int(game.get("rr_round_number"))
        return f"Round {number}" if number is not None else "Round robin"
    return _text(game.get("playoff_round") or game.get("playoff_game_code") or "Playoff")


def _side(team: dict[str, Any] | None, players: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not team:
        return {"team_id": None, "name": "To be determined", "participant_names": []}
    names: list[str] = []
    for field in ("player1_id", "player2_id"):
        player_id = _safe_int(team.get(field))
        if player_id is None:
            continue
        player = players.get(player_id) or {}
        names.append(
            _text(
                player.get("name")
                or player.get("display_name")
                or "Player name unavailable"
            )
        )
    label = _text(team.get("name") or team.get("team_name"))
    if not label:
        team_number = _safe_int(team.get("team_number"))
        label = " / ".join(names) or (
            f"Team {team_number}" if team_number is not None else "Team name unavailable"
        )
    return {"team_id": _text(team.get("id")) or None, "name": label, "participant_names": names}


def _draw_scheduled_for_day(
    draw: dict[str, Any], event: dict[str, Any] | None, day_id: str
) -> bool:
    if _text(draw.get("event_option_id")) and event is None:
        return False
    draw_day_id = _text(draw.get("registration_day_id"))
    if draw_day_id:
        if event:
            scheduled_value = event.get("scheduled_day_ids")
            if scheduled_value not in (None, []):
                if not isinstance(scheduled_value, list):
                    return False
                scheduled_ids = {_text(value) for value in scheduled_value}
                if scheduled_ids and draw_day_id not in scheduled_ids:
                    return False
            elif _text(event.get("registration_day_id")) not in {"", draw_day_id}:
                return False
        return draw_day_id == day_id
    if event:
        scheduled_value = event.get("scheduled_day_ids")
        if scheduled_value not in (None, []):
            if not isinstance(scheduled_value, list):
                return False
            scheduled_ids = {_text(value) for value in scheduled_value}
            if scheduled_ids:
                # A draw without an explicit owning day cannot safely inherit
                # a multi-day event membership.
                return len(scheduled_ids) == 1 and day_id in scheduled_ids
        return _text(event.get("registration_day_id")) == day_id
    return False


def _safe_operation(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_key": _text(row.get("operation_key")),
        "client_idempotency_key": _text(row.get("client_idempotency_key")),
        "action": _text(row.get("action")),
        "status": _text(row.get("status")),
        "entity_label": "Tournament day operation",
        "updated_at": row.get("updated_at"),
        "error_text": row.get("error_text"),
        "retryable": _text(row.get("status")) in {"failed", "recovery_required"},
    }


def build_admin_tournament_day_live_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str,
    exclude_operation_key: str | None = None,
) -> dict[str, Any]:
    """Build one authoritative, day-scoped multi-draw operator snapshot."""

    tournaments = _rows(
        supabase,
        "tournaments",
        filters=(("eq", "id", str(tournament_id)), ("eq", "club_id", str(club_id))),
    )
    tournament = next(
        (
            row
            for row in tournaments
            if _text(row.get("id")) == str(tournament_id)
            and _text(row.get("club_id")) == str(club_id)
        ),
        None,
    )
    if not tournament:
        raise ValueError("tournament not found for this club")

    all_days = [
        row
        for row in _rows(
            supabase,
            "tournament_registration_days",
            filters=(("eq", "tournament_id", str(tournament_id)),),
        )
        if _text(row.get("tournament_id")) == str(tournament_id)
    ]
    selected_day = next(
        (
            row
            for row in all_days
            if _text(row.get("id")) == str(registration_day_id)
        ),
        None,
    )
    if not selected_day:
        raise ValueError("registration day not found for this tournament")
    days = sorted(
        [
            row
            for row in all_days
            if bool(row.get("enabled", True))
            or _text(row.get("id")) == str(registration_day_id)
        ],
        key=lambda row: (_safe_int(row.get("sort_order"), 0), _text(row.get("id"))),
    )

    settings = next(
        (
            row
            for row in _rows(
                supabase,
                "tournament_registration_settings",
                filters=(("eq", "tournament_id", str(tournament_id)),),
            )
            if _text(row.get("tournament_id")) == str(tournament_id)
        ),
        {},
    )
    events = {
        _text(row.get("id")): row
        for row in _rows(
            supabase,
            "tournament_event_options",
            filters=(("eq", "tournament_id", str(tournament_id)),),
        )
        if _text(row.get("tournament_id")) == str(tournament_id)
    }
    tournament_runs = _rows(
        supabase,
        "tournament_day_live_runs",
        filters=(
            ("eq", "club_id", str(club_id)),
            ("eq", "tournament_id", str(tournament_id)),
        ),
    )
    runs = [
        row
        for row in tournament_runs
        if _text(row.get("club_id")) == str(club_id)
        and _text(row.get("tournament_id")) == str(tournament_id)
        and _text(row.get("registration_day_id")) == str(registration_day_id)
    ]
    run = runs[0] if runs else None
    run_id = _text((run or {}).get("id"))
    day_draw_rows = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_day_live_draws",
                filters=(("eq", "run_id", run_id),),
            )
            if run_id
            else []
        )
        if run_id and _text(row.get("run_id")) == run_id
    ]
    day_draw_by_draw = {_text(row.get("draw_id")): row for row in day_draw_rows}
    activated_draw_ids = set(day_draw_by_draw)
    draws = sorted(
        [
            row
            for row in _rows(
                supabase,
                "tournament_event_draws",
                filters=(("eq", "tournament_id", str(tournament_id)),),
            )
            if _text(row.get("tournament_id")) == str(tournament_id)
            and (
                _text(row.get("id")) in activated_draw_ids
                or (
                    _draw_scheduled_for_day(
                        row,
                        events.get(_text(row.get("event_option_id"))),
                        str(registration_day_id),
                    )
                    and not bool(row.get("hidden_from_primary_ops", False))
                )
            )
        ],
        key=lambda row: (_text(row.get("name")), _text(row.get("id"))),
    )
    draw_ids = {_text(row.get("id")) for row in draws}
    draw_by_id = {_text(row.get("id")): row for row in draws}
    teams = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_teams",
                filters=(
                    ("eq", "tournament_id", str(tournament_id)),
                    ("in", "draw_id", sorted(draw_ids)),
                ),
            )
            if draw_ids
            else []
        )
        if _text(row.get("tournament_id")) == str(tournament_id)
        and _text(row.get("draw_id")) in draw_ids
    ]
    teams_by_id = {_text(row.get("id")): row for row in teams}
    all_draw_games = sorted(
        [
            row
            for row in (
                _rows(
                    supabase,
                    "tournament_games",
                    filters=(
                        ("eq", "tournament_id", str(tournament_id)),
                        ("in", "draw_id", sorted(draw_ids)),
                    ),
                )
                if draw_ids
                else []
            )
            if _text(row.get("tournament_id")) == str(tournament_id)
            and _text(row.get("draw_id")) in draw_ids
        ],
        key=_game_sort_key,
    )
    games = [
        row
        for row in all_draw_games
        if _text(row.get("registration_day_id")) == str(registration_day_id)
        and _text(row.get("event_option_id"))
        == _text(
            (draw_by_id.get(_text(row.get("draw_id"))) or {}).get(
                "event_option_id"
            )
        )
    ]
    game_ids = sorted(
        {_text(row.get("id")) for row in all_draw_games if _text(row.get("id"))}
    )
    published_matches = [
        row
        for row in (
            _rows(
                supabase,
                "matches",
                filters=(("in", "tournament_game_id", game_ids),),
            )
            if game_ids
            else []
        )
        if _text(row.get("tournament_game_id")) in set(game_ids)
    ]
    published_game_ids = {
        _text(row.get("tournament_game_id")) for row in published_matches
    }
    draw_by_game_id = {
        _text(row.get("id")): _text(row.get("draw_id")) for row in all_draw_games
    }
    published_draw_ids = {
        draw_by_game_id[game_id]
        for game_id in published_game_ids
        if game_id in draw_by_game_id
    }
    podium_rows = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_podium",
                filters=(
                    ("eq", "tournament_id", str(tournament_id)),
                    ("in", "draw_id", sorted(draw_ids)),
                ),
            )
            if draw_ids
            else []
        )
        if _text(row.get("tournament_id")) == str(tournament_id)
        and _text(row.get("draw_id")) in draw_ids
    ]
    award_context_ids = sorted(
        {
            f"{tournament_id}:draw:{_text(row.get('draw_id'))}:podium:{_safe_int(row.get('placement'))}"
            for row in podium_rows
            if _text(row.get("draw_id"))
            and _safe_int(row.get("placement")) in {1, 2, 3}
        }
    )
    podium_badge_rows = [
        row
        for row in (
            _rows(
                supabase,
                "player_badges",
                filters=(
                    ("eq", "club_id", str(club_id)),
                    ("eq", "context_type", "tournament"),
                    ("in", "context_id", award_context_ids),
                ),
            )
            if award_context_ids
            else []
        )
        if _text(row.get("club_id")) == str(club_id)
        and _text(row.get("context_type")) == "tournament"
        and _text(row.get("context_id")) in set(award_context_ids)
    ]
    effective_player_ids = sorted(
        {
            player_id
            for team in teams
            for player_id in (_safe_int(team.get("player1_id")), _safe_int(team.get("player2_id")))
            if player_id is not None
        }
    )
    players = {
        int(row["id"]): row
        for row in (
            _rows(
                supabase,
                "players",
                filters=(
                    ("eq", "club_id", str(club_id)),
                    ("in", "id", effective_player_ids),
                ),
            )
            if effective_player_ids
            else []
        )
        if _safe_int(row.get("id")) is not None
    }
    registrations = [
        row
        for row in _rows(
            supabase,
            "tournament_registrations",
            filters=(("eq", "tournament_id", str(tournament_id)),),
        )
        if _text(row.get("tournament_id")) == str(tournament_id)
    ]
    registrations_by_player: dict[int, list[dict[str, Any]]] = {}
    for registration in registrations:
        player_id = _safe_int(registration.get("player_id"))
        if player_id is not None:
            registrations_by_player.setdefault(player_id, []).append(registration)

    active_registration_statuses = {"ACTIVE", "APPROVED", "CONFIRMED", "REGISTERED"}

    def player_registration_blockers(player_ids: set[int]) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        for player_id in sorted(player_ids):
            if player_id not in players:
                blockers.append(
                    _blocker(
                        "PLAYER_SCOPE_INVALID",
                        "A participant does not belong to this club's player directory.",
                    )
                )
                continue
            active_registrations = [
                row
                for row in registrations_by_player.get(player_id, [])
                if _text(row.get("status")).upper() in active_registration_statuses
            ]
            if len(active_registrations) != 1:
                blockers.append(
                    _blocker(
                        "REGISTRATION_AMBIGUOUS",
                        "A participant must resolve to exactly one active tournament registration.",
                    )
                )
        return list({row["code"]: row for row in blockers}.values())

    live_courts = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_day_live_courts",
                filters=(("eq", "run_id", run_id),),
            )
            if run_id
            else []
        )
        if run_id and _text(row.get("run_id")) == run_id
    ]
    queue_rows = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_day_live_queue",
                filters=(("eq", "run_id", run_id),),
            )
            if run_id
            else []
        )
        if run_id and _text(row.get("run_id")) == run_id
    ]
    queue_by_game = {_text(row.get("game_id")): row for row in queue_rows}
    claim_rows = [
        row
        for row in (
            _rows(
                supabase,
                "tournament_day_live_participant_claims",
                filters=(("eq", "run_id", run_id),),
            )
            if run_id
            else []
        )
    ]
    queue_by_court = {
        _text(row.get("court_id")): row
        for row in queue_rows
        if _text(row.get("state")).upper() in ACTIVE_QUEUE_STATES
        and not row.get("released_at")
        and row.get("court_id")
    }

    operation_entity_ids = sorted(
        {f"{tournament_id}:{registration_day_id}", *draw_ids}
    )
    raw_operations = [
        row
        for row in _rows(
            supabase,
            "tournament_admin_operations",
            filters=(
                ("eq", "club_id", str(club_id)),
                ("eq", "surface", TOURNAMENT_DAY_LIVE_SURFACE),
                ("in", "entity_id", operation_entity_ids),
            ),
        )
        if _text(row.get("club_id")) == str(club_id)
        and _text(row.get("surface")) == TOURNAMENT_DAY_LIVE_SURFACE
        and _text(row.get("operation_key")) != _text(exclude_operation_key)
        and (
            _text(row.get("entity_id")) in set(operation_entity_ids)
            or _text(row.get("lock_scope"))
            == f"tournament:{tournament_id}:day:{registration_day_id}"
        )
    ]
    raw_operations.sort(key=lambda row: _text(row.get("updated_at")), reverse=True)
    unsettled_operation_present = any(
        _text(row.get("status")) in ACTIVE_OPERATION_STATUSES
        for row in raw_operations
    )

    inventory = {
        _text(item.get("id")): item
        for item in list(settings.get("venue_courts_json") or [])
        if isinstance(item, dict) and _text(item.get("id"))
    }
    configured_court_plan: list[dict[str, Any]] = []
    configured_labels = list(selected_day.get("court_labels") or [])
    for position, court_key in enumerate(
        [str(value) for value in list(selected_day.get("available_court_ids") or [])],
        start=1,
    ):
        item = inventory.get(court_key) or {}
        configured_court_plan.append(
            {
                "court_key": court_key,
                "label": (
                    _text(item.get("title"))
                    or _text(item.get("label"))
                    or (
                        _text(configured_labels[position - 1])
                        if position <= len(configured_labels)
                        else ""
                    )
                    or f"Court {position}"
                ),
                "position": position,
            }
        )
    if live_courts:
        court_source = sorted(
            live_courts,
            key=lambda row: (_safe_int(row.get("position"), 0), _text(row.get("id"))),
        )
    else:
        court_source = [
            {
                "id": court["court_key"],
                **court,
                "state": "OPEN",
                "version": 0,
            }
            for court in configured_court_plan
        ]

    court_rows: list[dict[str, Any]] = []
    for court in court_source:
        court_id = _text(court.get("id"))
        assignment = queue_by_court.get(court_id)
        operator_state = (
            "ON_COURT"
            if assignment
            else ("AVAILABLE" if _text(court.get("state")).upper() == "OPEN" else _text(court.get("state")).upper())
        )
        assignment_row = None
        if assignment:
            assignment_row = {
                "id": _text(assignment.get("id")),
                "game_id": _text(assignment.get("game_id")),
                "state": _text(assignment.get("state")).upper(),
                "version": str(assignment.get("version") or "0"),
                "assigned_at": assignment.get("called_at") or assignment.get("held_at") or assignment.get("updated_at"),
                "started_at": assignment.get("started_at"),
            }
        court_rows.append(
            {
                "id": court_id,
                "label": _text(court.get("label") or f"Court {_safe_int(court.get('position'), 0) or len(court_rows) + 1}"),
                "position": _safe_int(court.get("position"), 0),
                "state": operator_state,
                "version": str(court.get("version") or "0"),
                "current_assignment": assignment_row,
            }
        )

    game_rows: list[dict[str, Any]] = []
    for game in games:
        game_id = _text(game.get("id"))
        queue = queue_by_game.get(game_id)
        team_a = teams_by_id.get(_text(game.get("team_a_id")))
        team_b = teams_by_id.get(_text(game.get("team_b_id")))
        draw = next((row for row in draws if _text(row.get("id")) == _text(game.get("draw_id"))), {})
        scoring = _game_scoring(events.get(_text(game.get("event_option_id"))))
        day_draw = day_draw_by_draw.get(_text(game.get("draw_id"))) or {}
        correction_blockers: list[dict[str, Any]] = []
        run_state_for_game = _text((run or {}).get("state") or "DRAFT").upper()
        if run_state_for_game not in {"ACTIVE", "PAUSED"}:
            correction_blockers.append(
                _blocker(
                    "DAY_NOT_OPEN",
                    "Completed scores can be corrected only while the tournament day is active or paused.",
                )
            )
        if not queue or _text(queue.get("state")).upper() != "COMPLETED":
            correction_blockers.append(
                _blocker(
                    "GAME_NOT_DAY_COMPLETED",
                    "Only a completed game owned by this day queue can be corrected.",
                )
            )
        elif (
            queue.get("court_id") not in (None, "")
            or queue.get("released_at") in (None, "")
            or any(
                _text(claim.get("queue_id")) == _text(queue.get("id"))
                and claim.get("released_at") in (None, "")
                for claim in claim_rows
            )
        ):
            correction_blockers.append(
                _blocker(
                    "GAME_RESOURCES_NOT_RELEASED",
                    "The completed game must have no court or active participant claim before correction.",
                )
            )
        if _text(game.get("stage")).upper() != "ROUND_ROBIN":
            correction_blockers.append(
                _blocker(
                    "PLAYOFF_CORRECTION_UNAVAILABLE",
                    "Playoff score correction requires an explicit downstream bracket reset and remains closed.",
                )
            )
        if any(
            _text(row.get("draw_id")) == _text(game.get("draw_id"))
            and _text(row.get("stage")).upper() == "PLAYOFF"
            for row in all_draw_games
        ):
            correction_blockers.append(
                _blocker(
                    "PLAYOFF_RESET_REQUIRED",
                    "Round-robin scores cannot be corrected after playoff games exist; review and reset the bracket first.",
                )
            )
        if any(
            _text(row.get("draw_id")) == _text(game.get("draw_id"))
            for row in podium_rows
        ):
            correction_blockers.append(
                _blocker(
                    "PODIUM_EXISTS",
                    "Remove or reset reviewed podium evidence before correcting this score.",
                )
            )
        if _text(game.get("draw_id")) in published_draw_ids:
            correction_blockers.append(
                _blocker(
                    "OFFICIAL_MATCH_EXISTS",
                    "This result is already published as an official match and cannot be corrected here.",
                )
            )
        if not day_draw or _text(day_draw.get("state")).upper() not in {"ACTIVE", "PAUSED"}:
            correction_blockers.append(
                _blocker(
                    "DRAW_NOT_DAY_OWNED",
                    "The game draw is not an active or paused member of this tournament day.",
                )
            )
        if not _is_finalized_non_tied(game):
            correction_blockers.append(
                _blocker(
                    "RESULT_EVIDENCE_INVALID",
                    "The current game must retain exact finalized result evidence before correction.",
                )
            )
        if _text(game.get("result_type") or "PLAYED").upper() != "PLAYED":
            correction_blockers.append(
                _blocker(
                    "NON_PLAYED_RESULT",
                    "Forfeit, no-show, and retirement outcomes must be reviewed as outcomes, not rewritten as played scores.",
                )
            )
        team_a_for_game = teams_by_id.get(_text(game.get("team_a_id"))) or {}
        team_b_for_game = teams_by_id.get(_text(game.get("team_b_id"))) or {}
        if (
            _text(team_a_for_game.get("draw_id")) != _text(game.get("draw_id"))
            or _text(team_b_for_game.get("draw_id")) != _text(game.get("draw_id"))
            or _text(team_a_for_game.get("tournament_id")) != str(tournament_id)
            or _text(team_b_for_game.get("tournament_id")) != str(tournament_id)
        ):
            correction_blockers.append(
                _blocker(
                    "TEAM_SCOPE_INVALID",
                    "Both corrected game sides must remain owned by this exact tournament draw.",
                )
            )
        if unsettled_operation_present:
            correction_blockers.append(
                _blocker(
                    "OPERATION_UNSETTLED",
                    "Reconcile unsettled tournament day operations before correcting a score.",
                )
            )
        game_rows.append(
            {
                "id": game_id,
                "draw_id": _text(game.get("draw_id")),
                "draw_name": _text(draw.get("name") or draw.get("label") or "Draw"),
                "state": _text((queue or {}).get("state") or ("COMPLETED" if _is_finalized_non_tied(game) else "UNQUEUED")).upper(),
                "stage": _text(game.get("stage")).upper(),
                "round_label": _round_label(game),
                "slot_label": (
                    f"Slot {_safe_int(game.get('rr_slot_number'))}"
                    if _safe_int(game.get("rr_slot_number")) is not None
                    else _text(game.get("playoff_game_code")) or None
                ),
                "team_a": _side(team_a, players),
                "team_b": _side(team_b, players),
                "score_a": _safe_int(game.get("score_a")),
                "score_b": _safe_int(game.get("score_b")),
                "scoring": scoring,
                "result_type": _text(game.get("result_type") or "PLAYED").upper(),
                "result_note": game.get("result_note"),
                "result_recorded_by": game.get("result_recorded_by"),
                "score_review": dict(game.get("score_review_json") or {}),
                "team_a_id": _text(game.get("team_a_id")) or None,
                "team_b_id": _text(game.get("team_b_id")) or None,
                "winner_team_id": _text(game.get("winner_team_id")) or None,
                "loser_team_id": _text(game.get("loser_team_id")) or None,
                "finalized_at": game.get("finalized_at"),
                "rr_round_number": _safe_int(game.get("rr_round_number")),
                "rr_slot_number": _safe_int(game.get("rr_slot_number")),
                "playoff_game_code": game.get("playoff_game_code"),
                "playoff_round": game.get("playoff_round"),
                "team_a_source": game.get("team_a_source"),
                "team_b_source": game.get("team_b_source"),
                "winner_name": (
                    _side(teams_by_id.get(_text(game.get("winner_team_id"))), players).get("name")
                    if game.get("winner_team_id")
                    else None
                ),
                "updated_at": game.get("updated_at"),
                "version": _text(game.get("updated_at")) or "0",
                "queue_entry_version": str((queue or {}).get("version") or "0"),
                "court_id": _text((queue or {}).get("court_id")) or None,
                "blockers": (
                    [_blocker(_text(queue.get("blocker_code")) or "BLOCKED", _text(queue.get("blocker_detail")) or "This game is not currently eligible.")]
                    if queue and _text(queue.get("state")).upper() == "BLOCKED"
                    else []
                ),
                "correction_readiness": _readiness(
                    correction_blockers,
                    COMMAND_CONFIRMATIONS["correct_completed_score"],
                ),
            }
        )

    draw_rows: list[dict[str, Any]] = []
    assignment_blockers_by_draw: dict[str, list[dict[str, Any]]] = {}
    close_progression_blockers: list[dict[str, Any]] = []
    podium_reviews_by_draw: dict[str, dict[str, Any]] = {}
    run_state = _text((run or {}).get("state") or "DRAFT").upper()
    unsettled_operations = [
        row for row in raw_operations if _text(row.get("status")) in ACTIVE_OPERATION_STATUSES
    ]
    for draw in draws:
        draw_id = _text(draw.get("id"))
        day_draw = day_draw_by_draw.get(draw_id)
        activation_state = _text((day_draw or {}).get("state") or "INACTIVE").upper()
        draw_games = [row for row in games if _text(row.get("draw_id")) == draw_id]
        foreign_day_games = [
            row
            for row in all_draw_games
            if _text(row.get("draw_id")) == draw_id
            and (
                _text(row.get("registration_day_id")) != str(registration_day_id)
                or _text(row.get("event_option_id"))
                != _text(draw.get("event_option_id"))
            )
        ]
        draw_queue = [row for row in queue_rows if _text(row.get("draw_id")) == draw_id]
        source_scope_blockers: list[dict[str, Any]] = []
        if selected_day.get("enabled") is not True:
            source_scope_blockers.append(
                _blocker(
                    "DAY_DISABLED",
                    "New court assignments are stopped because this tournament day is disabled.",
                )
            )
        activate_blockers: list[dict[str, Any]] = []
        if run_state != "ACTIVE":
            activate_blockers.append(
                _blocker(
                    "DAY_NOT_ACTIVE",
                    "The tournament day must be active before changing draw activation.",
                )
            )
        if activation_state != "INACTIVE":
            activate_blockers.append(_blocker("DRAW_ALREADY_ACTIVATED", "This draw is already part of the day run."))
        draw_status = _text(draw.get("status") or "DRAFT").upper()
        draw_kind = _text(draw.get("draw_kind") or "STANDARD").upper()
        if (
            draw_status in {"CANCELLED", "CANCELED", "ARCHIVED", "DISABLED"}
            or bool(draw.get("hidden_from_primary_ops", False))
            or draw_kind != "STANDARD"
        ):
            source_scope_blockers.append(
                _blocker(
                    "DRAW_UNAVAILABLE",
                    "New court assignments are stopped because this draw is inactive, hidden, or unsupported.",
                )
            )
        event = events.get(_text(draw.get("event_option_id")))
        if (
            not event
            or event.get("enabled") is not True
            or _text(event.get("status") or "DRAFT").upper()
            in {"CANCELLED", "CANCELED", "ARCHIVED", "DISABLED"}
            or not _draw_scheduled_for_day(draw, event, str(registration_day_id))
        ):
            source_scope_blockers.append(
                _blocker(
                    "DRAW_UNSCHEDULED",
                    "New court assignments are stopped until the enabled draw event is restored to this day.",
                )
            )
        if day_draw and _text(day_draw.get("source_draw_updated_at")) != _text(
            draw.get("updated_at")
        ):
            source_scope_blockers.append(
                _blocker(
                    "DRAW_SOURCE_CHANGED",
                    "New court assignments are stopped because draw setup changed; pause and resume from refreshed evidence.",
                )
            )
        source_scope_blockers = list(
            {row["code"]: row for row in source_scope_blockers}.values()
        )
        assignment_blockers_by_draw[draw_id] = source_scope_blockers
        activate_blockers.extend(source_scope_blockers)
        if not draw_games:
            activate_blockers.append(_blocker("NO_GAMES", "Generate and review the draw games before activation."))
        if foreign_day_games:
            activate_blockers.append(
                _blocker(
                    "GAME_DAY_SCOPE_INVALID",
                    "Every game in this draw must belong to the selected tournament day.",
                )
            )
        draw_event_id = _text(draw.get("event_option_id"))
        for game in draw_games:
            stage = _text(game.get("stage")).upper()
            if stage not in {"ROUND_ROBIN", "PLAYOFF"}:
                activate_blockers.append(
                    _blocker(
                        "GAME_STAGE_UNSUPPORTED",
                        "Every draw game must be a supported round-robin or playoff game.",
                    )
                )
            if stage == "PLAYOFF":
                activate_blockers.append(
                    _blocker(
                        "PLAYOFFS_ALREADY_GENERATED",
                        "Activate this draw from reviewed round-robin games before generating playoffs in Day Live.",
                    )
                )
            if game.get("finalized_at") is None and any(
                game.get(field) is not None
                for field in ("score_a", "score_b", "winner_team_id", "loser_team_id")
            ):
                activate_blockers.append(
                    _blocker(
                        "GAME_STATE_UNSETTLED",
                        "Partially scored games must be reconciled before draw activation.",
                    )
                )
            if game.get("finalized_at") is not None and not _is_finalized_non_tied(game):
                activate_blockers.append(
                    _blocker(
                        "GAME_STATE_UNSETTLED",
                        "Every finalized game must have a complete non-tied result.",
                    )
                )
            team_a_id = _text(game.get("team_a_id"))
            team_b_id = _text(game.get("team_b_id"))
            game_teams = [
                teams_by_id.get(team_a_id) if team_a_id else None,
                teams_by_id.get(team_b_id) if team_b_id else None,
            ]
            if (
                (
                    team_a_id
                    and (
                        game_teams[0] is None
                        or _text((game_teams[0] or {}).get("registration_day_id"))
                        != str(registration_day_id)
                        or _text((game_teams[0] or {}).get("event_option_id"))
                        != draw_event_id
                    )
                )
                or (
                    team_b_id
                    and (
                        game_teams[1] is None
                        or _text((game_teams[1] or {}).get("registration_day_id"))
                        != str(registration_day_id)
                        or _text((game_teams[1] or {}).get("event_option_id"))
                        != draw_event_id
                    )
                )
                or (team_a_id and team_b_id and team_a_id == team_b_id)
            ):
                activate_blockers.append(
                    _blocker(
                        "TEAM_SCOPE_INVALID",
                        "Every game side must belong to this exact tournament draw.",
                    )
                )
            effective = [
                player_id
                for team in game_teams
                if team
                for player_id in (
                    _safe_int(team.get("player1_id")),
                    _safe_int(team.get("player2_id")),
                )
                if player_id is not None
            ]
            if stage == "ROUND_ROBIN" and (
                not team_a_id
                or not team_b_id
                or team_a_id == team_b_id
                or any(team is None for team in game_teams)
            ):
                activate_blockers.append(
                    _blocker(
                        "ROUND_ROBIN_TEAMS_REQUIRED",
                        "Every round-robin game requires two distinct teams owned by this draw.",
                    )
                )
            if team_a_id and team_b_id and (
                any(team is None for team in game_teams)
                or not all(
                    _safe_int((team or {}).get("player1_id")) is not None
                    for team in game_teams
                )
                or team_a_id == team_b_id
                or len(effective) not in {2, 4}
                or len(effective) != len(set(effective))
            ):
                activate_blockers.append(
                    _blocker(
                        "PARTICIPANTS_INVALID",
                        "Every playable game requires two or four distinct effective players on valid teams.",
                    )
                )
        all_draw_teams = [row for row in teams if _text(row.get("draw_id")) == draw_id]
        draw_teams = [
            row
            for row in all_draw_teams
            if _text(row.get("draw_id")) == draw_id
            and _text(row.get("registration_day_id")) == str(registration_day_id)
            and _text(row.get("event_option_id")) == draw_event_id
        ]
        if len(draw_teams) != len(all_draw_teams):
            activate_blockers.append(
                _blocker(
                    "TEAM_DAY_SCOPE_INVALID",
                    "Every team in this draw must belong to the selected tournament day and draw event.",
                )
            )
        exact_draw_team_ids = {_text(row.get("id")) for row in draw_teams}
        rr_side_team_ids = {
            _text(game.get(field))
            for game in draw_games
            if _text(game.get("stage")).upper() == "ROUND_ROBIN"
            for field in ("team_a_id", "team_b_id")
            if _text(game.get(field))
        }
        roster_mismatch_blockers: list[dict[str, Any]] = []
        if exact_draw_team_ids != rr_side_team_ids:
            roster_mismatch_blockers.append(
                _blocker(
                    "ROUND_ROBIN_ROSTER_MISMATCH",
                    "Every exact draw team must appear in the reviewed round-robin schedule before activation or playoff generation.",
                )
            )
            activate_blockers.extend(roster_mismatch_blockers)
        exact_roster_player_ids = [
            player_id
            for team in draw_teams
            for player_id in (
                _safe_int(team.get("player1_id")),
                _safe_int(team.get("player2_id")),
            )
            if player_id is not None
        ]
        roster_uniqueness_blockers: list[dict[str, Any]] = []
        if len(exact_roster_player_ids) != len(set(exact_roster_player_ids)):
            roster_uniqueness_blockers.append(
                _blocker(
                    "DRAW_ROSTER_PLAYER_DUPLICATE",
                    "Each player may belong to only one exact team in this draw before activation or playoff generation.",
                )
            )
            activate_blockers.extend(roster_uniqueness_blockers)
        activate_blockers.extend(
            player_registration_blockers(set(exact_roster_player_ids))
        )
        allowed_advance_counts = [
            count for count in SUPPORTED_ADVANCE_COUNTS if count <= len(draw_teams)
        ]
        if not allowed_advance_counts:
            activate_blockers.append(
                _blocker(
                    "PLAYOFF_FORMAT_UNAVAILABLE",
                    "At least four exact in-draw teams are required to reach a supported playoff and closeout format.",
                )
            )
        activate_blockers = list(
            {row["code"]: row for row in activate_blockers}.values()
        )

        configured_advance_count = _safe_int(
            draw.get("playoff_advance_count")
            if draw.get("playoff_advance_count") not in (None, "")
            else tournament.get("playoff_advance_count")
        )
        default_advance_count = (
            configured_advance_count
            if configured_advance_count in allowed_advance_counts
            else None
        )
        rr_games = [row for row in draw_games if _text(row.get("stage")).upper() == "ROUND_ROBIN"]
        playoff_games = [row for row in draw_games if _text(row.get("stage")).upper() == "PLAYOFF"]
        playoff_blockers: list[dict[str, Any]] = []
        playoff_blockers.extend(source_scope_blockers)
        playoff_blockers.extend(roster_mismatch_blockers)
        playoff_blockers.extend(roster_uniqueness_blockers)
        if run_state != "ACTIVE":
            playoff_blockers.append(
                _blocker(
                    "DAY_NOT_ACTIVE",
                    "The tournament day must be active before generating playoffs.",
                )
            )
        if activation_state != "ACTIVE":
            playoff_blockers.append(_blocker("DRAW_NOT_ACTIVE", "Activate this draw before generating playoffs."))
        if not rr_games or not all(_is_finalized_non_tied(row) for row in rr_games):
            playoff_blockers.append(_blocker("ROUND_ROBIN_INCOMPLETE", "Every round-robin game must be finalized and non-tied."))
        if playoff_games:
            playoff_blockers.append(_blocker("PLAYOFFS_EXIST", "This draw already has playoff games."))
        if not allowed_advance_counts:
            playoff_blockers.append(
                _blocker(
                    "PLAYOFF_FORMAT_UNAVAILABLE",
                    "This draw does not have enough reviewed teams for a supported playoff format.",
                )
            )
        if any(_text(row.get("state")).upper() in ACTIVE_QUEUE_STATES for row in draw_queue):
            playoff_blockers.append(_blocker("DRAW_ON_COURT", "Finish this draw's current court assignments first."))
        if unsettled_operations:
            playoff_blockers.append(_blocker("OPERATION_UNSETTLED", "Reconcile unsettled tournament day operations first."))

        podium_blockers: list[dict[str, Any]] = []
        if not draw_games or not all(_is_finalized_non_tied(row) for row in draw_games):
            podium_blockers.append(_blocker("GAMES_INCOMPLETE", "Finalize every draw game before podium review."))
        draw_podium = [row for row in podium_rows if _text(row.get("draw_id")) == draw_id]
        podium_team_ids = [_text(row.get("team_id")) for row in draw_podium]
        podium_valid = (
            sorted(_safe_int(row.get("placement"), 0) for row in draw_podium)
            == [1, 2, 3]
            and len(set(podium_team_ids)) == 3
            and set(podium_team_ids).issubset({_text(row.get("id")) for row in draw_teams})
        )
        podium_matches_playoff_results = bool(
            podium_valid
            and _podium_matches_playoff_results(playoff_games, draw_podium)
        )
        review_fingerprint = build_admin_tournament_podium_review_fingerprint(
            draw=draw,
            teams=draw_teams,
            games=draw_games,
            podium=draw_podium,
        )
        podium_review = find_current_admin_tournament_podium_review(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=draw_id,
            review_fingerprint=review_fingerprint,
        )
        podium_reviews_by_draw[draw_id] = podium_review
        expected_awards = {
            (
                player_id,
                PODIUM_BADGE_MAP[int(placement)],
                f"{tournament_id}:draw:{draw_id}:podium:{placement}",
            )
            for podium in draw_podium
            for placement in [_safe_int(podium.get("placement"))]
            for team in [teams_by_id.get(_text(podium.get("team_id"))) or {}]
            for player_id in (
                _safe_int(team.get("player1_id")),
                _safe_int(team.get("player2_id")),
            )
            if placement in {1, 2, 3} and player_id is not None
        }
        context_prefix = f"{tournament_id}:draw:{draw_id}:podium:"
        actual_awards = {
            (
                _safe_int(row.get("player_id")),
                _text(row.get("badge_id")),
                _text(row.get("context_id")),
            )
            for row in podium_badge_rows
            if _text(row.get("context_id")).startswith(context_prefix)
            and row.get("revoked_at") in (None, "")
        }
        closeout_blockers: list[dict[str, Any]] = []
        if day_draw and activation_state != "REMOVED":
            if not playoff_games:
                closeout_blockers.append(
                    _blocker(
                        "PLAYOFFS_REQUIRED",
                        "Generate and finish this activated draw's playoffs before closing the day.",
                    )
                )
            elif not all(_is_finalized_non_tied(row) for row in draw_games):
                closeout_blockers.append(
                    _blocker(
                        "PLAYOFFS_INCOMPLETE",
                        "Finalize every round-robin and playoff game before closing the day.",
                    )
                )
            if not podium_valid:
                closeout_blockers.append(
                    _blocker(
                        "PODIUM_INCOMPLETE",
                        "Save exact first, second, and third place evidence before closing the day.",
                    )
                )
            elif not podium_matches_playoff_results:
                closeout_blockers.append(
                    _blocker(
                        "PODIUM_RESULT_MISMATCH",
                        "Podium placements must exactly match the finalized Final and Bronze playoff results.",
                    )
                )
            if not podium_review.get("current"):
                closeout_blockers.append(
                    _blocker(
                        "PODIUM_REVIEW_REQUIRED",
                        "Review the current podium evidence before closing the day.",
                    )
                )
            if not expected_awards or actual_awards != expected_awards:
                closeout_blockers.append(
                    _blocker(
                        "AWARDS_INCOMPLETE",
                        "Complete and verify every expected podium medal before closing the day.",
                    )
                )
        close_progression_blockers.extend(
            {
                **blocker,
                "draw_id": draw_id,
                "draw_name": _text(draw.get("name") or draw.get("label") or "Draw"),
            }
            for blocker in closeout_blockers
        )
        draw_rows.append(
            {
                "id": draw_id,
                "name": _text(draw.get("name") or draw.get("label") or events.get(_text(draw.get("event_option_id")), {}).get("label") or "Draw"),
                "state": activation_state,
                "activation_state": activation_state,
                "version": str((day_draw or {}).get("version") or draw.get("updated_at") or "0"),
                "source_updated_at": draw.get("updated_at"),
                "event_option_id": _text(draw.get("event_option_id")) or None,
                "stage": "PLAYOFF" if playoff_games else "ROUND_ROBIN",
                "total_games": len(draw_games),
                "finalized_games": sum(_is_finalized_non_tied(row) for row in draw_games),
                "queued_games": sum(_text(row.get("state")).upper() in {"WAITING", "BLOCKED"} for row in draw_queue),
                "active_games": sum(_text(row.get("state")).upper() in ACTIVE_QUEUE_STATES for row in draw_queue),
                "held_games": sum(_text(row.get("state")).upper() == "HELD" for row in draw_queue),
                "team_versions": sorted(
                    [
                        {"id": _text(row.get("id")), "updated_at": _text(row.get("updated_at"))}
                        for row in draw_teams
                    ],
                    key=lambda row: row["id"],
                ),
                "team_rows": sorted(
                    [
                        {
                            "id": _text(row.get("id")),
                            "team_number": _safe_int(row.get("team_number")),
                            "seed": _safe_int(row.get("seed")),
                        }
                        for row in draw_teams
                    ],
                    key=lambda row: (row.get("team_number") or 1_000_000, row["id"]),
                ),
                "source_game_versions": sorted(
                    [{"id": _text(row.get("id")), "updated_at": _text(row.get("updated_at"))} for row in draw_games],
                    key=lambda row: row["id"],
                ),
                "readiness": {
                    "activate": _readiness(activate_blockers, COMMAND_CONFIRMATIONS["activate_draw"]),
                    "pause": _readiness(
                        ([] if activation_state == "ACTIVE" else [_blocker("DRAW_NOT_ACTIVE", "Only an active draw can be paused.")])
                        + ([] if run_state == "ACTIVE" else [_blocker("DAY_NOT_ACTIVE", "The tournament day must be active before pausing a draw.")]),
                        COMMAND_CONFIRMATIONS["pause_draw"],
                    ),
                    "resume": _readiness(
                        ([] if activation_state == "PAUSED" else [_blocker("DRAW_NOT_PAUSED", "Only a paused draw can be resumed.")])
                        + ([] if run_state == "ACTIVE" else [_blocker("DAY_NOT_ACTIVE", "The tournament day must be active before resuming a draw.")])
                        + source_scope_blockers,
                        COMMAND_CONFIRMATIONS["resume_draw"],
                    ),
                    "assignments": _readiness(source_scope_blockers),
                    "generate_playoffs": {
                        **_readiness(
                            playoff_blockers,
                            COMMAND_CONFIRMATIONS["generate_playoffs"],
                        ),
                        "allowed_advance_counts": allowed_advance_counts,
                        "default_advance_count": default_advance_count,
                    },
                    "podium": _readiness(podium_blockers),
                    "closeout": _readiness(closeout_blockers),
                },
                "progression": {
                    "allowed_advance_counts": allowed_advance_counts,
                    "default_advance_count": default_advance_count,
                    "podium_href": f"/admin/tournaments/ops/draws?selectedTournamentId={tournament_id}&selectedDrawId={draw_id}",
                    "review_href": f"/admin/tournaments/live-operations/draws?selectedTournamentId={tournament_id}&selectedDrawId={draw_id}",
                },
            }
        )

    def queue_item(
        row: dict[str, Any],
        position: int,
        *,
        derived_blockers: list[dict[str, Any]] | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        blockers = list(derived_blockers or [])
        if row.get("blocker_code"):
            blockers.append(_blocker(_text(row.get("blocker_code")), _text(row.get("blocker_detail")) or "Game is blocked."))
        return {
            "game_id": _text(row.get("game_id")),
            "draw_id": _text(row.get("draw_id")),
            "position": position,
            "priority": _safe_int(row.get("priority"), position),
            "state": state or _text(row.get("state")).upper(),
            "version": str(row.get("version") or "0"),
            "court_id": _text(row.get("court_id")) or None,
            "eligible_since": row.get("eligible_since"),
            "reason": (blockers[0]["code"] if blockers else row.get("blocker_code")),
            "note": (blockers[0]["message"] if blockers else row.get("blocker_detail")),
            "held_at": row.get("held_at"),
            "blockers": blockers,
        }

    active_claimed_players = {
        int(row["player_id"])
        for row in claim_rows
        if row.get("released_at") is None and _safe_int(row.get("player_id")) is not None
    }
    active_assignments_by_draw = {
        draw_id: sum(
            _text(row.get("draw_id")) == draw_id
            and _text(row.get("state")).upper() in ACTIVE_QUEUE_STATES
            and not row.get("released_at")
            for row in queue_rows
        )
        for draw_id in draw_ids
    }

    def effective_players_for_queue(row: dict[str, Any]) -> set[int]:
        return {
            player_id
            for team_id in {
                _text(row.get("team_a_id")),
                _text(row.get("team_b_id")),
            }
            for player_id in (
                _safe_int((teams_by_id.get(team_id) or {}).get("player1_id")),
                _safe_int((teams_by_id.get(team_id) or {}).get("player2_id")),
            )
            if team_id and player_id is not None
        }

    def waiting_blockers(row: dict[str, Any]) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        draw_id = _text(row.get("draw_id"))
        day_draw = day_draw_by_draw.get(draw_id) or {}
        game = next((item for item in games if _text(item.get("id")) == _text(row.get("game_id"))), None)
        if _text(day_draw.get("state")).upper() != "ACTIVE":
            blockers.append(_blocker("DRAW_NOT_ACTIVE", "This draw is not active for court assignment."))
        blockers.extend(assignment_blockers_by_draw.get(draw_id, []))
        if not game or game.get("finalized_at") is not None:
            blockers.append(_blocker("GAME_NOT_OPEN", "This game is no longer open for court assignment."))
            return blockers
        team_ids_for_game = {_text(game.get("team_a_id")), _text(game.get("team_b_id"))}
        if (
            not all(team_ids_for_game)
            or _text(row.get("team_a_id")) != _text(game.get("team_a_id"))
            or _text(row.get("team_b_id")) != _text(game.get("team_b_id"))
        ):
            blockers.append(_blocker("TEAM_STATE_CHANGED", "The game's assigned teams changed after queueing."))
        effective = {
            player_id
            for team_id in team_ids_for_game
            for player_id in (
                _safe_int((teams_by_id.get(team_id) or {}).get("player1_id")),
                _safe_int((teams_by_id.get(team_id) or {}).get("player2_id")),
            )
            if player_id is not None
        }
        raw_effective_count = sum(
            player_id is not None
            for team_id in team_ids_for_game
            for player_id in (
                _safe_int((teams_by_id.get(team_id) or {}).get("player1_id")),
                _safe_int((teams_by_id.get(team_id) or {}).get("player2_id")),
            )
        )
        if raw_effective_count not in {2, 4} or len(effective) != raw_effective_count:
            blockers.append(_blocker("PARTICIPANTS_INVALID", "This game does not have two or four distinct effective players."))
        if effective & active_claimed_players:
            blockers.append(_blocker("PLAYER_ALREADY_CLAIMED", "A participant is already on another court for this tournament day."))
        blockers.extend(player_registration_blockers(effective))
        priority = _safe_int(row.get("priority"), 0) or 0
        if any(
            _text(earlier.get("draw_id")) == draw_id
            and (_safe_int(earlier.get("priority"), 0) or 0) < priority
            and _text(earlier.get("state")).upper() not in {"COMPLETED", "WITHDRAWN"}
            and bool(
                {_text(earlier.get("team_a_id")), _text(earlier.get("team_b_id"))}
                & team_ids_for_game
            )
            for earlier in queue_rows
        ):
            blockers.append(_blocker("EARLIER_GAME_UNFINISHED", "A team has an earlier unfinished game in this draw."))
        if _text(game.get("stage")).upper() == "PLAYOFF" and any(
            _text(other.get("draw_id")) == draw_id
            and _text(other.get("stage")).upper() == "ROUND_ROBIN"
            and not _is_finalized_non_tied(other)
            for other in games
        ):
            blockers.append(
                _blocker(
                    "ROUND_ROBIN_INCOMPLETE",
                    "Every round-robin game in this draw must finish before a playoff can take a court.",
                )
            )
        return list({item["code"]: item for item in blockers}.values())

    def fairness_key(
        row: dict[str, Any],
        *,
        assignment_counts: dict[str, int] | None = None,
        last_assigned_values: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        draw_id = _text(row.get("draw_id"))
        day_draw = day_draw_by_draw.get(draw_id) or {}
        game = next((item for item in games if _text(item.get("id")) == _text(row.get("game_id"))), {})
        counts = assignment_counts or active_assignments_by_draw
        last_assigned = (
            last_assigned_values.get(draw_id)
            if last_assigned_values is not None
            else day_draw.get("last_assigned_at")
        )
        return (
            counts.get(draw_id, 0),
            0 if last_assigned is None else 1,
            _text(last_assigned),
            _text(day_draw.get("activated_at")),
            draw_id,
            *_game_sort_key(game),
        )

    ordered_queue = sorted(queue_rows, key=lambda row: (_safe_int(row.get("priority"), 0), _text(row.get("id"))))
    eligible_rows: list[dict[str, Any]] = []
    derived_blocked: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for row in queue_rows:
        if _text(row.get("state")).upper() != "WAITING":
            continue
        blockers = waiting_blockers(row)
        if blockers:
            derived_blocked.append((row, blockers))
        else:
            eligible_rows.append(row)
    # PostgreSQL recomputes fairness after every court assignment. Simulate
    # that sequence so visible queue positions match an atomic multi-court fill
    # instead of sorting once and letting one draw monopolize the front.
    remaining = list(eligible_rows)
    eligible_rows = []
    virtual_counts = dict(active_assignments_by_draw)
    virtual_last = {
        draw_id: (day_draw_by_draw.get(draw_id) or {}).get("last_assigned_at")
        for draw_id in draw_ids
    }
    assignment_ordinal = 0
    virtual_claimed_players = set(active_claimed_players)
    available_assignment_slots = sum(
        row.get("state") == "AVAILABLE" for row in court_rows
    )
    while remaining and assignment_ordinal < available_assignment_slots:
        candidates = [
            row
            for row in remaining
            if not (
                effective_players_for_queue(row) & virtual_claimed_players
            )
        ]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda row: fairness_key(
                row,
                assignment_counts=virtual_counts,
                last_assigned_values=virtual_last,
            ),
        )
        remaining.remove(chosen)
        eligible_rows.append(chosen)
        chosen_draw_id = _text(chosen.get("draw_id"))
        virtual_counts[chosen_draw_id] = virtual_counts.get(chosen_draw_id, 0) + 1
        assignment_ordinal += 1
        virtual_claimed_players.update(effective_players_for_queue(chosen))
        virtual_last[chosen_draw_id] = f"9999-12-31T23:59:59.{assignment_ordinal:06d}Z"
    # Positions through the available-court window are the exact next atomic
    # fill sequence. Rows after that window remain eligible but are ordered for
    # display only because future releases change the scheduler state.
    remaining.sort(
        key=lambda row: fairness_key(
            row,
            assignment_counts=virtual_counts,
            last_assigned_values=virtual_last,
        )
    )
    eligible_rows.extend(remaining)
    eligible_queue = []
    for index, row in enumerate(eligible_rows, start=1):
        item = queue_item(row, index)
        item["immediate_fill_candidate"] = index <= assignment_ordinal
        eligible_queue.append(item)
    held_games = [queue_item(row, index) for index, row in enumerate([row for row in ordered_queue if _text(row.get("state")).upper() == "HELD"], start=1)]
    persisted_blocked = [(row, []) for row in ordered_queue if _text(row.get("state")).upper() == "BLOCKED"]
    blocked_games = [
        queue_item(row, index, derived_blockers=blockers, state="BLOCKED")
        for index, (row, blockers) in enumerate(persisted_blocked + sorted(derived_blocked, key=lambda item: fairness_key(item[0])), start=1)
    ]

    day_run = {
        "id": run_id,
        "registration_day_id": str(registration_day_id),
        "state": _text((run or {}).get("state") or "DRAFT").upper(),
        "version": str((run or {}).get("version") or "0"),
        "updated_at": (run or {}).get("updated_at"),
    }
    state_evidence = {
        "tournament": {"id": tournament_id, "updated_at": tournament.get("updated_at")},
        "day": {"id": registration_day_id, "updated_at": selected_day.get("updated_at")},
        "configured_court_plan": configured_court_plan,
        "run": run,
        "tournament_runs": [
            {
                "id": row.get("id"),
                "registration_day_id": row.get("registration_day_id"),
                "state": row.get("state"),
                "version": row.get("version"),
                "updated_at": row.get("updated_at"),
            }
            for row in tournament_runs
        ],
        "day_draws": day_draw_rows,
        "courts": live_courts,
        "queue": queue_rows,
        "participant_claims": claim_rows,
        "draws": [{"id": row.get("id"), "updated_at": row.get("updated_at")} for row in draws],
        "teams": [{"id": row.get("id"), "updated_at": row.get("updated_at")} for row in teams],
        "games": [
            {
                "id": row.get("id"),
                "updated_at": row.get("updated_at"),
                "score_a": row.get("score_a"),
                "score_b": row.get("score_b"),
                "winner_team_id": row.get("winner_team_id"),
                "result_type": row.get("result_type") or "PLAYED",
                "result_note": row.get("result_note"),
                "score_review_json": row.get("score_review_json") or {},
            }
            for row in games
        ],
        "official_matches": [
            {
                "id": row.get("id"),
                "tournament_game_id": row.get("tournament_game_id"),
                "updated_at": row.get("updated_at"),
            }
            for row in published_matches
        ],
        "podium": [
            {
                "id": row.get("id"),
                "draw_id": row.get("draw_id"),
                "placement": row.get("placement"),
                "team_id": row.get("team_id"),
                "source": row.get("source"),
                "updated_at": row.get("updated_at"),
            }
            for row in podium_rows
        ],
        "podium_badges": [
            {
                "id": row.get("id"),
                "player_id": row.get("player_id"),
                "badge_id": row.get("badge_id"),
                "context_id": row.get("context_id"),
                "revoked_at": row.get("revoked_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in podium_badge_rows
        ],
        "podium_reviews": {
            draw_id: {
                "current": review.get("current"),
                "review_fingerprint": review.get("review_fingerprint"),
                "reviewed_at": review.get("reviewed_at"),
            }
            for draw_id, review in podium_reviews_by_draw.items()
        },
        "draw_game_days": [
            {
                "id": row.get("id"),
                "draw_id": row.get("draw_id"),
                "registration_day_id": row.get("registration_day_id"),
            }
            for row in all_draw_games
        ],
        # Registration identity/status is structural evidence for a rostered
        # participant. Payment is intentionally omitted so a commerce update
        # cannot stale an otherwise safe live-day command.
        "registrations": [
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "player_id": row.get("player_id"),
            }
            for row in sorted(registrations, key=lambda item: _text(item.get("id")))
        ],
    }
    state_fingerprint = _fingerprint(state_evidence)

    activation_blockers: list[dict[str, Any]] = []
    if selected_day.get("enabled") is not True:
        activation_blockers.append(
            _blocker(
                "DAY_DISABLED",
                "Enable this tournament day before activation.",
            )
        )
    if run:
        activation_blockers.append(_blocker("DAY_ALREADY_ACTIVATED", "This day already has a durable run."))
    if any(
        _text(row.get("registration_day_id")) != str(registration_day_id)
        and _text(row.get("state")).upper() in {"ACTIVE", "PAUSED"}
        for row in tournament_runs
    ):
        activation_blockers.append(
            _blocker(
                "ANOTHER_DAY_ACTIVE",
                "Close the tournament's current live day before activating another day.",
            )
        )
    supported_scheduled_draw_ids = {
        _text(row.get("id"))
        for row in draw_rows
        if row.get("readiness", {}).get("assignments", {}).get("ready") is True
    }
    if not supported_scheduled_draw_ids:
        activation_blockers.append(
            _blocker(
                "NO_SUPPORTED_DRAWS",
                "Configure at least one supported draw on this tournament day before activation.",
            )
        )
    if not court_rows:
        activation_blockers.append(_blocker("NO_COURTS", "Configure at least one venue court for this day."))
    fill_blockers: list[dict[str, Any]] = []
    if day_run["state"] != "ACTIVE":
        fill_blockers.append(_blocker("DAY_NOT_ACTIVE", "Activate the tournament day before filling courts."))
    if not any(row["activation_state"] == "ACTIVE" for row in draw_rows):
        fill_blockers.append(_blocker("NO_ACTIVE_DRAWS", "Activate at least one draw before filling courts."))
    if not any(row["state"] == "AVAILABLE" for row in court_rows):
        fill_blockers.append(_blocker("NO_AVAILABLE_COURTS", "No day court is currently available."))
    if not eligible_queue:
        fill_blockers.append(_blocker("NO_ELIGIBLE_GAMES", "No game is currently eligible for a court."))
    close_blockers: list[dict[str, Any]] = []
    if day_run["state"] not in {"ACTIVE", "PAUSED"}:
        close_blockers.append(_blocker("DAY_NOT_OPEN", "Only an active or paused tournament day can be closed."))
    if any(
        _text(row.get("state")).upper() in ACTIVE_QUEUE_STATES
        and not row.get("released_at")
        for row in queue_rows
    ):
        close_blockers.append(_blocker("COURT_ASSIGNMENTS_ACTIVE", "Finish every current court assignment before closing the day."))
    if any(row.get("released_at") is None for row in claim_rows):
        close_blockers.append(_blocker("PLAYER_CLAIMS_ACTIVE", "Release every active participant claim before closing the day."))
    if any(
        _text(row.get("state")).upper() not in {"COMPLETED", "WITHDRAWN"}
        for row in queue_rows
    ):
        close_blockers.append(_blocker("GAMES_UNFINISHED", "Every owned game must be completed or withdrawn before closing the day."))
    represented_draw_ids = {
        _text(row.get("draw_id"))
        for row in day_draw_rows
        if _text(row.get("state")).upper() != "REMOVED"
    }
    if not represented_draw_ids:
        close_blockers.append(
            _blocker(
                "NO_ACTIVATED_DRAWS",
                "Activate and complete at least one scheduled draw before closing the tournament day.",
            )
        )
    missing_draw_ids = supported_scheduled_draw_ids - represented_draw_ids
    if missing_draw_ids:
        close_blockers.append(
            _blocker(
                "SCHEDULED_DRAWS_NOT_ACTIVATED",
                "Every currently supported draw scheduled for this day must be activated and completed before close.",
                draw_count=len(missing_draw_ids),
            )
        )
    close_blockers.extend(close_progression_blockers)
    if unsettled_operations:
        close_blockers.append(_blocker("OPERATION_UNSETTLED", "Reconcile unsettled day operations before closing the day."))

    runtime_state = tournament_admin_mutation_status()
    surface_state = runtime_state.get("surface_flags", {}).get(TOURNAMENT_DAY_LIVE_SURFACE, {})
    runtime = {
        "writes_enabled": bool(
            is_admin_tournament_admin_enabled()
            and runtime_state.get("environment") == "staging"
            and surface_state.get("enabled")
            and runtime_state.get("service_role_ready")
        ),
        "environment": runtime_state.get("environment"),
        "warnings": [],
    }
    warnings: list[str] = []
    if not runtime["writes_enabled"]:
        warnings.append("Tournament day writes are staging-gated; readiness remains independent of this environment gate.")

    return {
        "ok": True,
        "mode": "tournament_day_live",
        "scope": {
            "club_id": str(club_id),
            "tournament_id": str(tournament_id),
            "registration_day_id": str(registration_day_id),
        },
        "tournament": {
            "id": str(tournament_id),
            "name": _text(tournament.get("name") or tournament.get("title") or "Tournament"),
            "status": tournament.get("status"),
        },
        "day_scope": {
            "selected_day_id": str(registration_day_id),
            "selected_day": _day_option(selected_day),
            "available_days": [_day_option(row) for row in days],
        },
        "day_run": day_run,
        "state_fingerprint": state_fingerprint,
        "queue_version": str((run or {}).get("queue_version") or "0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "courts": len(court_rows),
            "available_courts": sum(row["state"] == "AVAILABLE" for row in court_rows),
            "active_draws": sum(row["activation_state"] == "ACTIVE" for row in draw_rows),
            "eligible_games": len(eligible_queue),
            "held_games": len(held_games),
            "completed_games": sum(_text(row.get("state")).upper() == "COMPLETED" for row in queue_rows),
        },
        "draws": draw_rows,
        "activated_draws": [row for row in draw_rows if row["activation_state"] != "INACTIVE"],
        "courts": court_rows,
        "games": game_rows,
        "eligible_queue": eligible_queue,
        "held_games": held_games,
        "blocked_games": blocked_games,
        "operations": [_safe_operation(row) for row in raw_operations[:25]],
        "readiness": {
            "activate_day": _readiness(activation_blockers, COMMAND_CONFIRMATIONS["activate_day"]),
            "auto_fill_courts": _readiness(fill_blockers, COMMAND_CONFIRMATIONS["auto_fill_courts"]),
            "close_day": _readiness(close_blockers, COMMAND_CONFIRMATIONS["close_day"]),
            "correct_completed_score": _readiness(
                (
                    []
                    if any(
                        row.get("correction_readiness", {}).get("ready")
                        for row in game_rows
                    )
                    else [
                        _blocker(
                            "NO_CORRECTABLE_COMPLETED_SCORE",
                            "No completed round-robin score is currently safe to correct.",
                        )
                    ]
                ),
                COMMAND_CONFIRMATIONS["correct_completed_score"],
            ),
        },
        "runtime": runtime,
        "warnings": warnings,
    }


def _require_permission(actor_role: str, action: str) -> None:
    permission = (
        PERMISSION_ENTER_SCORES
        if action in {
            "score_and_release",
            "correct_completed_score",
            "record_non_played_result",
        }
        else PERMISSION_MANAGE_TOURNAMENTS
    )
    if not has_permission(actor_role, permission):
        raise PermissionError("insufficient permission for this tournament day command")


def _normalize_request(request: dict[str, Any]) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    action = _text(request.get("action")).lower()
    if action not in COMMAND_CONFIRMATIONS:
        raise ValueError("Unsupported tournament day command.")
    confirmation = _text(request.get("confirmation_text"))
    required = COMMAND_CONFIRMATIONS[action]
    if confirmation != required:
        raise ValueError(f"Type {required} exactly to continue.")
    idempotency_key = _text(request.get("client_idempotency_key"))
    try:
        uuid.UUID(idempotency_key)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("A UUID client idempotency key is required.") from exc
    expected = dict(request.get("expected") or {})
    fingerprint = _text(expected.get("state_fingerprint"))
    if not STATE_HASH_RE.fullmatch(fingerprint):
        raise StaleTournamentAdminStateError("A reviewed tournament day fingerprint is required.")
    if expected.get("day_run_version") in (None, ""):
        raise StaleTournamentAdminStateError("A reviewed tournament day run version is required.")
    payload = dict(request.get("payload") or {})
    if action in {"score_and_release", "correct_completed_score"} and set(payload) == {
        "game_id", "score_a", "score_b"
    }:
        payload["unusual_score_acknowledgement"] = False
    payload_keys = {
        "activate_day": set(),
        "auto_fill_courts": set(),
        "close_day": set(),
        "assign_next_court": {"game_id"},
        "assign_game_to_court": {"game_id", "court_id"},
        "requeue_game": {"game_id"},
        "move_game_to_court": {"game_id", "court_id"},
        "activate_draw": {"draw_id"},
        "pause_draw": {"draw_id"},
        "resume_draw": {"draw_id"},
        "generate_playoffs": {"draw_id", "advance_count"},
        "score_and_release": {
            "game_id", "score_a", "score_b", "unusual_score_acknowledgement"
        },
        "correct_completed_score": {
            "game_id", "score_a", "score_b", "unusual_score_acknowledgement"
        },
        "record_non_played_result": {
            "game_id", "result_type", "winner_team_id", "result_note"
        },
    }[action]
    if set(payload) != payload_keys or any(payload[key] is None for key in payload_keys):
        raise ValueError(
            f"{action.replace('_', ' ').title()} requires its exact command payload."
        )
    return action, idempotency_key, fingerprint, expected, payload


def _selected_draw(snapshot: dict[str, Any], draw_id: str) -> dict[str, Any]:
    draw = next((row for row in snapshot.get("draws", []) if _text(row.get("id")) == draw_id), None)
    if not draw:
        raise ValueError("Selected draw does not belong to this tournament day.")
    return dict(draw)


def _selected_game(snapshot: dict[str, Any], game_id: str) -> dict[str, Any] | None:
    game = next((row for row in snapshot.get("games", []) if _text(row.get("id")) == game_id), None)
    return dict(game) if game else None


def _validate_review(snapshot: dict[str, Any], expected: dict[str, Any]) -> None:
    if _text(snapshot.get("state_fingerprint")) != _text(expected.get("state_fingerprint")):
        raise StaleTournamentAdminStateError("Tournament day data changed after review. Reload the day workspace.")
    if str(snapshot.get("day_run", {}).get("version") or "0") != str(expected.get("day_run_version")):
        raise StaleTournamentAdminStateError("Tournament day run version changed after review.")
    if expected.get("queue_version") not in (None, "") and str(snapshot.get("queue_version") or "0") != str(expected.get("queue_version")):
        raise StaleTournamentAdminStateError("Tournament day queue changed after review.")


def _preflight(snapshot: dict[str, Any], action: str, expected: dict[str, Any], payload: dict[str, Any]) -> None:
    _validate_review(snapshot, expected)
    if action == "activate_day":
        if payload:
            raise ValueError(
                "Activate the day without draws, then activate each reviewed draw individually."
            )
        if not snapshot.get("readiness", {}).get("activate_day", {}).get("ready"):
            raise ValueError("This tournament day is not ready to activate.")
        return
    if action == "close_day":
        if not snapshot.get("readiness", {}).get("close_day", {}).get("ready"):
            raise ValueError("This tournament day is not ready to close.")
        return
    if _text(snapshot.get("day_run", {}).get("state")).upper() != "ACTIVE":
        raise ValueError("Tournament day must be active for this command.")
    if action in {"activate_draw", "pause_draw", "resume_draw", "generate_playoffs"}:
        draw = _selected_draw(snapshot, _text(payload.get("draw_id")))
        readiness_name = {
            "activate_draw": "activate",
            "pause_draw": "pause",
            "resume_draw": "resume",
            "generate_playoffs": "generate_playoffs",
        }[action]
        if not draw.get("readiness", {}).get(readiness_name, {}).get("ready"):
            raise ValueError(f"Draw {draw.get('name') or 'name unavailable'} is not ready for {action.replace('_', ' ')}.")
        if expected.get("draw_version") in (None, "") or str(expected.get("draw_version")) != str(draw.get("version")):
            raise StaleTournamentAdminStateError("Tournament draw version changed after review.")
        if action == "generate_playoffs":
            advance_count = _safe_int(payload.get("advance_count"))
            allowed = {
                int(value)
                for value in draw.get("readiness", {})
                .get("generate_playoffs", {})
                .get("allowed_advance_counts", [])
                if _safe_int(value) is not None
            }
            if advance_count not in allowed:
                raise ValueError(
                    "Choose a supported advancing-team count from the reviewed draw."
                )
    elif action == "auto_fill_courts":
        # Rechecking the database RPC under locks is final authority. Allow a
        # no-op fill when no game is ready so operators can safely refresh it.
        return
    elif action in {"assign_next_court", "assign_game_to_court"}:
        game_id = _text(payload.get("game_id"))
        game = _selected_game(snapshot, game_id)
        queue_entry = next(
            (
                row
                for row in snapshot.get("eligible_queue", [])
                if _text(row.get("game_id")) == game_id
            ),
            None,
        )
        if not game or not queue_entry:
            raise ValueError("Choose a currently eligible queued game.")
        if expected.get("game_version") in (None, "") or str(
            expected.get("game_version")
        ) != str(game.get("version")):
            raise StaleTournamentAdminStateError(
                "Tournament game version changed after court-assignment review."
            )
        if expected.get("queue_entry_version") in (None, "") or str(
            expected.get("queue_entry_version")
        ) != str(queue_entry.get("version")):
            raise StaleTournamentAdminStateError(
                "Tournament queue entry changed after court-assignment review."
            )
        available_courts = [
            row
            for row in snapshot.get("courts", [])
            if _text(row.get("state")).upper() == "AVAILABLE"
            and not row.get("current_assignment")
        ]
        if not available_courts:
            raise ValueError("No tournament-day court is currently available.")
        if action == "assign_game_to_court":
            court_id = _text(payload.get("court_id"))
            court = next(
                (row for row in available_courts if _text(row.get("id")) == court_id),
                None,
            )
            if not court:
                raise StaleTournamentAdminStateError(
                    "The selected tournament-day court is no longer available."
                )
            if expected.get("court_version") in (None, "") or str(
                expected.get("court_version")
            ) != str(court.get("version")):
                raise StaleTournamentAdminStateError(
                    "Selected court version changed after assignment review."
                )
        elif expected.get("court_version") not in (None, ""):
            raise StaleTournamentAdminStateError(
                "Next-open-court assignment must let the server select the court."
            )
        return
    elif action in {"requeue_game", "move_game_to_court"}:
        game_id = _text(payload.get("game_id"))
        game = _selected_game(snapshot, game_id)
        source_court = next(
            (
                row
                for row in snapshot.get("courts", [])
                if _text((row.get("current_assignment") or {}).get("game_id"))
                == game_id
            ),
            None,
        )
        if not game or not source_court:
            raise StaleTournamentAdminStateError(
                "This game is no longer assigned to a tournament-day court."
            )
        if expected.get("game_version") in (None, "") or str(
            expected.get("game_version")
        ) != str(game.get("version")):
            raise StaleTournamentAdminStateError(
                "Tournament game version changed after assignment review."
            )
        if expected.get("queue_entry_version") in (None, "") or str(
            expected.get("queue_entry_version")
        ) != str(game.get("queue_entry_version")):
            raise StaleTournamentAdminStateError(
                "Tournament queue entry changed after assignment review."
            )
        if expected.get("court_version") in (None, "") or str(
            expected.get("court_version")
        ) != str(source_court.get("version")):
            raise StaleTournamentAdminStateError(
                "Assigned court version changed after assignment review."
            )
        if action == "requeue_game":
            if expected.get("target_court_version") not in (None, ""):
                raise StaleTournamentAdminStateError(
                    "Returning a game to the queue cannot include a target court."
                )
            return
        target_court_id = _text(payload.get("court_id"))
        target_court = next(
            (
                row
                for row in snapshot.get("courts", [])
                if _text(row.get("id")) == target_court_id
                and _text(row.get("state")).upper() == "AVAILABLE"
                and not row.get("current_assignment")
            ),
            None,
        )
        if not target_court or target_court_id == _text(source_court.get("id")):
            raise StaleTournamentAdminStateError(
                "The selected destination court is no longer available."
            )
        if expected.get("target_court_version") in (None, "") or str(
            expected.get("target_court_version")
        ) != str(target_court.get("version")):
            raise StaleTournamentAdminStateError(
                "Destination court version changed after assignment review."
            )
        return
    elif action in {"score_and_release", "correct_completed_score"}:
        game_id = _text(payload.get("game_id"))
        if not game_id:
            raise ValueError("Choose the on-court game to score.")
        score_a = _safe_int(payload.get("score_a"))
        score_b = _safe_int(payload.get("score_b"))
        if score_a is None or score_b is None or score_a < 0 or score_b < 0 or score_a == score_b:
            raise ValueError("A finalized non-tied score is required.")
        game = _selected_game(snapshot, game_id)
        if not game:
            raise ValueError("Choose a game owned by this tournament day.")
        _score_review(
            game,
            score_a,
            score_b,
            acknowledged=payload.get("unusual_score_acknowledgement") is True,
        )
        if action == "correct_completed_score":
            if not game.get("correction_readiness", {}).get("ready"):
                blockers = game.get("correction_readiness", {}).get("blockers") or []
                code = _text((blockers[0] if blockers else {}).get("code"))
                if code == "PLAYOFF_RESET_REQUIRED":
                    raise ValueError(
                        "PLAYOFF_RESET_REQUIRED: reset the bracket before correcting this round-robin score."
                    )
                raise ValueError("This completed score is not safe to correct.")
            draw = _selected_draw(snapshot, _text(game.get("draw_id")))
            if expected.get("draw_version") in (None, "") or str(
                expected.get("draw_version")
            ) != str(draw.get("version")):
                raise StaleTournamentAdminStateError(
                    "Tournament draw version changed after correction review."
                )
            observed_game_version = _text(game.get("version"))
            if expected.get("game_version") in (None, "") or str(
                expected.get("game_version")
            ) != observed_game_version:
                raise StaleTournamentAdminStateError(
                    "Tournament game version changed after correction review."
                )
            return
        court = next(
            (
                row
                for row in snapshot.get("courts", [])
                if _text((row.get("current_assignment") or {}).get("game_id")) == game_id
            ),
            None,
        )
        if not court:
            raise StaleTournamentAdminStateError("This game is no longer on a tournament day court.")
        if expected.get("court_version") in (None, "") or str(expected.get("court_version")) != str(court.get("version")):
            raise StaleTournamentAdminStateError("Assigned court version changed after review.")
        observed_game_version = _text((game or {}).get("version"))
        if not observed_game_version:
            observed_game_version = next(
                (
                    _text(version.get("updated_at"))
                    for draw in snapshot.get("draws", [])
                    for version in draw.get("source_game_versions", [])
                    if _text(version.get("id")) == game_id
                ),
                "",
            )
        if expected.get("game_version") in (None, "") or (observed_game_version and str(expected.get("game_version")) != observed_game_version):
            raise StaleTournamentAdminStateError("Tournament game version changed after review.")
    elif action == "record_non_played_result":
        game = _selected_game(snapshot, _text(payload.get("game_id")))
        if not game:
            raise ValueError("Choose a game owned by this tournament day.")
        result_type = _text(payload.get("result_type")).upper()
        if result_type not in NON_PLAYED_RESULT_TYPES:
            raise ValueError("Choose forfeit, no-show, or retirement.")
        team_ids = {
            _text((game.get("team_a") or {}).get("team_id")),
            _text((game.get("team_b") or {}).get("team_id")),
        }
        winner_team_id = _text(payload.get("winner_team_id"))
        if not winner_team_id or winner_team_id not in team_ids:
            raise ValueError("Choose the winning team from this reviewed matchup.")
        note = _text(payload.get("result_note"))
        if not note:
            raise ValueError("Add an operator note explaining the non-played result.")
        if len(note) > 500:
            raise ValueError("The non-played result note is limited to 500 characters.")
        if _text(game.get("state")).upper() not in {
            "WAITING", "HELD", "CALLED", "ON_COURT", "BLOCKED"
        }:
            raise ValueError("Only an unfinished queued game can receive a non-played result.")
        if expected.get("game_version") in (None, "") or str(
            expected.get("game_version")
        ) != str(game.get("version")):
            raise StaleTournamentAdminStateError(
                "Tournament game version changed after outcome review."
            )
        if expected.get("queue_entry_version") in (None, "") or str(
            expected.get("queue_entry_version")
        ) != str(game.get("queue_entry_version")):
            raise StaleTournamentAdminStateError(
                "Tournament queue entry changed after outcome review."
            )
        court_id = _text(game.get("court_id"))
        if court_id:
            court = next(
                (row for row in snapshot.get("courts", []) if _text(row.get("id")) == court_id),
                None,
            )
            if not court or expected.get("court_version") in (None, "") or str(
                expected.get("court_version")
            ) != str(court.get("version")):
                raise StaleTournamentAdminStateError(
                    "Assigned court changed after outcome review."
                )
        elif expected.get("court_version") not in (None, "", 0, "0"):
            raise StaleTournamentAdminStateError(
                "This reviewed outcome no longer has the same court assignment."
            )


def _playoff_rows(
    snapshot: dict[str, Any],
    draw: dict[str, Any],
    tournament_id: str,
    advance_count: int,
) -> list[dict[str, Any]]:
    draw_id = _text(draw.get("id"))
    source_games = [row for row in snapshot.get("games", []) if _text(row.get("draw_id")) == draw_id]
    rr_games: list[dict[str, Any]] = []
    for game in source_games:
        if _text(game.get("stage")).upper() != "ROUND_ROBIN":
            continue
        team_a_id = _text(game.get("team_a_id") or (game.get("team_a") or {}).get("team_id"))
        team_b_id = _text(game.get("team_b_id") or (game.get("team_b") or {}).get("team_id"))
        score_a = _safe_int(game.get("score_a"))
        score_b = _safe_int(game.get("score_b"))
        if score_a is None or score_b is None or score_a == score_b:
            raise ValueError("Every reviewed round-robin game must have a non-tied score.")
        winner_team_id = team_a_id if score_a > score_b else team_b_id
        rr_games.append(
            {
                "id": game.get("id"),
                "stage": "ROUND_ROBIN",
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "score_a": score_a,
                "score_b": score_b,
                "winner_team_id": winner_team_id,
            }
        )
    teams = [
        {
            "id": _text(row.get("id")),
            "team_number": _safe_int(row.get("team_number")),
            "seed": _safe_int(row.get("seed")),
        }
        for row in list(draw.get("team_rows") or [])
    ]
    if not teams or any(not row["id"] or row["team_number"] is None for row in teams):
        raise ValueError("The reviewed draw is missing exact team-number evidence.")
    if len({row["id"] for row in teams}) != len(teams) or len(
        {row["team_number"] for row in teams}
    ) != len(teams):
        raise ValueError("The reviewed draw team identities or numbers are duplicated.")
    count = _safe_int(advance_count)
    if count not in SUPPORTED_ADVANCE_COUNTS or count > len(teams):
        raise ValueError(
            "Playoff generation requires a reviewed advance count of 4, 5, or 6 within the draw team count."
        )
    standings = compute_round_robin_standings(teams, rr_games)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            **row,
            "id": str(uuid.uuid4()),
            "draw_id": draw_id,
            "registration_day_id": snapshot.get("day_scope", {}).get("selected_day_id"),
            "event_option_id": draw.get("event_option_id"),
            "created_at": now,
            "updated_at": now,
        }
        for row in build_playoff_games(
            tournament_id=str(tournament_id), advance_count=count, standings=standings
        )
    ]


def _score_evidence(
    snapshot: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    game_id = _text(payload.get("game_id"))
    game = _selected_game(snapshot, game_id) or {}
    team_a_id = _text(game.get("team_a_id") or (game.get("team_a") or {}).get("team_id"))
    team_b_id = _text(game.get("team_b_id") or (game.get("team_b") or {}).get("team_id"))
    score_a = int(payload["score_a"])
    score_b = int(payload["score_b"])
    score_review = _score_review(
        game,
        score_a,
        score_b,
        acknowledged=payload.get("unusual_score_acknowledgement") is True,
    )
    raw_game = {
        "id": game_id,
        "team_a_id": team_a_id or None,
        "team_b_id": team_b_id or None,
        "score_a": score_a,
        "score_b": score_b,
    }
    patch = (
        finalize_game(raw_game)
        if team_a_id and team_b_id
        else {"score_a": score_a, "score_b": score_b}
    )
    patch["updated_at"] = datetime.now(timezone.utc).isoformat()
    draw_id = _text(game.get("draw_id"))
    draw = _selected_draw(snapshot, draw_id) if draw_id else next(
        (
            row
            for row in snapshot.get("draws", [])
            if any(
                _text(version.get("id")) == game_id
                for version in row.get("source_game_versions", [])
            )
        ),
        {},
    )
    dependencies: list[dict[str, Any]] = []
    if _text(game.get("stage")).upper() == "PLAYOFF":
        prospective = []
        for row in snapshot.get("games", []):
            if _text(row.get("draw_id")) != draw_id:
                continue
            candidate = {
                "id": row.get("id"),
                "draw_id": row.get("draw_id"),
                "stage": row.get("stage"),
                "playoff_game_code": row.get("playoff_game_code"),
                "team_a_id": row.get("team_a_id") or (row.get("team_a") or {}).get("team_id"),
                "team_b_id": row.get("team_b_id") or (row.get("team_b") or {}).get("team_id"),
                "team_a_source": row.get("team_a_source"),
                "team_b_source": row.get("team_b_source"),
                "score_a": row.get("score_a"),
                "score_b": row.get("score_b"),
                "winner_team_id": row.get("winner_team_id"),
                "loser_team_id": row.get("loser_team_id"),
                "finalized_at": row.get("finalized_at"),
            }
            if _text(row.get("id")) == game_id:
                candidate.update(patch)
            prospective.append(candidate)
        versions = {
            row["id"]: row["updated_at"]
            for row in draw.get("source_game_versions", [])
        }
        dependencies = [
            {**row, "expected_updated_at": versions.get(_text(row.get("id")))}
            for row in resolve_playoff_dependencies(prospective)
        ]
    return {
        "draw_id": _text(draw.get("id")),
        "source_draw_updated_at": draw.get("source_updated_at") or draw.get("version"),
        "game_patch": patch,
        "dependency_updates": dependencies,
        "score_review": score_review,
    }


def _non_played_evidence(
    snapshot: dict[str, Any], payload: dict[str, Any], actor_email: str
) -> dict[str, Any]:
    game = _selected_game(snapshot, _text(payload.get("game_id"))) or {}
    scoring = dict(game.get("scoring") or {})
    target = _safe_int(scoring.get("target"))
    if target is None or target <= 0:
        raise ValueError(
            "The configured scoring target is unavailable for this non-played outcome."
        )
    winner_team_id = _text(payload.get("winner_team_id"))
    team_a_id = _text((game.get("team_a") or {}).get("team_id"))
    score_payload = {
        "game_id": _text(game.get("id")),
        "score_a": target if winner_team_id == team_a_id else 0,
        "score_b": 0 if winner_team_id == team_a_id else target,
        "unusual_score_acknowledgement": False,
    }
    score_evidence = _score_evidence(snapshot, score_payload)
    return {
        **score_evidence,
        "outcome": {
            "result_type": _text(payload.get("result_type")).upper(),
            "winner_team_id": winner_team_id,
            "result_note": _text(payload.get("result_note")),
            "result_recorded_by": _text(actor_email),
            "synthetic_progression_score": True,
            "rating_publish_eligible": False,
        },
    }


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise RuntimeError("Tournament day atomic RPC returned no result.")


def _rpc(supabase: Any, name: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        return _rpc_payload(supabase.rpc(name, params).execute())
    except Exception as exc:
        detail = str(exc)
        if any(marker in detail for marker in ("_STALE", "JUPR_TOURNAMENT_GAME_STALE", "JUPR_TOURNAMENT_DRAW_STALE", "JUPR_TOURNAMENT_DEPENDENCY_STALE")):
            raise StaleTournamentAdminStateError(
                "Tournament day state changed while the command was committing. Reload the day workspace."
            ) from exc
        if "JUPR_TOURNAMENT_DAY_LIVE" in detail:
            raise ValueError(detail.split("JUPR_TOURNAMENT_DAY_LIVE", 1)[-1].lstrip("_: ")) from exc
        raise RuntimeError("Atomic tournament day command failed; recovery evidence must be reviewed before retrying.") from exc


def _operation_payload_base(action: str, expected: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    command_payload = dict(payload)
    operator_review: dict[str, Any] | None = None
    if action in {"score_and_release", "correct_completed_score"}:
        operator_review = {
            "unusual_score_acknowledgement": command_payload.pop(
                "unusual_score_acknowledgement", False
            )
            is True
        }
    result = {"action": action, "expected": dict(expected), "payload": command_payload}
    if operator_review is not None:
        result["operator_review"] = operator_review
    return result


def _activation_evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    courts = sorted(
        list(snapshot.get("courts") or []),
        key=lambda row: (_safe_int(row.get("position"), 0), _text(row.get("id"))),
    )
    return {
        "courts": [
            {
                "court_key": _text(row.get("id")),
                "label": _text(row.get("label")),
                "position": _safe_int(row.get("position"), 0),
            }
            for row in courts
        ]
    }


def _day_operation_evidence_rows(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str,
    operation_key: str,
) -> list[dict[str, Any]]:
    runs = _rows(
        supabase,
        "tournament_day_live_runs",
        filters=(
            ("eq", "club_id", str(club_id)),
            ("eq", "tournament_id", str(tournament_id)),
            ("eq", "registration_day_id", str(registration_day_id)),
        ),
    )
    run = runs[0] if len(runs) == 1 else None
    if not run:
        return []
    run_id = _text(run.get("id"))
    evidence: list[dict[str, Any]] = []
    if _text(run.get("last_operation_key")) == str(operation_key):
        evidence.append({"table": "tournament_day_live_runs", "id": run_id})
    for table in (
        "tournament_day_live_draws",
        "tournament_day_live_courts",
        "tournament_day_live_queue",
        "tournament_day_live_participant_claims",
    ):
        rows = _rows(
            supabase,
            table,
            filters=(
                ("eq", "run_id", run_id),
                ("eq", "last_operation_key", str(operation_key)),
            ),
        )
        evidence.extend({"table": table, "id": row.get("id")} for row in rows)
    return evidence


def execute_admin_tournament_day_live_command(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str,
    request: dict[str, Any],
    actor_email: str,
    actor_role: str,
) -> dict[str, Any]:
    require_tournament_admin_mutation_runtime(TOURNAMENT_DAY_LIVE_SURFACE)
    action, idempotency_key, expected_state, expected, submitted_payload = _normalize_request(request)
    _require_permission(actor_role, action)

    existing = get_tournament_admin_operation_record_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_DAY_LIVE_SURFACE,
        idempotency_key=idempotency_key,
    )
    base_payload = _operation_payload_base(action, expected, submitted_payload)
    if existing:
        request_json = existing.get("request_json") if isinstance(existing.get("request_json"), dict) else {}
        stored = request_json.get("payload") if isinstance(request_json.get("payload"), dict) else {}
        stored_base = {key: stored.get(key) for key in base_payload}
        if _canonical(stored_base) != _canonical(base_payload) or _text(existing.get("expected_state")) != expected_state:
            raise ValueError("This idempotency key was already used for a different tournament day request.")
        operation_payload = dict(stored)
    else:
        reviewed = build_admin_tournament_day_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
        )
        _preflight(reviewed, action, expected, submitted_payload)
        operation_payload = dict(base_payload)
        if action == "activate_day":
            operation_payload["activation_evidence"] = _activation_evidence(
                reviewed
            )
        elif action in {"activate_draw", "pause_draw", "resume_draw"}:
            draw = _selected_draw(reviewed, _text(submitted_payload.get("draw_id")))
            operation_payload["draw_evidence"] = {
                "source_draw_updated_at": draw.get("source_updated_at")
                or draw.get("version")
            }
        elif action in {"score_and_release", "correct_completed_score"}:
            operation_payload["score_evidence"] = _score_evidence(
                reviewed, submitted_payload
            )
        elif action == "record_non_played_result":
            operation_payload["operator_authorization"] = {
                "email": _text(actor_email).lower(),
                "role": _text(actor_role).lower(),
            }
            operation_payload["score_evidence"] = _non_played_evidence(
                reviewed, submitted_payload, actor_email
            )
        elif action == "generate_playoffs":
            draw = _selected_draw(reviewed, _text(submitted_payload.get("draw_id")))
            operation_payload["playoff_games"] = _playoff_rows(
                reviewed,
                draw,
                str(tournament_id),
                int(submitted_payload["advance_count"]),
            )
            operation_payload["playoff_evidence"] = {
                "source_draw_updated_at": draw.get("source_updated_at")
                or draw.get("version"),
                "team_versions": list(draw.get("team_versions") or []),
                "source_game_versions": list(
                    draw.get("source_game_versions") or []
                ),
            }

    captured: dict[str, Any] | None = None

    def current_state() -> str:
        nonlocal captured
        captured = build_admin_tournament_day_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
            exclude_operation_key=identity["operation_key"],
        )
        return _text(captured.get("state_fingerprint"))

    def reviewed_snapshot() -> dict[str, Any]:
        if captured is None:
            current_state()
        assert captured is not None
        _preflight(captured, action, expected, submitted_payload)
        return captured

    entity_id = f"{tournament_id}:{registration_day_id}"
    lock_scope = f"tournament:{tournament_id}:day:{registration_day_id}"
    identity = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface=TOURNAMENT_DAY_LIVE_SURFACE,
        action=COMMAND_ACTIONS[action],
        entity_type=TOURNAMENT_DAY_LIVE_ENTITY,
        entity_id=entity_id,
        lock_scope=lock_scope,
        expected_state=expected_state,
        payload=operation_payload,
        idempotency_key=idempotency_key,
    )

    def mutate() -> dict[str, Any]:
        snapshot = reviewed_snapshot()
        common = {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_registration_day_id": str(registration_day_id),
            "p_expected_run_version": int(expected.get("day_run_version") or 0),
            "p_expected_queue_version": int(expected.get("queue_version") or 0),
            "p_operation_key": identity["operation_key"],
            "p_request_fingerprint": identity["request_fingerprint"],
            "p_actor": str(actor_email or ""),
        }
        if action == "activate_day":
            return _rpc(
                supabase,
                "admin_activate_tournament_day_live_cas",
                {
                    **common,
                    "p_activation_fingerprint": expected_state,
                    "p_activation_evidence": dict(
                        operation_payload.get("activation_evidence") or {}
                    ),
                },
            )
        if action == "record_non_played_result":
            game_id = _text(submitted_payload.get("game_id"))
            game = _selected_game(snapshot, game_id) or {}
            score_evidence = dict(operation_payload.get("score_evidence") or {})
            return _rpc(
                supabase,
                "admin_record_non_played_tournament_day_game_cas",
                {
                    **common,
                    "p_game_id": game_id,
                    "p_expected_queue_entry_version": int(
                        expected.get("queue_entry_version") or 0
                    ),
                    "p_expected_court_version": (
                        int(expected.get("court_version"))
                        if expected.get("court_version") not in (None, "")
                        else None
                    ),
                    "p_expected_game_updated_at": expected.get("game_version"),
                    "p_expected_draw_updated_at": score_evidence.get(
                        "source_draw_updated_at"
                    ),
                    "p_game_patch": dict(score_evidence.get("game_patch") or {}),
                    "p_dependency_updates": list(
                        score_evidence.get("dependency_updates") or []
                    ),
                    "p_result_type": _text(
                        submitted_payload.get("result_type")
                    ).upper(),
                    "p_winner_team_id": _text(
                        submitted_payload.get("winner_team_id")
                    ),
                    "p_result_note": _text(submitted_payload.get("result_note")),
                    "p_actor_role": _text(actor_role).lower(),
                },
            )
        if action in {"activate_draw", "pause_draw", "resume_draw"}:
            draw = _selected_draw(snapshot, _text(submitted_payload.get("draw_id")))
            return _rpc(
                supabase,
                "admin_transition_tournament_day_draw_cas",
                {
                    **common,
                    "p_action": {"activate_draw": "ACTIVATE", "pause_draw": "PAUSE", "resume_draw": "RESUME"}[action],
                    "p_draw_id": draw["id"],
                    "p_expected_day_draw_version": int(expected.get("draw_version") or 0) if action != "activate_draw" else 0,
                    "p_expected_draw_updated_at": (
                        operation_payload.get("draw_evidence") or {}
                    ).get("source_draw_updated_at"),
                },
            )
        if action == "auto_fill_courts":
            return _rpc(supabase, "admin_fill_tournament_day_courts_cas", common)
        if action in {"assign_next_court", "assign_game_to_court"}:
            return _rpc(
                supabase,
                "admin_assign_tournament_day_game_cas",
                {
                    **common,
                    "p_action": (
                        "NEXT_OPEN"
                        if action == "assign_next_court"
                        else "SELECTED"
                    ),
                    "p_game_id": _text(submitted_payload.get("game_id")),
                    "p_court_id": (
                        _text(submitted_payload.get("court_id")) or None
                    ),
                    "p_expected_queue_entry_version": int(
                        expected.get("queue_entry_version") or 0
                    ),
                    "p_expected_game_updated_at": expected.get("game_version"),
                    "p_expected_court_version": (
                        int(expected.get("court_version"))
                        if expected.get("court_version") not in (None, "")
                        else None
                    ),
                },
            )
        if action in {"requeue_game", "move_game_to_court"}:
            return _rpc(
                supabase,
                "admin_reassign_tournament_day_game_cas",
                {
                    **common,
                    "p_action": "REQUEUE" if action == "requeue_game" else "MOVE",
                    "p_game_id": _text(submitted_payload.get("game_id")),
                    "p_target_court_id": (
                        _text(submitted_payload.get("court_id")) or None
                    ),
                    "p_expected_queue_entry_version": int(
                        expected.get("queue_entry_version") or 0
                    ),
                    "p_expected_game_updated_at": expected.get("game_version"),
                    "p_expected_source_court_version": int(
                        expected.get("court_version") or 0
                    ),
                    "p_expected_target_court_version": (
                        int(expected.get("target_court_version"))
                        if expected.get("target_court_version") not in (None, "")
                        else None
                    ),
                },
            )
        if action == "close_day":
            return _rpc(supabase, "admin_close_tournament_day_live_cas", common)
        if action == "score_and_release":
            game_id = _text(submitted_payload.get("game_id"))
            score_evidence = dict(operation_payload.get("score_evidence") or {})
            return _rpc(
                supabase,
                "admin_score_release_tournament_day_game_cas",
                {
                    **common,
                    "p_game_id": game_id,
                    "p_expected_court_version": int(expected.get("court_version") or 0),
                    "p_expected_game_updated_at": expected.get("game_version"),
                    "p_expected_draw_updated_at": score_evidence.get("source_draw_updated_at"),
                    "p_game_patch": dict(score_evidence.get("game_patch") or {}),
                    "p_dependency_updates": list(score_evidence.get("dependency_updates") or []),
                },
            )
        if action == "correct_completed_score":
            game_id = _text(submitted_payload.get("game_id"))
            game = _selected_game(snapshot, game_id) or {}
            draw = _selected_draw(snapshot, _text(game.get("draw_id")))
            score_evidence = dict(operation_payload.get("score_evidence") or {})
            return _rpc(
                supabase,
                "admin_correct_completed_tournament_day_game_cas",
                {
                    **common,
                    "p_game_id": game_id,
                    "p_expected_day_draw_version": int(
                        expected.get("draw_version") or 0
                    ),
                    "p_expected_game_updated_at": expected.get("game_version"),
                    "p_expected_draw_updated_at": score_evidence.get(
                        "source_draw_updated_at"
                    ),
                    "p_game_patch": dict(score_evidence.get("game_patch") or {}),
                    "p_dependency_updates": list(
                        score_evidence.get("dependency_updates") or []
                    ),
                },
            )
        if action == "generate_playoffs":
            draw = _selected_draw(snapshot, _text(submitted_payload.get("draw_id")))
            playoff_evidence = dict(operation_payload.get("playoff_evidence") or {})
            return _rpc(
                supabase,
                "admin_generate_tournament_day_playoffs_cas",
                {
                    **common,
                    "p_draw_id": draw["id"],
                    "p_advance_count": int(submitted_payload["advance_count"]),
                    "p_expected_day_draw_version": int(expected.get("draw_version") or 0),
                    "p_expected_draw_version": playoff_evidence.get("source_draw_updated_at"),
                    "p_expected_team_versions": list(playoff_evidence.get("team_versions") or []),
                    "p_expected_source_game_versions": list(playoff_evidence.get("source_game_versions") or []),
                    "p_games": list(operation_payload.get("playoff_games") or []),
                },
            )
        raise ValueError("Unsupported tournament day command.")

    def reconcile_lost_response(operation: dict[str, Any]) -> dict[str, Any] | None:
        snapshot = build_admin_tournament_day_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
        )
        operation_key = _text(operation.get("operation_key"))
        evidence = _day_operation_evidence_rows(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
            operation_key=operation_key,
        )
        if operation_key and evidence:
            return {"ok": True, "mode": "tournament_day_live_recovered", "action": action}
        return None

    result = run_tournament_admin_guarded_operation(
        supabase=supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_DAY_LIVE_SURFACE,
        action=COMMAND_ACTIONS[action],
        entity_type=TOURNAMENT_DAY_LIVE_ENTITY,
        entity_id=entity_id,
        lock_scope=lock_scope,
        expected_state=expected_state,
        current_state=current_state,
        payload=operation_payload,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=f"next_tournament_day_live_{action}",
        preflight=lambda: _preflight(reviewed_snapshot(), action, expected, submitted_payload),
        reconcile=reconcile_lost_response,
        mutate=mutate,
        idempotency_key=idempotency_key,
    )
    snapshot = build_admin_tournament_day_live_snapshot(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        registration_day_id=str(registration_day_id),
    )
    operation = {
        "operation_key": _text(result.get("operation_key")),
        "client_idempotency_key": _text(result.get("client_idempotency_key") or idempotency_key),
        "action": COMMAND_ACTIONS[action],
        "status": _text(result.get("status") or "completed"),
        "entity_label": _text(snapshot.get("day_scope", {}).get("selected_day", {}).get("label")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error_text": result.get("error_text"),
        "retryable": False,
        "idempotent_replay": bool(result.get("idempotent_replay")),
    }
    return {
        "command": {
            "action": action,
            "confirmation_text": COMMAND_CONFIRMATIONS[action],
            "idempotent_replay": bool(result.get("idempotent_replay")),
        },
        "operation": operation,
        "snapshot": snapshot,
    }


def reconcile_admin_tournament_day_live_operation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str,
    operation_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
) -> dict[str, Any]:
    require_tournament_admin_mutation_runtime(TOURNAMENT_DAY_LIVE_SURFACE)
    if _text(confirmation_text) != TOURNAMENT_DAY_LIVE_RECONCILE_CONFIRMATION:
        raise ValueError(f"Type {TOURNAMENT_DAY_LIVE_RECONCILE_CONFIRMATION} exactly to reconcile this day operation.")
    if not (
        has_permission(actor_role, PERMISSION_MANAGE_TOURNAMENTS)
        or has_permission(actor_role, PERMISSION_ENTER_SCORES)
    ):
        raise PermissionError("insufficient permission for tournament day recovery")
    entity_id = f"{tournament_id}:{registration_day_id}"
    lock_scope = f"tournament:{tournament_id}:day:{registration_day_id}"
    operation = get_tournament_admin_operation_record(
        supabase, club_id=str(club_id), operation_key=str(operation_key)
    )
    if not operation:
        raise ValueError("Tournament day operation not found for this club.")
    if (
        _text(operation.get("surface")) != TOURNAMENT_DAY_LIVE_SURFACE
        or _text(operation.get("entity_type")) != TOURNAMENT_DAY_LIVE_ENTITY
        or _text(operation.get("entity_id")) != entity_id
        or _text(operation.get("lock_scope")) != lock_scope
    ):
        raise ValueError("Operation does not belong to this tournament day.")
    command = ACTION_COMMANDS.get(_text(operation.get("action")))
    if not command:
        raise ValueError("Operation is not a recognized tournament day command.")
    required_permission = (
        PERMISSION_ENTER_SCORES
        if command in {"score_and_release", "correct_completed_score"}
        else PERMISSION_MANAGE_TOURNAMENTS
    )
    if not has_permission(actor_role, required_permission):
        raise PermissionError(
            "insufficient permission to reconcile this tournament day operation"
        )

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        key = _text(row.get("operation_key"))
        matched = _day_operation_evidence_rows(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
            operation_key=key,
        )
        if matched:
            return {
                "status": "completed",
                "result": {"ok": True, "mode": "tournament_day_live_recovered"},
                "evidence": {"authority": "day_live_last_operation_key", "matched_rows": matched},
            }
        snapshot = build_admin_tournament_day_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
        )
        if _text(snapshot.get("state_fingerprint")) == _text(row.get("expected_state")):
            return {"status": "not_applied", "result": {}, "evidence": {"authority": "unchanged_day_fingerprint"}}
        return {"status": "uncertain", "result": {}, "evidence": {"authority": "day_live_last_operation_key"}}

    result = reconcile_tournament_admin_guarded_operation(
        supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_DAY_LIVE_SURFACE,
        operation_key=str(operation_key),
        entity_type=TOURNAMENT_DAY_LIVE_ENTITY,
        entity_id=entity_id,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source="next_tournament_day_live_reconcile",
        verify_outcome=verify,
    )
    return {
        "command": {"action": "reconcile", "confirmation_text": TOURNAMENT_DAY_LIVE_RECONCILE_CONFIRMATION, "idempotent_replay": True},
        "operation": {
            "operation_key": _text(result.get("operation_key") or operation_key),
            "client_idempotency_key": _text(result.get("client_idempotency_key")),
            "action": _text(operation.get("action")),
            "status": "completed" if result.get("recovery_disposition") != "not_applied" else "failed",
            "entity_label": "Tournament day operation",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error_text": None,
            "retryable": False,
            "idempotent_replay": True,
        },
        "snapshot": build_admin_tournament_day_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            registration_day_id=str(registration_day_id),
        ),
    }


__all__ = [
    "COMMAND_CONFIRMATIONS",
    "TOURNAMENT_DAY_LIVE_RECONCILE_CONFIRMATION",
    "TOURNAMENT_DAY_LIVE_SURFACE",
    "build_admin_tournament_day_live_snapshot",
    "execute_admin_tournament_day_live_command",
    "reconcile_admin_tournament_day_live_operation",
]
