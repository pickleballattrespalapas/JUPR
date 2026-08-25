from __future__ import annotations

from itertools import combinations
from typing import Any

from jupr_app.domain.player_activity import coerce_utc_datetime
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.domain.tournaments import SUPPORTED_TEAM_COUNTS
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_state_fingerprint,
)


SCHEDULE_ACTIONS = {
    "ops_round_robin_reconcile",
    "ops_round_robin_rebuild",
}
EMPTY_DRAW_ACTION = "ops_empty_draw_cancel"
EMPTY_EVENT_ACTION = "ops_empty_event_cancel"
RECOVERY_ACTIONS = SCHEDULE_ACTIONS | {EMPTY_DRAW_ACTION, EMPTY_EVENT_ACTION}
INACTIVE_STATUSES = {
    "archived",
    "cancelled",
    "canceled",
    "deleted",
    "disabled",
    "inactive",
    "void",
    "voided",
}


def _strict_rows(
    supabase: Any,
    table_name: str,
    *,
    filters: tuple[tuple[str, Any], ...],
    limit: int = 5000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_size = min(500, max(1, int(limit)))
    offset = 0
    while True:
        query = supabase.table(table_name).select("*")
        for key, value in filters:
            query = query.eq(str(key), value)
        if hasattr(query, "order"):
            query = query.order("id", desc=False)
        supports_range = hasattr(query, "range")
        if supports_range:
            query = query.range(offset, offset + page_size - 1)
        else:
            query = query.limit(page_size)
        data = getattr(query.execute(), "data", None)
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise RuntimeError(f"{table_name} recovery evidence is unavailable")
        rows.extend(dict(row) for row in data)
        if len(rows) >= int(limit):
            raise RuntimeError(f"{table_name} recovery evidence exceeded its safe bound")
        if not supports_range:
            if len(data) >= page_size:
                raise RuntimeError(f"{table_name} recovery evidence is incomplete")
            break
        if len(data) < page_size:
            break
        offset += page_size
    return rows


def _request_payload(operation: dict[str, Any]) -> dict[str, Any] | None:
    request = operation.get("request_json")
    if not isinstance(request, dict):
        return None
    payload = request.get("payload")
    if not isinstance(payload, dict):
        return None
    identity_fields = (
        "operation_key",
        "request_fingerprint",
        "club_id",
        "surface",
        "action",
        "entity_type",
        "entity_id",
        "lock_scope",
        "expected_state",
    )
    if any(
        str(request.get(field) or "") != str(operation.get(field) or "")
        for field in identity_fields
    ):
        return None
    rebuilt = build_tournament_admin_operation_request(
        club_id=str(request.get("club_id") or ""),
        surface=str(request.get("surface") or ""),
        action=str(request.get("action") or ""),
        entity_type=str(request.get("entity_type") or ""),
        entity_id=str(request.get("entity_id") or ""),
        lock_scope=str(request.get("lock_scope") or ""),
        expected_state=str(request.get("expected_state") or ""),
        payload=dict(payload),
        idempotency_key=str(request.get("idempotency_key") or "") or None,
    )
    if any(
        str(rebuilt.get(field) or "") != str(request.get(field) or "")
        for field in ("request_fingerprint", "operation_key")
    ):
        return None
    return dict(payload)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_unstarted(game: dict[str, Any]) -> bool:
    return all(
        game.get(field) in (None, "")
        for field in (
            "score_a",
            "score_b",
            "winner_team_id",
            "loser_team_id",
            "finalized_at",
        )
    )


def _is_complete_result(game: dict[str, Any]) -> bool:
    if _is_unstarted(game):
        return True
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    team_a = str(game.get("team_a_id") or "")
    team_b = str(game.get("team_b_id") or "")
    expected_winner = team_a if (score_a or 0) > (score_b or 0) else team_b
    expected_loser = team_b if expected_winner == team_a else team_a
    return bool(
        score_a is not None
        and score_b is not None
        and score_a >= 0
        and score_b >= 0
        and score_a != score_b
        and game.get("finalized_at")
        and str(game.get("winner_team_id") or "") == expected_winner
        and str(game.get("loser_team_id") or "") == expected_loser
    )


def _exact_round_robin(
    *, teams: list[dict[str, Any]], games: list[dict[str, Any]]
) -> bool:
    team_ids = [str(row.get("id") or "") for row in teams]
    if (
        len(team_ids) not in SUPPORTED_TEAM_COUNTS
        or any(not team_id for team_id in team_ids)
        or len(team_ids) != len(set(team_ids))
    ):
        return False
    expected_pairs = {
        tuple(sorted(pair)) for pair in combinations(team_ids, 2)
    }
    observed_pairs: list[tuple[str, str]] = []
    for game in games:
        team_a = str(game.get("team_a_id") or "")
        team_b = str(game.get("team_b_id") or "")
        if (
            str(game.get("stage") or "").upper() != "ROUND_ROBIN"
            or not team_a
            or not team_b
            or team_a == team_b
            or not _is_complete_result(game)
        ):
            return False
        observed_pairs.append(tuple(sorted((team_a, team_b))))
    return (
        len(observed_pairs) == len(expected_pairs)
        and len(observed_pairs) == len(set(observed_pairs))
        and set(observed_pairs) == expected_pairs
    )


def _versions_match(
    rows: list[dict[str, Any]], expected: Any
) -> bool:
    if not isinstance(expected, list) or not expected:
        return False
    expected_versions = {
        str(row.get("id") or ""): str(row.get("updated_at") or "")
        for row in expected
        if isinstance(row, dict)
        and str(row.get("id") or "")
        and str(row.get("updated_at") or "")
    }
    observed_versions = {
        str(row.get("id") or ""): str(row.get("updated_at") or "")
        for row in rows
        if str(row.get("id") or "") and str(row.get("updated_at") or "")
    }
    return (
        len(expected_versions) == len(expected)
        and len(observed_versions) == len(rows)
        and observed_versions == expected_versions
    )


def _draw_dependencies_clear(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    games: list[dict[str, Any]],
) -> bool:
    for table_name in (
        "tournament_podium",
        "tournament_day_live_draws",
        "tournament_day_live_queue",
    ):
        if _strict_rows(
            supabase,
            table_name,
            filters=(("tournament_id", tournament_id), ("draw_id", draw_id)),
        ):
            return False
    game_ids = {str(row.get("id") or "") for row in games if row.get("id")}
    matches = _strict_rows(
        supabase,
        "matches",
        filters=(("club_id", club_id), ("tournament_id", tournament_id)),
    )
    if any(str(row.get("tournament_game_id") or "") in game_ids for row in matches):
        return False
    badge_prefix = f"{tournament_id}:draw:{draw_id}:podium:"
    badges = _strict_rows(
        supabase,
        "player_badges",
        filters=(("club_id", club_id), ("context_type", "tournament")),
    )
    return not any(
        str(row.get("context_id") or "").startswith(badge_prefix)
        for row in badges
    )


def _reconcile_schedule(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    action: str,
    payload: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any] | None:
    draws = _strict_rows(
        supabase,
        "tournament_event_draws",
        filters=(("tournament_id", tournament_id), ("id", draw_id)),
    )
    if (
        len(draws) != 1
        or str(draws[0].get("updated_at") or "")
        != str(payload.get("expected_draw_updated_at") or "")
    ):
        return None
    teams = _strict_rows(
        supabase,
        "tournament_teams",
        filters=(("tournament_id", tournament_id), ("draw_id", draw_id)),
    )
    if not _versions_match(teams, payload.get("expected_team_versions")):
        return None
    games = _strict_rows(
        supabase,
        "tournament_games",
        filters=(("tournament_id", tournament_id), ("draw_id", draw_id)),
    )
    if not _exact_round_robin(teams=teams, games=games):
        return None
    if not _draw_dependencies_clear(
        supabase,
        club_id=club_id,
        tournament_id=tournament_id,
        draw_id=draw_id,
        games=games,
    ):
        return None

    if action == "ops_round_robin_rebuild":
        operation_created_at = coerce_utc_datetime(operation.get("created_at"))
        created_times = [coerce_utc_datetime(row.get("created_at")) for row in games]
        if (
            operation_created_at is None
            or any(created_at is None for created_at in created_times)
            or any(created_at < operation_created_at for created_at in created_times if created_at)
            or any(not _is_unstarted(row) for row in games)
        ):
            return None

    return {
        "ok": True,
        "mode": (
            "tournament_round_robin_reconcile"
            if action == "ops_round_robin_reconcile"
            else "tournament_round_robin_rebuild"
        ),
        "draw_id": draw_id,
        "game_count": len(games),
        "games": games,
        "warnings": [],
        "response_loss_reconciled": True,
    }


def _reconcile_empty_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    draws = _strict_rows(
        supabase,
        "tournament_event_draws",
        filters=(("tournament_id", tournament_id), ("id", draw_id)),
    )
    if len(draws) != 1 or str(draws[0].get("status") or "").lower() not in INACTIVE_STATUSES:
        return None
    if str(draws[0].get("updated_at") or "") == str(
        payload.get("expected_draw_updated_at") or ""
    ):
        return None
    games = _strict_rows(
        supabase,
        "tournament_games",
        filters=(("tournament_id", tournament_id), ("draw_id", draw_id)),
    )
    if games or _strict_rows(
        supabase,
        "tournament_teams",
        filters=(("tournament_id", tournament_id), ("draw_id", draw_id)),
    ):
        return None
    if not _draw_dependencies_clear(
        supabase,
        club_id=club_id,
        tournament_id=tournament_id,
        draw_id=draw_id,
        games=games,
    ):
        return None
    return {
        "ok": True,
        "mode": "tournament_empty_draw_cancel",
        "draw": draws[0],
        "dependencies": {},
        "warnings": [],
        "response_loss_reconciled": True,
    }


def _reconcile_empty_event(
    supabase: Any,
    *,
    tournament_id: str,
    event_option_id: str,
) -> dict[str, Any] | None:
    events = _strict_rows(
        supabase,
        "tournament_event_options",
        filters=(("tournament_id", tournament_id), ("id", event_option_id)),
    )
    if len(events) != 1:
        return None
    enabled = events[0].get("enabled", True)
    if enabled is not False or str(events[0].get("status") or "").lower() not in INACTIVE_STATUSES:
        return None
    for table_name in (
        "tournament_registration_selections",
        "tournament_registration_team_links",
        "tournament_registration_team_members",
        "tournament_event_draws",
        "tournament_teams",
        "tournament_games",
    ):
        if _strict_rows(
            supabase,
            table_name,
            filters=(
                ("tournament_id", tournament_id),
                ("event_option_id", event_option_id),
            ),
        ):
            return None
    return {
        "ok": True,
        "mode": "tournament_empty_event_cancel",
        "event_option": events[0],
        "dependencies": {},
        "warnings": [],
        "response_loss_reconciled": True,
    }


def reconcile_admin_tournament_ops_recovery(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    action: str,
    entity_id: str,
    operation: dict[str, Any],
) -> dict[str, Any] | None:
    """Prove a response-lost recovery mutation from exact current DB state.

    The guarded runner calls this only for its matching recovery-required intent.
    Every unavailable, advanced, cross-scope, or ambiguous state returns ``None``
    so the durable lock remains in place.
    """

    clean_action = str(action or "")
    clean_tournament_id = str(tournament_id or "")
    clean_entity_id = str(entity_id or "")
    try:
        if (
            clean_action not in RECOVERY_ACTIONS
            or str(operation.get("club_id") or "") != str(club_id)
            or str(operation.get("surface") or "") != "operations"
            or str(operation.get("action") or "") != clean_action
            or str(operation.get("entity_id") or "") != clean_entity_id
            or str(operation.get("lock_scope") or "") != clean_tournament_id
            or str(operation.get("status") or "") != "recovery_required"
        ):
            return None
        expected_entity_type = (
            "tournament_event_option"
            if clean_action == EMPTY_EVENT_ACTION
            else "tournament_event_draw"
        )
        if str(operation.get("entity_type") or "") != expected_entity_type:
            return None
        payload = _request_payload(operation)
        if payload is None:
            return None
        expected_state = str(operation.get("expected_state") or "")
        if not expected_state:
            return None
        current_state = get_admin_tournament_ops_state_fingerprint(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
        )
        if not current_state or current_state == expected_state:
            return None
        tournaments = _strict_rows(
            supabase,
            "tournaments",
            filters=(("club_id", str(club_id)), ("id", clean_tournament_id)),
        )
        if len(tournaments) != 1 or str(tournaments[0].get("status") or "").upper() in {
            "COMPLETED",
            "ARCHIVED",
        }:
            return None

        if clean_action in SCHEDULE_ACTIONS:
            intent_marker = (
                "preserve_existing_games"
                if clean_action == "ops_round_robin_reconcile"
                else "replace_unstarted_games"
            )
            if (
                str(payload.get("draw_id") or "") != clean_entity_id
                or payload.get(intent_marker) is not True
            ):
                return None
            return _reconcile_schedule(
                supabase,
                club_id=str(club_id),
                tournament_id=clean_tournament_id,
                draw_id=clean_entity_id,
                action=clean_action,
                payload=payload,
                operation=operation,
            )
        if clean_action == EMPTY_DRAW_ACTION:
            if (
                str(payload.get("draw_id") or "") != clean_entity_id
                or str(payload.get("status") or "").lower() != "cancelled"
            ):
                return None
            return _reconcile_empty_draw(
                supabase,
                club_id=str(club_id),
                tournament_id=clean_tournament_id,
                draw_id=clean_entity_id,
                payload=payload,
            )
        if (
            str(payload.get("event_option_id") or "") != clean_entity_id
            or payload.get("enabled") is not False
            or str(payload.get("status") or "").lower() != "cancelled"
        ):
            return None
        return _reconcile_empty_event(
            supabase,
            tournament_id=clean_tournament_id,
            event_option_id=clean_entity_id,
        )
    except Exception:
        return None


__all__ = ["reconcile_admin_tournament_ops_recovery"]
