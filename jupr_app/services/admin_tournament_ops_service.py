from __future__ import annotations

import os
from typing import Any

from jupr_app.config import get_email_mode
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.domain.tournament_team_canonical_publish import (
    classify_team_child_publish_state,
)
from jupr_app.services.admin_player_updates_service import is_auto_player_updates_enabled
from jupr_app.services.production_tournament_guard import production_tournament_writes_enabled, require_production_tournament_writes
from jupr_app.services.admin_tournament_guarded_operation import tournament_admin_guarded_runtime_enabled
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    is_admin_team_tournament_enabled,
)

OPS_TABLES = {
    "draws": "tournament_event_draws",
    "teams": "tournament_teams",
    "games": "tournament_games",
    "podium": "tournament_podium",
}
OPS_STATE_TABLES = (
    "tournament_event_draws",
    "tournament_teams",
    "tournament_games",
    "tournament_podium",
    "tournament_registration_days",
    "tournament_event_options",
    "tournament_registrations",
    "tournament_registration_selections",
    "tournament_team_match_games",
)
TRUTHY = {"1", "true", "yes", "y", "on"}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _table_rows(supabase: Any, table_name: str, *, tournament_id: str, limit: int = 1000) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        rows = _safe_rows(
            supabase.table(table_name)
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .limit(max(1, min(int(limit), 2000)))
            .execute()
        )
        return rows, []
    except Exception as exc:  # noqa: BLE001 - surface operational table availability without failing the whole ops view
        return [], [f"{table_name} unavailable: {exc.__class__.__name__}"]


def _player_options(supabase: Any, *, club_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,active,is_active")
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - player options are helpful but not required for the ops snapshot
        return [], [f"players unavailable: {exc.__class__.__name__}"]
    return _player_options_from_rows(rows), []


def _player_options_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for row in rows:
        try:
            player_id = int(float(row.get("id")))
        except Exception:
            continue
        name = _clean_text(row.get("name"), limit=160)
        if not name:
            continue
        active_value = row.get("active", row.get("is_active", True))
        players.append({"id": player_id, "name": name, "active": bool(active_value)})
    return sorted(players, key=lambda row: str(row.get("name") or "").lower())


def _sort_rows(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(key) or "") for key in keys)

    return sorted(rows, key=sort_key)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def _canonical_state_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove no fields, but impose a stable order before hashing DB state."""

    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            str(row.get("id") or ""),
            str(row.get("draw_id") or ""),
            str(row.get("registration_id") or ""),
            str(row.get("team_number") or ""),
            str(row.get("placement") or ""),
        ),
    )


def _strict_paginated_rows(
    supabase: Any,
    table_name: str,
    *,
    filters: tuple[tuple[str, Any], ...],
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Read a complete deterministic table slice or fail closed.

    Real Supabase builders expose ``range``; small in-memory test doubles do not,
    so they use one bounded read and are rejected if that bound is saturated.
    """

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = supabase.table(table_name).select("*")
        for key, value in filters:
            query = query.eq(str(key), value)
        if hasattr(query, "order"):
            query = query.order("id", desc=False)
        supports_range = hasattr(query, "range")
        if supports_range:
            query = query.range(offset, offset + int(page_size) - 1)
        else:
            query = query.limit(int(page_size))
        page = _safe_rows(query.execute())
        rows.extend(page)
        if not supports_range:
            if len(page) >= int(page_size):
                raise RuntimeError(f"{table_name} state exceeded the safe non-paginated read bound")
            break
        if len(page) < int(page_size):
            break
        offset += int(page_size)
        if offset > 100_000:
            raise RuntimeError(f"{table_name} state exceeded the Tournament Ops safety bound")
    return rows


def _load_admin_tournament_ops_state(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    tournament: dict[str, Any] | None = None,
    include_team_competition: bool | None = None,
) -> dict[str, Any]:
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    current_tournament = tournament or _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not current_tournament or str(current_tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    state: dict[str, Any] = {"tournament": dict(current_tournament), "tables": {}}
    team_feature_enabled = (
        is_admin_team_tournament_enabled()
        if include_team_competition is None
        else bool(include_team_competition)
    )
    state_tables = (
        OPS_STATE_TABLES
        if team_feature_enabled
        else tuple(
            table_name
            for table_name in OPS_STATE_TABLES
            if table_name != "tournament_team_match_games"
        )
    )
    for table_name in state_tables:
        state["tables"][table_name] = _canonical_state_rows(
            _strict_paginated_rows(
                supabase,
                table_name,
                filters=(("tournament_id", clean_tournament_id),),
            )
        )
    player_rows = _strict_paginated_rows(supabase, "players", filters=(("club_id", str(club_id)),))
    published_rows = _strict_paginated_rows(
        supabase,
        "matches",
        filters=(("club_id", str(club_id)), ("tournament_id", clean_tournament_id)),
    )
    badge_rows = _strict_paginated_rows(
        supabase,
        "player_badges",
        filters=(("club_id", str(club_id)), ("context_type", "tournament")),
    )
    badge_prefix = f"{clean_tournament_id}:"
    state["players"] = _canonical_state_rows(player_rows)
    state["published_matches"] = _canonical_state_rows(published_rows)
    state["podium_badges"] = _canonical_state_rows(
        [row for row in badge_rows if str(row.get("context_id") or "").startswith(badge_prefix)]
    )
    return state


def get_admin_tournament_ops_state_fingerprint(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> str:
    """Fingerprint every authoritative input/output touched by Tournament Ops.

    Unlike the display snapshot, this helper is strict: an unavailable table
    aborts a staging mutation before intent is recorded. The operation ledger
    and activity log are deliberately excluded so a guard can recheck state
    after acquiring its per-tournament lock.
    """

    state = _load_admin_tournament_ops_state(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    return stable_tournament_admin_fingerprint(state)


def require_admin_tournament_official_publish_runtime() -> None:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment == "production":
        require_production_tournament_writes()
    elif environment != "staging":
        return
    if not _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH"):
        raise PermissionError("Tournament official publishing is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH only with the approved tournament write gate.")
    if is_auto_player_updates_enabled():
        if not _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF"):
            raise PermissionError("Automatic tournament player-update email handoff is disabled.")
        allowed_modes = {"dry_run"} if environment == "production" else {"dry_run", "staging_redirect"}
        if get_email_mode() not in allowed_modes:
            raise PermissionError("Tournament official publishing requires a non-live email mode.")


def build_admin_tournament_ops_runtime_status() -> dict[str, Any]:
    environment = os.getenv("JUPR_ENV", "").strip().lower() or "local"
    hosted = environment == "staging" or production_tournament_writes_enabled()
    return {
        "environment": environment,
        "tournament_mutations_enabled": tournament_admin_guarded_runtime_enabled("tournament"),
        "operations_mutations_enabled": tournament_admin_guarded_runtime_enabled("operations"),
        "official_publish_enabled": hosted and _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH"),
        "email_handoff_enabled": hosted and _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF"),
        "auto_player_updates_enabled": is_auto_player_updates_enabled(),
        "email_mode": get_email_mode(),
        "staging_only": False,
        "production_authorized": production_tournament_writes_enabled(),
    }


def get_admin_tournament_ops_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    if not clean_tournament_id:
        raise ValueError("tournament_id is required")
    team_feature_enabled = is_admin_team_tournament_enabled()
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    warnings: list[str] = []
    state_fingerprint: str | None = None
    try:
        state = _load_admin_tournament_ops_state(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            tournament=tournament,
            include_team_competition=team_feature_enabled,
        )
        state_tables = dict(state.get("tables") or {})
        draws = list(state_tables.get(OPS_TABLES["draws"]) or [])
        teams = list(state_tables.get(OPS_TABLES["teams"]) or [])
        games = list(state_tables.get(OPS_TABLES["games"]) or [])
        podium = list(state_tables.get(OPS_TABLES["podium"]) or [])
        registration_days = list(state_tables.get("tournament_registration_days") or [])
        event_options = list(state_tables.get("tournament_event_options") or [])
        team_match_games = (
            list(state_tables.get("tournament_team_match_games") or [])
            if team_feature_enabled
            else []
        )
        published_matches = list(state.get("published_matches") or [])
        players = _player_options_from_rows(list(state.get("players") or []))
        state_fingerprint = stable_tournament_admin_fingerprint(state)
    except Exception as exc:  # retain read-only recovery, but never pair it with an accepted write fingerprint
        warnings.append(f"Tournament Ops state fingerprint unavailable: {exc.__class__.__name__}")
        draws, draw_warnings = _table_rows(supabase, OPS_TABLES["draws"], tournament_id=clean_tournament_id)
        teams, team_warnings = _table_rows(supabase, OPS_TABLES["teams"], tournament_id=clean_tournament_id)
        games, game_warnings = _table_rows(supabase, OPS_TABLES["games"], tournament_id=clean_tournament_id)
        podium, podium_warnings = _table_rows(supabase, OPS_TABLES["podium"], tournament_id=clean_tournament_id)
        registration_days, day_warnings = _table_rows(supabase, "tournament_registration_days", tournament_id=clean_tournament_id)
        event_options, event_warnings = _table_rows(supabase, "tournament_event_options", tournament_id=clean_tournament_id)
        if team_feature_enabled:
            team_match_games, child_warnings = _table_rows(
                supabase,
                "tournament_team_match_games",
                tournament_id=clean_tournament_id,
            )
            published_matches, published_warnings = _table_rows(
                supabase,
                "matches",
                tournament_id=clean_tournament_id,
            )
            published_matches = [
                row
                for row in published_matches
                if str(row.get("club_id") or "") == str(club_id)
            ]
        else:
            team_match_games, child_warnings = [], []
            published_matches, published_warnings = [], []
        players, player_warnings = _player_options(supabase, club_id=str(club_id))
        warnings.extend([*draw_warnings, *team_warnings, *game_warnings, *podium_warnings, *day_warnings, *event_warnings, *child_warnings, *published_warnings, *player_warnings])

    # Keep every authoritative game row available for CAS/version evidence and
    # team-child classification.  SERIES_GAME rows are rating leaves for a
    # best-of-three matchup, not independently scheduled operator games.
    all_games = list(games)

    protected_rating_child_draws = [
        row
        for row in draws
        if str(row.get("draw_kind") or "").upper() == "TEAM_RATING_CHILD"
    ]
    rating_child_draws = (
        protected_rating_child_draws if team_feature_enabled else []
    )
    rating_child_ids = {
        str(row.get("id") or "") for row in protected_rating_child_draws
    }
    team_parent_ids = {
        str(row.get("id") or "")
        for row in draws
        if str(row.get("draw_kind") or "").upper() == "TEAM_PARENT"
    }
    hidden_draw_ids = {
        str(row.get("id") or "")
        for row in draws
        if bool(row.get("hidden_from_primary_ops"))
    }
    protected_team_draw_ids = rating_child_ids | team_parent_ids | hidden_draw_ids
    games_by_id = {str(row.get("id") or ""): row for row in all_games}
    canonical_by_game: dict[str, list[dict[str, Any]]] = {}
    for row in published_matches:
        tournament_game_id = str(row.get("tournament_game_id") or "")
        if tournament_game_id:
            canonical_by_game.setdefault(tournament_game_id, []).append(row)
    children_by_draw: dict[str, list[dict[str, Any]]] = {}
    for child in team_match_games:
        rating_draw_id = str(child.get("rating_draw_id") or "")
        if rating_draw_id:
            children_by_draw.setdefault(rating_draw_id, []).append(child)
    rating_child_publish_queue: list[dict[str, Any]] = []
    for draw in rating_child_draws:
        rating_draw_id = str(draw.get("id") or "")
        child_games = children_by_draw.get(rating_draw_id, [])
        if len(child_games) != 1:
            rating_child_publish_queue.append(
                {
                    "draw": draw,
                    "child_game": child_games[0] if child_games else None,
                    "tournament_game": None,
                    "publish_state": "RECONCILE_REQUIRED",
                    "canonical_match_count": 0,
                }
            )
            continue
        child = child_games[0]
        tournament_game_id = str(child.get("tournament_game_id") or "")
        canonical = canonical_by_game.get(tournament_game_id, [])
        publish_state = classify_team_child_publish_state(
            child=child,
            tournament_game=games_by_id.get(tournament_game_id),
            canonical_matches=canonical,
        )
        rating_child_publish_queue.append(
            {
                "draw": draw,
                "child_game": child,
                "tournament_game": games_by_id.get(tournament_game_id),
                "publish_state": publish_state,
                "canonical_match_count": len(canonical),
            }
        )

    draws = [
        row
        for row in draws
        if str(row.get("id") or "") not in protected_team_draw_ids
    ]
    teams = [
        row
        for row in teams
        if str(row.get("draw_id") or "") not in protected_team_draw_ids
    ]
    source_games = [
        row
        for row in all_games
        if str(row.get("draw_id") or "") not in protected_team_draw_ids
    ]
    games = [
        row
        for row in source_games
        if not row.get("series_parent_game_id")
        and str(row.get("stage") or "").upper() != "SERIES_GAME"
    ]
    podium = [
        row
        for row in podium
        if str(row.get("draw_id") or "") not in protected_team_draw_ids
    ]

    clean_draw_id = _clean_text(draw_id, limit=120) or None
    if clean_draw_id:
        teams = [row for row in teams if str(row.get("draw_id") or "") == clean_draw_id]
        games = [row for row in games if str(row.get("draw_id") or "") == clean_draw_id]
        source_games = [
            row
            for row in source_games
            if str(row.get("draw_id") or "") == clean_draw_id
        ]
        podium = [row for row in podium if str(row.get("draw_id") or "") == clean_draw_id]
        draws = [row for row in draws if str(row.get("id") or "") == clean_draw_id]
        rating_child_publish_queue = [
            row
            for row in rating_child_publish_queue
            if str((row.get("draw") or {}).get("id") or "") == clean_draw_id
        ]

    draws = _sort_rows(draws, "registration_day_id", "event_option_id", "name", "id")
    teams = _sort_rows(teams, "draw_id", "team_number", "id")
    games = _sort_rows(games, "draw_id", "stage", "rr_round_number", "rr_slot_number", "game_number", "id")
    source_games = _sort_rows(
        source_games,
        "draw_id",
        "stage",
        "rr_round_number",
        "rr_slot_number",
        "series_game_number",
        "game_number",
        "id",
    )
    podium = _sort_rows(podium, "draw_id", "placement", "id")
    registration_days = _sort_rows(registration_days, "event_date", "sort_order", "id")
    event_options = _sort_rows(event_options, "registration_day_id", "sort_order", "id")

    return {
        "ok": True,
        "mode": "tournament_ops_snapshot",
        "tournament": _tournament_payload(tournament),
        "draw_id": clean_draw_id,
        "summary": {
            "draws": len(draws),
            "teams": len(teams),
            "games": len(games),
            "podium": len(podium),
            "rating_children": len(rating_child_publish_queue),
            "completed_games": len([row for row in games if str(row.get("status") or "").lower() in {"complete", "completed", "final"} or row.get("winner_team_id")]),
        },
        "draws": draws,
        "teams": teams,
        "games": games,
        "source_game_versions": [
            {
                "id": str(row.get("id") or ""),
                "draw_id": str(row.get("draw_id") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            }
            for row in source_games
            if row.get("id")
        ],
        "podium": podium,
        "registration_days": registration_days,
        "event_options": event_options,
        "rating_child_draws": rating_child_draws,
        "rating_child_publish_queue": rating_child_publish_queue,
        "players": players,
        "state_fingerprint": state_fingerprint,
        "state_ready": bool(state_fingerprint),
        "operation_runtime": build_admin_tournament_ops_runtime_status(),
        "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or "https://juprtrespalapas.streamlit.app",
        "warnings": warnings,
    }
