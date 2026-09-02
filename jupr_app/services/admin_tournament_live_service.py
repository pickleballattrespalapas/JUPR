from __future__ import annotations

import os
import re
import uuid
from typing import Any

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_TOURNAMENTS,
    has_permission,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
    stable_tournament_admin_fingerprint,
)
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.domain.tournaments import (
    SUPPORTED_TEAM_COUNTS,
    build_playoff_games,
    build_round_robin_games,
    compute_podium_from_playoffs,
    compute_podium_from_rr,
    compute_round_robin_standings,
    finalize_game,
    resolve_playoff_dependencies,
)
from jupr_app.domain.tournaments.score_policy import (
    SUPPORTED_SCORING_FORMATS,
    require_best_of_three_game_scores,
    require_tournament_score,
    resolve_tournament_scoring_format,
)
from jupr_app.services.admin_tournament_award_service import award_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_game_service import generate_admin_tournament_round_robin_games
from jupr_app.services.admin_tournament_guarded_operation import (
    TOURNAMENT_ADMIN_OPERATION_TABLE,
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
    get_tournament_admin_operation_record,
    get_tournament_admin_operation_record_by_idempotency_key,
    reconcile_tournament_admin_guarded_operation,
    require_tournament_admin_mutation_runtime,
    run_tournament_admin_guarded_operation,
    tournament_admin_guarded_runtime_enabled,
    tournament_admin_mutation_status,
)
from jupr_app.services.admin_tournament_lifecycle_service import (
    build_admin_tournament_lifecycle,
    build_tournament_rating_game_plan,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    build_admin_tournament_official_publish_plan,
    publish_admin_tournament_draw_matches,
    reconcile_admin_tournament_official_publish,
)
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_state_fingerprint,
    get_admin_tournament_ops_snapshot,
    require_admin_tournament_official_publish_runtime,
)
from jupr_app.services.admin_tournament_playoff_service import SUPPORTED_ADVANCE_COUNTS, generate_admin_tournament_playoff_games
from jupr_app.services.admin_tournament_podium_service import generate_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_score_service import update_admin_tournament_game_score
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled


TOURNAMENT_LIVE_SURFACE = "tournament_live"
TOURNAMENT_LIVE_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES"
TOURNAMENT_OFFICIAL_PUBLISH_WRITE_FLAG = (
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH"
)
TOURNAMENT_LIVE_FALLBACK = "https://juprtrespalapas.streamlit.app"
TOURNAMENT_LIVE_RECONCILE_CONFIRMATION = "RECONCILE TOURNAMENT LIVE"
ACTIVE_OPERATION_STATUSES = {"intent", "mutated", "recovery_required"}
COMMAND_CONFIRMATIONS = {
    "save_score": "SAVE SCORE",
    "generate_round_robin": "GENERATE GAMES",
    "generate_playoffs": "GENERATE PLAYOFFS",
    "generate_podium": "GENERATE PODIUM",
    "award_podium": "AWARD PODIUM",
    "publish_official_matches": "PUBLISH MATCHES",
}
COMMAND_ACTIONS = {
    "save_score": "tournament_live_score",
    "generate_round_robin": "tournament_live_round_robin",
    "generate_playoffs": "tournament_live_playoffs",
    "generate_podium": "tournament_live_podium",
    "award_podium": "tournament_live_awards",
    "publish_official_matches": "tournament_live_official_publish",
}
ACTION_COMMANDS = {action: command for command, action in COMMAND_ACTIONS.items()}
STATE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PROVEN_PRE_MUTATION_PODIUM_VERSION_ERRORS = {
    "A complete reviewed podium version set is required. Reload the live board.",
    "The reviewed podium version set is malformed. Reload the live board.",
    "The reviewed podium version set is incomplete or duplicated. Reload the live board.",
}
ATOMIC_PODIUM_AWARD_ROLLBACK_ERROR = (
    "Atomic tournament podium awards failed; no badge set was committed."
)
PROVEN_NO_WRITE_PODIUM_AWARD_ERRORS = (
    PROVEN_PRE_MUTATION_PODIUM_VERSION_ERRORS
    | {ATOMIC_PODIUM_AWARD_ROLLBACK_ERROR}
)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_live_command_permission(actor_role: str, command: str) -> None:
    """Enforce the same permission boundary as the underlying admin route."""

    if command == "save_score":
        permitted = has_permission(actor_role, PERMISSION_ENTER_SCORES)
    elif command == "publish_official_matches":
        permitted = all(
            has_permission(actor_role, permission)
            for permission in (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_MATCHES)
        )
    else:
        permitted = has_permission(actor_role, PERMISSION_MANAGE_TOURNAMENTS)
    if not permitted:
        raise PermissionError("insufficient permission for this Tournament Live command")


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_scored(game: dict[str, Any]) -> bool:
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    return score_a is not None and score_b is not None and score_a != score_b and bool(game.get("winner_team_id"))


def _is_series_game_child(game: dict[str, Any]) -> bool:
    """Keep rating-game leaves out of every operational matchup surface."""

    return bool(str(game.get("series_parent_game_id") or "").strip()) or (
        str(game.get("stage") or "").strip().upper() == "SERIES_GAME"
    )


def _competition_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in games:
        if _is_series_game_child(row):
            continue
        game = dict(row)
        review = (
            game.get("score_review_json")
            if isinstance(game.get("score_review_json"), dict)
            else {}
        )
        reviewed_games = review.get("game_scores")
        if isinstance(reviewed_games, list):
            game["game_scores"] = [
                {
                    "game_number": _safe_int(item.get("game_number")),
                    "score_a": _safe_int(item.get("score_a")),
                    "score_b": _safe_int(item.get("score_b")),
                }
                for item in reviewed_games
                if isinstance(item, dict)
            ]
        projected.append(game)
    return projected


def _authoritative_draw_games(
    supabase: Any,
    *,
    tournament_id: str,
    draw_id: str,
    expected_versions: Any,
) -> list[dict[str, Any]]:
    """Reload parent and rating-leaf rows, proving the Ops read did not race."""

    try:
        rows = _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the complete tournament game source set; the live board was refused."
        ) from exc
    expected = (
        _canonical_version_rows(expected_versions, label="source game")
        if expected_versions
        else []
    )
    observed = _snapshot_version_rows(rows, label="source game") if rows else []
    if expected != observed:
        raise StaleTournamentAdminStateError(
            "The tournament game source set changed while the live board was loading. Reload the draw."
        )
    return rows


def _is_rating_publish_eligible(game: dict[str, Any]) -> bool:
    """Only genuinely played results may become official rated matches."""

    result_type = str(game.get("result_type") or "PLAYED").strip().upper()
    parent_result_only = game.get("parent_result_only") is True or str(
        game.get("parent_result_only") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    return (
        result_type == "PLAYED"
        and game.get("rating_publish_eligible") is not False
        and not parent_result_only
    )


def _project(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def _sorted_projection(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    projected = [_project(row, fields) for row in rows]
    return sorted(projected, key=lambda row: tuple(str(row.get(field) or "") for field in fields))


def _canonical_version_rows(value: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"A complete reviewed {label} version set is required. Reload the live board.")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"The reviewed {label} version set is malformed. Reload the live board.")
        row_id = str(item.get("id") or "").strip()
        updated_at = str(item.get("updated_at") or "").strip()
        if not row_id or not updated_at or row_id in seen:
            raise ValueError(f"The reviewed {label} version set is incomplete or duplicated. Reload the live board.")
        seen.add(row_id)
        rows.append({"id": row_id, "updated_at": updated_at})
    return sorted(rows, key=lambda row: row["id"])


def _snapshot_version_rows(rows: list[dict[str, Any]], *, label: str) -> list[dict[str, str]]:
    return _canonical_version_rows(
        [{"id": row.get("id"), "updated_at": row.get("updated_at")} for row in rows],
        label=label,
    )


def _round_robin_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "ROUND_ROBIN",
        "rr_round_number": _safe_int(row.get("rr_round_number")),
        "rr_slot_number": _safe_int(row.get("rr_slot_number")),
        "team_a_id": str(row.get("team_a_id") or ""),
        "team_b_id": str(row.get("team_b_id") or ""),
    }


def _round_robin_plan(*, tournament_id: str, teams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    team_ids_by_number: dict[int, str] = {}
    for team in teams:
        team_number = _safe_int(team.get("team_number"))
        team_id = str(team.get("id") or "").strip()
        if team_number is None or not team_id or team_number in team_ids_by_number:
            raise ValueError("Every reviewed team must have one unique team number and id.")
        team_ids_by_number[int(team_number)] = team_id
    if len(team_ids_by_number) not in SUPPORTED_TEAM_COUNTS:
        raise ValueError(f"Round-robin generation supports {SUPPORTED_TEAM_COUNTS}; this draw has {len(team_ids_by_number)} teams.")
    if sorted(team_ids_by_number) != list(range(1, len(team_ids_by_number) + 1)):
        raise ValueError("Team numbers must be contiguous before generating round-robin games.")
    return sorted(
        [
            _round_robin_projection(row)
            for row in build_round_robin_games(
                tournament_id=str(tournament_id),
                team_ids_by_number=team_ids_by_number,
            )
        ],
        key=lambda row: (int(row["rr_round_number"] or 0), int(row["rr_slot_number"] or 0)),
    )


def _playoff_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "PLAYOFF",
        "playoff_game_code": str(row.get("playoff_game_code") or ""),
        "playoff_round": str(row.get("playoff_round") or ""),
        "team_a_id": str(row.get("team_a_id") or "") or None,
        "team_b_id": str(row.get("team_b_id") or "") or None,
        "team_a_source": dict(row.get("team_a_source")) if isinstance(row.get("team_a_source"), dict) else None,
        "team_b_source": dict(row.get("team_b_source")) if isinstance(row.get("team_b_source"), dict) else None,
    }


def _playoff_plan(
    *,
    tournament_id: str,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    advance_count: int,
) -> list[dict[str, Any]]:
    rr_games = [row for row in games if str(row.get("stage") or "").upper() == "ROUND_ROBIN"]
    if not rr_games or not all(_is_scored(row) for row in rr_games):
        raise ValueError("Every reviewed round-robin game must be finalized before generating playoffs.")
    standings = compute_round_robin_standings(teams, rr_games)
    return sorted(
        [
            _playoff_projection(row)
            for row in build_playoff_games(
                tournament_id=str(tournament_id),
                advance_count=int(advance_count),
                standings=standings,
            )
        ],
        key=lambda row: row["playoff_game_code"],
    )


def _podium_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "placement": _safe_int(row.get("placement")),
        "team_id": str(row.get("team_id") or ""),
        "source": str(row.get("source") or "").upper(),
    }


def _podium_plan(*, teams: list[dict[str, Any]], games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    playoff_games = [row for row in games if str(row.get("stage") or "").upper() == "PLAYOFF"]
    rr_games = [row for row in games if str(row.get("stage") or "").upper() == "ROUND_ROBIN"]
    source = "PLAYOFF" if playoff_games else "ROUND_ROBIN"
    podium = compute_podium_from_playoffs(playoff_games) if playoff_games else compute_podium_from_rr(teams, rr_games)
    if not podium:
        raise ValueError("The reviewed games do not produce a complete podium.")
    rows = [_podium_projection({**row, "source": source}) for row in podium[:3]]
    if {row["placement"] for row in rows} != {1, 2, 3} or any(not row["team_id"] for row in rows):
        raise ValueError("The reviewed games do not produce an exact first/second/third podium.")
    return sorted(rows, key=lambda row: int(row["placement"] or 0))


def _score_game_projection(row: dict[str, Any]) -> dict[str, Any]:
    projection = {
        "id": str(row.get("id") or ""),
        "stage": str(row.get("stage") or "").upper(),
        "playoff_game_code": str(row.get("playoff_game_code") or "") or None,
        "playoff_round": str(row.get("playoff_round") or "") or None,
        "team_a_id": str(row.get("team_a_id") or "") or None,
        "team_b_id": str(row.get("team_b_id") or "") or None,
        "team_a_source": dict(row.get("team_a_source")) if isinstance(row.get("team_a_source"), dict) else None,
        "team_b_source": dict(row.get("team_b_source")) if isinstance(row.get("team_b_source"), dict) else None,
        "score_a": _safe_int(row.get("score_a")),
        "score_b": _safe_int(row.get("score_b")),
        "winner_team_id": str(row.get("winner_team_id") or "") or None,
        "loser_team_id": str(row.get("loser_team_id") or "") or None,
        "finalized": bool(row.get("finalized_at")),
    }
    scoring_format = str(row.get("scoring_format") or "").strip().upper()
    if scoring_format:
        projection["scoring_format"] = scoring_format
    return projection


def _score_plan(*, games: list[dict[str, Any]], game_id: str, score_a: int, score_b: int) -> dict[str, Any]:
    target = next((dict(row) for row in games if str(row.get("id") or "") == str(game_id)), None)
    if not target:
        raise ValueError("The selected game does not belong to the reviewed draw.")
    finalized = finalize_game({**target, "score_a": int(score_a), "score_b": int(score_b)})
    after_target = {**target, **finalized}
    prospective = [after_target if str(row.get("id") or "") == str(game_id) else dict(row) for row in games]
    dependency_updates = resolve_playoff_dependencies(prospective) if str(target.get("stage") or "").upper() == "PLAYOFF" else []
    by_id = {str(row.get("id") or ""): dict(row) for row in prospective}
    for update in dependency_updates:
        update_id = str(update.get("id") or "")
        if update_id in by_id:
            by_id[update_id].update({key: value for key, value in update.items() if key != "id"})
    dependencies = sorted(
        [
            _score_game_projection(row)
            for row in by_id.values()
            if str(row.get("stage") or "").upper() == "PLAYOFF" and str(row.get("id") or "") != str(game_id)
        ],
        key=lambda row: row["id"],
    )
    return {"game": _score_game_projection(after_target), "downstream_games": dependencies}


def _snapshot_scoring_format(
    snapshot: dict[str, Any],
    draw_id: str,
    game_id: str,
) -> str:
    game = next(
        (
            row
            for row in snapshot.get("games") or []
            if str(row.get("id") or "") == str(game_id)
        ),
        None,
    )
    if not game:
        raise ValueError("The reviewed game scoring authority is unavailable.")
    game_format = str(game.get("scoring_format") or "").strip().upper()
    if game_format:
        if game_format not in SUPPORTED_SCORING_FORMATS:
            raise ValueError(
                "This playoff game's scoring format is unsupported. Review and "
                "regenerate the bracket before recording a result."
            )
        return game_format
    draw = next(
        (row for row in snapshot.get("draws") or [] if str(row.get("id") or "") == str(draw_id)),
        None,
    )
    if not draw:
        raise ValueError("The reviewed draw scoring authority is unavailable.")
    event = next(
        (
            row
            for row in snapshot.get("event_options") or []
            if str(row.get("id") or "") == str(draw.get("event_option_id") or "")
        ),
        None,
    )
    return resolve_tournament_scoring_format(event)


def _score_review_for_payload(
    snapshot: dict[str, Any],
    *,
    draw_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate one reviewed score command against its frozen game format."""

    scoring_format = _snapshot_scoring_format(
        snapshot,
        draw_id,
        str(payload.get("game_id") or ""),
    )
    game_scores = payload.get("game_scores")
    acknowledged = bool(payload.get("unusual_score_acknowledged"))
    if scoring_format == "BEST_2_OF_3":
        if not isinstance(game_scores, list):
            raise ValueError(
                "BEST_2_OF_3 requires the individual Game 1, Game 2, and, when needed, Game 3 scores."
            )
        review = require_best_of_three_game_scores(
            game_scores,
            unusual_score_acknowledged=acknowledged,
        )
        if (
            int(payload.get("score_a") or 0) != int(review["score_a"])
            or int(payload.get("score_b") or 0) != int(review["score_b"])
        ):
            raise ValueError(
                "The best-of-three aggregate must match the individual game winners."
            )
        return review
    if game_scores is not None:
        raise ValueError(
            "Individual game scores are accepted only for BEST_2_OF_3 matchups."
        )
    return require_tournament_score(
        int(payload.get("score_a") or 0),
        int(payload.get("score_b") or 0),
        scoring_format=scoring_format,
        unusual_score_acknowledged=acknowledged,
    )


def _active_award_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "player_id": _safe_int(row.get("player_id")),
                "badge_id": str(row.get("badge_id") or ""),
                "context_id": str(row.get("context_id") or ""),
            }
            for row in rows
            if not row.get("revoked_at")
        ],
        key=lambda row: (row["context_id"], row["badge_id"], int(row["player_id"] or 0)),
    )


DERIVED_EVIDENCE_KEYS = {
    "score_plan",
    "score_review",
    "round_robin_plan",
    "playoff_plan",
    "podium_plan",
    "award_podium_plan",
    "award_plan",
    "publish_plan",
}


def _operation_store_ready(supabase: Any | None) -> tuple[bool, str | None]:
    if supabase is None:
        return False, "Tournament Live operation storage was not checked."
    try:
        supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE).select("operation_key,client_idempotency_key").limit(1).execute()
    except Exception:
        return False, "Apply the order-28 Tournament Live operation migration before opening the write gate."
    return True, None


def _audit_store_ready(supabase: Any | None) -> tuple[bool, str | None]:
    if supabase is None:
        return False, "Tournament Live audit storage was not checked."
    try:
        supabase.table("admin_activity_log").select("id").limit(1).execute()
    except Exception:
        return False, "Required Tournament Live audit storage is unavailable; keep the write gate closed."
    return True, None


def build_admin_tournament_live_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    mutation_runtime = tournament_admin_mutation_status()
    surface_flag = mutation_runtime.get("surface_flags", {}).get(TOURNAMENT_LIVE_SURFACE, {})
    operation_store_ready, store_warning = _operation_store_ready(supabase) if is_admin_tournament_admin_enabled() else (False, None)
    audit_store_ready, audit_warning = _audit_store_ready(supabase) if is_admin_tournament_admin_enabled() else (False, None)
    environment = str(mutation_runtime.get("environment") or "local")
    service_role_ready = bool(mutation_runtime.get("service_role_ready"))
    writes_enabled = bool(
        is_admin_tournament_admin_enabled()
        and surface_flag.get("enabled")
        and service_role_ready
        and operation_store_ready
        and audit_store_ready
    )
    official_publish_writes_enabled = False
    official_publish_runtime_warning: str | None = None
    if writes_enabled:
        try:
            require_admin_tournament_official_publish_runtime()
            official_publish_writes_enabled = True
        except (PermissionError, RuntimeError) as exc:
            official_publish_runtime_warning = str(exc)
    warnings: list[str] = []
    if not is_admin_tournament_admin_enabled():
        warnings.append("Tournament Admin reads are disabled on FastAPI.")
    if environment not in {"staging", "production"}:
        warnings.append("Tournament Live writes require an explicit staging or production runtime.")
    elif not surface_flag.get("enabled"):
        warnings.append(
            f"Tournament Live writes are closed. Enable {TOURNAMENT_LIVE_WRITE_FLAG} only with the approved environment write gate."
        )
    if environment in {"staging", "production"} and not service_role_ready:
        warnings.append("FastAPI does not have the server-only Supabase service role required for Tournament Live writes.")
    if store_warning:
        warnings.append(store_warning)
    if audit_warning:
        warnings.append(audit_warning)
    if official_publish_runtime_warning:
        warnings.append(official_publish_runtime_warning)
    return {
        "enabled": is_admin_tournament_admin_enabled(),
        "status": "write_ready" if writes_enabled else "read_only_fallback",
        "authority": "python_fastapi",
        "product_boundary": "draw_scoped_tournament_runner_not_jupr_live",
        "club_id": str(club_id),
        "environment": environment,
        "staging_only": False,
        "writes_enabled": writes_enabled,
        "official_publish_writes_enabled": official_publish_writes_enabled,
        "service_role_ready": service_role_ready,
        "operation_store_ready": operation_store_ready,
        "audit_store_ready": audit_store_ready,
        "write_flag": {"name": TOURNAMENT_LIVE_WRITE_FLAG, "enabled": bool(surface_flag.get("enabled"))},
        "official_publish_write_flag": {
            "name": TOURNAMENT_OFFICIAL_PUBLISH_WRITE_FLAG,
            "enabled": _truthy_env(TOURNAMENT_OFFICIAL_PUBLISH_WRITE_FLAG),
        },
        "snapshot_endpoint": "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/snapshot",
        "command_endpoint": "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands",
        "reconcile_endpoint": "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile",
        "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or TOURNAMENT_LIVE_FALLBACK,
        "warnings": warnings,
    }


def require_tournament_live_write_runtime() -> None:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment not in {"staging", "production"}:
        raise PermissionError("Tournament Live mutations require an explicit staging or production runtime.")
    require_tournament_admin_mutation_runtime(TOURNAMENT_LIVE_SURFACE)


def _published_matches_for_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    game_ids: list[str],
) -> tuple[list[dict[str, Any]], bool, str | None]:
    if not game_ids:
        return [], True, None
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .in_("tournament_game_id", game_ids)
            .execute()
        )
    except Exception:
        return [], False, "Official tournament-match links could not be verified; publishing and score edits are blocked."
    rows = [row for row in rows if str(row.get("tournament_game_id") or "") in set(game_ids)]
    return rows, True, None


def _expected_awards(
    *,
    tournament_id: str,
    draw_id: str,
    teams: list[dict[str, Any]],
    podium: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    teams_by_id = {str(row.get("id") or ""): row for row in teams}
    expected: list[dict[str, Any]] = []
    for row in podium:
        placement = _safe_int(row.get("placement"))
        badge_id = PODIUM_BADGE_MAP.get(int(placement or 0))
        team = teams_by_id.get(str(row.get("team_id") or ""))
        if not badge_id or not team:
            continue
        context_id = f"{tournament_id}:draw:{draw_id}:podium:{placement}"
        for player_id in (team.get("player1_id"), team.get("player2_id")):
            normalized_player_id = _safe_int(player_id)
            if normalized_player_id is not None:
                expected.append(
                    {
                        "player_id": normalized_player_id,
                        "badge_id": badge_id,
                        "context_id": context_id,
                    }
                )
    return sorted(expected, key=lambda row: (row["context_id"], row["badge_id"], row["player_id"]))


def _award_rows_for_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    context_prefix = f"{tournament_id}:draw:{draw_id}:podium:"
    try:
        rows = _safe_rows(
            supabase.table("player_badges")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("context_type", "tournament")
            .execute()
        )
    except Exception:
        return [], False, "Podium award evidence could not be verified; award and official-publish actions are blocked."
    return [row for row in rows if str(row.get("context_id") or "").startswith(context_prefix)], True, None


def _require_podium_badge_catalog(
    supabase: Any,
    *,
    award_plan: list[dict[str, Any]],
) -> None:
    required_badge_ids = sorted(
        {
            str(row.get("badge_id") or "").strip()
            for row in award_plan
            if str(row.get("badge_id") or "").strip()
        }
    )
    if not required_badge_ids:
        raise ValueError("No podium badge definitions were requested.")
    try:
        catalog_rows = _safe_rows(
            supabase.table("badges")
            .select("badge_id")
            .in_("badge_id", required_badge_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "The tournament podium badge catalog is unavailable; no durable intent was created."
        ) from exc
    present_badge_ids = {
        str(row.get("badge_id") or "").strip()
        for row in catalog_rows
        if str(row.get("badge_id") or "").strip()
    }
    missing_badge_ids = sorted(set(required_badge_ids) - present_badge_ids)
    if missing_badge_ids:
        raise ValueError(
            "The tournament podium badge catalog is incomplete; missing "
            + ", ".join(missing_badge_ids)
            + ". No durable intent was created."
        )


def _award_key_sets(
    expected_awards: list[dict[str, Any]],
    award_rows: list[dict[str, Any]],
) -> tuple[set[tuple[int | None, str, str]], set[tuple[int | None, str, str]]]:
    expected_keys = {
        (int(row["player_id"]), str(row["badge_id"]), str(row["context_id"]))
        for row in expected_awards
    }
    awarded_keys = {
        (_safe_int(row.get("player_id")), str(row.get("badge_id") or ""), str(row.get("context_id") or ""))
        for row in award_rows
        if not row.get("revoked_at")
    }
    return expected_keys, awarded_keys


def _audit_rows_for_draw(supabase: Any, *, club_id: str, draw_id: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table("admin_activity_log")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("entity_id", str(draw_id))
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
    except Exception:
        return [], "Tournament Live audit evidence is unavailable."
    return rows, None


def _operation_audit_evidence(operation_key: str, audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions: list[str] = []
    timestamps: list[str] = []
    for row in audit_rows:
        after_json = row.get("after_json") if isinstance(row.get("after_json"), dict) else {}
        marker = after_json.get("audit_marker") if isinstance(after_json.get("audit_marker"), dict) else {}
        if str(marker.get("operation_key") or "") != str(operation_key):
            continue
        actions.append(str(row.get("action_type") or ""))
        if row.get("created_at"):
            timestamps.append(str(row.get("created_at")))
    return {
        "actions": actions,
        "intent_present": any(action.endswith("_intent") for action in actions),
        "completion_present": any(action.endswith("_completion") or action.endswith("_reconciliation") for action in actions),
        "failure_present": any(action.endswith("_failure") for action in actions),
        "latest_at": max(timestamps) if timestamps else None,
    }


def _operations_for_draw(
    supabase: Any,
    *,
    club_id: str,
    draw_id: str,
    audit_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("surface", TOURNAMENT_LIVE_SURFACE)
            .eq("entity_id", str(draw_id))
            .order("updated_at", desc=True)
            .limit(20)
            .execute()
        )
    except Exception:
        return [], "Tournament Live durable operation history is unavailable; keep the write gate closed."
    operations: list[dict[str, Any]] = []
    for row in rows:
        request_json = row.get("request_json") if isinstance(row.get("request_json"), dict) else {}
        result_json = row.get("result_json") if isinstance(row.get("result_json"), dict) else {}
        operations.append(
            {
                "operation_key": str(row.get("operation_key") or ""),
                "request_fingerprint": str(row.get("request_fingerprint") or ""),
                "client_idempotency_key": str(row.get("client_idempotency_key") or request_json.get("idempotency_key") or ""),
                "action": str(row.get("action") or ""),
                "command": ACTION_COMMANDS.get(str(row.get("action") or "")),
                "status": str(row.get("status") or ""),
                "expected_state": str(row.get("expected_state") or ""),
                "attempt_count": int(row.get("attempt_count") or 1),
                "error_text": str(row.get("error_text") or "")[:500] or None,
                "result_mode": str(result_json.get("mode") or "") or None,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "completion_audited_at": row.get("completion_audited_at"),
                "audit_evidence": _operation_audit_evidence(str(row.get("operation_key") or ""), audit_rows),
            }
        )
    return operations, None


def _state_fingerprint(
    snapshot: dict[str, Any],
    *,
    published_matches: list[dict[str, Any]],
    award_rows: list[dict[str, Any]],
    publication_visible: bool,
    awards_visible: bool,
) -> str:
    state = {
        "contract": "jupr:tournament-live:draw-state:v1",
        "tournament": _project(snapshot.get("tournament") or {}, ("id", "status", "start_date", "end_date", "updated_at")),
        "draw_id": str(snapshot.get("draw_id") or ""),
        "draws": _sorted_projection(
            snapshot.get("draws") or [],
            ("id", "tournament_id", "registration_day_id", "event_option_id", "name", "status", "updated_at"),
        ),
        # Bind both legacy event-scoped scoring and frozen per-game playoff
        # formats so an operator can never save against a format that changed
        # after the board was loaded.
        "event_options": _sorted_projection(
            snapshot.get("event_options") or [],
            (
                "id",
                "tournament_id",
                "registration_day_id",
                "scoring_default",
                "scoring_override",
                "division_scoring",
                "updated_at",
            ),
        ),
        "teams": _sorted_projection(
            snapshot.get("teams") or [],
            ("id", "draw_id", "team_number", "player1_id", "player2_id", "seed", "source", "updated_at"),
        ),
        "games": _sorted_projection(
            snapshot.get("games") or [],
            (
                "id",
                "draw_id",
                "stage",
                "series_parent_game_id",
                "series_game_number",
                "parent_result_only",
                "rr_round_number",
                "rr_slot_number",
                "playoff_game_code",
                "playoff_round",
                "scoring_format",
                "team_a_id",
                "team_b_id",
                "team_a_source",
                "team_b_source",
                "score_a",
                "score_b",
                "winner_team_id",
                "loser_team_id",
                "finalized_at",
                "updated_at",
            ),
        ),
        "podium": _sorted_projection(
            snapshot.get("podium") or [],
            ("id", "draw_id", "placement", "team_id", "source", "created_at", "updated_at"),
        ),
        "publication_visible": publication_visible,
        "published_matches": _sorted_projection(
            published_matches,
            ("id", "tournament_id", "tournament_game_id", "context_type", "context_id", "created_at", "updated_at"),
        ),
        "awards_visible": awards_visible,
        "awards": _sorted_projection(
            award_rows,
            ("id", "player_id", "badge_id", "context_type", "context_id", "earned_at", "revoked_at"),
        ),
    }
    return stable_tournament_admin_fingerprint(state)


def _readiness(
    *,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    rating_publish_games: list[dict[str, Any]],
    rating_publish_errors: list[dict[str, Any]],
    podium: list[dict[str, Any]],
    expected_awards: list[dict[str, Any]],
    award_rows: list[dict[str, Any]],
    published_matches: list[dict[str, Any]],
    publication_visible: bool,
    awards_visible: bool,
    writes_enabled: bool,
    official_publish_writes_enabled: bool,
    active_operation: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    rr_games = [row for row in games if str(row.get("stage") or "").upper() == "ROUND_ROBIN"]
    playoff_games = [row for row in games if str(row.get("stage") or "").upper() == "PLAYOFF"]
    all_rr_scored = bool(rr_games) and all(_is_scored(row) for row in rr_games)
    all_playoffs_scored = bool(playoff_games) and all(_is_scored(row) for row in playoff_games)
    published_ids = {str(row.get("tournament_game_id") or "") for row in published_matches}
    publication_counts: dict[str, int] = {}
    for row in published_matches:
        game_id = str(row.get("tournament_game_id") or "")
        if game_id:
            publication_counts[game_id] = publication_counts.get(game_id, 0) + 1
    duplicate_published_ids = {game_id for game_id, count in publication_counts.items() if count > 1}
    expected_keys, awarded_keys = _award_key_sets(expected_awards, award_rows)
    unexpected_awarded_keys = awarded_keys - expected_keys
    awards_complete = bool(expected_keys) and expected_keys.issubset(awarded_keys)
    awards_partial = bool(expected_keys.intersection(awarded_keys)) and not awards_complete

    common: list[str] = []
    if not writes_enabled:
        common.append("The dedicated staging write gate, service role, or durable operation store is not ready.")
    if active_operation:
        common.append(f"Operation {str(active_operation.get('operation_key') or '')[:12]}… is {active_operation.get('status')}; reconcile it before another draw write.")

    blockers: dict[str, list[str]] = {command: list(common) for command in COMMAND_CONFIRMATIONS}
    if not official_publish_writes_enabled:
        blockers["publish_official_matches"].append(
            "The separate official-publish runtime gate is closed."
        )
    if not games:
        blockers["save_score"].append("This draw has no games.")
    if published_ids:
        blockers["save_score"].append(
            "At least one game is already an official rated match. Linked source results are immutable; use tournament recovery and reconciliation, never a generic Match Log edit."
        )
    if podium:
        blockers["save_score"].append("Scores are locked after podium generation; use Tournament Ops for a reviewed correction.")
    if award_rows:
        blockers["save_score"].append("Scores are locked after draw-scoped podium awards exist.")
    if not publication_visible:
        blockers["save_score"].append("Official match visibility is unavailable, so score safety cannot be proven.")

    if len(teams) not in SUPPORTED_TEAM_COUNTS:
        blockers["generate_round_robin"].append(f"Round robin requires one of {SUPPORTED_TEAM_COUNTS} teams; this draw has {len(teams)}.")
    if games:
        blockers["generate_round_robin"].append("This draw already has games.")
    if podium or award_rows:
        blockers["generate_round_robin"].append("Round-robin generation is locked after podium or award evidence exists.")
    team_numbers = [_safe_int(row.get("team_number")) for row in teams]
    if teams and (any(value is None for value in team_numbers) or sorted(int(value) for value in team_numbers if value is not None) != list(range(1, len(teams) + 1))):
        blockers["generate_round_robin"].append("Team numbers must be contiguous from 1 through the draw size.")

    if not rr_games:
        blockers["generate_playoffs"].append("Generate round-robin games first.")
    elif not all_rr_scored:
        blockers["generate_playoffs"].append("Every round-robin game must have a finalized non-tied score.")
    if playoff_games:
        blockers["generate_playoffs"].append("This draw already has playoff games.")
    if podium or award_rows:
        blockers["generate_playoffs"].append("Playoff generation is locked after podium or award evidence exists.")

    if podium:
        blockers["generate_podium"].append("A draw-scoped podium already exists; reconcile or use Tournament Ops for a reviewed correction.")
    if award_rows:
        blockers["generate_podium"].append("Draw-scoped award evidence already exists; reconcile before podium generation.")
    if playoff_games and not all_playoffs_scored:
        blockers["generate_podium"].append("Every playoff game must be finalized before podium generation.")
    if not playoff_games and not all_rr_scored:
        blockers["generate_podium"].append("Every round-robin game must be finalized before podium generation.")

    placements = {_safe_int(row.get("placement")) for row in podium}
    if placements != {1, 2, 3}:
        blockers["award_podium"].append("A complete first/second/third draw podium is required.")
    if not awards_visible:
        blockers["award_podium"].append("Award evidence is unavailable.")
    if not expected_awards:
        blockers["award_podium"].append("No linked-player award candidates exist for this podium.")
    if awards_partial:
        blockers["award_podium"].append("Only part of the expected podium award set exists; reconcile before any retry.")
    if unexpected_awarded_keys:
        blockers["award_podium"].append("Unexpected draw-scoped podium awards exist; reconcile before any retry.")
    if awards_complete:
        blockers["award_podium"].append("Every expected podium award is already present.")

    rating_eligible_game_ids = {
        str(row.get("id") or "")
        for row in rating_publish_games
        if row.get("id") and _is_rating_publish_eligible(row)
    }
    if not games or not all(_is_scored(row) for row in games):
        blockers["publish_official_matches"].append("Every tournament game must have a finalized non-tied score.")
    if games and not rating_eligible_game_ids:
        blockers["publish_official_matches"].append(
            "This draw has no played result eligible for official rating publication."
        )
    if rating_publish_errors:
        blockers["publish_official_matches"].append(
            "Best-two-of-three individual game evidence is incomplete or inconsistent; reconcile it before publishing."
        )
    if placements != {1, 2, 3}:
        blockers["publish_official_matches"].append("Generate and review the draw podium first.")
    if not awards_complete:
        blockers["publish_official_matches"].append("Complete and verify all podium awards before official match publishing.")
    if unexpected_awarded_keys:
        blockers["publish_official_matches"].append("Unexpected draw-scoped podium awards must be reconciled before publishing.")
    if not publication_visible:
        blockers["publish_official_matches"].append("Official match links are unavailable.")
    if published_ids:
        if duplicate_published_ids:
            blockers["publish_official_matches"].append(
                "Duplicate official links exist for this draw; stop and reconcile the tournament publication before any retry."
            )
        elif published_ids == rating_eligible_game_ids:
            blockers["publish_official_matches"].append("Every played game is already published as an official match.")
        else:
            blockers["publish_official_matches"].append("Only part of this draw is published; reconcile before any retry.")

    return {
        command: {
            "ready": not reasons,
            "confirmation": COMMAND_CONFIRMATIONS[command],
            "blockers": reasons,
        }
        for command, reasons in blockers.items()
    }


def build_admin_tournament_live_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    base = get_admin_tournament_ops_snapshot(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        draw_id=str(draw_id) if draw_id else None,
    )
    status = build_admin_tournament_live_status(supabase, club_id=str(club_id))
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        selected_draw_id=str(draw_id) if draw_id else None,
    )
    lifecycle["runtime_capability"] = {
        **dict(lifecycle.get("runtime_capability") or {}),
        "live_writes_enabled": bool(status.get("writes_enabled")),
        "official_publish_writes_enabled": bool(
            status.get("official_publish_writes_enabled")
        ),
        "live_operation_store_ready": bool(status.get("operation_store_ready")),
        "live_audit_store_ready": bool(status.get("audit_store_ready")),
    }
    if not draw_id:
        selector_games = _competition_games(list(base.get("games") or []))
        selector_summary = {
            **dict(base.get("summary") or {}),
            "games": len(selector_games),
            "completed_games": len(
                [
                    row
                    for row in selector_games
                    if str(row.get("status") or "").lower()
                    in {"complete", "completed", "final"}
                    or row.get("winner_team_id")
                ]
            ),
        }
        return {
            **base,
            "summary": selector_summary,
            "games": selector_games,
            "mode": "tournament_live_draw_selector",
            "scope": "tournament_selector",
            "authority": "python_fastapi",
            "product_boundary": "draw_scoped_tournament_runner_not_jupr_live",
            "state_fingerprint": None,
            "ops_state_fingerprint": None,
            "runtime": status,
            "lifecycle": lifecycle,
            "readiness": {},
            "operations": [],
        }
    if len(base.get("draws") or []) != 1 or str((base.get("draws") or [{}])[0].get("id") or "") != str(draw_id):
        raise ValueError("draw not found for this tournament")

    warnings = list(base.get("warnings") or [])
    try:
        ops_state_fingerprint = get_admin_tournament_ops_state_fingerprint(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
        )
    except Exception:
        ops_state_fingerprint = None
        warnings.append(
            "Tournament Ops guarded state is unavailable; podium review remains closed."
        )
    all_games = _authoritative_draw_games(
        supabase,
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
        expected_versions=base.get("source_game_versions"),
    )
    games = _competition_games(all_games)
    rating_game_plan = build_tournament_rating_game_plan(all_games)
    rating_publish_games = [
        dict(row) for row in (rating_game_plan.get("rating_games") or [])
    ]
    rating_publish_errors = [
        dict(row) for row in (rating_game_plan.get("errors") or [])
    ]
    teams = list(base.get("teams") or [])
    podium = list(base.get("podium") or [])
    all_game_ids = [
        str(row.get("id") or "") for row in all_games if row.get("id")
    ]
    published_matches, publication_visible, publication_warning = _published_matches_for_draw(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        game_ids=all_game_ids,
    )
    if publication_warning:
        warnings.append(publication_warning)
    expected_awards = _expected_awards(
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
        teams=teams,
        podium=podium,
    )
    award_rows, awards_visible, award_warning = _award_rows_for_draw(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
    )
    if award_warning:
        warnings.append(award_warning)
    audit_rows, audit_warning = _audit_rows_for_draw(supabase, club_id=str(club_id), draw_id=str(draw_id))
    if audit_warning:
        warnings.append(audit_warning)
    operations, operation_warning = _operations_for_draw(
        supabase,
        club_id=str(club_id),
        draw_id=str(draw_id),
        audit_rows=audit_rows,
    )
    if operation_warning:
        warnings.append(operation_warning)
    active_operation = next((row for row in operations if str(row.get("status") or "") in ACTIVE_OPERATION_STATUSES), None)
    expected_award_keys, awarded_keys = _award_key_sets(expected_awards, award_rows)
    verified_award_count = len(expected_award_keys.intersection(awarded_keys))
    unexpected_award_count = len(awarded_keys - expected_award_keys)
    publication_counts: dict[str, int] = {}
    for row in published_matches:
        published_game_id = str(row.get("tournament_game_id") or "")
        if published_game_id:
            publication_counts[published_game_id] = publication_counts.get(published_game_id, 0) + 1
    duplicate_published_game_ids = sorted(
        game_id for game_id, count in publication_counts.items() if count > 1
    )
    published_game_ids = sorted(publication_counts)
    rating_eligible_game_ids = sorted(
        str(row.get("id") or "")
        for row in rating_publish_games
        if row.get("id") and _is_rating_publish_eligible(row)
    )
    publication_complete = (
        bool(rating_eligible_game_ids)
        and not duplicate_published_game_ids
        and set(published_game_ids) == set(rating_eligible_game_ids)
    )
    fingerprint = _state_fingerprint(
        {**base, "games": all_games},
        published_matches=published_matches,
        award_rows=award_rows,
        publication_visible=publication_visible,
        awards_visible=awards_visible,
    )
    readiness = _readiness(
        teams=teams,
        games=games,
        rating_publish_games=rating_publish_games,
        rating_publish_errors=rating_publish_errors,
        podium=podium,
        expected_awards=expected_awards,
        award_rows=award_rows,
        published_matches=published_matches,
        publication_visible=publication_visible,
        awards_visible=awards_visible,
        writes_enabled=bool(status.get("writes_enabled")) and not audit_warning and not operation_warning,
        official_publish_writes_enabled=bool(
            status.get("official_publish_writes_enabled")
        ),
        active_operation=active_operation,
    )
    lifecycle_draw = next(
        (
            row
            for row in lifecycle.get("draws") or []
            if str(row.get("draw_id") or "") == str(draw_id)
        ),
        {},
    )
    review_evidence = (
        lifecycle_draw.get("review_evidence")
        if isinstance(lifecycle_draw.get("review_evidence"), dict)
        else {}
    )
    if not bool(review_evidence.get("current")):
        review_message = str(
            (review_evidence.get("blockers") or ["Explicitly review the current podium before awarding trophies."])[0]
        )
        for command in ("award_podium", "publish_official_matches"):
            command_readiness = readiness.get(command) or {}
            command_readiness["blockers"] = list(command_readiness.get("blockers") or []) + [review_message]
            command_readiness["ready"] = False
    publish_domain = (lifecycle.get("domain_readiness") or {}).get("official_publish") or {}
    if not bool(publish_domain.get("ready")):
        command_readiness = readiness.get("publish_official_matches") or {}
        command_readiness["blockers"] = list(command_readiness.get("blockers") or []) + [
            str(row.get("message") or "")
            for row in publish_domain.get("blockers") or []
            if str(row.get("message") or "")
        ]
        command_readiness["blockers"] = list(dict.fromkeys(command_readiness["blockers"]))
        command_readiness["ready"] = False
    source_game_versions = _snapshot_version_rows(
        all_games,
        label="source game",
    ) if all_games else []
    operational_summary = {
        **dict(base.get("summary") or {}),
        "games": len(games),
        "completed_games": len(
            [
                row
                for row in games
                if str(row.get("status") or "").lower()
                in {"complete", "completed", "final"}
                or row.get("winner_team_id")
            ]
        ),
    }
    return {
        **base,
        "summary": operational_summary,
        "games": games,
        "mode": "tournament_live_draw_snapshot",
        "scope": "draw",
        "authority": "python_fastapi",
        "product_boundary": "draw_scoped_tournament_runner_not_jupr_live",
        "state_fingerprint": fingerprint,
        "ops_state_fingerprint": ops_state_fingerprint,
        "source_game_versions": source_game_versions,
        "publication_source_game_versions": source_game_versions,
        "publication_rating_game_ids": rating_eligible_game_ids,
        "runtime": status,
        "lifecycle": lifecycle,
        "progression": {
            "phase": "published" if publication_complete else "podium" if podium else "playoffs" if any(str(row.get("stage") or "").upper() == "PLAYOFF" for row in games) else "round_robin" if games else "teams_ready",
            "open_games": len([row for row in games if not _is_scored(row)]),
            "completed_games": len([row for row in games if _is_scored(row)]),
            "published_games": len(published_game_ids),
            "expected_awards": len(expected_awards),
            "verified_awards": verified_award_count,
        },
        "publication_evidence": {
            "available": publication_visible,
            "published_game_ids": published_game_ids,
            "match_count": len(published_matches),
            "duplicate_game_ids": duplicate_published_game_ids,
            "complete": publication_complete,
        },
        "award_evidence": {
            "available": awards_visible,
            "expected": expected_awards,
            "verified_count": verified_award_count,
            "active_row_count": len(awarded_keys),
            "unexpected_count": unexpected_award_count,
        },
        "readiness": readiness,
        "active_operation": active_operation,
        "operations": operations,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _normalize_idempotency_key(value: Any) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("idempotency_key is required for every Tournament Live command.")
    try:
        parsed = uuid.UUID(clean)
    except (ValueError, AttributeError) as exc:
        raise ValueError("idempotency_key must be a UUID generated once and retained for an exact retry.") from exc
    return str(parsed)


def _normalize_command_request(request: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    command = str(request.get("command") or "").strip().lower()
    if command not in COMMAND_CONFIRMATIONS:
        raise ValueError("Unsupported Tournament Live command.")
    expected_state = str(request.get("expected_state_fingerprint") or "").strip().lower()
    if not STATE_HASH_RE.fullmatch(expected_state):
        raise ValueError("A valid 64-character draw state fingerprint is required. Reload the live board.")
    idempotency_key = _normalize_idempotency_key(request.get("idempotency_key"))
    expected_confirmation = COMMAND_CONFIRMATIONS[command]
    if str(request.get("confirmation_text") or "").strip() != expected_confirmation:
        raise ValueError(f"Type {expected_confirmation} exactly to run this Tournament Live command.")

    expected_draw_updated_at = str(request.get("expected_draw_updated_at") or "").strip()
    if not expected_draw_updated_at:
        raise ValueError("A reviewed draw version is required. Reload the live board.")
    payload: dict[str, Any] = {"expected_draw_updated_at": expected_draw_updated_at}
    if command == "save_score":
        game_id = str(request.get("game_id") or "").strip()
        score_a = _safe_int(request.get("score_a"))
        score_b = _safe_int(request.get("score_b"))
        raw_game_scores = request.get("game_scores")
        expected_game_updated_at = str(request.get("expected_game_updated_at") or "").strip()
        if not game_id:
            raise ValueError("game_id is required for score entry.")
        if not expected_game_updated_at:
            raise ValueError("A reviewed game version is required. Reload the live board.")
        normalized_game_scores: list[dict[str, int]] | None = None
        if raw_game_scores is not None:
            if not isinstance(raw_game_scores, list):
                raise ValueError("game_scores must be a list of individual game scores.")
            normalized_game_scores = []
            for raw in raw_game_scores:
                if not isinstance(raw, dict):
                    raise ValueError("Each individual game score must be an object.")
                game_number = _safe_int(raw.get("game_number"))
                game_score_a = _safe_int(raw.get("score_a"))
                game_score_b = _safe_int(raw.get("score_b"))
                if game_number is None or game_score_a is None or game_score_b is None:
                    raise ValueError(
                        "Each individual game requires a game number and both whole-number scores."
                    )
                normalized_game_scores.append(
                    {
                        "game_number": game_number,
                        "score_a": game_score_a,
                        "score_b": game_score_b,
                    }
                )
            series_review = require_best_of_three_game_scores(
                normalized_game_scores,
                unusual_score_acknowledged=bool(
                    request.get("unusual_score_acknowledged", False)
                ),
            )
            derived_a = int(series_review["score_a"])
            derived_b = int(series_review["score_b"])
            if score_a is not None and score_a != derived_a:
                raise ValueError(
                    "score_a does not match the aggregate derived from the individual games."
                )
            if score_b is not None and score_b != derived_b:
                raise ValueError(
                    "score_b does not match the aggregate derived from the individual games."
                )
            score_a = derived_a
            score_b = derived_b
        if score_a is None or score_b is None or score_a < 0 or score_b < 0 or score_a == score_b:
            raise ValueError("Tournament scores must be non-negative, whole-number, and non-tied.")
        payload.update(
            {
                "game_id": game_id,
                "score_a": score_a,
                "score_b": score_b,
                "expected_game_updated_at": expected_game_updated_at,
                "unusual_score_acknowledged": bool(
                    request.get("unusual_score_acknowledged", False)
                ),
            }
        )
        if normalized_game_scores is not None:
            payload["game_scores"] = normalized_game_scores
    elif command == "generate_round_robin":
        payload["expected_team_versions"] = _canonical_version_rows(
            request.get("expected_team_versions"),
            label="team",
        )
    elif command == "generate_playoffs":
        advance_count = _safe_int(request.get("advance_count"))
        if advance_count not in SUPPORTED_ADVANCE_COUNTS:
            raise ValueError("Playoff generation supports advance counts of 4, 5, or 6.")
        payload.update(
            {
                "advance_count": int(advance_count),
                "expected_team_versions": _canonical_version_rows(request.get("expected_team_versions"), label="team"),
                "expected_source_game_versions": _canonical_version_rows(
                    request.get("expected_source_game_versions"),
                    label="source game",
                ),
            }
        )
    elif command in {"generate_podium", "award_podium"}:
        payload.update(
            {
                "expected_team_versions": _canonical_version_rows(request.get("expected_team_versions"), label="team"),
                "expected_source_game_versions": _canonical_version_rows(
                    request.get("expected_source_game_versions"),
                    label="source game",
                ),
            }
        )
    elif command == "publish_official_matches":
        bonus = _safe_float(request.get("playoff_winner_bonus_elo"))
        if bonus is None:
            bonus = 0.0
        if bonus < 0 or bonus > 40:
            raise ValueError("Playoff winner bonus must be between 0 and 40 Elo points.")
        payload.update(
            {
                "playoff_winner_bonus_elo": float(bonus),
                "expected_team_versions": _canonical_version_rows(request.get("expected_team_versions"), label="team"),
                "expected_source_game_versions": _canonical_version_rows(
                    request.get("expected_source_game_versions"),
                    label="source game",
                ),
            }
        )
    return command, expected_state, idempotency_key, payload


def _validate_reviewed_versions(snapshot: dict[str, Any], *, command: str, payload: dict[str, Any]) -> None:
    draws = list(snapshot.get("draws") or [])
    if len(draws) != 1:
        raise StaleTournamentAdminStateError("The reviewed draw is unavailable. Reload the live board.")
    if str(draws[0].get("updated_at") or "") != str(payload.get("expected_draw_updated_at") or ""):
        raise StaleTournamentAdminStateError("This tournament draw changed after review. Reload the live board.")
    if command == "save_score":
        game = next(
            (row for row in snapshot.get("games") or [] if str(row.get("id") or "") == str(payload.get("game_id") or "")),
            None,
        )
        if not game or str(game.get("updated_at") or "") != str(payload.get("expected_game_updated_at") or ""):
            raise StaleTournamentAdminStateError("The selected tournament game changed after review. Reload the live board.")
        return
    expected_teams = payload.get("expected_team_versions")
    if expected_teams != _snapshot_version_rows(list(snapshot.get("teams") or []), label="team"):
        raise StaleTournamentAdminStateError("The tournament team set changed after review. Reload the live board.")
    if command in {"generate_playoffs", "generate_podium", "award_podium", "publish_official_matches"}:
        expected_games = payload.get("expected_source_game_versions")
        current_game_versions = list(snapshot.get("source_game_versions") or [])
        if expected_games != current_game_versions:
            raise StaleTournamentAdminStateError("The tournament source game set changed after review. Reload the live board.")


def _build_command_evidence(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    snapshot: dict[str, Any],
    command: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    teams = list(snapshot.get("teams") or [])
    games = list(snapshot.get("games") or [])
    podium = list(snapshot.get("podium") or [])
    if command == "save_score":
        score_review = _score_review_for_payload(
            snapshot,
            draw_id=draw_id,
            payload=payload,
        )
        return {
            "score_plan": _score_plan(
                games=games,
                game_id=str(payload.get("game_id") or ""),
                score_a=int(payload.get("score_a") or 0),
                score_b=int(payload.get("score_b") or 0),
            ),
            "score_review": score_review,
        }
    if command == "generate_round_robin":
        return {"round_robin_plan": _round_robin_plan(tournament_id=str(tournament_id), teams=teams)}
    if command == "generate_playoffs":
        return {
            "playoff_plan": _playoff_plan(
                tournament_id=str(tournament_id),
                teams=teams,
                games=games,
                advance_count=int(payload.get("advance_count") or 0),
            )
        }
    if command == "generate_podium":
        return {"podium_plan": _podium_plan(teams=teams, games=games)}
    if command == "award_podium":
        award_plan = _expected_awards(
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            teams=teams,
            podium=podium,
        )
        _require_podium_badge_catalog(
            supabase,
            award_plan=award_plan,
        )
        return {
            "award_podium_plan": sorted(
                [_podium_projection(row) for row in podium],
                key=lambda row: int(row["placement"] or 0),
            ),
            "award_plan": award_plan,
        }
    if command == "publish_official_matches":
        publish_plan = build_admin_tournament_official_publish_plan(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            playoff_winner_bonus_elo=float(payload.get("playoff_winner_bonus_elo") or 0.0),
        )
        reviewed_game_ids = sorted(
            str(value)
            for value in (snapshot.get("publication_rating_game_ids") or [])
            if str(value)
        )
        planned_game_ids = sorted(str(value) for value in (publish_plan.get("tournament_game_ids") or []))
        if (
            not reviewed_game_ids
            or planned_game_ids != reviewed_game_ids
            or str(publish_plan.get("draw_updated_at") or "") != str(payload.get("expected_draw_updated_at") or "")
            or publish_plan.get("team_versions") != payload.get("expected_team_versions")
            or publish_plan.get("game_versions") != payload.get("expected_source_game_versions")
            or not publish_plan.get("match_payload_projections")
            or not publish_plan.get("match_payload_fingerprints")
        ):
            raise StaleTournamentAdminStateError(
                "The official-publish game plan did not match the reviewed draw. Reload the live board."
            )
        return {"publish_plan": {**publish_plan, "tournament_game_ids": planned_game_ids}}
    raise ValueError("Unsupported Tournament Live command.")


def _validate_command_evidence(
    snapshot: dict[str, Any],
    *,
    tournament_id: str,
    draw_id: str,
    command: str,
    payload: dict[str, Any],
) -> None:
    teams = list(snapshot.get("teams") or [])
    games = list(snapshot.get("games") or [])
    podium = list(snapshot.get("podium") or [])
    if command == "save_score":
        expected_review = _score_review_for_payload(
            snapshot,
            draw_id=draw_id,
            payload=payload,
        )
        expected = _score_plan(
            games=games,
            game_id=str(payload.get("game_id") or ""),
            score_a=int(payload.get("score_a") or 0),
            score_b=int(payload.get("score_b") or 0),
        )
        if payload.get("score_plan") != expected or payload.get("score_review") != expected_review:
            raise StaleTournamentAdminStateError("The score/dependency plan changed after review. Reload the live board.")
    elif command == "generate_round_robin":
        if payload.get("round_robin_plan") != _round_robin_plan(tournament_id=str(tournament_id), teams=teams):
            raise StaleTournamentAdminStateError("The round-robin plan changed after review. Reload the live board.")
    elif command == "generate_playoffs":
        expected = _playoff_plan(
            tournament_id=str(tournament_id),
            teams=teams,
            games=games,
            advance_count=int(payload.get("advance_count") or 0),
        )
        if payload.get("playoff_plan") != expected:
            raise StaleTournamentAdminStateError("The playoff plan changed after review. Reload the live board.")
    elif command == "generate_podium":
        if payload.get("podium_plan") != _podium_plan(teams=teams, games=games):
            raise StaleTournamentAdminStateError("The podium plan changed after review. Reload the live board.")
    elif command == "award_podium":
        expected_podium = sorted(
            [_podium_projection(row) for row in podium],
            key=lambda row: int(row["placement"] or 0),
        )
        expected = _expected_awards(
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            teams=teams,
            podium=podium,
        )
        if (
            not expected
            or payload.get("award_podium_plan") != expected_podium
            or payload.get("award_plan") != expected
        ):
            raise StaleTournamentAdminStateError("The podium award recipient set changed after review. Reload the live board.")
    elif command == "publish_official_matches":
        publish_plan = payload.get("publish_plan") if isinstance(payload.get("publish_plan"), dict) else {}
        reviewed_ids = sorted(
            str(value)
            for value in (snapshot.get("publication_rating_game_ids") or [])
            if str(value)
        )
        planned_ids = sorted(str(value) for value in (publish_plan.get("tournament_game_ids") or []))
        if (
            not planned_ids
            or planned_ids != reviewed_ids
            or str(publish_plan.get("draw_id") or "") != str(draw_id)
            or str(publish_plan.get("draw_updated_at") or "") != str(payload.get("expected_draw_updated_at") or "")
            or publish_plan.get("team_versions") != payload.get("expected_team_versions")
            or publish_plan.get("game_versions") != payload.get("expected_source_game_versions")
            or not publish_plan.get("match_payload_fingerprints")
        ):
            raise StaleTournamentAdminStateError("The official-publish game set changed after review. Reload the live board.")


def _preflight_command(snapshot: dict[str, Any], *, command: str, payload: dict[str, Any]) -> None:
    _validate_reviewed_versions(snapshot, command=command, payload=payload)
    if command == "award_podium":
        # Podium rows are a first-class reviewed dependency. Validate their
        # version contract before durable intent so a schema/projection defect
        # cannot strand the draw behind a recovery lock.
        _snapshot_version_rows(
            list(snapshot.get("podium") or []),
            label="podium",
        )
    readiness = (snapshot.get("readiness") or {}).get(command) or {}
    blockers = [str(value) for value in (readiness.get("blockers") or []) if str(value)]
    if blockers:
        raise ValueError("Tournament Live command is not ready: " + " ".join(blockers))
    if command == "save_score":
        game = next((row for row in snapshot.get("games") or [] if str(row.get("id") or "") == str(payload.get("game_id") or "")), None)
        if not game:
            raise ValueError("The selected game does not belong to this draw.")
        if not game.get("team_a_id") or not game.get("team_b_id"):
            raise ValueError("Both teams must be assigned before scoring this game.")
        if str(game.get("stage") or "").upper() == "ROUND_ROBIN" and any(
            str(row.get("stage") or "").upper() == "PLAYOFF" for row in snapshot.get("games") or []
        ):
            raise ValueError("Round-robin scores are locked after playoff generation; use Tournament Ops recovery.")
    if command == "generate_playoffs" and int(payload.get("advance_count") or 0) > len(snapshot.get("teams") or []):
        raise ValueError("Advance count cannot exceed the number of teams in this draw.")


def _require_command_runtime(command: str) -> None:
    if command == "publish_official_matches":
        require_admin_tournament_official_publish_runtime()


def _mutate_command(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    command: str,
    payload: dict[str, Any],
    reviewed_snapshot: dict[str, Any],
    actor_email: str,
    actor_role: str,
    guarded_operation_key: str,
    guarded_request_fingerprint: str,
    client_idempotency_key: str,
) -> dict[str, Any]:
    source = f"next_tournament_live_{command}"
    confirmation = COMMAND_CONFIRMATIONS[command]
    common = {
        "supabase": supabase,
        "club_id": str(club_id),
        "tournament_id": str(tournament_id),
        "actor_email": actor_email,
        "actor_role": actor_role,
        "confirmation_text": confirmation,
        "source": source,
    }
    draws = list(reviewed_snapshot.get("draws") or [])
    if len(draws) != 1 or str(reviewed_snapshot.get("state_fingerprint") or "") == "":
        raise StaleTournamentAdminStateError("The post-lock reviewed draw snapshot is unavailable.")
    reviewed_draw_updated_at = str(draws[0].get("updated_at") or "")
    reviewed_team_versions = _snapshot_version_rows(list(reviewed_snapshot.get("teams") or []), label="team")
    reviewed_source_game_versions = list(
        reviewed_snapshot.get("source_game_versions") or []
    )
    if command == "save_score":
        reviewed_game = next(
            (
                row
                for row in reviewed_snapshot.get("games") or []
                if str(row.get("id") or "") == str(payload.get("game_id") or "")
            ),
            None,
        )
        if not reviewed_game:
            raise StaleTournamentAdminStateError("The post-lock reviewed game snapshot is unavailable.")
        return update_admin_tournament_game_score(
            **common,
            game_id=str(payload["game_id"]),
            score_a=int(payload["score_a"]),
            score_b=int(payload["score_b"]),
            game_scores=(
                [dict(row) for row in payload.get("game_scores") or []]
                if payload.get("game_scores") is not None
                else None
            ),
            unusual_score_acknowledged=bool(
                payload.get("unusual_score_acknowledged")
            ),
            expected_updated_at=str(reviewed_game.get("updated_at") or ""),
            expected_draw_updated_at=reviewed_draw_updated_at,
            expected_source_game_versions=reviewed_source_game_versions,
            atomic=True,
        )
    if command == "generate_round_robin":
        return generate_admin_tournament_round_robin_games(
            **common,
            draw_id=str(draw_id),
            expected_draw_updated_at=reviewed_draw_updated_at,
            expected_team_versions=reviewed_team_versions,
            atomic=True,
        )
    if command == "generate_playoffs":
        return generate_admin_tournament_playoff_games(
            **common,
            draw_id=str(draw_id),
            advance_count=int(payload["advance_count"]),
            expected_draw_updated_at=reviewed_draw_updated_at,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_source_game_versions,
            atomic=True,
        )
    if command == "generate_podium":
        return generate_admin_tournament_draw_podium(
            **common,
            draw_id=str(draw_id),
            expected_draw_updated_at=reviewed_draw_updated_at,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_source_game_versions,
            atomic=True,
        )
    if command == "award_podium":
        reviewed_podium_versions = _snapshot_version_rows(
            list(reviewed_snapshot.get("podium") or []),
            label="podium",
        )
        return award_admin_tournament_draw_podium(
            **common,
            draw_id=str(draw_id),
            expected_draw_updated_at=reviewed_draw_updated_at,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_source_game_versions,
            expected_podium_versions=reviewed_podium_versions,
            expected_podium=list(payload.get("award_podium_plan") or []),
            expected_awards=list(payload.get("award_plan") or []),
            atomic=True,
        )
    if command == "publish_official_matches":
        require_admin_tournament_official_publish_runtime()
        return publish_admin_tournament_draw_matches(
            **common,
            draw_id=str(draw_id),
            playoff_winner_bonus_elo=float(payload.get("playoff_winner_bonus_elo") or 0.0),
            expected_plan=dict(payload.get("publish_plan") or {}),
            guarded_operation_key=str(guarded_operation_key),
            guarded_request_fingerprint=str(guarded_request_fingerprint),
            client_idempotency_key=str(client_idempotency_key),
        )
    raise ValueError("Unsupported Tournament Live command.")


def execute_admin_tournament_live_command(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    request: dict[str, Any],
    actor_email: str,
    actor_role: str,
) -> dict[str, Any]:
    require_tournament_live_write_runtime()
    command, expected_state, idempotency_key, normalized_payload = _normalize_command_request(request)
    _require_live_command_permission(actor_role, command)
    existing = get_tournament_admin_operation_record_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_LIVE_SURFACE,
        idempotency_key=idempotency_key,
    )
    if existing:
        request_json = existing.get("request_json") if isinstance(existing.get("request_json"), dict) else {}
        stored_payload = request_json.get("payload") if isinstance(request_json.get("payload"), dict) else {}
        stored_base_payload = {
            key: value for key, value in stored_payload.items() if key not in DERIVED_EVIDENCE_KEYS
        }
        if (
            str(existing.get("expected_state") or "") != expected_state
            or stored_base_payload != {"command": command, **normalized_payload}
        ):
            raise ValueError(
                "This idempotency key was already used for a different Tournament Live request. Reload and create a new command."
            )
        payload = dict(stored_payload)
        if str(existing.get("status") or "") == "recovery_required":
            recovery = _verified_recovery_outcome(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                operation=existing,
            )
            if str(recovery.get("status") or "") == "not_applied":
                reconciled = reconcile_tournament_admin_guarded_operation(
                    supabase,
                    club_id=str(club_id),
                    surface=TOURNAMENT_LIVE_SURFACE,
                    operation_key=str(existing.get("operation_key") or ""),
                    entity_type="tournament_event_draw",
                    entity_id=str(draw_id),
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=f"next_tournament_live_{command}_exact_retry",
                    verify_outcome=lambda _operation: recovery,
                )
                return {
                    **reconciled,
                    "authority": "python_fastapi",
                    "tournament_id": str(tournament_id),
                    "draw_id": str(draw_id),
                    "command": command,
                }
    else:
        # The separate rating/email gate is a pre-intent capability. A closed
        # official-publish gate must never create a durable recovery lock.
        _require_command_runtime(command)
        seed_snapshot = build_admin_tournament_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
        )
        if str(seed_snapshot.get("state_fingerprint") or "") != expected_state:
            raise StaleTournamentAdminStateError(
                "Tournament Admin data changed after it was loaded. Reload the authoritative detail, review the impact, and submit again."
            )
        _preflight_command(seed_snapshot, command=command, payload=normalized_payload)
        payload = {
            "command": command,
            **normalized_payload,
            **_build_command_evidence(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                snapshot=seed_snapshot,
                command=command,
                payload=normalized_payload,
            ),
        }

    captured_snapshot: dict[str, Any] | None = None

    def current_snapshot() -> dict[str, Any]:
        nonlocal captured_snapshot
        captured_snapshot = build_admin_tournament_live_snapshot(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
        )
        return captured_snapshot

    def require_captured_snapshot(*, include_readiness: bool) -> dict[str, Any]:
        snapshot = captured_snapshot
        if not snapshot or str(snapshot.get("state_fingerprint") or "") != expected_state:
            raise StaleTournamentAdminStateError(
                "The captured Tournament Live snapshot no longer matches the reviewed draw. Reload before continuing."
            )
        if include_readiness:
            _require_command_runtime(command)
            _preflight_command(snapshot, command=command, payload=payload)
        else:
            _validate_reviewed_versions(snapshot, command=command, payload=payload)
        _validate_command_evidence(
            snapshot,
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            command=command,
            payload=payload,
        )
        return snapshot

    def reconcile_response_loss(operation: dict[str, Any]) -> dict[str, Any] | None:
        outcome = _verified_recovery_outcome(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            operation=operation,
        )
        if str(outcome.get("status") or "") == "completed" and isinstance(outcome.get("result"), dict):
            return dict(outcome["result"])
        return None

    operation_identity = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface=TOURNAMENT_LIVE_SURFACE,
        action=COMMAND_ACTIONS[command],
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        lock_scope=f"tournament:{tournament_id}:draw:{draw_id}",
        expected_state=expected_state,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    result = run_tournament_admin_guarded_operation(
        supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_LIVE_SURFACE,
        action=COMMAND_ACTIONS[command],
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        lock_scope=f"tournament:{tournament_id}:draw:{draw_id}",
        expected_state=expected_state,
        current_state=lambda: str(current_snapshot().get("state_fingerprint") or ""),
        payload=payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=f"next_tournament_live_{command}",
        preflight=lambda: require_captured_snapshot(include_readiness=True),
        reconcile=reconcile_response_loss,
        mutate=lambda: _mutate_command(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            command=command,
            payload=payload,
            reviewed_snapshot=require_captured_snapshot(include_readiness=False),
            actor_email=actor_email,
            actor_role=actor_role,
            guarded_operation_key=str(operation_identity["operation_key"]),
            guarded_request_fingerprint=str(operation_identity["request_fingerprint"]),
            client_idempotency_key=idempotency_key,
        ),
        idempotency_key=idempotency_key,
    )
    return {
        **result,
        "authority": "python_fastapi",
        "tournament_id": str(tournament_id),
        "draw_id": str(draw_id),
        "command": command,
    }


def _verified_recovery_outcome(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    operation: dict[str, Any],
) -> dict[str, Any]:
    command = ACTION_COMMANDS.get(str(operation.get("action") or ""))
    request_json = operation.get("request_json") if isinstance(operation.get("request_json"), dict) else {}
    payload = request_json.get("payload") if isinstance(request_json.get("payload"), dict) else {}
    if not command:
        return {"status": "uncertain", "evidence": {"reason": "unknown_action"}}
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
    )
    games = list(snapshot.get("games") or [])
    teams = list(snapshot.get("teams") or [])
    podium = list(snapshot.get("podium") or [])
    evidence: dict[str, Any] = {
        "authority": "python_fastapi",
        "command": command,
        "expected_state": str(operation.get("expected_state") or ""),
        "observed_state": str(snapshot.get("state_fingerprint") or ""),
    }
    verified = False
    result: dict[str, Any] = {
        "ok": True,
        "mode": "tournament_live_recovered",
        "draw_id": str(draw_id),
        "command": command,
    }

    if command == "save_score":
        game = next((row for row in games if str(row.get("id") or "") == str(payload.get("game_id") or "")), None)
        score_plan = payload.get("score_plan") if isinstance(payload.get("score_plan"), dict) else {}
        expected_game = score_plan.get("game") if isinstance(score_plan.get("game"), dict) else None
        expected_dependencies = score_plan.get("downstream_games") if isinstance(score_plan.get("downstream_games"), list) else None
        observed_game = _score_game_projection(game) if game else None
        observed_dependencies = sorted(
            [
                _score_game_projection(row)
                for row in games
                if str(row.get("stage") or "").upper() == "PLAYOFF"
                and str(row.get("id") or "") != str(payload.get("game_id") or "")
            ],
            key=lambda row: row["id"],
        )
        verified = bool(
            expected_game
            and expected_dependencies is not None
            and observed_game == expected_game
            and observed_dependencies == expected_dependencies
        )
        evidence.update(
            {
                "game_id": payload.get("game_id"),
                "score_and_identity_match": observed_game == expected_game,
                "downstream_dependency_set_match": observed_dependencies == expected_dependencies,
                "expected_downstream_games": expected_dependencies,
                "observed_downstream_games": observed_dependencies,
            }
        )
        result["game"] = game
    elif command == "generate_round_robin":
        rr_games = [row for row in games if str(row.get("stage") or "").upper() == "ROUND_ROBIN"]
        expected_plan = payload.get("round_robin_plan") if isinstance(payload.get("round_robin_plan"), list) else []
        observed_plan = sorted(
            [_round_robin_projection(row) for row in rr_games],
            key=lambda row: (int(row["rr_round_number"] or 0), int(row["rr_slot_number"] or 0)),
        )
        verified = bool(expected_plan) and observed_plan == expected_plan
        evidence.update(
            {
                "exact_round_robin_set_match": verified,
                "expected_games": expected_plan,
                "observed_games": observed_plan,
            }
        )
        result.update({"game_count": len(rr_games), "games": rr_games})
    elif command == "generate_playoffs":
        playoff_games = [row for row in games if str(row.get("stage") or "").upper() == "PLAYOFF"]
        advance_count = _safe_int(payload.get("advance_count")) or 0
        expected_plan = payload.get("playoff_plan") if isinstance(payload.get("playoff_plan"), list) else []
        observed_plan = sorted(
            [_playoff_projection(row) for row in playoff_games],
            key=lambda row: row["playoff_game_code"],
        )
        verified = bool(expected_plan) and observed_plan == expected_plan
        evidence.update(
            {
                "advance_count": advance_count,
                "exact_playoff_set_match": verified,
                "expected_games": expected_plan,
                "observed_games": observed_plan,
            }
        )
        result.update({"advance_count": advance_count, "game_count": len(playoff_games), "games": playoff_games})
    elif command == "generate_podium":
        expected_plan = payload.get("podium_plan") if isinstance(payload.get("podium_plan"), list) else []
        observed_plan = sorted([_podium_projection(row) for row in podium], key=lambda row: int(row["placement"] or 0))
        verified = bool(expected_plan) and observed_plan == expected_plan
        evidence.update(
            {
                "exact_podium_set_match": verified,
                "expected_podium": expected_plan,
                "observed_podium": observed_plan,
            }
        )
        result["podium"] = podium
    elif command == "award_podium":
        expected = payload.get("award_plan") if isinstance(payload.get("award_plan"), list) else []
        award_rows, awards_visible, _award_warning = _award_rows_for_draw(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
        )
        observed = _active_award_projection(award_rows)
        verified = bool(awards_visible) and bool(expected) and observed == expected
        evidence.update(
            {
                "award_evidence_available": awards_visible,
                "exact_award_set_match": verified,
                "expected_awards": expected,
                "observed_awards": observed,
            }
        )
        result.update({"candidate_count": len(expected), "awarded_count": len(observed)})
        if (
            str(operation.get("error_text") or "")
            in PROVEN_NO_WRITE_PODIUM_AWARD_ERRORS
            and awards_visible
            and bool(expected)
            and not observed
        ):
            return {
                "status": "not_applied",
                "result": {},
                "evidence": {
                    **evidence,
                    "pre_mutation_podium_version_rejection": (
                        str(operation.get("error_text") or "")
                        in PROVEN_PRE_MUTATION_PODIUM_VERSION_ERRORS
                    ),
                    "atomic_award_rollback": (
                        str(operation.get("error_text") or "")
                        == ATOMIC_PODIUM_AWARD_ROLLBACK_ERROR
                    ),
                    "no_award_rows_observed": True,
                },
            }
    elif command == "publish_official_matches":
        publish_plan = payload.get("publish_plan") if isinstance(payload.get("publish_plan"), dict) else {}
        try:
            result = reconcile_admin_tournament_official_publish(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                expected_plan=publish_plan,
                guarded_operation_key=str(operation.get("operation_key") or ""),
                guarded_request_fingerprint=str(operation.get("request_fingerprint") or ""),
                client_idempotency_key=str(
                    operation.get("client_idempotency_key")
                    or request_json.get("idempotency_key")
                    or ""
                ),
            )
            verified = True
            evidence.update(
                {
                    "exact_official_publish_set_match": True,
                    "expected_tournament_game_ids": list(publish_plan.get("tournament_game_ids") or []),
                    "observed_tournament_game_ids": list(result.get("tournament_game_ids") or []),
                }
            )
        except TournamentAdminRecoveryRequiredError as exc:
            verified = False
            evidence.update(
                {
                    "exact_official_publish_set_match": False,
                    "expected_tournament_game_ids": list(publish_plan.get("tournament_game_ids") or []),
                    "reason": str(exc),
                }
            )

    if verified:
        return {"status": "completed", "result": result, "evidence": evidence}
    if str(snapshot.get("state_fingerprint") or "") == str(operation.get("expected_state") or ""):
        return {"status": "not_applied", "result": {}, "evidence": {**evidence, "unchanged_draw_state": True}}
    return {"status": "uncertain", "result": {}, "evidence": evidence}


def reconcile_admin_tournament_live_operation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    operation_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
) -> dict[str, Any]:
    require_tournament_live_write_runtime()
    if str(confirmation_text or "").strip() != TOURNAMENT_LIVE_RECONCILE_CONFIRMATION:
        raise ValueError(f"Type {TOURNAMENT_LIVE_RECONCILE_CONFIRMATION} exactly to reconcile this draw operation.")
    operation = get_tournament_admin_operation_record(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
    )
    if not operation:
        raise ValueError("Tournament Live operation not found for this club.")
    if str(operation.get("surface") or "") != TOURNAMENT_LIVE_SURFACE:
        raise ValueError("Operation is not a Tournament Live command.")
    if str(operation.get("entity_type") or "") != "tournament_event_draw" or str(operation.get("entity_id") or "") != str(draw_id):
        raise ValueError("Tournament Live operation does not belong to this draw.")
    if str(operation.get("lock_scope") or "") != f"tournament:{tournament_id}:draw:{draw_id}":
        raise ValueError("Tournament Live operation does not belong to this tournament draw.")
    request_json = operation.get("request_json") if isinstance(operation.get("request_json"), dict) else {}
    if str(request_json.get("payload", {}).get("command") if isinstance(request_json.get("payload"), dict) else "") not in COMMAND_CONFIRMATIONS:
        raise ValueError("Operation is not a recognized Tournament Live command.")
    command = str(request_json.get("payload", {}).get("command"))
    _require_live_command_permission(actor_role, command)
    return reconcile_tournament_admin_guarded_operation(
        supabase,
        club_id=str(club_id),
        surface=TOURNAMENT_LIVE_SURFACE,
        operation_key=str(operation_key),
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        actor_email=actor_email,
        actor_role=actor_role,
        source="next_tournament_live_reconcile",
        verify_outcome=lambda row: _verified_recovery_outcome(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            operation=row,
        ),
    )


def tournament_live_runtime_is_enabled() -> bool:
    """Small public hook used by status/static contract tests."""

    return tournament_admin_guarded_runtime_enabled(TOURNAMENT_LIVE_SURFACE) and bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


__all__ = [
    "COMMAND_CONFIRMATIONS",
    "TOURNAMENT_LIVE_RECONCILE_CONFIRMATION",
    "TOURNAMENT_LIVE_SURFACE",
    "TOURNAMENT_LIVE_WRITE_FLAG",
    "TournamentAdminRecoveryRequiredError",
    "build_admin_tournament_live_snapshot",
    "build_admin_tournament_live_status",
    "execute_admin_tournament_live_command",
    "reconcile_admin_tournament_live_operation",
    "require_tournament_live_write_runtime",
    "tournament_live_runtime_is_enabled",
]
