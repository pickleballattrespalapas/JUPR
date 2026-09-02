from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import SUPPORTED_TEAM_COUNTS, build_round_robin_games
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminMutationNotAppliedError,
)
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_GENERATE_GAMES = "GENERATE GAMES"
CONFIRM_REBUILD_GAMES = "REBUILD GAMES"
CONFIRM_RECONCILE_GAMES = "RECONCILE GAMES"
DEFINITELY_NOT_APPLIED_GAME_RPC_SQLSTATES = frozenset(
    {
        # unique_violation: PostgreSQL returned a statement rejection from the
        # atomic RPC, so its transaction was rolled back before commit.
        "23505",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _server_database_error_code(exc: Exception) -> str:
    """Return a structured PostgREST/Postgres error code, never parsed prose."""

    code = str(getattr(exc, "code", "") or "").strip().upper()
    if code:
        return code
    for value in getattr(exc, "args", ()):
        if isinstance(value, dict):
            code = str(value.get("code") or "").strip().upper()
            if code:
                return code
    return ""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_atomic_recovery(
    *, atomic: bool, allow_non_atomic_test_adapter: bool
) -> None:
    if atomic:
        return
    if (
        os.getenv("JUPR_ENV", "").strip().lower() == "test"
        and allow_non_atomic_test_adapter
    ):
        return
    raise PermissionError(
        "Tournament schedule recovery requires its atomic database RPC; the non-atomic adapter is unit-test-only."
    )


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _fetch_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(draw_id))
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; game generation was refused.") from exc
    return rows[0] if rows else None


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load draw teams; game generation was refused.") from exc
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify whether this draw already has games; game generation was refused.") from exc


def _is_series_game_child(game: dict[str, Any]) -> bool:
    return bool(str(game.get("series_parent_game_id") or "").strip()) or (
        str(game.get("stage") or "").strip().upper() == "SERIES_GAME"
    )


def _competition_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return scheduled matchups, never their rating-only series leaves."""

    return [dict(row) for row in games if not _is_series_game_child(row)]


def _rows_for_draw(
    supabase: Any,
    table_name: str,
    *,
    tournament_id: str,
    draw_id: str,
) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table(table_name)
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not verify {table_name}; round-robin rebuild was refused."
        ) from exc


def _round_robin_rows(
    *,
    tournament_id: str,
    draw_id: str,
    draw: dict[str, Any],
    team_ids_by_number: dict[int, str],
) -> list[dict[str, Any]]:
    now = _now_iso()
    return [
        {
            **row,
            "id": str(uuid.uuid4()),
            "draw_id": draw_id,
            "registration_day_id": _clean_text(
                draw.get("registration_day_id"), limit=120
            )
            or None,
            "event_option_id": _clean_text(
                draw.get("event_option_id"), limit=120
            )
            or None,
            "created_at": now,
            "updated_at": now,
        }
        for row in build_round_robin_games(
            tournament_id=tournament_id,
            team_ids_by_number=team_ids_by_number,
        )
    ]


def _missing_round_robin_rows(
    *,
    tournament_id: str,
    draw_id: str,
    draw: dict[str, Any],
    team_ids_by_number: dict[int, str],
    existing_games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_pairs = {
        tuple(
            sorted(
                (
                    str(row.get("team_a_id") or ""),
                    str(row.get("team_b_id") or ""),
                )
            )
        )
        for row in existing_games
    }
    max_round = max(
        (int(_safe_int(row.get("rr_round_number")) or 0) for row in existing_games),
        default=0,
    )
    generated = build_round_robin_games(
        tournament_id=tournament_id,
        team_ids_by_number=team_ids_by_number,
    )
    source_rounds = sorted(
        {
            int(row["rr_round_number"])
            for row in generated
            if tuple(sorted((str(row["team_a_id"]), str(row["team_b_id"]))))
            not in existing_pairs
        }
    )
    round_map = {
        source_round: max_round + offset
        for offset, source_round in enumerate(source_rounds, start=1)
    }
    now = _now_iso()
    slots_by_round: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for row in generated:
        pair = tuple(
            sorted((str(row["team_a_id"]), str(row["team_b_id"])))
        )
        if pair in existing_pairs:
            continue
        target_round = round_map[int(row["rr_round_number"])]
        slots_by_round[target_round] = slots_by_round.get(target_round, 0) + 1
        rows.append(
            {
                **row,
                "id": str(uuid.uuid4()),
                "draw_id": draw_id,
                "registration_day_id": _clean_text(
                    draw.get("registration_day_id"), limit=120
                )
                or None,
                "event_option_id": _clean_text(
                    draw.get("event_option_id"), limit=120
                )
                or None,
                "rr_round_number": target_round,
                "rr_slot_number": slots_by_round[target_round],
                "created_at": now,
                "updated_at": now,
            }
        )
    return rows


def _validate_reconcilable_round_robin_games(
    *,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
) -> int:
    if not games:
        raise ValueError(
            "This draw has no partial schedule to reconcile. Use Generate Games."
        )
    team_ids = {str(row.get("id") or "") for row in teams if row.get("id")}
    seen_pairs: set[tuple[str, str]] = set()
    finalized_count = 0
    for game in games:
        if str(game.get("stage") or "").upper() != "ROUND_ROBIN":
            raise ValueError(
                "Round-robin reconciliation is unavailable after playoff games exist."
            )
        team_a = str(game.get("team_a_id") or "")
        team_b = str(game.get("team_b_id") or "")
        if not team_a or not team_b or team_a == team_b or team_a not in team_ids or team_b not in team_ids:
            raise ValueError(
                "The partial schedule contains a game outside the current roster; reconcile the roster before rebuilding pairings."
            )
        pair = tuple(sorted((team_a, team_b)))
        if pair in seen_pairs:
            raise ValueError(
                "The partial schedule contains a duplicate team pairing; resolve it explicitly before reconciliation."
            )
        seen_pairs.add(pair)
        score_a = _safe_int(game.get("score_a"))
        score_b = _safe_int(game.get("score_b"))
        has_result = any(
            game.get(field) not in (None, "")
            for field in (
                "score_a",
                "score_b",
                "winner_team_id",
                "loser_team_id",
                "finalized_at",
            )
        )
        if not has_result:
            continue
        expected_winner = team_a if score_a is not None and score_b is not None and score_a > score_b else team_b
        expected_loser = team_b if expected_winner == team_a else team_a
        if (
            score_a is None
            or score_b is None
            or score_a < 0
            or score_b < 0
            or score_a == score_b
            or not game.get("finalized_at")
            or str(game.get("winner_team_id") or "") != expected_winner
            or str(game.get("loser_team_id") or "") != expected_loser
        ):
            raise ValueError(
                "The partial schedule contains a partially scored or inconsistent game; correct it before reconciliation."
            )
        finalized_count += 1
    return finalized_count


def _require_unstarted_round_robin_rebuild(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    games: list[dict[str, Any]],
) -> None:
    if not games:
        raise ValueError(
            "This draw has no games to rebuild. Use Generate Games for a new schedule."
        )
    if any(str(row.get("stage") or "").upper() != "ROUND_ROBIN" for row in games):
        raise ValueError(
            "Round-robin rebuild is unavailable after playoff games exist."
        )
    if any(
        row.get(field) not in (None, "")
        for row in games
        for field in (
            "score_a",
            "score_b",
            "winner_team_id",
            "loser_team_id",
            "finalized_at",
        )
    ):
        raise ValueError(
            "Round-robin rebuild is available only before any score or result evidence exists."
        )
    if _rows_for_draw(
        supabase,
        "tournament_podium",
        tournament_id=tournament_id,
        draw_id=draw_id,
    ):
        raise ValueError(
            "Round-robin rebuild is unavailable after podium evidence exists."
        )

    game_ids = {str(row.get("id") or "") for row in games if row.get("id")}
    try:
        published = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
        day_queue = _safe_rows(
            supabase.table("tournament_day_live_queue")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
        day_draws = _safe_rows(
            supabase.table("tournament_day_live_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
        awards = _safe_rows(
            supabase.table("player_badges")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("context_type", "tournament")
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not verify published, award, or day-live dependencies; round-robin rebuild was refused."
        ) from exc
    if any(str(row.get("tournament_game_id") or "") in game_ids for row in published):
        raise ValueError(
            "Round-robin rebuild is unavailable after an official Match Log link exists."
        )
    if day_queue or any(
        str(row.get("state") or "").upper() in {"ACTIVE", "PAUSED"}
        for row in day_draws
    ):
        raise ValueError(
            "Round-robin rebuild is unavailable while the draw has day-live queue evidence. Close or reconcile the day run first."
        )
    context_prefix = f"{tournament_id}:draw:{draw_id}:podium:"
    if any(
        str(row.get("context_id") or "").startswith(context_prefix)
        and row.get("revoked_at") in (None, "")
        for row in awards
    ):
        raise ValueError(
            "Round-robin rebuild is unavailable after podium award evidence exists."
        )


def _require_round_robin_reconcile_dependencies_clear(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    games: list[dict[str, Any]],
) -> None:
    if _rows_for_draw(
        supabase,
        "tournament_podium",
        tournament_id=tournament_id,
        draw_id=draw_id,
    ):
        raise ValueError(
            "Round-robin reconciliation is unavailable after podium evidence exists."
        )
    game_ids = {str(row.get("id") or "") for row in games if row.get("id")}
    try:
        published = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
        day_queue = _safe_rows(
            supabase.table("tournament_day_live_queue")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
        day_draws = _safe_rows(
            supabase.table("tournament_day_live_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
        awards = _safe_rows(
            supabase.table("player_badges")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("context_type", "tournament")
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not verify published, award, or day-live dependencies; round-robin reconciliation was refused."
        ) from exc
    if any(str(row.get("tournament_game_id") or "") in game_ids for row in published):
        raise ValueError(
            "Round-robin reconciliation is unavailable after an official Match Log link exists."
        )
    if day_queue or any(
        str(row.get("state") or "").upper() in {"ACTIVE", "PAUSED"}
        for row in day_draws
    ):
        raise ValueError(
            "Round-robin reconciliation is unavailable while the draw has day-live queue evidence. Close or reconcile the day run first."
        )
    context_prefix = f"{tournament_id}:draw:{draw_id}:podium:"
    if any(
        str(row.get("context_id") or "").startswith(context_prefix)
        and row.get("revoked_at") in (None, "")
        for row in awards
    ):
        raise ValueError(
            "Round-robin reconciliation is unavailable after podium award evidence exists."
        )


def _game_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "draw_id": _clean_text(row.get("draw_id"), limit=120) or None,
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120) or None,
        "event_option_id": _clean_text(row.get("event_option_id"), limit=120) or None,
        "stage": _clean_text(row.get("stage"), limit=80),
        "scoring_format": _clean_text(row.get("scoring_format"), limit=80).upper()
        or None,
        "rr_round_number": _safe_int(row.get("rr_round_number")),
        "rr_slot_number": _safe_int(row.get("rr_slot_number")),
        "team_a_id": _clean_text(row.get("team_a_id"), limit=120) or None,
        "team_b_id": _clean_text(row.get("team_b_id"), limit=120) or None,
        "score_a": _safe_int(row.get("score_a")),
        "score_b": _safe_int(row.get("score_b")),
        "winner_team_id": _clean_text(row.get("winner_team_id"), limit=120) or None,
        "loser_team_id": _clean_text(row.get("loser_team_id"), limit=120) or None,
        "finalized_at": row.get("finalized_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _require_reviewed_draw_version(
    draw: dict[str, Any],
    *,
    expected_draw_updated_at: str | None,
    atomic: bool,
) -> str:
    reviewed = str(expected_draw_updated_at or "").strip()
    if atomic and not reviewed:
        raise StaleTournamentAdminStateError(
            "A reviewed draw version is required for staging game generation. Reload the Ops snapshot."
        )
    if reviewed and str(draw.get("updated_at") or "") != reviewed:
        raise StaleTournamentAdminStateError(
            "This tournament draw changed after it was reviewed. Reload the Ops snapshot before generating games."
        )
    return reviewed


def _canonical_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except Exception:
        return text


def _require_reviewed_row_versions(
    current_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None,
    *,
    label: str,
    atomic: bool,
) -> list[dict[str, str]]:
    reviewed = [
        {"id": str(row.get("id") or "").strip(), "updated_at": str(row.get("updated_at") or "").strip()}
        for row in (expected_rows or [])
    ]
    if atomic and not reviewed:
        raise StaleTournamentAdminStateError(
            f"A reviewed {label} snapshot is required for this staging mutation. Reload the Ops snapshot."
        )
    expected_map = {
        row["id"]: _canonical_timestamp(row["updated_at"])
        for row in reviewed
        if row["id"] and row["updated_at"]
    }
    if len(expected_map) != len(reviewed):
        raise StaleTournamentAdminStateError(
            f"The reviewed {label} snapshot is incomplete or duplicated. Reload the Ops snapshot."
        )
    current_map = {
        str(row.get("id") or "").strip(): _canonical_timestamp(row.get("updated_at"))
        for row in current_rows
        if str(row.get("id") or "").strip() and str(row.get("updated_at") or "").strip()
    }
    if reviewed and (len(current_map) != len(current_rows) or current_map != expected_map):
        raise StaleTournamentAdminStateError(
            f"The tournament {label} changed after review. Reload the Ops snapshot before continuing."
        )
    return sorted(reviewed, key=lambda row: row["id"])


def _insert_tournament_draw_games_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_source_game_versions: list[dict[str, str]],
    mode: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_insert_tournament_draw_games_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_mode": str(mode),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_source_games": list(expected_source_game_versions),
                "p_games": list(rows),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if any(
            marker in detail
            for marker in (
                "JUPR_TOURNAMENT_DRAW_STALE",
                "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE",
            )
        ):
            raise StaleTournamentAdminStateError(
                "The draw, team set, or source game set changed while games were being generated. Reload the Ops snapshot."
            ) from exc
        database_error_code = _server_database_error_code(exc)
        if database_error_code in DEFINITELY_NOT_APPLIED_GAME_RPC_SQLSTATES:
            raise TournamentAdminMutationNotAppliedError(
                "The database rejected the atomic game schedule; no games were created. "
                f"Database code: {database_error_code}. Reload the draw before another attempt."
            ) from exc
        raise RuntimeError("Atomic tournament game generation failed; no game set was committed.") from exc
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        saved = data.get("games")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        saved = data[0].get("games")
    else:
        saved = None
    if not isinstance(saved, list):
        raise RuntimeError("Atomic tournament game generation returned no saved game set.")
    return [dict(row) for row in saved if isinstance(row, dict)]


def _rebuild_tournament_round_robin_games_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_source_game_versions: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_rebuild_tournament_round_robin_games_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_source_games": list(expected_source_game_versions),
                "p_games": list(rows),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if any(
            marker in detail
            for marker in (
                "JUPR_TOURNAMENT_DRAW_STALE",
                "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE",
            )
        ):
            raise StaleTournamentAdminStateError(
                "The draw, team set, or partial game set changed while games were being rebuilt. Reload the Ops snapshot."
            ) from exc
        if "JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED" in detail:
            raise ValueError(
                "Round-robin rebuild is blocked because score, playoff, podium, award, official-match, or day-live evidence exists."
            ) from exc
        raise RuntimeError(
            "Atomic round-robin rebuild failed; the previous game set was retained."
        ) from exc
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        saved = data.get("games")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        saved = data[0].get("games")
    else:
        saved = None
    if not isinstance(saved, list):
        raise RuntimeError(
            "Atomic round-robin rebuild returned no saved game set."
        )
    return [dict(row) for row in saved if isinstance(row, dict)]


def _reconcile_tournament_round_robin_games_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_source_game_versions: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_reconcile_tournament_round_robin_games_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_source_games": list(expected_source_game_versions),
                "p_games": list(rows),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if any(
            marker in detail
            for marker in (
                "JUPR_TOURNAMENT_DRAW_STALE",
                "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE",
            )
        ):
            raise StaleTournamentAdminStateError(
                "The draw, current roster, or partial game set changed while missing games were reconciled. Reload the Ops snapshot."
            ) from exc
        if "JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED" in detail:
            raise ValueError(
                "Round-robin reconciliation is blocked because the current games are inconsistent or official, podium, award, or day-live evidence exists."
            ) from exc
        raise RuntimeError(
            "Atomic round-robin reconciliation failed; existing games were retained and no partial insert should be retried blindly."
        ) from exc
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        saved = data.get("games")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        saved = data[0].get("games")
    else:
        saved = None
    if not isinstance(saved, list):
        raise RuntimeError(
            "Atomic round-robin reconciliation returned no authoritative game set."
        )
    return [dict(row) for row in saved if isinstance(row, dict)]


def generate_admin_tournament_round_robin_games(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_generate_round_robin",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_GENERATE_GAMES:
        raise ValueError(f"Type {CONFIRM_GENERATE_GAMES} to generate round-robin games.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )
    if _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id):
        raise ValueError("This draw already has games. Delete/recreate the draw or clear games before regenerating.")

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    team_count = len(teams)
    if team_count not in SUPPORTED_TEAM_COUNTS:
        raise ValueError(f"Round-robin generation supports {SUPPORTED_TEAM_COUNTS}; this draw has {team_count} teams.")
    team_ids_by_number: dict[int, str] = {}
    for team in teams:
        team_number = _safe_int(team.get("team_number"))
        team_id = _clean_text(team.get("id"), limit=120)
        if team_number is None or not team_id:
            raise ValueError("Every team must have a team number and id before generating games.")
        team_ids_by_number[int(team_number)] = team_id
    if sorted(team_ids_by_number) != list(range(1, team_count + 1)):
        raise ValueError("Team numbers must be contiguous from 1 through the draw size before generating games.")

    game_rows = _round_robin_rows(
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        draw=draw,
        team_ids_by_number=team_ids_by_number,
    )
    if dry_run:
        games = [_game_payload(row) for row in game_rows]
        return {
            "ok": True,
            "mode": "tournament_round_robin_generate_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "game_count": len(games),
            "games": games,
            "warnings": [],
        }
    inserted = (
        _insert_tournament_draw_games_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=[],
            mode="ROUND_ROBIN",
            rows=game_rows,
        )
        if atomic
        else (_safe_rows(supabase.table("tournament_games").insert(game_rows).execute()) if game_rows else [])
    )
    games = [_game_payload(row) for row in (inserted or game_rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="generate_tournament_round_robin_games_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "teams": len(teams), "games": 0},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "game_count": len(games),
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_round_robin_generate", "draw_id": clean_draw_id, "game_count": len(games), "games": games, "warnings": warnings}


def rebuild_admin_tournament_round_robin_games(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_rebuild_round_robin",
    dry_run: bool = False,
    atomic: bool = False,
    allow_non_atomic_test_adapter: bool = False,
) -> dict[str, Any]:
    """Replace an unstarted partial RR schedule with one complete schedule.

    This is deliberately not a generic delete-games escape hatch.  Any score,
    playoff, podium, award, official Match Log, or day-live evidence makes the
    operation fail closed.  The explicit confirmation and CAS snapshots keep
    the recovery path useful without silently discarding tournament work.
    """

    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    _require_atomic_recovery(
        atomic=atomic,
        allow_non_atomic_test_adapter=allow_non_atomic_test_adapter,
    )
    if str(confirmation_text or "").strip().upper() != CONFIRM_REBUILD_GAMES:
        raise ValueError(
            f"Type {CONFIRM_REBUILD_GAMES} to clear and rebuild the unstarted round-robin schedule."
        )

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )

    teams = _teams_for_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    team_count = len(teams)
    if team_count not in SUPPORTED_TEAM_COUNTS:
        raise ValueError(
            f"Round-robin generation supports {SUPPORTED_TEAM_COUNTS[0]}-{SUPPORTED_TEAM_COUNTS[-1]} teams; this draw has {team_count} teams."
        )
    team_ids_by_number: dict[int, str] = {}
    for team in teams:
        team_number = _safe_int(team.get("team_number"))
        team_id = _clean_text(team.get("id"), limit=120)
        if team_number is None or not team_id:
            raise ValueError(
                "Every team must have a team number and id before rebuilding games."
            )
        team_ids_by_number[int(team_number)] = team_id
    if sorted(team_ids_by_number) != list(range(1, team_count + 1)):
        raise ValueError(
            "Team numbers must be contiguous from 1 through the draw size before rebuilding games."
        )

    existing_games = _games_for_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    _require_unstarted_round_robin_rebuild(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        games=existing_games,
    )
    reviewed_game_versions = _require_reviewed_row_versions(
        existing_games,
        [
            {
                "id": str(row.get("id") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            }
            for row in existing_games
        ],
        label="partial game set",
        atomic=atomic,
    )
    game_rows = _round_robin_rows(
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        draw=draw,
        team_ids_by_number=team_ids_by_number,
    )
    preview_games = [_game_payload(row) for row in game_rows]
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_round_robin_rebuild_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "replaced_game_count": len(existing_games),
            "game_count": len(preview_games),
            "games": preview_games,
            "warnings": [],
        }

    if atomic:
        saved_rows = _rebuild_tournament_round_robin_games_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_game_versions,
            rows=game_rows,
        )
    else:
        # Local/test fallback only. Deployed staging always uses the atomic RPC.
        supabase.table("tournament_games").delete().eq(
            "tournament_id", clean_tournament_id
        ).eq("draw_id", clean_draw_id).execute()
        saved_rows = _safe_rows(
            supabase.table("tournament_games").insert(game_rows).execute()
        )
    games = [_game_payload(row) for row in (saved_rows or game_rows)]
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="rebuild_tournament_round_robin_games_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={
            "draw": _draw_payload(draw),
            "team_count": len(teams),
            "game_count": len(existing_games),
        },
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "game_count": len(games),
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_round_robin_rebuild",
        "draw_id": clean_draw_id,
        "replaced_game_count": len(existing_games),
        "game_count": len(games),
        "games": games,
        "warnings": warnings,
    }


def reconcile_admin_tournament_round_robin_games(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_reconcile_round_robin",
    dry_run: bool = False,
    atomic: bool = False,
    allow_non_atomic_test_adapter: bool = False,
) -> dict[str, Any]:
    """Preserve valid existing games and append every missing roster pairing."""

    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    _require_atomic_recovery(
        atomic=atomic,
        allow_non_atomic_test_adapter=allow_non_atomic_test_adapter,
    )
    if str(confirmation_text or "").strip().upper() != CONFIRM_RECONCILE_GAMES:
        raise ValueError(
            f"Type {CONFIRM_RECONCILE_GAMES} to preserve existing results and add missing round-robin games."
        )
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )
    teams = _teams_for_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    team_count = len(teams)
    if team_count not in SUPPORTED_TEAM_COUNTS:
        raise ValueError(
            f"Round-robin generation supports {SUPPORTED_TEAM_COUNTS[0]}-{SUPPORTED_TEAM_COUNTS[-1]} teams; this draw has {team_count} teams."
        )
    team_ids_by_number: dict[int, str] = {}
    for team in teams:
        team_number = _safe_int(team.get("team_number"))
        team_id = _clean_text(team.get("id"), limit=120)
        if team_number is None or not team_id:
            raise ValueError(
                "Every team must have a team number and id before reconciling games."
            )
        team_ids_by_number[int(team_number)] = team_id
    if sorted(team_ids_by_number) != list(range(1, team_count + 1)):
        raise ValueError(
            "Team numbers must be contiguous from 1 through the draw size before reconciling games."
        )
    all_existing_games = _games_for_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    )
    existing_games = _competition_games(all_existing_games)
    finalized_count = _validate_reconcilable_round_robin_games(
        teams=teams,
        games=existing_games,
    )
    _require_round_robin_reconcile_dependencies_clear(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        games=all_existing_games,
    )
    reviewed_game_versions = _require_reviewed_row_versions(
        all_existing_games,
        [
            {
                "id": str(row.get("id") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            }
            for row in all_existing_games
        ],
        label="partial game set",
        atomic=atomic,
    )
    missing_rows = _missing_round_robin_rows(
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        draw=draw,
        team_ids_by_number=team_ids_by_number,
        existing_games=existing_games,
    )
    if not missing_rows:
        raise ValueError(
            "This draw already contains exactly one game for every current-roster pairing."
        )
    expected_total = team_count * (team_count - 1) // 2
    if len(existing_games) + len(missing_rows) != expected_total:
        raise ValueError(
            "The partial schedule cannot be reconciled into one exact round robin without changing existing games."
        )
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_round_robin_reconcile_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "preserved_game_count": len(existing_games),
            "preserved_finalized_game_count": finalized_count,
            "inserted_game_count": len(missing_rows),
            "game_count": expected_total,
            "games": [_game_payload(row) for row in missing_rows],
            "warnings": [],
        }
    if atomic:
        authoritative_rows = _reconcile_tournament_round_robin_games_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_game_versions,
            rows=missing_rows,
        )
    else:
        inserted = _safe_rows(
            supabase.table("tournament_games").insert(missing_rows).execute()
        )
        authoritative_rows = [*all_existing_games, *(inserted or missing_rows)]
    games = [
        _game_payload(row)
        for row in _competition_games(authoritative_rows)
    ]
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="reconcile_tournament_round_robin_games_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={
            "draw": _draw_payload(draw),
            "team_count": len(teams),
            "game_count": len(existing_games),
            "finalized_game_count": finalized_count,
        },
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "preserved_game_count": len(existing_games),
            "inserted_game_count": len(missing_rows),
            "game_count": len(games),
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_round_robin_reconcile",
        "draw_id": clean_draw_id,
        "preserved_game_count": len(existing_games),
        "preserved_finalized_game_count": finalized_count,
        "inserted_game_count": len(missing_rows),
        "game_count": len(games),
        "games": games,
        "warnings": warnings,
    }
