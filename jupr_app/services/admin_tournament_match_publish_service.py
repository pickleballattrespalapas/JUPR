from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.matches.side_effects import queue_player_updates, run_badge_side_effects
from jupr_app.domain.matches import normalize_rating_scope
from jupr_app.domain.player_activity import coerce_utc_datetime
from jupr_app.domain.singles_match_processing import process_singles_matches
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.services.admin_player_updates_service import auto_send_player_updates_for_match_payloads
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
)
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    is_admin_tournament_admin_enabled,
)

CONFIRM_PUBLISH_MATCHES = "PUBLISH MATCHES"
MAX_PLAYOFF_WINNER_BONUS_ELO = 40.0
BONUS_PLAYOFF_ROUNDS = {
    "SF": "semifinal",
    "SEMIFINAL": "semifinal",
    "SEMIFINALS": "semifinal",
    "BRONZE": "bronze",
    "BRONZE MEDAL MATCH": "bronze",
    "FINAL": "gold",
    "GOLD": "gold",
    "GOLD MEDAL MATCH": "gold",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
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


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
        raise RuntimeError("Could not verify the tournament draw before official publish; no matches were published.") from exc
    return rows[0] if rows else None


def _fetch_event_option(supabase: Any, *, tournament_id: str, event_option_id: str | None) -> dict[str, Any] | None:
    clean_event_option_id = _clean_text(event_option_id, limit=120)
    if not clean_event_option_id:
        return None
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_options")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", clean_event_option_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament event option before official publish; no matches were published.") from exc
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
        raise RuntimeError("Could not verify tournament teams before official publish; no matches were published.") from exc
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify tournament games before official publish; no matches were published.") from exc
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("stage") or ""),
            int(_safe_int(row.get("rr_round_number")) or 0),
            int(_safe_int(row.get("rr_slot_number")) or 0),
            str(row.get("playoff_game_code") or ""),
            str(row.get("id") or ""),
        ),
    )


def _existing_published_game_ids(supabase: Any, *, club_id: str, tournament_id: str, game_ids: list[str]) -> set[str]:
    if not game_ids:
        return set()
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id,tournament_game_id")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .in_("tournament_game_id", game_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify whether tournament games were already published; official publish was refused.") from exc
    return {str(row.get("tournament_game_id")) for row in rows if row.get("tournament_game_id")}


def _table_frame(supabase: Any, table_name: str, *, club_id: str | None = None) -> pd.DataFrame:
    try:
        query = supabase.table(table_name).select("*")
        if club_id and table_name in {"players", "league_ratings", "leagues_metadata"}:
            query = query.eq("club_id", str(club_id))
        rows = _safe_rows(query.execute())
    except Exception as exc:
        raise RuntimeError(f"Could not load required {table_name} state before official publish; no matches were published.") from exc
    return pd.DataFrame(rows)


def _safe_rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    return {}


def _apply_official_rating_plan_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    guarded_operation_key: str,
    guarded_request_fingerprint: str,
    publish_plan_fingerprint: str,
    write_plan: dict[str, Any],
) -> dict[str, Any]:
    """Apply the Python-calculated rating core in one fail-closed transaction."""

    try:
        response = supabase.rpc(
            "admin_apply_tournament_official_rating_plan_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_operation_key": str(guarded_operation_key),
                "p_request_fingerprint": str(guarded_request_fingerprint),
                "p_publish_plan_fingerprint": str(publish_plan_fingerprint),
                "p_publish_plan": dict(write_plan.get("publish_plan") or {}),
                "p_match_rows": list(write_plan.get("match_rows") or []),
                "p_player_updates": list(write_plan.get("player_updates") or []),
                "p_league_rating_updates": list(write_plan.get("league_rating_updates") or []),
                "p_league_metadata_expectations": list(
                    write_plan.get("league_metadata_expectations") or []
                ),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        explicit_rejections = (
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAN_INVALID",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_OPERATION_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_TOURNAMENT_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_DRAW_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_TEAM_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_GAME_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_EVENT_OPTION_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_PLAN_INVALID",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_WRITE_INCOMPLETE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_PLAN_INVALID",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_METADATA_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_WRITE_INCOMPLETE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_EXISTS",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_PAYLOAD_STALE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_INSERT_INCOMPLETE",
        )
        if any(marker in detail for marker in explicit_rejections):
            raise StaleTournamentAdminStateError(
                "Official publish dependencies changed before the atomic rating transaction. No plan was applied."
            ) from exc
        raise RuntimeError(
            "Official publish atomic rating response is ambiguous. Keep the guarded operation recovery-locked; never republish."
        ) from exc
    result = _safe_rpc_payload(response)
    if not result or not bool(result.get("ok")):
        raise RuntimeError("Official publish atomic rating RPC returned no completion evidence.")
    return result


def _division_label(event_option: dict[str, Any] | None, draw: dict[str, Any]) -> str:
    event_option = event_option or {}
    family = _clean_text(event_option.get("event_family_label") or event_option.get("label"), limit=120)
    division = _clean_text(event_option.get("division_name") or event_option.get("label"), limit=120)
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or _clean_text(draw.get("name"), limit=160) or "Tournament Draw"


def _published_date(tournament: dict[str, Any], draw: dict[str, Any], game: dict[str, Any]) -> str:
    for key_source in (
        game.get("finalized_at"),
        tournament.get("start_date"),
        tournament.get("end_date"),
        draw.get("created_at"),
        game.get("created_at"),
    ):
        value = _clean_text(key_source, limit=80)
        if value:
            return value
    return _now_iso()


def _validate_scored_game(game: dict[str, Any], *, game_index: int) -> tuple[int, int]:
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    if score_a is None or score_b is None:
        raise ValueError(f"Game {game_index} is missing a score.")
    if score_a == score_b:
        raise ValueError(f"Game {game_index} has a tied score; official matches do not support ties.")
    if not _clean_text(game.get("winner_team_id"), limit=120):
        raise ValueError(f"Game {game_index} is not finalized with a winner.")
    if not _clean_text(game.get("team_a_id"), limit=120) or not _clean_text(game.get("team_b_id"), limit=120):
        raise ValueError(f"Game {game_index} is missing team assignments.")
    return int(score_a), int(score_b)


def _validate_bonus_elo(value: Any) -> float:
    bonus = _safe_float(value)
    if bonus is None:
        return 0.0
    if bonus < 0:
        raise ValueError("Playoff winner bonus cannot be negative.")
    if bonus > MAX_PLAYOFF_WINNER_BONUS_ELO:
        raise ValueError(f"Playoff winner bonus is capped at {MAX_PLAYOFF_WINNER_BONUS_ELO:g} Elo points per winning player.")
    return float(bonus)


def _bonus_label_for_game(game: dict[str, Any]) -> str | None:
    if _clean_text(game.get("stage"), limit=80).upper() != "PLAYOFF":
        return None
    round_key = _clean_text(game.get("playoff_round"), limit=80).upper()
    return BONUS_PLAYOFF_ROUNDS.get(round_key)


def _team_shape(team: dict[str, Any]) -> str:
    p1 = _safe_int(team.get("player1_id"))
    p2 = _safe_int(team.get("player2_id"))
    if p1 is not None and p2 is not None:
        return "doubles"
    if p1 is not None and p2 is None:
        return "singles"
    return "invalid"


def _build_official_match_payloads(
    *,
    tournament: dict[str, Any],
    draw: dict[str, Any],
    event_option: dict[str, Any] | None,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    playoff_winner_bonus_elo: float = 0.0,
) -> list[dict[str, Any]]:
    teams_by_id = {str(row.get("id")): row for row in teams if row.get("id")}
    tournament_name = _clean_text(tournament.get("name"), limit=160) or "Tournament"
    division_label = _division_label(event_option, draw)
    league_name = f"Tournament · {tournament_name} · {division_label}"
    week_tag = _clean_text(draw.get("name"), limit=120) or division_label
    bonus_elo = _validate_bonus_elo(playoff_winner_bonus_elo)

    payloads: list[dict[str, Any]] = []
    detected_format: str | None = None
    for index, game in enumerate(games, start=1):
        score_a, score_b = _validate_scored_game(game, game_index=index)
        team_a = teams_by_id.get(str(game.get("team_a_id") or ""))
        team_b = teams_by_id.get(str(game.get("team_b_id") or ""))
        if not team_a or not team_b:
            raise ValueError(f"Game {index} references a team that is not in this draw.")
        shape_a, shape_b = _team_shape(team_a), _team_shape(team_b)
        if shape_a != shape_b or shape_a == "invalid":
            raise ValueError("Official match publishing requires each game to use either two singles teams or two doubles teams with linked JUPR players.")
        if detected_format and detected_format != shape_a:
            raise ValueError("A draw cannot mix singles and doubles games when publishing official rating matches.")
        detected_format = shape_a
        a1, a2 = _safe_int(team_a.get("player1_id")), _safe_int(team_a.get("player2_id"))
        b1, b2 = _safe_int(team_b.get("player1_id")), _safe_int(team_b.get("player2_id"))
        payload = {
            "date": _published_date(tournament, draw, game),
            "league": league_name,
            "week_tag": week_tag,
            "match_type": "Tournament Singles" if shape_a == "singles" else "Tournament",
            "t1_p1": a1,
            "t2_p1": b1,
            "score_t1": score_a,
            "score_t2": score_b,
            "context_type": "tournament_game",
            "context_id": _clean_text(game.get("id"), limit=120),
            "tournament_id": _clean_text(tournament.get("id"), limit=120),
            "tournament_game_id": _clean_text(game.get("id"), limit=120),
            "rating_scope": "",
            "match_format": shape_a,
        }
        if shape_a == "doubles":
            payload["t1_p2"] = a2
            payload["t2_p2"] = b2
        bonus_label = _bonus_label_for_game(game)
        if bonus_elo > 0 and bonus_label:
            payload["winner_bonus_elo"] = bonus_elo
            payload["winner_bonus_reason"] = f"tournament_{bonus_label}_winner_bonus"
            payload["rating_bonus_elo"] = bonus_elo
            payload["rating_bonus_reason"] = f"tournament_{bonus_label}_winner_bonus"
        payloads.append(payload)
    return payloads


def _reviewed_versions(rows: list[dict[str, Any]], *, label: str) -> list[dict[str, str]]:
    versions = sorted(
        [
            {"id": str(row.get("id") or ""), "updated_at": str(row.get("updated_at") or "")}
            for row in rows
        ],
        key=lambda row: row["id"],
    )
    if not versions or any(not row["id"] or not row["updated_at"] for row in versions):
        raise StaleTournamentAdminStateError(
            f"Official publish requires a complete reviewed {label} version set. Reload Tournament Ops."
        )
    if len({row["id"] for row in versions}) != len(versions):
        raise StaleTournamentAdminStateError(
            f"Official publish found duplicate reviewed {label} identities. Reload Tournament Ops."
        )
    return versions


def _official_match_projection(row: dict[str, Any], *, club_id: str) -> dict[str, Any]:
    """Project fields that must survive unchanged into the official match row."""

    bonus_elo = _safe_float(row.get("rating_bonus_elo", row.get("winner_bonus_elo"))) or 0.0
    parsed_date = coerce_utc_datetime(row.get("date"))
    return {
        "club_id": str(row.get("club_id") or club_id),
        "date": parsed_date.isoformat() if parsed_date else str(row.get("date") or ""),
        "league": str(row.get("league") or ""),
        "week_tag": str(row.get("week_tag") or ""),
        "match_type": str(row.get("match_type") or ""),
        "match_format": "singles" if str(row.get("match_format") or "").lower() == "singles" else "doubles",
        "t1_p1": _safe_int(row.get("t1_p1")),
        "t1_p2": _safe_int(row.get("t1_p2")),
        "t2_p1": _safe_int(row.get("t2_p1")),
        "t2_p2": _safe_int(row.get("t2_p2")),
        "score_t1": _safe_int(row.get("score_t1")),
        "score_t2": _safe_int(row.get("score_t2")),
        "context_type": str(row.get("context_type") or ""),
        "context_id": str(row.get("context_id") or ""),
        "tournament_id": str(row.get("tournament_id") or ""),
        "tournament_game_id": str(row.get("tournament_game_id") or ""),
        "rating_scope": normalize_rating_scope(row),
        "rating_bonus_elo": float(bonus_elo),
        "rating_bonus_reason": str(
            row.get("rating_bonus_reason") or row.get("winner_bonus_reason") or ""
        ),
    }


def _official_publish_plan_from_state(
    *,
    club_id: str,
    tournament: dict[str, Any],
    event_option: dict[str, Any] | None,
    draw: dict[str, Any],
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    match_payloads: list[dict[str, Any]],
    bonus_elo: float,
) -> dict[str, Any]:
    game_ids = [str(row.get("tournament_game_id") or "") for row in match_payloads]
    if len(game_ids) != len(set(game_ids)) or any(not value for value in game_ids):
        raise ValueError("Official publish requires one unique tournament game id per reviewed match.")
    projections = [_official_match_projection(row, club_id=str(club_id)) for row in match_payloads]
    fingerprints = sorted(
        [
            {
                "tournament_game_id": str(row.get("tournament_game_id") or ""),
                "payload_fingerprint": stable_tournament_admin_fingerprint(row),
            }
            for row in projections
        ],
        key=lambda row: row["tournament_game_id"],
    )
    singles_count = sum(1 for row in projections if row["match_format"] == "singles")
    bonus_game_ids = sorted(
        str(row.get("tournament_game_id") or "")
        for row in projections
        if float(row.get("rating_bonus_elo") or 0.0) > 0
    )
    return {
        "tournament_metadata": {
            field: tournament.get(field)
            for field in ("id", "club_id", "name", "status", "start_date", "end_date", "updated_at")
        },
        "event_option_metadata": (
            {
                field: event_option.get(field)
                for field in ("id", "tournament_id", "label", "event_family_label", "division_name")
            }
            if event_option
            else None
        ),
        "draw_id": str(draw.get("id") or ""),
        "draw_updated_at": str(draw.get("updated_at") or ""),
        "team_versions": _reviewed_versions(teams, label="team"),
        "game_versions": _reviewed_versions(games, label="game"),
        "tournament_game_ids": sorted(game_ids),
        "match_payload_projections": sorted(
            projections,
            key=lambda row: str(row.get("tournament_game_id") or ""),
        ),
        "match_payload_fingerprints": fingerprints,
        "match_count": len(game_ids),
        "singles_match_count": singles_count,
        "doubles_match_count": len(game_ids) - singles_count,
        "playoff_winner_bonus_elo": float(bonus_elo),
        "bonus_tournament_game_ids": bonus_game_ids,
    }


def build_admin_tournament_official_publish_plan(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    playoff_winner_bonus_elo: Any = 0.0,
) -> dict[str, Any]:
    """Build immutable evidence for a guarded official-publish request.

    Unlike the mutation preflight this intentionally does not reject existing
    matches. That makes the same deterministic request reproducible after a
    response loss, while the guarded runner decides whether read-back evidence
    is exact enough to complete the operation without invoking the processor.
    """

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    bonus_elo = _validate_bonus_elo(playoff_winner_bonus_elo)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not games:
        raise ValueError("This draw has no tournament games to publish.")
    event_option = _fetch_event_option(
        supabase,
        tournament_id=clean_tournament_id,
        event_option_id=_clean_text(draw.get("event_option_id"), limit=120),
    )
    match_payloads = _build_official_match_payloads(
        tournament=tournament,
        draw=draw,
        event_option=event_option,
        teams=teams,
        games=games,
        playoff_winner_bonus_elo=bonus_elo,
    )
    return _official_publish_plan_from_state(
        club_id=str(club_id),
        tournament=tournament,
        event_option=event_option,
        draw=draw,
        teams=teams,
        games=games,
        match_payloads=match_payloads,
        bonus_elo=bonus_elo,
    )


def reconcile_admin_tournament_official_publish(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_plan: dict[str, Any],
    guarded_operation_key: str,
    guarded_request_fingerprint: str,
    client_idempotency_key: str = "",
) -> dict[str, Any]:
    """Reconstruct a lost publish result from an exact authoritative set.

    Zero, partial, duplicate, changed-draw, and cross-club evidence all remain
    recovery-required. This function is read-only and never calls either match
    processor, so replay cannot create a second official match or rating write.
    """

    clean_operation_key = str(guarded_operation_key or "").strip()
    clean_request_fingerprint = str(guarded_request_fingerprint or "").strip()
    clean_idempotency_key = str(client_idempotency_key or "").strip()
    if not clean_operation_key or not clean_request_fingerprint:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery is missing its exact guarded operation identity; do not repeat the mutation."
        )
    expected_ids = [str(value) for value in (expected_plan.get("tournament_game_ids") or []) if str(value)]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery has no valid deterministic game set; do not repeat the mutation."
        )
    try:
        current_plan = build_admin_tournament_official_publish_plan(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            draw_id=str(draw_id),
            playoff_winner_bonus_elo=float(expected_plan.get("playoff_winner_bonus_elo") or 0.0),
        )
    except Exception as exc:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery could not reconstruct the reviewed match plan; do not repeat the mutation."
        ) from exc
    if current_plan != expected_plan:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery found changed draw/team/game versions or match payloads. The operation remains recovery-required; do not repeat it."
        )
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .in_("tournament_game_id", expected_ids)
            .execute()
        )
    except Exception as exc:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery could not read authoritative match evidence; do not repeat the mutation."
        ) from exc
    actual_ids = [str(row.get("tournament_game_id")) for row in rows if row.get("tournament_game_id")]
    exact = (
        len(actual_ids) == len(expected_ids)
        and len(actual_ids) == len(set(actual_ids))
        and set(actual_ids) == set(expected_ids)
    )
    if not exact:
        evidence = "zero" if not actual_ids else "partial or duplicate"
        raise TournamentAdminRecoveryRequiredError(
            f"Official publish recovery found {evidence} match evidence. The operation remains recovery-required; do not repeat it."
        )
    expected_fingerprints = list(expected_plan.get("match_payload_fingerprints") or [])
    actual_fingerprints = sorted(
        [
            {
                "tournament_game_id": str(row.get("tournament_game_id") or ""),
                "payload_fingerprint": stable_tournament_admin_fingerprint(
                    _official_match_projection(row, club_id=str(club_id))
                ),
            }
            for row in rows
        ],
        key=lambda row: row["tournament_game_id"],
    )
    if not expected_fingerprints or actual_fingerprints != expected_fingerprints:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery found changed official-match content. The operation remains recovery-required; do not repeat it."
        )
    expected_plan_fingerprint = stable_tournament_admin_fingerprint(expected_plan)
    try:
        receipt_rows = _safe_rows(
            supabase.table("admin_activity_log")
            .select("action_type,entity_id,after_json")
            .eq("club_id", str(club_id))
            .eq("entity_id", str(draw_id))
            .eq("action_type", "publish_tournament_games_to_matches_admin")
            .execute()
        )
    except Exception as exc:
        raise TournamentAdminRecoveryRequiredError(
            "Official publish recovery could not verify the post-processor completion receipt; do not repeat the mutation."
        ) from exc
    matching_receipts = [
        row
        for row in receipt_rows
        if isinstance(row.get("after_json"), dict)
        and str(row["after_json"].get("publish_plan_fingerprint") or "") == expected_plan_fingerprint
        and str(row["after_json"].get("guarded_operation_key") or "") == clean_operation_key
        and str(row["after_json"].get("guarded_request_fingerprint") or "") == clean_request_fingerprint
        and str(row["after_json"].get("client_idempotency_key") or "") == clean_idempotency_key
    ]
    if len(matching_receipts) != 1:
        raise TournamentAdminRecoveryRequiredError(
            "Official match rows exist without one exact post-processor completion receipt. Rating/player updates are not proven; keep recovery locked."
        )
    singles_count = int(expected_plan.get("singles_match_count") or 0)
    doubles_count = int(expected_plan.get("doubles_match_count") or 0)
    bonus_ids = [str(value) for value in (expected_plan.get("bonus_tournament_game_ids") or [])]
    return {
        "ok": True,
        "mode": "tournament_official_matches_publish",
        "draw_id": str(draw_id),
        "match_count": len(expected_ids),
        "singles_match_count": singles_count,
        "doubles_match_count": doubles_count,
        "game_count": len(expected_ids),
        "tournament_game_ids": expected_ids,
        "playoff_winner_bonus_elo": float(expected_plan.get("playoff_winner_bonus_elo") or 0.0),
        "bonus_match_count": len(bonus_ids),
        "bonus_tournament_game_ids": bonus_ids,
        "process_result": {"inserted": len(expected_ids), "reconciled_from_authoritative_matches": True},
        "auto_player_updates": {"mode": "reconciliation_readback_only"},
        "warnings": ["Response-loss recovery completed from the exact official-match set; no publish mutation was repeated."],
    }


def publish_admin_tournament_draw_matches(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    playoff_winner_bonus_elo: Any = 0.0,
    expected_plan: dict[str, Any] | None = None,
    guarded_operation_key: str = "",
    guarded_request_fingerprint: str = "",
    client_idempotency_key: str = "",
    source: str = "next_tournament_admin_publish_matches",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_PUBLISH_MATCHES:
        raise ValueError(f"Type {CONFIRM_PUBLISH_MATCHES} to publish official tournament matches.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    bonus_elo = _validate_bonus_elo(playoff_winner_bonus_elo)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not games:
        raise ValueError("This draw has no tournament games to publish.")

    game_ids = [_clean_text(row.get("id"), limit=120) for row in games if _clean_text(row.get("id"), limit=120)]
    already_published = _existing_published_game_ids(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        game_ids=game_ids,
    )
    if already_published:
        raise ValueError("Some tournament games are already published as official matches: " + ", ".join(sorted(already_published)))

    event_option = _fetch_event_option(
        supabase,
        tournament_id=clean_tournament_id,
        event_option_id=_clean_text(draw.get("event_option_id"), limit=120),
    )
    match_payloads = _build_official_match_payloads(
        tournament=tournament,
        draw=draw,
        event_option=event_option,
        teams=teams,
        games=games,
        playoff_winner_bonus_elo=bonus_elo,
    )
    current_plan = _official_publish_plan_from_state(
        club_id=str(club_id),
        tournament=tournament,
        event_option=event_option,
        draw=draw,
        teams=teams,
        games=games,
        match_payloads=match_payloads,
        bonus_elo=bonus_elo,
    )
    if expected_plan is not None and current_plan != expected_plan:
        raise StaleTournamentAdminStateError(
            "The draw, team set, game versions, or official match payload changed after review. No matches were published."
        )
    publish_plan_fingerprint = stable_tournament_admin_fingerprint(current_plan)
    bonus_game_ids = [str(payload.get("tournament_game_id")) for payload in match_payloads if _safe_float(payload.get("winner_bonus_elo"))]
    singles_payloads = [row for row in match_payloads if str(row.get("match_format")) == "singles"]
    doubles_payloads = [row for row in match_payloads if str(row.get("match_format")) != "singles"]

    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_official_matches_publish_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "match_count": len(match_payloads),
            "singles_match_count": len(singles_payloads),
            "doubles_match_count": len(doubles_payloads),
            "game_count": len(games),
            "tournament_game_ids": game_ids,
            "playoff_winner_bonus_elo": bonus_elo,
            "bonus_match_count": len(bonus_game_ids),
            "bonus_tournament_game_ids": bonus_game_ids,
            "auto_player_updates": {"mode": "preview_only"},
            "warnings": [],
        }

    if (
        os.getenv("JUPR_ENV", "").strip().lower() == "staging"
        and (
            not str(guarded_operation_key or "").strip()
            or not str(guarded_request_fingerprint or "").strip()
        )
    ):
        raise RuntimeError(
            "Staging official publish requires an exact guarded operation key and request fingerprint before any mutation."
        )

    df_players_all = _table_frame(supabase, "players", club_id=str(club_id))
    df_leagues = _table_frame(supabase, "league_ratings", club_id=str(club_id))
    df_meta = _table_frame(supabase, "leagues_metadata", club_id=str(club_id))
    if expected_plan is not None:
        final_plan = build_admin_tournament_official_publish_plan(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            playoff_winner_bonus_elo=bonus_elo,
        )
        if final_plan != expected_plan:
            raise StaleTournamentAdminStateError(
                "The official match payload changed immediately before processing. No matches were published."
            )
    process_result: dict[str, Any] = {"doubles": {"inserted": 0}, "singles": {"inserted": 0}}
    inserted_count = 0
    atomic_identity_ready = bool(
        str(guarded_operation_key or "").strip()
        and str(guarded_request_fingerprint or "").strip()
    )
    if atomic_identity_ready:
        combined_write_plan: dict[str, list[dict[str, Any]]] = {
            "match_rows": [],
            "player_updates": [],
            "league_rating_updates": [],
            "league_metadata_expectations": [],
        }
        side_effect_contexts: list[dict[str, Any]] = []
        if doubles_payloads:
            doubles_result = process_matches(
                doubles_payloads,
                supabase=supabase,
                club_id=str(club_id),
                name_to_id={},
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
                build_write_plan_only=True,
            )
            process_result["doubles"] = {
                key: value
                for key, value in doubles_result.items()
                if key not in {"write_plan", "side_effect_context"}
            }
            for key in combined_write_plan:
                combined_write_plan[key].extend(list((doubles_result.get("write_plan") or {}).get(key) or []))
            side_effect_contexts.append(dict(doubles_result.get("side_effect_context") or {}))
        if singles_payloads:
            singles_result = process_singles_matches(
                singles_payloads,
                supabase=supabase,
                club_id=str(club_id),
                name_to_id={},
                df_players_all=df_players_all,
                build_write_plan_only=True,
            )
            process_result["singles"] = {
                key: value
                for key, value in singles_result.items()
                if key not in {"write_plan", "side_effect_context"}
            }
            for key in combined_write_plan:
                combined_write_plan[key].extend(list((singles_result.get("write_plan") or {}).get(key) or []))
            side_effect_contexts.append(dict(singles_result.get("side_effect_context") or {}))
        atomic_result = _apply_official_rating_plan_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            guarded_operation_key=str(guarded_operation_key),
            guarded_request_fingerprint=str(guarded_request_fingerprint),
            publish_plan_fingerprint=publish_plan_fingerprint,
            write_plan={**combined_write_plan, "publish_plan": current_plan},
        )
        inserted_count = int(atomic_result.get("inserted") or 0)
        affected_players = {
            int(value)
            for context in side_effect_contexts
            for value in (context.get("affected_player_ids") or [])
        }
        successful_dates = [
            str(value)
            for context in side_effect_contexts
            for value in (context.get("successful_match_dates") or [])
        ]
        side_effect_match_payloads = [
            dict(value)
            for context in side_effect_contexts
            for value in (context.get("match_payloads") or [])
            if isinstance(value, dict)
        ]
        badge_summary = run_badge_side_effects(
            supabase=supabase,
            club_id=str(club_id),
            has_badge_eligible_match=any(
                bool(context.get("has_badge_eligible_match")) for context in side_effect_contexts
            ),
            affected_players=affected_players,
            db_matches=list(combined_write_plan["match_rows"]),
            match_payloads=side_effect_match_payloads,
        )
        player_update_queue = queue_player_updates(
            supabase=supabase,
            club_id=str(club_id),
            db_matches=list(combined_write_plan["match_rows"]),
            affected_players=affected_players,
            successful_match_dates=successful_dates,
        )
        if (
            str(badge_summary.get("mode") or "").endswith("error")
            or str(player_update_queue.get("mode") or "") == "error"
            or int(player_update_queue.get("failed") or 0) > 0
        ):
            raise RuntimeError(
                "Official match/rating core committed, but a required post-processor failed. Keep the guarded operation recovery-locked."
            )
        process_result["atomic_core"] = atomic_result
        process_result["badge_summary"] = badge_summary
        process_result["player_update_queue"] = player_update_queue
    else:
        if doubles_payloads:
            doubles_result = process_matches(
                doubles_payloads,
                supabase=supabase,
                club_id=str(club_id),
                name_to_id={},
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
            )
            process_result["doubles"] = doubles_result
            inserted_count += int(doubles_result.get("inserted") or 0)
        if singles_payloads:
            singles_result = process_singles_matches(
                singles_payloads,
                supabase=supabase,
                club_id=str(club_id),
                name_to_id={},
                df_players_all=df_players_all,
            )
            process_result["singles"] = singles_result
            inserted_count += int(singles_result.get("inserted") or 0)
    process_result["inserted"] = inserted_count
    if inserted_count != len(match_payloads):
        raise RuntimeError(f"Official match publish inserted {inserted_count} of {len(match_payloads)} tournament games.")

    auto_player_updates = auto_send_player_updates_for_match_payloads(
        supabase,
        club_id=str(club_id),
        match_payloads=match_payloads,
        source=source,
    )

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="publish_tournament_games_to_matches_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "game_count": len(games)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "match_count": inserted_count,
            "singles_match_count": len(singles_payloads),
            "doubles_match_count": len(doubles_payloads),
            "tournament_game_ids": game_ids,
            "playoff_winner_bonus_elo": bonus_elo,
            "bonus_tournament_game_ids": bonus_game_ids,
            "publish_plan_fingerprint": publish_plan_fingerprint,
            "guarded_operation_key": str(guarded_operation_key or ""),
            "guarded_request_fingerprint": str(guarded_request_fingerprint or ""),
            "client_idempotency_key": str(client_idempotency_key or ""),
            "process_result": process_result,
            "auto_player_updates": auto_player_updates,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if atomic_identity_ready and not audit_write.ok:
        raise RuntimeError(
            "Official match/rating core completed, but its exact operation-bound post-processor receipt did not persist. Keep recovery locked."
        )
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    return {
        "ok": True,
        "mode": "tournament_official_matches_publish",
        "draw_id": clean_draw_id,
        "match_count": inserted_count,
        "singles_match_count": len(singles_payloads),
        "doubles_match_count": len(doubles_payloads),
        "game_count": len(games),
        "tournament_game_ids": game_ids,
        "playoff_winner_bonus_elo": bonus_elo,
        "bonus_match_count": len(bonus_game_ids),
        "bonus_tournament_game_ids": bonus_game_ids,
        "process_result": process_result,
        "auto_player_updates": auto_player_updates,
        "warnings": warnings,
    }
