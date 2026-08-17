from __future__ import annotations

from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    is_admin_tournament_admin_enabled,
)


CONFIRM_REVIEW_PODIUM = "REVIEW PODIUM"
PODIUM_REVIEW_ACTION = "review_tournament_draw_podium_admin"
PODIUM_REVIEW_CONTRACT = "jupr:tournament-podium-review:v1"


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


def _table_rows(supabase: Any, table_name: str, *, tournament_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table(table_name)
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not verify {table_name} before podium review; no review evidence was recorded."
        ) from exc


def _canonical_versions(rows: list[dict[str, Any]], *, label: str) -> list[dict[str, str]]:
    versions = sorted(
        [
            {
                "id": str(row.get("id") or "").strip(),
                "updated_at": str(row.get("updated_at") or "").strip(),
            }
            for row in rows
        ],
        key=lambda row: row["id"],
    )
    if not versions or any(not row["id"] or not row["updated_at"] for row in versions):
        raise StaleTournamentAdminStateError(
            f"Podium review requires a complete {label} version set. Reload Tournament Live."
        )
    if len({row["id"] for row in versions}) != len(versions):
        raise StaleTournamentAdminStateError(
            f"Podium review found duplicate {label} identities. Reload Tournament Live."
        )
    return versions


def _canonical_expected_versions(value: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise StaleTournamentAdminStateError(
            f"A reviewed {label} version set is required. Reload Tournament Live."
        )
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise StaleTournamentAdminStateError(
                f"The reviewed {label} version set is malformed. Reload Tournament Live."
            )
        rows.append(item)
    return _canonical_versions(rows, label=label)


def _draw_projection(draw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: draw.get(key)
        for key in (
            "id",
            "tournament_id",
            "registration_day_id",
            "event_option_id",
            "name",
            "status",
            "draw_kind",
            "hidden_from_primary_ops",
        )
    }


def _team_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id",
            "draw_id",
            "team_number",
            "player1_id",
            "player2_id",
            "seed",
            "source",
            "updated_at",
        )
    }


def _game_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id",
            "draw_id",
            "stage",
            "rr_round_number",
            "rr_slot_number",
            "playoff_game_code",
            "playoff_round",
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
        )
    }


def _podium_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "draw_id": str(row.get("draw_id") or ""),
        "placement": _safe_int(row.get("placement")),
        "team_id": str(row.get("team_id") or ""),
        "source": str(row.get("source") or "").upper(),
        "updated_at": str(row.get("updated_at") or ""),
    }


def build_admin_tournament_podium_review_fingerprint(
    *,
    draw: dict[str, Any],
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    podium: list[dict[str, Any]],
) -> str:
    """Hash only authoritative state whose drift invalidates a podium review.

    Audit rows, awards, official matches, and unrelated tournament draws are
    intentionally excluded. A later award or an unrelated review therefore
    cannot make otherwise-current evidence stale.
    """

    state = {
        "contract": PODIUM_REVIEW_CONTRACT,
        "draw": _draw_projection(draw),
        "teams": sorted(
            [_team_projection(row) for row in teams],
            key=lambda row: str(row.get("id") or ""),
        ),
        "games": sorted(
            [_game_projection(row) for row in games],
            key=lambda row: str(row.get("id") or ""),
        ),
        "podium": sorted(
            [_podium_projection(row) for row in podium],
            key=lambda row: (
                int(row.get("placement") or 0),
                str(row.get("id") or ""),
            ),
        ),
    }
    return stable_tournament_admin_fingerprint(state)


def _validate_reviewable_podium(
    *,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    podium: list[dict[str, Any]],
) -> None:
    if not games:
        raise ValueError("A podium cannot be reviewed until this draw has tournament games.")
    team_ids = {str(row.get("id") or "") for row in teams if row.get("id")}
    for index, game in enumerate(games, start=1):
        score_a = _safe_int(game.get("score_a"))
        score_b = _safe_int(game.get("score_b"))
        team_a = str(game.get("team_a_id") or "")
        team_b = str(game.get("team_b_id") or "")
        winner = str(game.get("winner_team_id") or "")
        loser = str(game.get("loser_team_id") or "")
        expected_winner = team_a if score_a is not None and score_b is not None and score_a > score_b else team_b
        expected_loser = team_b if expected_winner == team_a else team_a
        if (
            score_a is None
            or score_b is None
            or score_a < 0
            or score_b < 0
            or score_a == score_b
            or not game.get("finalized_at")
            or not team_a
            or not team_b
            or team_a == team_b
            or team_a not in team_ids
            or team_b not in team_ids
            or winner != expected_winner
            or loser != expected_loser
        ):
            raise ValueError(
                f"Game {index} is not a finalized, non-tied result with valid teams and winner evidence."
            )

    normalized = sorted(
        [_podium_projection(row) for row in podium],
        key=lambda row: (
            int(row.get("placement") or 0),
            str(row.get("id") or ""),
        ),
    )
    placements = [row["placement"] for row in normalized]
    podium_team_ids = [str(row["team_id"]) for row in normalized]
    if (
        placements != [1, 2, 3]
        or len(set(podium_team_ids)) != 3
        or any(team_id not in team_ids for team_id in podium_team_ids)
    ):
        raise ValueError(
            "Podium review requires exactly one valid first-, second-, and third-place team."
        )


def find_current_admin_tournament_podium_review(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    review_fingerprint: str,
) -> dict[str, Any]:
    try:
        rows = _safe_rows(
            supabase.table("admin_activity_log")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("entity_type", "tournament_event_draw")
            .eq("entity_id", str(draw_id))
            .eq("action_type", PODIUM_REVIEW_ACTION)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
    except Exception:
        return {
            "available": False,
            "reviewed": False,
            "current": False,
            "review_fingerprint": str(review_fingerprint),
            "reviewed_at": None,
            "reviewed_by": None,
            "blockers": ["Podium review evidence is unavailable."],
        }

    for row in rows:
        after_json = row.get("after_json") if isinstance(row.get("after_json"), dict) else {}
        evidence = (
            after_json.get("podium_review_evidence")
            if isinstance(after_json.get("podium_review_evidence"), dict)
            else {}
        )
        if (
            str(evidence.get("contract") or "") == PODIUM_REVIEW_CONTRACT
            and str(evidence.get("tournament_id") or "") == str(tournament_id)
            and str(evidence.get("draw_id") or "") == str(draw_id)
            and str(evidence.get("review_fingerprint") or "") == str(review_fingerprint)
        ):
            return {
                "available": True,
                "reviewed": True,
                "current": True,
                "review_fingerprint": str(review_fingerprint),
                "reviewed_at": row.get("created_at"),
                "reviewed_by": row.get("actor_email"),
                "evidence": evidence,
                "blockers": [],
            }

    had_review = bool(rows)
    return {
        "available": True,
        "reviewed": had_review,
        "current": False,
        "review_fingerprint": str(review_fingerprint),
        "reviewed_at": rows[0].get("created_at") if rows else None,
        "reviewed_by": rows[0].get("actor_email") if rows else None,
        "blockers": [
            "The draw, teams, games, or podium changed after review; review the current podium again."
            if had_review
            else "The current podium has not been explicitly reviewed."
        ],
    }


def review_admin_tournament_draw_podium(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_state_fingerprint: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, Any]],
    expected_source_game_versions: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_review_podium",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_REVIEW_PODIUM:
        raise ValueError(f"Type {CONFIRM_REVIEW_PODIUM} to record podium review evidence.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    expected_state = str(expected_state_fingerprint or "").strip().lower()
    if len(expected_state) != 64:
        raise StaleTournamentAdminStateError(
            "A complete reviewed Tournament Ops fingerprint is required. Reload Tournament Live."
        )
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    draws = [
        row
        for row in _table_rows(supabase, "tournament_event_draws", tournament_id=clean_tournament_id)
        if str(row.get("id") or "") == clean_draw_id
    ]
    if len(draws) != 1:
        raise ValueError("draw not found for this tournament")
    draw = draws[0]
    teams = [
        row
        for row in _table_rows(supabase, "tournament_teams", tournament_id=clean_tournament_id)
        if str(row.get("draw_id") or "") == clean_draw_id
    ]
    games = [
        row
        for row in _table_rows(supabase, "tournament_games", tournament_id=clean_tournament_id)
        if str(row.get("draw_id") or "") == clean_draw_id
    ]
    podium = [
        row
        for row in _table_rows(supabase, "tournament_podium", tournament_id=clean_tournament_id)
        if str(row.get("draw_id") or "") == clean_draw_id
    ]

    if str(draw.get("updated_at") or "") != str(expected_draw_updated_at or ""):
        raise StaleTournamentAdminStateError(
            "This draw changed after podium review was opened. Reload Tournament Live."
        )
    current_team_versions = _canonical_versions(teams, label="team")
    current_game_versions = _canonical_versions(games, label="source game")
    if current_team_versions != _canonical_expected_versions(expected_team_versions, label="team"):
        raise StaleTournamentAdminStateError(
            "The team set changed after podium review was opened. Reload Tournament Live."
        )
    if current_game_versions != _canonical_expected_versions(
        expected_source_game_versions,
        label="source game",
    ):
        raise StaleTournamentAdminStateError(
            "The game set changed after podium review was opened. Reload Tournament Live."
        )
    current_state = get_admin_tournament_ops_state_fingerprint(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
    )
    if current_state != expected_state:
        raise StaleTournamentAdminStateError(
            "Tournament Ops state changed after podium review was opened. Reload Tournament Live."
        )
    _validate_reviewable_podium(teams=teams, games=games, podium=podium)

    review_fingerprint = build_admin_tournament_podium_review_fingerprint(
        draw=draw,
        teams=teams,
        games=games,
        podium=podium,
    )
    evidence = {
        "contract": PODIUM_REVIEW_CONTRACT,
        "tournament_id": clean_tournament_id,
        "draw_id": clean_draw_id,
        "review_fingerprint": review_fingerprint,
        "state_fingerprint": current_state,
        "draw_updated_at": str(draw.get("updated_at") or ""),
        "draw": _draw_projection(draw),
        "team_versions": current_team_versions,
        "game_versions": current_game_versions,
        "podium": sorted(
            [_podium_projection(row) for row in podium],
            key=lambda row: (
                int(row.get("placement") or 0),
                str(row.get("id") or ""),
            ),
        ),
    }
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=PODIUM_REVIEW_ACTION,
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json=None,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": str(source),
            "podium_review_evidence": evidence,
        },
        source_page=str(source),
        flagged_for_review=False,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok:
        raise RuntimeError(
            "Podium review evidence could not be recorded. Awards and official publishing remain blocked."
        )
    return {
        "ok": True,
        "mode": "tournament_draw_podium_review",
        "tournament_id": clean_tournament_id,
        "draw_id": clean_draw_id,
        "reviewed": True,
        "current": True,
        "review_fingerprint": review_fingerprint,
        "review_evidence": evidence,
        "warnings": [],
    }


__all__ = [
    "CONFIRM_REVIEW_PODIUM",
    "PODIUM_REVIEW_ACTION",
    "PODIUM_REVIEW_CONTRACT",
    "build_admin_tournament_podium_review_fingerprint",
    "find_current_admin_tournament_podium_review",
    "review_admin_tournament_draw_podium",
]
