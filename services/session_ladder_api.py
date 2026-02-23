from __future__ import annotations

import csv
import io
from typing import Any

from jupr_app.config import FEATURE_SESSION_LADDER
from jupr_app.domain.player_ops import get_or_create_player
from jupr_app.domain.session_ladder_engine import computeCourtStandings, resolveTies
from jupr_app.domain.session_ladder_service import (
    closeRound,
    completeSession,
    createSession,
    lockSeeding,
    publishSession,
    seedCourtsByRating,
    startRound,
    submitGameResult,
    updateRosterStatus,
)


def post_create_session(*, supabase: Any, auth: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    session = createSession(
        supabase,
        club_id=str(auth.get("club_id") or ""),
        league_id=str(payload.get("league_id") or ""),
        season_id=payload.get("season_id"),
        session_starts_at=str(payload.get("session_starts_at") or ""),
        courts_available=int(payload.get("courts_available") or 0),
        players_per_court=int(payload.get("players_per_court") or 4),
        rounds_planned=int(payload.get("rounds_planned") or 2),
        created_by=str(auth.get("user_id") or auth.get("email") or "api"),
    )
    return {"session": session}


def get_session_details(*, supabase: Any, auth: dict[str, Any], session_id: str) -> dict[str, Any]:
    _require_feature()
    _require_read_access(auth)

    session = _single(supabase.table("session_ladder_sessions").select("*").eq("id", str(session_id)).execute().data)
    if not session:
        raise ValueError("session not found")
    _assert_club_scope(auth, str(session.get("club_id") or ""))

    pods = (
        supabase.table("session_ladder_court_pods")
        .select("*")
        .eq("session_id", str(session_id))
        .execute()
        .data
        or []
    )

    details: list[dict[str, Any]] = []
    for pod in sorted(pods, key=lambda r: (int(r.get("round_number") or 0), int(r.get("court_number") or 0))):
        players = (
            supabase.table("session_ladder_court_pod_players")
            .select("*")
            .eq("court_pod_id", str(pod["id"]))
            .execute()
            .data
            or []
        )
        games = (
            supabase.table("session_ladder_games")
            .select("*")
            .eq("court_pod_id", str(pod["id"]))
            .execute()
            .data
            or []
        )

        player_ids = [int(r["player_id"]) for r in sorted(players, key=lambda x: int(x.get("player_order") or 0))]
        completed_games = [
            {
                "teamA": list(game.get("team_a_player_ids") or []),
                "teamB": list(game.get("team_b_player_ids") or []),
                "scoreA": game.get("score_a"),
                "scoreB": game.get("score_b"),
            }
            for game in games
            if game.get("score_a") is not None and game.get("score_b") is not None
        ]
        standings = resolveTies(computeCourtStandings(completed_games, player_ids), completed_games)
        details.append(
            {
                "pod": pod,
                "players": sorted(players, key=lambda x: int(x.get("player_order") or 0)),
                "games": sorted(games, key=lambda x: int(x.get("game_number") or 0)),
                "standings": standings,
                "court_sheet_route": _court_sheet_route(str(session_id), int(pod.get("round_number") or 0), int(pod.get("court_number") or 0)),
            }
        )

    return {"session": session, "courts": details}


def post_add_roster_entries(*, supabase: Any, auth: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    session_id = str(payload.get("session_id") or "")
    mode = str(payload.get("mode") or "manual_existing").strip().lower()

    added: list[dict[str, Any]] = []
    if mode == "manual_existing":
        player_id = int(payload.get("player_id"))
        rating = float(payload.get("rating_snapshot") or 1200)
        added.append(
            updateRosterStatus(
                supabase,
                sessionId=session_id,
                player_id=player_id,
                status=str(payload.get("status") or "EXPECTED"),
                rating_snapshot=rating,
                updated_by=str(auth.get("user_id") or "api"),
            )
        )
    elif mode == "create_new_player":
        row = _create_player_from_payload(supabase, auth, payload)
        added.append(
            updateRosterStatus(
                supabase,
                sessionId=session_id,
                player_id=int(row["id"]),
                status=str(payload.get("status") or "WALK_IN"),
                rating_snapshot=float(row.get("rating") or 1200),
                updated_by=str(auth.get("user_id") or "api"),
            )
        )
    elif mode in {"bulk_text", "csv_upload"}:
        entries = _parse_bulk_entries(payload)
        for entry in entries:
            row = _create_player_from_payload(supabase, auth, entry)
            added.append(
                updateRosterStatus(
                    supabase,
                    sessionId=session_id,
                    player_id=int(row["id"]),
                    status=str(entry.get("status") or "EXPECTED"),
                    rating_snapshot=float(row.get("rating") or entry.get("rating", 1200)),
                    updated_by=str(auth.get("user_id") or "api"),
                )
            )
    else:
        raise ValueError(f"Unsupported roster mode: {mode}")

    return {"session_id": session_id, "added": added}


def post_seed_courts_by_rating(*, supabase: Any, auth: dict[str, Any], session_id: str) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    pods = seedCourtsByRating(supabase, sessionId=str(session_id), seeded_by=str(auth.get("user_id") or "api"))
    return {"session_id": str(session_id), "pods": pods}


def post_lock_seeding(*, supabase: Any, auth: dict[str, Any], session_id: str) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    return {"session": lockSeeding(supabase, sessionId=str(session_id), updated_by=str(auth.get("user_id") or "api"))}


def post_start_round(*, supabase: Any, auth: dict[str, Any], session_id: str, round_number: int) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    return {"session": startRound(supabase, sessionId=str(session_id), roundNumber=int(round_number), updated_by=str(auth.get("user_id") or "api"))}


def post_submit_game_result(*, supabase: Any, auth: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    game = submitGameResult(
        supabase,
        sessionId=str(payload.get("session_id")),
        court_pod_id=str(payload.get("court_pod_id")),
        game_number=int(payload.get("game_number")),
        teamA_player_ids=list(payload.get("teamA_player_ids") or []),
        teamB_player_ids=list(payload.get("teamB_player_ids") or []),
        scoreA=int(payload.get("scoreA")),
        scoreB=int(payload.get("scoreB")),
        edited_by=str(auth.get("user_id") or "api"),
    )
    return {"game": game}


def post_close_round(
    *,
    supabase: Any,
    auth: dict[str, Any],
    session_id: str,
    round_number: int,
    movers_per_court: int = 1,
    allow_override: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    result = closeRound(
        supabase,
        sessionId=str(session_id),
        roundNumber=int(round_number),
        updated_by=str(auth.get("user_id") or "api"),
        movers_per_court=int(movers_per_court),
        allow_override=bool(allow_override),
        override_reason=override_reason,
    )
    for court in result.get("courts", []):
        court["court_sheet_route"] = _court_sheet_route(str(session_id), int(round_number), int(court.get("court_number") or 0))
    return result


def post_complete_session(*, supabase: Any, auth: dict[str, Any], session_id: str) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    return {"session": completeSession(supabase, sessionId=str(session_id), updated_by=str(auth.get("user_id") or "api"))}


def post_publish_session(*, supabase: Any, auth: dict[str, Any], session_id: str) -> dict[str, Any]:
    _require_feature()
    _require_mutation_access(auth)
    return {"session": publishSession(supabase, sessionId=str(session_id), updated_by=str(auth.get("user_id") or "api"))}


def _create_player_from_payload(supabase: Any, auth: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    club_id = str(auth.get("club_id") or "")
    if not club_id:
        raise ValueError("club_id is required")

    ok, row, err = get_or_create_player(
        supabase=supabase,
        club_id=club_id,
        normalized_name=" ".join(name.lower().split()),
        payload={
            "club_id": club_id,
            "name": name,
            "normalized_name": " ".join(name.lower().split()),
            "rating": float(payload.get("rating") or 1200),
            "active": True,
        },
    )
    if not ok or row is None:
        raise RuntimeError(err or "Unable to create player")
    return dict(row)


def _parse_bulk_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("mode") == "csv_upload":
        text = str(payload.get("csv_text") or "")
        rows: list[dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            rows.append(
                {
                    "name": str(row.get("name") or "").strip(),
                    "rating": float(row.get("rating") or 1200),
                    "status": str(row.get("status") or "EXPECTED"),
                }
            )
        return [row for row in rows if row.get("name")]

    text = str(payload.get("bulk_text") or "")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        rating = float(parts[1]) if len(parts) >= 2 and parts[1] else 1200
        status = parts[2] if len(parts) >= 3 and parts[2] else "EXPECTED"
        rows.append({"name": name, "rating": rating, "status": status})
    return rows


def _court_sheet_route(session_id: str, round_number: int, court_number: int) -> str:
    return f"/sessions/{session_id}/rounds/{int(round_number)}/courts/{int(court_number)}"


def _single(rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not rows:
        return None
    return dict(rows[0])


def _require_feature() -> None:
    if not bool(FEATURE_SESSION_LADDER):
        raise RuntimeError("feature.sessionLadder disabled")


def _assert_club_scope(auth: dict[str, Any], row_club_id: str) -> None:
    if str(auth.get("club_id") or "") != str(row_club_id or ""):
        raise PermissionError("club scope mismatch")


def _require_read_access(auth: dict[str, Any]) -> None:
    if not str(auth.get("club_id") or ""):
        raise PermissionError("club_id required")


def _require_mutation_access(auth: dict[str, Any]) -> None:
    _require_read_access(auth)
    role = str(auth.get("role") or "").strip().lower()
    if bool(auth.get("admin_logged_in")):
        return
    if role not in {"admin", "manager"}:
        raise PermissionError("manager/admin role required")
