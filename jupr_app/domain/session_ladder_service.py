from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert
from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.session_ladder_engine import (
    applyMovement,
    computeCourtStandings,
    generateRoundGames,
    getMovers,
    resolveTies,
)

SESSION_STATES = [
    "DRAFT",
    "ROSTER_OPEN",
    "SEEDED_LOCKED",
    "ROUND_1_ACTIVE",
    "ROUND_1_CLOSED",
    "ROUND_2_ACTIVE",
    "ROUND_2_CLOSED",
    "ROUND_3_ACTIVE",
    "ROUND_3_CLOSED",
    "COMPLETED",
    "PUBLISHED",
]

_VALID_TRANSITIONS = {
    "DRAFT": {"ROSTER_OPEN"},
    "ROSTER_OPEN": {"SEEDED_LOCKED"},
    "SEEDED_LOCKED": {"ROUND_1_ACTIVE"},
    "ROUND_1_ACTIVE": {"ROUND_1_CLOSED"},
    "ROUND_1_CLOSED": {"ROUND_2_ACTIVE"},
    "ROUND_2_ACTIVE": {"ROUND_2_CLOSED"},
    "ROUND_2_CLOSED": {"ROUND_3_ACTIVE", "COMPLETED"},
    "ROUND_3_ACTIVE": {"ROUND_3_CLOSED"},
    "ROUND_3_CLOSED": {"COMPLETED"},
    "COMPLETED": {"PUBLISHED"},
    "PUBLISHED": set(),
}


@dataclass(frozen=True)
class SessionCompleteness:
    ok: bool
    required_games_per_court: int
    missing: list[dict[str, Any]]


def canTransition(from_state: str, to_state: str) -> bool:
    src = str(from_state or "").strip().upper()
    dst = str(to_state or "").strip().upper()
    return dst in _VALID_TRANSITIONS.get(src, set())


def _fetch_rows(supabase: Any, table: str, **filters: Any) -> list[dict[str, Any]]:
    query = supabase.table(table).select("*")
    for key, value in filters.items():
        query = query.eq(key, value)
    return list((query.execute().data or []))


def _require_session(supabase: Any, sessionId: str) -> dict[str, Any]:
    rows = _fetch_rows(supabase, "session_ladder_sessions", id=str(sessionId))
    if not rows:
        raise ValueError(f"Session not found: {sessionId}")
    return dict(rows[0])


def _required_games(players_per_court: int) -> int:
    return 3 if int(players_per_court) == 4 else 5


def validateSessionCompleteness(supabase: Any, sessionId: str) -> SessionCompleteness:
    session = _require_session(supabase, sessionId)
    required = _required_games(int(session.get("players_per_court") or 4))
    pods = _fetch_rows(supabase, "session_ladder_court_pods", session_id=str(sessionId))
    pods = [p for p in pods if str(p.get("state") or "planned") != "planned"]

    missing: list[dict[str, Any]] = []
    for pod in pods:
        games = _fetch_rows(supabase, "session_ladder_games", court_pod_id=str(pod["id"]))
        scored = [g for g in games if g.get("score_a") is not None and g.get("score_b") is not None]
        if len(scored) < required:
            missing.append(
                {
                    "court_pod_id": str(pod["id"]),
                    "round_number": int(pod.get("round_number") or 0),
                    "court_number": int(pod.get("court_number") or 0),
                    "scored_games": len(scored),
                    "required_games": required,
                }
            )
    return SessionCompleteness(ok=(len(missing) == 0), required_games_per_court=required, missing=missing)


def createSession(
    supabase: Any,
    *,
    club_id: str,
    league_id: str,
    season_id: str | None,
    session_starts_at: str,
    courts_available: int,
    players_per_court: int,
    rounds_planned: int = 2,
    created_by: str,
) -> dict[str, Any]:
    payload = {
        "club_id": str(club_id),
        "league_id": str(league_id),
        "season_id": str(season_id) if season_id is not None else None,
        "session_starts_at": str(session_starts_at),
        "courts_available": int(courts_available),
        "players_per_court": int(players_per_court),
        "rounds_planned": max(2, min(int(rounds_planned), 3)),
        "state": "DRAFT",
        "created_by": str(created_by),
        "updated_by": str(created_by),
    }
    return dict((sb_insert(supabase, "session_ladder_sessions", payload).data or [payload])[0])


def updateRosterStatus(
    supabase: Any,
    *,
    sessionId: str,
    player_id: int,
    status: str,
    rating_snapshot: float,
    updated_by: str,
) -> dict[str, Any]:
    session = _require_session(supabase, sessionId)
    payload = {
        "club_id": str(session["club_id"]),
        "session_id": str(sessionId),
        "player_id": int(player_id),
        "status": str(status).upper(),
        "rating_snapshot": float(rating_snapshot),
        "updated_by": str(updated_by),
        "created_by": str(updated_by),
    }
    result = sb_upsert(
        supabase,
        "session_ladder_roster_entries",
        payload,
        conflict="session_id,player_id",
    )
    return dict((result.data or [payload])[0])


def seedCourtsByRating(supabase: Any, *, sessionId: str, seeded_by: str) -> list[dict[str, Any]]:
    session = _require_session(supabase, sessionId)
    players_per_court = int(session.get("players_per_court") or 4)
    courts_available = int(session.get("courts_available") or 0)

    roster = _fetch_rows(supabase, "session_ladder_roster_entries", session_id=str(sessionId))
    eligible = [
        r for r in roster if str(r.get("status") or "").upper() in {"CHECKED_IN", "WALK_IN", "EXPECTED"}
    ]
    eligible.sort(key=lambda r: float(r.get("rating_snapshot") or 0), reverse=True)

    total_slots = max(0, courts_available * players_per_court)
    seeded = eligible[:total_slots]

    by_court: list[list[dict[str, Any]]] = []
    for i in range(0, len(seeded), players_per_court):
        chunk = seeded[i : i + players_per_court]
        if len(chunk) == players_per_court:
            by_court.append(chunk)

    created: list[dict[str, Any]] = []
    for idx, players in enumerate(by_court, start=1):
        pod_payload = {
            "club_id": str(session["club_id"]),
            "session_id": str(sessionId),
            "round_number": 1,
            "court_number": idx,
            "state": "planned",
            "created_by": str(seeded_by),
            "updated_by": str(seeded_by),
        }
        pod_row = sb_upsert(
            supabase,
            "session_ladder_court_pods",
            pod_payload,
            conflict="session_id,round_number,court_number",
        ).data[0]
        created.append(dict(pod_row))

        for order, row in enumerate(players, start=1):
            sb_upsert(
                supabase,
                "session_ladder_court_pod_players",
                {
                    "club_id": str(session["club_id"]),
                    "session_id": str(sessionId),
                    "court_pod_id": str(pod_row["id"]),
                    "player_id": int(row["player_id"]),
                    "player_order": int(order),
                    "player_label": f"P{order}",
                },
                conflict="court_pod_id,player_id",
            )
    return created


def lockSeeding(supabase: Any, *, sessionId: str, updated_by: str) -> dict[str, Any]:
    return _transition_state(supabase, sessionId=sessionId, to_state="SEEDED_LOCKED", updated_by=updated_by)


def startRound(supabase: Any, *, sessionId: str, roundNumber: int, updated_by: str) -> dict[str, Any]:
    state = f"ROUND_{int(roundNumber)}_ACTIVE"
    out = _transition_state(supabase, sessionId=sessionId, to_state=state, updated_by=updated_by)
    sb_update(
        supabase,
        "session_ladder_court_pods",
        {"state": "in_progress", "updated_by": str(updated_by)},
        filters={"session_id": str(sessionId), "round_number": int(roundNumber)},
    )
    return out


def submitGameResult(
    supabase: Any,
    *,
    sessionId: str,
    court_pod_id: str,
    game_number: int,
    teamA_player_ids: list[int],
    teamB_player_ids: list[int],
    scoreA: int,
    scoreB: int,
    edited_by: str,
) -> dict[str, Any]:
    session = _require_session(supabase, sessionId)
    state = str(session.get("state") or "").upper()
    if state in {"COMPLETED", "PUBLISHED"}:
        raise RuntimeError("Session is locked; game results cannot be edited after completion/publish")
    payload = {
        "club_id": str(session["club_id"]),
        "session_id": str(sessionId),
        "court_pod_id": str(court_pod_id),
        "game_number": int(game_number),
        "team_a_player_ids": [int(x) for x in teamA_player_ids],
        "team_b_player_ids": [int(x) for x in teamB_player_ids],
        "score_a": int(scoreA),
        "score_b": int(scoreB),
        "edited_by": str(edited_by),
    }
    result = sb_upsert(
        supabase,
        "session_ladder_games",
        payload,
        conflict="court_pod_id,game_number",
    )
    return dict((result.data or [payload])[0])


def closeRound(
    supabase: Any,
    *,
    sessionId: str,
    roundNumber: int,
    updated_by: str,
    movers_per_court: int = 1,
    allow_override: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    session = _require_session(supabase, sessionId)
    pods = _fetch_rows(supabase, "session_ladder_court_pods", session_id=str(sessionId), round_number=int(roundNumber))
    pods = sorted(pods, key=lambda row: int(row.get("court_number") or 0))

    required_games = _required_games(int(session.get("players_per_court") or 4))
    incomplete_courts: list[int] = []
    for pod in pods:
        games = _fetch_rows(supabase, "session_ladder_games", court_pod_id=str(pod["id"]))
        complete_count = len([g for g in games if g.get("score_a") is not None and g.get("score_b") is not None])
        if complete_count < required_games:
            incomplete_courts.append(int(pod.get("court_number") or 0))

    if incomplete_courts and not bool(allow_override):
        raise RuntimeError(f"Cannot close round: incomplete courts {incomplete_courts}")

    state = f"ROUND_{int(roundNumber)}_CLOSED"
    _transition_state(supabase, sessionId=sessionId, to_state=state, updated_by=updated_by)

    ranked_courts: list[list[int]] = []
    round_summaries: list[dict[str, Any]] = []
    for pod in pods:
        pod_players = _fetch_rows(supabase, "session_ladder_court_pod_players", court_pod_id=str(pod["id"]))
        players = [int(item["player_id"]) for item in sorted(pod_players, key=lambda r: int(r.get("player_order") or 0))]
        games = _fetch_rows(supabase, "session_ladder_games", court_pod_id=str(pod["id"]))

        normalized_games = [
            {
                "teamA": list(game.get("team_a_player_ids") or []),
                "teamB": list(game.get("team_b_player_ids") or []),
                "scoreA": game.get("score_a"),
                "scoreB": game.get("score_b"),
            }
            for game in games
            if game.get("score_a") is not None and game.get("score_b") is not None
        ]
        standings = computeCourtStandings(normalized_games, players)
        resolved = resolveTies(standings, normalized_games)
        movers = getMovers(resolved, int(movers_per_court))
        ranked = [int(item["player_id"]) for item in sorted(resolved, key=lambda r: int(r["rank"]))]

        ranked_courts.append(ranked)
        round_summaries.append(
            {
                "court_pod_id": str(pod["id"]),
                "court_number": int(pod.get("court_number") or 0),
                "standings": resolved,
                "movers": movers,
                "playoff_required": any(bool(item.get("playoff_required")) for item in resolved),
            }
        )

    unresolved_playoffs = [item["court_number"] for item in round_summaries if bool(item.get("playoff_required"))]
    if unresolved_playoffs and not bool(allow_override):
        raise RuntimeError(f"Cannot close round: unresolved playoffs {unresolved_playoffs}")

    sb_update(
        supabase,
        "session_ladder_court_pods",
        {"state": "complete", "updated_by": str(updated_by)},
        filters={"session_id": str(sessionId), "round_number": int(roundNumber)},
    )

    rounds_planned = int(session.get("rounds_planned") or 2)
    next_round = int(roundNumber) + 1
    next_round_pods = _fetch_rows(supabase, "session_ladder_court_pods", session_id=str(sessionId), round_number=next_round)
    generated_next_round = False

    if ranked_courts and next_round <= rounds_planned and not next_round_pods:
        moved = applyMovement(ranked_courts, int(movers_per_court))
        generated_next_round = True
        for court_number, players in enumerate(moved, start=1):
            pod = sb_upsert(
                supabase,
                "session_ladder_court_pods",
                {
                    "club_id": str(session["club_id"]),
                    "session_id": str(sessionId),
                    "round_number": next_round,
                    "court_number": int(court_number),
                    "state": "planned",
                    "created_by": str(updated_by),
                    "updated_by": str(updated_by),
                },
                conflict="session_id,round_number,court_number",
            ).data[0]

            for order, pid in enumerate(players, start=1):
                sb_upsert(
                    supabase,
                    "session_ladder_court_pod_players",
                    {
                        "club_id": str(session["club_id"]),
                        "session_id": str(sessionId),
                        "court_pod_id": str(pod["id"]),
                        "player_id": int(pid),
                        "player_order": int(order),
                        "player_label": f"P{order}",
                    },
                    conflict="court_pod_id,player_id",
                )

            template = generateRoundGames(players, "4p" if int(session.get("players_per_court") or 4) == 4 else "5p")
            for game in template:
                sb_upsert(
                    supabase,
                    "session_ladder_games",
                    {
                        "club_id": str(session["club_id"]),
                        "session_id": str(sessionId),
                        "court_pod_id": str(pod["id"]),
                        "game_number": int(game["game_number"]),
                        "team_a_player_ids": [int(x) for x in game["teamA"]],
                        "team_b_player_ids": [int(x) for x in game["teamB"]],
                        "score_a": None,
                        "score_b": None,
                        "edited_by": None,
                    },
                    conflict="court_pod_id,game_number",
                )

    auto_completed = False
    if int(roundNumber) >= rounds_planned:
        completeSession(supabase, sessionId=sessionId, updated_by=updated_by)
        auto_completed = True

    return {
        "session_id": str(sessionId),
        "round_number": int(roundNumber),
        "next_round": next_round,
        "generated_next_round": bool(generated_next_round),
        "courts": round_summaries,
        "incomplete_courts": incomplete_courts,
        "unresolved_playoffs": unresolved_playoffs,
        "override_used": bool(allow_override),
        "override_reason": str(override_reason or "").strip() if allow_override else None,
        "auto_completed": auto_completed,
    }


def completeSession(supabase: Any, *, sessionId: str, updated_by: str) -> dict[str, Any]:
    completeness = validateSessionCompleteness(supabase, sessionId)
    if not completeness.ok:
        raise RuntimeError(f"Session incomplete: {completeness.missing}")
    result = _transition_state(supabase, sessionId=sessionId, to_state="COMPLETED", updated_by=updated_by)
    rating_summary = _apply_session_rating_updates(supabase, session_id=sessionId, updated_by=updated_by)
    result["rating_update_hook_triggered"] = True
    result["rating_update"] = rating_summary
    return result


def publishSession(supabase: Any, *, sessionId: str, updated_by: str) -> dict[str, Any]:
    session = _require_session(supabase, sessionId)
    if str(session.get("state") or "").upper() != "PUBLISHED":
        _transition_state(supabase, sessionId=sessionId, to_state="PUBLISHED", updated_by=updated_by)

    attendance = _apply_attendance_for_session(supabase, session_id=sessionId, updated_by=updated_by)
    recap = _build_session_recap(supabase, session_id=sessionId)
    leaderboard = _build_ratings_leaderboard(supabase, club_id=str(session.get("club_id") or ""))
    sb_update(
        supabase,
        "session_ladder_sessions",
        {
            "published_at": datetime.now(timezone.utc).isoformat(),
            "recap_json": recap,
            "leaderboard_json": leaderboard,
            "updated_by": str(updated_by),
        },
        filters={"id": str(sessionId)},
    )
    out = _require_session(supabase, sessionId)
    out["recap"] = recap
    out["leaderboard"] = leaderboard
    out["attendance"] = attendance
    return out


def _apply_session_rating_updates(supabase: Any, *, session_id: str, updated_by: str) -> dict[str, Any]:
    session = _require_session(supabase, session_id)
    if session.get("ratings_applied_at"):
        return {"applied": False, "reason": "already_applied"}

    games = sorted(
        _fetch_rows(supabase, "session_ladder_games", session_id=str(session_id)),
        key=lambda g: (str(g.get("court_pod_id") or ""), int(g.get("game_number") or 0)),
    )
    scored_games = [g for g in games if g.get("score_a") is not None and g.get("score_b") is not None]

    players = {int(row.get("id")): dict(row) for row in _fetch_rows(supabase, "players", club_id=str(session.get("club_id") or ""))}
    current = {pid: float(row.get("rating") or 1200.0) for pid, row in players.items()}
    delta_by_player: dict[int, float] = {}

    for game in scored_games:
        team_a = [int(x) for x in list(game.get("team_a_player_ids") or [])]
        team_b = [int(x) for x in list(game.get("team_b_player_ids") or [])]
        if len(team_a) != 2 or len(team_b) != 2:
            continue
        avg_a = sum(float(current.get(pid, 1200.0)) for pid in team_a) / 2.0
        avg_b = sum(float(current.get(pid, 1200.0)) for pid in team_b) / 2.0
        d_a, d_b = calculate_hybrid_elo(avg_a, avg_b, int(game.get("score_a") or 0), int(game.get("score_b") or 0))
        for pid in team_a:
            delta_by_player[pid] = float(delta_by_player.get(pid, 0.0)) + float(d_a)
            current[pid] = float(current.get(pid, 1200.0)) + float(d_a)
        for pid in team_b:
            delta_by_player[pid] = float(delta_by_player.get(pid, 0.0)) + float(d_b)
            current[pid] = float(current.get(pid, 1200.0)) + float(d_b)

    history_rows = 0
    league_id = str(session.get("league_id") or "")
    for pid, delta in delta_by_player.items():
        before = float(players.get(pid, {}).get("rating") or 1200.0)
        after = float(before) + float(delta)
        sb_update(
            supabase,
            "players",
            {"rating": after, "updated_by": str(updated_by)},
            filters={"id": int(pid), "club_id": str(session.get("club_id") or "")},
            derived_from_match_history=True,
        )
        sb_upsert(
            supabase,
            "league_ratings",
            {
                "club_id": str(session.get("club_id") or ""),
                "player_id": int(pid),
                "league_name": league_id,
                "rating": after,
            },
            conflict="club_id,player_id,league_name",
            derived_from_match_history=True,
        )
        sb_upsert(
            supabase,
            "session_ladder_rating_history",
            {
                "club_id": str(session.get("club_id") or ""),
                "session_id": str(session_id),
                "player_id": int(pid),
                "league_id": league_id,
                "rating_before": before,
                "rating_after": after,
                "rating_delta": float(delta),
                "created_by": str(updated_by),
            },
            conflict="session_id,player_id",
        )
        history_rows += 1

    sb_update(
        supabase,
        "session_ladder_sessions",
        {"ratings_applied_at": datetime.now(timezone.utc).isoformat(), "updated_by": str(updated_by)},
        filters={"id": str(session_id)},
    )
    return {"applied": True, "players_updated": len(delta_by_player), "rating_history_rows": history_rows}


def _apply_attendance_for_session(supabase: Any, *, session_id: str, updated_by: str) -> dict[str, Any]:
    session = _require_session(supabase, session_id)
    roster = _fetch_rows(supabase, "session_ladder_roster_entries", session_id=str(session_id))
    attended = [
        int(r.get("player_id"))
        for r in roster
        if str(r.get("status") or "").upper() in {"CHECKED_IN", "WALK_IN"}
    ]
    new_records = 0
    for pid in attended:
        existing = _fetch_rows(supabase, "session_ladder_attendance", session_id=str(session_id), player_id=int(pid))
        if existing:
            continue
        sb_insert(
            supabase,
            "session_ladder_attendance",
            {
                "club_id": str(session.get("club_id") or ""),
                "league_id": str(session.get("league_id") or ""),
                "season_id": str(session.get("season_id") or "") if session.get("season_id") else None,
                "session_id": str(session_id),
                "player_id": int(pid),
                "created_by": str(updated_by),
            },
        )
        new_records += 1

        existing_progress = _fetch_rows(
            supabase,
            "session_ladder_awards_attendance",
            club_id=str(session.get("club_id") or ""),
            league_id=str(session.get("league_id") or ""),
            season_id=str(session.get("season_id") or "") if session.get("season_id") else None,
            player_id=int(pid),
        )
        current_sessions = int(existing_progress[0].get("sessions_attended") or 0) if existing_progress else 0
        sb_upsert(
            supabase,
            "session_ladder_awards_attendance",
            {
                "club_id": str(session.get("club_id") or ""),
                "league_id": str(session.get("league_id") or ""),
                "season_id": str(session.get("season_id") or "") if session.get("season_id") else None,
                "player_id": int(pid),
                "sessions_attended": current_sessions + 1,
                "updated_by": str(updated_by),
            },
            conflict="club_id,league_id,season_id,player_id",
        )

    return {"attended_players": len(attended), "new_attendance_records": new_records}


def _build_session_recap(supabase: Any, *, session_id: str) -> dict[str, Any]:
    session = _require_session(supabase, session_id)
    roster = _fetch_rows(supabase, "session_ladder_roster_entries", session_id=str(session_id))
    games = _fetch_rows(supabase, "session_ladder_games", session_id=str(session_id))
    scored_games = [g for g in games if g.get("score_a") is not None and g.get("score_b") is not None]
    history = _fetch_rows(supabase, "session_ladder_rating_history", session_id=str(session_id))
    top_gainers = sorted(history, key=lambda row: float(row.get("rating_delta") or 0.0), reverse=True)[:5]
    return {
        "session_id": str(session_id),
        "state": str(session.get("state") or ""),
        "players": len(roster),
        "games_scored": len(scored_games),
        "top_gainers": top_gainers,
    }


def _build_ratings_leaderboard(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _fetch_rows(supabase, "players", club_id=str(club_id))
    ranked = sorted(rows, key=lambda row: float(row.get("rating") or 0.0), reverse=True)
    return [
        {
            "rank": idx,
            "player_id": int(row.get("id") or 0),
            "name": str(row.get("name") or f"#{row.get('id')}") if row.get("id") is not None else "",
            "rating": float(row.get("rating") or 0.0),
        }
        for idx, row in enumerate(ranked[:20], start=1)
    ]


def _transition_state(supabase: Any, *, sessionId: str, to_state: str, updated_by: str) -> dict[str, Any]:
    session = _require_session(supabase, sessionId)
    current = str(session.get("state") or "").upper()
    target = str(to_state).upper()
    if current == target:
        return session
    if not canTransition(current, target):
        raise RuntimeError(f"Invalid state transition: {current} -> {target}")

    sb_update(
        supabase,
        "session_ladder_sessions",
        {"state": target, "updated_by": str(updated_by)},
        filters={"id": str(sessionId)},
    )
    session["state"] = target
    session["updated_by"] = str(updated_by)
    return session
