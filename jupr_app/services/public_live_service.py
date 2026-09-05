from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.live_beta_engine import (
    league_aggregate_standings,
    league_round_summary,
    match_is_scored,
    resolve_display_name,
    round_robin_current_round_number,
    round_robin_standings,
    tournament_bracket_rows,
    tournament_champion,
)

PUBLIC_LIVE_STATUSES = {"active", "completed"}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_expired(row: dict[str, Any], *, now: datetime | None = None) -> bool:
    expires_at = _parse_datetime(row.get("expires_at"))
    if expires_at is None:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expires_at <= current.astimezone(timezone.utc)


def is_public_live_session_row(row: dict[str, Any], *, now: datetime | None = None) -> bool:
    status = str((row or {}).get("status") or "").strip().lower()
    if status not in PUBLIC_LIVE_STATUSES:
        return False
    if status == "active" and _is_expired(row, now=now):
        return False
    return True


def _state_payload(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    return state if isinstance(state, dict) else {}


def _page_state(row: dict[str, Any]) -> dict[str, Any]:
    state = _state_payload(row)
    page_state = state.get("page_state")
    return page_state if isinstance(page_state, dict) else {}


def _event(row: dict[str, Any]) -> dict[str, Any]:
    page_state = _page_state(row)
    event = page_state.get("event")
    return event if isinstance(event, dict) else {}


def _event_name(row: dict[str, Any]) -> str:
    state = _state_payload(row)
    event = _event(row)
    page_state = _page_state(row)
    return str(
        row.get("title")
        or state.get("event_name")
        or event.get("name")
        or page_state.get("event_name")
        or "JUPR Live Session"
    )


def _event_type(row: dict[str, Any]) -> str:
    state = _state_payload(row)
    event = _event(row)
    page_state = _page_state(row)
    return str(event.get("type") or state.get("event_type") or page_state.get("type_label") or "").strip()


def _participant_names(event: dict[str, Any], match_id: str, ids: list[Any]) -> list[str]:
    names: list[str] = []
    for participant_id in ids or []:
        name = resolve_display_name(event, match_id, str(participant_id))
        if name:
            names.append(str(name))
    return names


def _score(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _winner_label(team_a: list[str], team_b: list[str], score_a: int | None, score_b: int | None) -> str | None:
    if score_a is None or score_b is None or score_a == score_b:
        return None
    return " / ".join(team_a if score_a > score_b else team_b)


def _public_match(
    event: dict[str, Any],
    match: dict[str, Any],
    *,
    round_number: int,
    court_number: int | None = None,
    mini_round_number: int | None = None,
) -> dict[str, Any]:
    match_id = str(match.get("id") or f"r{round_number}-m{match.get('slot', '')}")
    team_a = _participant_names(event, match_id, [str(x) for x in (match.get("teamA") or [])])
    team_b = _participant_names(event, match_id, [str(x) for x in (match.get("teamB") or [])])
    score_a = _score(match.get("scoreA"))
    score_b = _score(match.get("scoreB"))
    fallback_label = (
        f"Court {court_number}, game {mini_round_number}"
        if court_number is not None and mini_round_number is not None
        else f"Court {court_number}"
        if court_number is not None
        else f"Round {round_number} match"
    )
    payload: dict[str, Any] = {
        "id": match_id,
        "round_number": int(round_number),
        "label": str(match.get("desc") or match.get("name") or fallback_label),
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "is_scored": match_is_scored(match),
        "winner": _winner_label(team_a, team_b, score_a, score_b),
    }
    if court_number is not None:
        payload["court_number"] = int(court_number)
    if mini_round_number is not None:
        payload["mini_round_number"] = int(mini_round_number)
    return payload


def _round_robin_rounds(event: dict[str, Any]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        number = int(round_data.get("number") or 0)
        rounds.append(
            {
                "number": number,
                "matches": [
                    _public_match(event, match, round_number=number)
                    for match in (round_data.get("matches") or [])
                ],
            }
        )
    return rounds


def _league_rounds(event: dict[str, Any]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        round_number = int(round_data.get("number") or 0)
        courts: list[dict[str, Any]] = []
        round_matches: list[dict[str, Any]] = []
        for court in round_data.get("courts") or []:
            court_number = int(court.get("courtNumber") or 0)
            court_matches: list[dict[str, Any]] = []
            for mini_round in court.get("miniRounds") or []:
                mini_number = int(mini_round.get("number") or 0)
                for match in mini_round.get("matches") or []:
                    projected = _public_match(
                        event,
                        match,
                        round_number=round_number,
                        court_number=court_number,
                        mini_round_number=mini_number,
                    )
                    court_matches.append(projected)
                    round_matches.append(projected)
            courts.append(
                {
                    "court_number": court_number,
                    "size": int(court.get("size") or 0),
                    "matches": court_matches,
                }
            )
        rounds.append({"number": round_number, "courts": courts, "matches": round_matches})
    return rounds


def _tournament_team_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(team.get("id")): dict(team) for team in (event.get("teams") or []) if team.get("id")}


def _tournament_rounds(event: dict[str, Any]) -> list[dict[str, Any]]:
    team_map = _tournament_team_map(event)
    rounds: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        round_number = int(round_data.get("number") or 0)
        matches: list[dict[str, Any]] = []
        for match in round_data.get("matches") or []:
            slot = int(match.get("slot") or 0)
            team_a = team_map.get(str(match.get("participantAId")), {})
            team_b = team_map.get(str(match.get("participantBId")), {})
            winner = team_map.get(str(match.get("winnerId")), {})
            matches.append(
                {
                    "id": f"r{round_number}-s{slot}",
                    "round_number": round_number,
                    "slot": slot,
                    "label": str(match.get("name") or f"Round {round_number} Match {slot}"),
                    "team_a": [str(team_a.get("name") or "TBD")],
                    "team_b": [str(team_b.get("name") or "TBD")],
                    "score_a": _score(match.get("scoreA")),
                    "score_b": _score(match.get("scoreB")),
                    "is_scored": match_is_scored(match),
                    "winner": str(winner.get("name") or "") or None,
                }
            )
        rounds.append({"number": round_number, "matches": matches})
    return rounds


def _standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = str(event.get("type") or "")
    try:
        if event_type == "round_robin":
            return round_robin_standings(event)
        if event_type == "league":
            return league_aggregate_standings(event)
    except Exception:
        return []
    return []


def _bracket(event: dict[str, Any]) -> dict[str, Any] | None:
    if str(event.get("type") or "") != "tournament":
        return None
    team_map = _tournament_team_map(event)
    champion_id = tournament_champion(event)
    champion = team_map.get(str(champion_id), {}).get("name") if champion_id else None
    try:
        rows = tournament_bracket_rows(event)
    except Exception:
        rows = []
    return {"champion": champion, "rows": rows}


def _current_round(event: dict[str, Any]) -> int | None:
    event_type = str(event.get("type") or "")
    try:
        if event_type == "round_robin":
            return int(round_robin_current_round_number(event))
        if event_type == "league":
            return int(event.get("currentRoundNumber") or 1)
        rounds = event.get("rounds") or []
        for round_data in rounds:
            if any(not match_is_scored(match) for match in (round_data.get("matches") or [])):
                return int(round_data.get("number") or 1)
        if rounds:
            return int(rounds[-1].get("number") or 1)
    except Exception:
        return None
    return None


def _rounds(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = str(event.get("type") or "")
    if event_type == "round_robin":
        return _round_robin_rounds(event)
    if event_type == "league":
        return _league_rounds(event)
    if event_type == "tournament":
        return _tournament_rounds(event)
    return []


def public_live_session_summary(row: dict[str, Any]) -> dict[str, Any]:
    event = _event(row)
    state = _state_payload(row)
    session_key = str(row.get("session_key") or state.get("session_key") or "")
    state_mode = str(state.get("mode") or "public_quick_session")
    return {
        "session_key": session_key,
        "title": _event_name(row),
        "status": str(row.get("status") or "active"),
        "version": int(row.get("version") or 1),
        "live_mode": "club_social" if state_mode == "public_club_social" else "quick",
        "event_type": _event_type(row),
        "current_round": _current_round(event) if event else None,
        "has_event": bool(event),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "last_seen_at": row.get("last_seen_at"),
        "expires_at": row.get("expires_at"),
        "completed_at": row.get("completed_at"),
    }


def public_live_session_detail(row: dict[str, Any]) -> dict[str, Any]:
    event = _event(row)
    state = _state_payload(row)
    summary = public_live_session_summary(row)
    participants = [
        {
            "id": str(participant.get("id") or ""),
            "name": str(participant.get("name") or ""),
            "player_id": participant.get("player_id"),
        }
        for participant in (event.get("participants") or [])
        if str(participant.get("id") or "")
    ]
    substitutions = [
        {
            "id": str(substitution.get("id") or ""),
            "scope": str(substitution.get("scope") or ""),
            "round_number": substitution.get("round_number"),
            "match_id": substitution.get("match_id"),
            "original_participant_id": str(substitution.get("original_participant_id") or ""),
            "original_player_name": str(substitution.get("original_player_name") or ""),
            "substitute_name": str(substitution.get("substitute_name") or ""),
            "affected_match_ids": [str(value) for value in (substitution.get("affected_match_ids") or [])],
        }
        for substitution in (event.get("substitutions") or [])
    ]
    social = state.get("social") if isinstance(state.get("social"), dict) else {}
    submission = social.get("submission") if isinstance(social.get("submission"), dict) else None
    payload = {
        **summary,
        "rounds": _rounds(event) if event else [],
        "standings": _standings(event) if event else [],
        "bracket": _bracket(event) if event else None,
        "participants": participants,
        "substitutions": substitutions,
        "social": {
            "enabled": bool(social.get("enabled")),
            "skill_levels": [str(value) for value in (social.get("skill_levels") or [])],
            "submission_status": str(submission.get("status") or "") or None if submission else None,
        },
    }
    if str(event.get("type") or "") == "league":
        try:
            payload["court_standings"] = league_round_summary(event)
        except Exception:
            payload["court_standings"] = []
    return payload


def public_live_sessions_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = 20,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for row in rows or []:
        if not is_public_live_session_row(row, now=now):
            continue
        summary = public_live_session_summary(row)
        if not summary.get("session_key"):
            continue
        sessions.append(summary)
        if len(sessions) >= int(limit):
            break
    return sessions
