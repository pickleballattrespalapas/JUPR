"""Read-only public archive of completed, public round robins.

The directory projects small JSON fields in PostgREST. Full schedules and
standings are read/computed only when someone opens one session.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from jupr_app.domain.adaptive_play_engine import generator_event_standings
from jupr_app.domain.moneyball_expectations import is_moneyball


SUMMARY_FIELDS = (
    "session_key,title,created_at,completed_at,"
    "play_format:state->>play_format,"
    "participant_count:state->page_state->>participant_count,"
    "event_format:state->page_state->event->>eventFormat,"
    "rating_mode:state->page_state->event->>ratingMode,"
    "scoring_mode:state->page_state->event->>scoringMode"
)


def _archive_query(db: Any, club_id: str, fields: str, kind: str = "all") -> Any:
    # Shared admin sessions and officially published results may be discovered.
    # Finishing a private admin draft alone does not publish it. Public-created
    # sessions already have a public read route, including after completion.
    visibility = (
        "state->>mode.eq.public_play_generator,"
        "state->generator_sharing->>enabled.eq.true,"
        "state->official_publish->>published_at.not.is.null,"
        "state->generator_submission->>status.eq.approved"
    )
    conditions = [
        f"or({visibility})",
        "or(state->page_state->event->>hubSupplementalDraft.is.null,"
        "state->page_state->event->>hubSupplementalDraft.eq.false)",
    ]
    if kind == "moneyball":
        conditions.append("state->page_state->event->>eventFormat.eq.moneyball_expectations")
    elif kind == "round_robin":
        conditions.append("or(state->page_state->event->>eventFormat.is.null,"
                          "state->page_state->event->>eventFormat.neq.moneyball_expectations)")
    return (db.table("live_sessions").select(fields)
            .eq("club_id", str(club_id)).eq("status", "completed")
            .eq("state->>generator_kind", "round_robin")
            .in_("state->>mode", ["admin_play_generator", "public_play_generator"])
            .or_(f"and({','.join(conditions)})"))


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_key": row["session_key"],
        "title": str(row.get("title") or "Round robin"),
        "created_at": row.get("created_at"),
        "completed_at": row.get("completed_at"),
        "play_format": row.get("play_format") or "doubles",
        "participant_count": int(row.get("participant_count") or 0),
        "kind": "moneyball" if row.get("event_format") == "moneyball_expectations" else "round_robin",
        "rating_mode": row.get("rating_mode") or "unrated",
        "scoring_mode": row.get("scoring_mode") or "scored",
    }


def list_round_robin_history(db: Any, *, club_id: str, kind: str = "all",
                            from_date: date | None = None, to_date: date | None = None,
                            offset: int = 0, limit: int = 24,
                            timezone: str = "America/Mazatlan") -> dict[str, Any]:
    if from_date and to_date and from_date > to_date:
        raise ValueError("Choose an end date on or after the start date.")
    query = _archive_query(db, club_id, SUMMARY_FIELDS, kind)
    tz = ZoneInfo(timezone)
    if from_date:
        query = query.gte("created_at", datetime.combine(from_date, time.min, tz).isoformat())
    if to_date and to_date < date.max:
        query = query.lt("created_at", datetime.combine(to_date + timedelta(days=1), time.min, tz).isoformat())
    # One extra lightweight row determines whether there is another page.
    rows = (query.order("created_at", desc=True).order("session_key")
            .range(offset, offset + limit).execute().data or [])
    return {"sessions": [_summary(row) for row in rows[:limit]],
            "has_more": len(rows) > limit, "offset": offset, "limit": limit}


def get_round_robin_history(db: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    rows = (_archive_query(db, club_id, "session_key,title,created_at,completed_at,state")
            .eq("session_key", session_key).limit(1).execute().data or [])
    if not rows:
        raise LookupError("This round robin is not available.")
    row = rows[0]
    state = row.get("state") or {}
    event = (state.get("page_state") or {}).get("event") or {}
    summary = _summary({**row,
        "play_format": event.get("playFormat"), "event_format": event.get("eventFormat"),
        "participant_count": len(event.get("participants") or []),
        "rating_mode": event.get("ratingMode"), "scoring_mode": event.get("scoringMode")})
    names = {str(p["id"]): str(p.get("name") or "Player") for p in event.get("participants") or []}
    rounds = []
    for r in event.get("rounds") or []:
        matches = r.get("matches") or [m for c in r.get("courts") or [] for m in c.get("matches") or []]
        rounds.append({"number": r["number"], "status": r.get("status"), "matches": [
            {"court": m.get("court"),
             "side_a": [names.get(str(pid), "Player") for pid in m.get("sideA") or m.get("teamA") or []],
             "side_b": [names.get(str(pid), "Player") for pid in m.get("sideB") or m.get("teamB") or []],
             "score_a": m.get("scoreA"), "score_b": m.get("scoreB"), "status": m.get("status")}
            for m in matches]})
    # Never serialize state, organizer identity, edit hashes, or handoff material.
    return {"session": {**summary, "standings": generator_event_standings(event),
                        "standings_sort": "moneyball_expectations" if is_moneyball(event) else event.get("standingsSort", "wins"),
                        "rounds": rounds}}
