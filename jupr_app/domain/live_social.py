from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from jupr_app.domain.live_beta_engine import normalize_name, round_robin_standings


def normalize_person_name(value: object) -> str:
    return normalize_name(value).casefold()


def _event_date_from_event(event: dict) -> str:
    value = str(event.get("eventDate") or "").strip()
    if value:
        return value
    return datetime.now(timezone.utc).date().isoformat()


def _normalized_name_to_player_id(name_to_id: dict[object, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_name, raw_id in (name_to_id or {}).items():
        key = normalize_person_name(raw_name)
        if not key:
            continue
        result[key] = int(raw_id)
    return result


def resolve_or_create_club_person(
    ctx,
    *,
    display_name: str,
    event_date: str,
) -> tuple[dict, bool, bool]:
    """Returns (club_person_row, created_new, matched_competitive_player)."""
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    display = normalize_name(display_name)
    normalized = normalize_person_name(display)
    if not normalized:
        raise ValueError("Participant display name is required.")

    player_map = _normalized_name_to_player_id(getattr(ctx, "name_to_id", {}))
    linked_player_id = player_map.get(normalized)

    if linked_player_id is not None:
        linked_rows = (
            supabase.table("club_people")
            .select("*")
            .eq("club_id", club_id)
            .eq("linked_player_id", linked_player_id)
            .execute()
            .data
            or []
        )
        if linked_rows:
            row = dict(linked_rows[0])
            supabase.table("club_people").update(
                {
                    "display_name": display,
                    "normalized_name": normalized,
                    "last_seen_on": event_date,
                    "first_seen_on": row.get("first_seen_on") or event_date,
                }
            ).eq("id", row["id"]).execute()
            row.update(
                {
                    "display_name": display,
                    "normalized_name": normalized,
                    "linked_player_id": int(linked_player_id),
                    "last_seen_on": event_date,
                    "first_seen_on": row.get("first_seen_on") or event_date,
                }
            )
            return row, False, True

        inserted = (
            supabase.table("club_people")
            .insert(
                {
                    "club_id": club_id,
                    "display_name": display,
                    "normalized_name": normalized,
                    "linked_player_id": int(linked_player_id),
                    "source": "social",
                    "first_seen_on": event_date,
                    "last_seen_on": event_date,
                }
            )
            .execute()
            .data
            or []
        )
        return dict(inserted[0]), True, True

    existing_rows = (
        supabase.table("club_people")
        .select("*")
        .eq("club_id", club_id)
        .eq("normalized_name", normalized)
        .execute()
        .data
        or []
    )
    if len(existing_rows) == 1:
        row = dict(existing_rows[0])
        supabase.table("club_people").update(
            {
                "display_name": display,
                "last_seen_on": event_date,
                "first_seen_on": row.get("first_seen_on") or event_date,
            }
        ).eq("id", row["id"]).execute()
        row.update(
            {
                "display_name": display,
                "first_seen_on": row.get("first_seen_on") or event_date,
                "last_seen_on": event_date,
            }
        )
        return row, False, False

    inserted = (
        supabase.table("club_people")
        .insert(
            {
                "club_id": club_id,
                "display_name": display,
                "normalized_name": normalized,
                "linked_player_id": None,
                "source": "social",
                "first_seen_on": event_date,
                "last_seen_on": event_date,
            }
        )
        .execute()
        .data
        or []
    )
    return dict(inserted[0]), True, False


def social_round_robin_match_rows_from_event(event: dict) -> list[dict]:
    rows: list[dict] = []
    played_on = _event_date_from_event(event)
    for round_data in event.get("rounds") or []:
        for match in round_data.get("matches") or []:
            if match.get("scoreA") is None or match.get("scoreB") is None:
                continue
            team_a = [str(x) for x in (match.get("teamA") or [])]
            team_b = [str(x) for x in (match.get("teamB") or [])]
            if len(team_a) != 2 or len(team_b) != 2:
                continue
            rows.append(
                {
                    "match_key": str(match.get("id")),
                    "played_on": played_on,
                    "round_number": int(round_data.get("number") or 0) or None,
                    "court_number": None,
                    "mini_round_number": None,
                    "t1_p1_key": team_a[0],
                    "t1_p2_key": team_a[1],
                    "t2_p1_key": team_b[0],
                    "t2_p2_key": team_b[1],
                    "score_t1": int(match.get("scoreA") or 0),
                    "score_t2": int(match.get("scoreB") or 0),
                }
            )
    return rows


def build_social_event_summary(event: dict, *, match_count: int) -> dict:
    standings = round_robin_standings(event)
    leader = standings[0] if standings else None
    return {
        "participant_count": len(event.get("participants") or []),
        "match_count": int(match_count),
        "schedule_mode": str(event.get("scheduleMode") or ""),
        "leader": (
            {
                "name": leader.get("name"),
                "wins": int(leader.get("wins") or 0),
                "differential": int(leader.get("differential") or 0),
            }
            if leader
            else None
        ),
    }


def _event_uid(event: dict) -> str:
    return str(
        event.get("sourceEventUid")
        or event.get("source_event_uid")
        or event.get("event_uid")
        or f"social-{uuid4()}"
    )


def save_social_round_robin(ctx, event: dict, *, saved_by: str = "admin") -> dict:
    if str(event.get("type") or "") != "round_robin":
        raise ValueError("Only round_robin social saves are supported.")

    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    event_date = _event_date_from_event(event)
    source_event_uid = _event_uid(event)
    participants = list(event.get("participants") or [])

    created_people_count = 0
    linked_existing_players_count = 0
    participant_rows: list[dict] = []

    for participant in participants:
        display_name = str(participant.get("name") or participant.get("id") or "")
        club_person, created_new, matched_player = resolve_or_create_club_person(
            ctx,
            display_name=display_name,
            event_date=event_date,
        )
        created_people_count += int(bool(created_new))
        linked_existing_players_count += int(bool(matched_player))
        participant_rows.append(
            {
                "participant_key": str(participant.get("id")),
                "club_person_id": club_person["id"],
                "linked_player_id": club_person.get("linked_player_id"),
                "display_name_snapshot": display_name,
                "seed": participant.get("seed"),
            }
        )

    match_rows = social_round_robin_match_rows_from_event(event)
    summary_json = build_social_event_summary(event, match_count=len(match_rows))

    upsert_payload = {
        "club_id": club_id,
        "source_event_uid": source_event_uid,
        "name": str(event.get("name") or "JUPR Live Social Round Robin"),
        "event_type": "round_robin",
        "result_mode": "social_unrated",
        "event_date": event_date,
        "status": "saved",
        "raw_event_json": event,
        "summary_json": summary_json,
        "updated_by": saved_by,
    }
    upsert_payload.pop("updated_by", None)

    supabase.table("live_events").upsert(
        upsert_payload,
        on_conflict="club_id,source_event_uid",
    ).execute()

    event_rows = (
        supabase.table("live_events")
        .select("id")
        .eq("club_id", club_id)
        .eq("source_event_uid", source_event_uid)
        .execute()
        .data
        or []
    )
    if not event_rows:
        raise RuntimeError("Unable to resolve saved live_events row.")
    event_id = str(event_rows[0]["id"])

    supabase.table("live_event_matches").delete().eq("event_id", event_id).execute()
    supabase.table("live_event_participants").delete().eq("event_id", event_id).execute()

    if participant_rows:
        supabase.table("live_event_participants").insert(
            [{**row, "event_id": event_id} for row in participant_rows]
        ).execute()

    participants_saved = (
        supabase.table("live_event_participants")
        .select("id,participant_key")
        .eq("event_id", event_id)
        .execute()
        .data
        or []
    )
    participant_key_to_id = {str(row["participant_key"]): str(row["id"]) for row in participants_saved}

    if match_rows:
        payloads = []
        for row in match_rows:
            payloads.append(
                {
                    "event_id": event_id,
                    "match_key": row["match_key"],
                    "played_on": row["played_on"],
                    "round_number": row["round_number"],
                    "court_number": row["court_number"],
                    "mini_round_number": row["mini_round_number"],
                    "t1_p1_participant_id": participant_key_to_id[row["t1_p1_key"]],
                    "t1_p2_participant_id": participant_key_to_id[row["t1_p2_key"]],
                    "t2_p1_participant_id": participant_key_to_id[row["t2_p1_key"]],
                    "t2_p2_participant_id": participant_key_to_id[row["t2_p2_key"]],
                    "score_t1": row["score_t1"],
                    "score_t2": row["score_t2"],
                }
            )
        supabase.table("live_event_matches").insert(payloads).execute()

    return {
        "event_id": event_id,
        "participant_count": len(participant_rows),
        "match_count": len(match_rows),
        "created_people_count": created_people_count,
        "linked_existing_players_count": linked_existing_players_count,
    }
