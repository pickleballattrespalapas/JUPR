from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

import pandas as pd

from jupr_app.domain.event_tags import (
    SKILL_LEVEL_OPTIONS,
    derive_default_date_tags,
    merge_event_tags,
    normalize_skill_levels,
)
from jupr_app.domain.live_beta_engine import (
    league_aggregate_standings,
    match_is_scored,
    matches_for_round,
    normalize_name,
    round_robin_standings,
)

SOCIAL_SKILL_LEVEL_OPTIONS = SKILL_LEVEL_OPTIONS
SOCIAL_TABLES_INSTALL_MESSAGE = (
    "Club Social tables are not installed in Supabase yet. Apply the social live events migration first."
)
_SOCIAL_TABLE_NAMES = ("club_people", "live_events", "live_event_participants", "live_event_matches")


class SocialTablesNotInstalledError(RuntimeError):
    """Raised when Club Social persistence tables are missing in Supabase."""


def _error_payload_text(exc: Exception) -> str:
    pieces = [str(exc)]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("text",):
            value = getattr(response, attr, None)
            if value:
                pieces.append(str(value))
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                payload = json_fn()
            except Exception:
                payload = None
            if payload:
                pieces.append(str(payload))
    return " | ".join(pieces).lower()


def is_missing_social_tables_error(exc: Exception) -> bool:
    payload = _error_payload_text(exc)
    if not payload:
        return False
    has_missing_code = "pgrst205" in payload or "42p01" in payload
    has_table_marker = "table" in payload or "relation" in payload
    mentions_social_table = any(table in payload for table in _SOCIAL_TABLE_NAMES)
    return has_missing_code and has_table_marker and mentions_social_table


def submitted_by_name_from_row(row: dict) -> str:
    return normalize_name((row or {}).get("submitted_by_name") or (row or {}).get("submitted_by"))


def _upsert_live_event_row(supabase, payload: dict) -> None:
    attempts = [dict(payload)]
    if "submitted_by" in payload:
        attempts.append({k: v for k, v in payload.items() if k != "submitted_by"})
    if "submitted_by_name" in payload:
        attempts.append({k: v for k, v in payload.items() if k != "submitted_by_name"})
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            supabase.table("live_events").upsert(
                attempt,
                on_conflict="club_id,source_event_uid",
            ).execute()
            return
        except Exception as exc:
            payload_text = _error_payload_text(exc)
            if "submitted_by" in payload_text or "submitted_by_name" in payload_text or "schema cache" in payload_text:
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to upsert live_events row.")


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


def normalized_player_name_map(df_players_all: pd.DataFrame | None) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    if df_players_all is None or df_players_all.empty:
        return result
    for _, row in df_players_all.iterrows():
        player_id = row.get("id")
        if pd.isna(player_id):
            continue
        normalized = normalize_person_name(row.get("name"))
        if not normalized:
            continue
        result.setdefault(normalized, []).append(
            {
                "id": int(player_id),
                "name": str(row.get("name") or "").strip(),
            }
        )
    return result


def find_exact_player_link_candidates(
    club_people_rows: list[dict],
    player_name_map: dict[str, list[dict[str, object]]],
) -> dict[str, int]:
    matches: dict[str, int] = {}
    for row in club_people_rows or []:
        if row.get("linked_player_id") is not None:
            continue
        club_person_id = row.get("id")
        if club_person_id is None:
            continue
        normalized = normalize_person_name(row.get("normalized_name") or row.get("display_name"))
        candidates = player_name_map.get(normalized, [])
        if len(candidates) == 1:
            matches[str(club_person_id)] = int(candidates[0]["id"])
    return matches


def auto_link_exact_matches(
    supabase,
    *,
    club_id: str,
    club_people_rows: list[dict],
    df_players_all: pd.DataFrame | None,
) -> dict[str, int]:
    player_map = normalized_player_name_map(df_players_all)
    candidates = find_exact_player_link_candidates(club_people_rows, player_map)
    linked_count = 0
    for club_person_id, player_id in candidates.items():
        supabase.table("club_people").update({"linked_player_id": int(player_id)}).eq("club_id", club_id).eq(
            "id", club_person_id
        ).execute()
        linked_count += 1
    return {
        "linked_count": linked_count,
        "candidate_count": len(candidates),
        "skipped_count": max(0, len(club_people_rows or []) - len(candidates)),
    }


def social_person_rollup_rows(supabase, club_id: str) -> list[dict]:
    club_people = (
        supabase.table("club_people")
        .select("id,display_name,normalized_name,linked_player_id,first_seen_on,last_seen_on")
        .eq("club_id", club_id)
        .order("last_seen_on", desc=True)
        .execute()
        .data
        or []
    )
    if not club_people:
        return []

    cp_ids = [str(row["id"]) for row in club_people if row.get("id")]
    participants = (
        supabase.table("live_event_participants")
        .select("id,event_id,club_person_id")
        .in_("club_person_id", cp_ids)
        .execute()
        .data
        or []
    )
    participant_ids = [str(row["id"]) for row in participants if row.get("id")]
    matches = []
    if participant_ids:
        matches = (
            supabase.table("live_event_matches")
            .select(
                "id,t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id,score_t1,score_t2"
            )
            .or_(
                ",".join(
                    [
                        f"t1_p1_participant_id.in.({','.join(participant_ids)})",
                        f"t1_p2_participant_id.in.({','.join(participant_ids)})",
                        f"t2_p1_participant_id.in.({','.join(participant_ids)})",
                        f"t2_p2_participant_id.in.({','.join(participant_ids)})",
                    ]
                )
            )
            .execute()
            .data
            or []
        )

    participant_to_person = {
        str(row["id"]): str(row["club_person_id"])
        for row in participants
        if row.get("id") and row.get("club_person_id")
    }
    event_sets: dict[str, set[str]] = {str(cp["id"]): set() for cp in club_people if cp.get("id")}
    for row in participants:
        cp_id = str(row.get("club_person_id") or "")
        event_id = str(row.get("event_id") or "")
        if cp_id and event_id:
            event_sets.setdefault(cp_id, set()).add(event_id)

    match_counts: dict[str, int] = {str(cp["id"]): 0 for cp in club_people if cp.get("id")}
    for row in matches:
        scored = (int(row.get("score_t1") or 0) + int(row.get("score_t2") or 0)) > 0
        if not scored:
            continue
        seen_people: set[str] = set()
        for key in ("t1_p1_participant_id", "t1_p2_participant_id", "t2_p1_participant_id", "t2_p2_participant_id"):
            participant_id = str(row.get(key) or "")
            cp_id = participant_to_person.get(participant_id)
            if cp_id:
                seen_people.add(cp_id)
        for cp_id in seen_people:
            match_counts[cp_id] = match_counts.get(cp_id, 0) + 1

    rows: list[dict] = []
    for cp in club_people:
        cp_id = str(cp.get("id") or "")
        if not cp_id:
            continue
        rows.append(
            {
                **cp,
                "social_event_count": len(event_sets.get(cp_id, set())),
                "social_match_count": int(match_counts.get(cp_id, 0)),
            }
        )
    return rows


def resolve_or_create_club_person(
    ctx,
    *,
    display_name: str,
    event_date: str,
    club_id: str | None = None,
    explicit_linked_player_id: int | None = None,
    allow_name_auto_link: bool = True,
) -> tuple[dict, bool, bool]:
    """Returns (club_person_row, created_new, matched_competitive_player)."""
    supabase = ctx.supabase
    club_id = str(club_id or ctx.club_id)
    display = normalize_name(display_name)
    normalized = normalize_person_name(display)
    if not normalized:
        raise ValueError("Participant display name is required.")

    linked_player_id = None
    if explicit_linked_player_id is not None:
        linked_player_id = int(explicit_linked_player_id)
    elif allow_name_auto_link:
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


def social_league_match_rows_from_event(event: dict) -> list[dict]:
    rows: list[dict] = []
    played_on = _event_date_from_event(event)
    for round_data in event.get("rounds") or []:
        round_number = int(round_data.get("number") or 0) or None
        for match in matches_for_round(event, int(round_data.get("number") or 0)):
            if not match_is_scored(match):
                continue
            team_a = [str(x) for x in (match.get("teamA") or [])]
            team_b = [str(x) for x in (match.get("teamB") or [])]
            if len(team_a) != 2 or len(team_b) != 2:
                continue
            rows.append(
                {
                    "match_key": str(match.get("id")),
                    "played_on": played_on,
                    "round_number": round_number,
                    "court_number": int(match.get("courtNumber") or 0) or None,
                    "mini_round_number": int(match.get("miniRoundNumber") or 0) or None,
                    "t1_p1_key": team_a[0],
                    "t1_p2_key": team_a[1],
                    "t2_p1_key": team_b[0],
                    "t2_p2_key": team_b[1],
                    "score_t1": int(match.get("scoreA") or 0),
                    "score_t2": int(match.get("scoreB") or 0),
                }
            )
    return rows


def build_social_event_summary(
    event: dict,
    *,
    match_count: int,
    event_tags: dict | None = None,
    skill_levels: list[str] | None = None,
) -> dict:
    event_type = str(event.get("type") or "")
    standings = (
        round_robin_standings(event)
        if event_type == "round_robin"
        else league_aggregate_standings(event)
    )
    leader = standings[0] if standings else None
    return {
        "participant_count": len(event.get("participants") or []),
        "match_count": int(match_count),
        "event_type": event_type,
        "schedule_mode": str(event.get("scheduleMode") or ""),
        "event_tags": merge_event_tags(
            event.get("event_tags"),
            event_tags
            if isinstance(event_tags, dict)
            else {"skill_levels": skill_levels if skill_levels is not None else None},
            default_skill_all=True,
        ),
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


def _status_for_submission_mode(submission_mode: str) -> str:
    return "saved" if str(submission_mode or "").strip().lower() == "admin" else "pending"


def _moderated_by(ctx, fallback: str = "admin") -> str:
    name = normalize_name(getattr(ctx, "admin_name", "") or getattr(ctx, "user_name", ""))
    return name or fallback


def _saved_rounds(event: dict) -> list[int | str]:
    if str(event.get("type") or "") == "round_robin":
        return ["rr"]
    rounds: set[int] = set()
    for round_data in event.get("rounds") or []:
        round_number = int(round_data.get("number") or 0)
        if round_number <= 0:
            continue
        if any(match_is_scored(match) for match in matches_for_round(event, round_number)):
            rounds.add(round_number)
    return sorted(rounds)


def save_social_live_event(
    ctx,
    event: dict,
    *,
    target_club_id: str,
    submission_mode: str,
    host_name: str,
    skill_levels: object = None,
) -> dict:
    event_type = str(event.get("type") or "")
    if event_type not in {"round_robin", "league"}:
        raise ValueError("Social saves only support round_robin and league events.")
    supabase = ctx.supabase
    club_id = str(target_club_id or "").strip()
    if not club_id:
        raise ValueError("target_club_id is required for social saves.")
    event_date = _event_date_from_event(event)
    source_event_uid = _event_uid(event)
    participants = list(event.get("participants") or [])

    created_people_count = 0
    linked_existing_players_count = 0
    participant_rows: list[dict] = []

    try:
        for participant in participants:
            display_name = str(participant.get("name") or participant.get("id") or "")
            explicit_player_id = participant.get("player_id")
            match_status = str(participant.get("match_status") or "").strip().lower()
            auto_link_enabled = match_status not in {"new_social", "new social person"}
            club_person, created_new, matched_player = resolve_or_create_club_person(
                ctx,
                display_name=display_name,
                event_date=event_date,
                club_id=club_id,
                explicit_linked_player_id=(
                    int(explicit_player_id)
                    if explicit_player_id is not None and str(explicit_player_id).strip() != ""
                    else None
                ),
                allow_name_auto_link=auto_link_enabled,
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

        match_rows = (
            social_round_robin_match_rows_from_event(event)
            if event_type == "round_robin"
            else social_league_match_rows_from_event(event)
        )
        default_date_tags = derive_default_date_tags(event_date=event_date)
        normalized_event_tags = merge_event_tags(
            event.get("event_tags"),
            {
                "skill_levels": skill_levels if skill_levels is not None else None,
                "date_tags": [*(event.get("event_tags") or {}).get("date_tags", []), *default_date_tags],
            },
            default_skill_all=True,
        )
        summary_json = build_social_event_summary(
            event,
            match_count=len(match_rows),
            event_tags=normalized_event_tags,
        )
        normalized_submission_mode = str(submission_mode or "").strip().lower() or "public"
        status = _status_for_submission_mode(normalized_submission_mode)
        submitted_by_name = normalize_name(host_name) or ("admin" if status == "saved" else "guest")
        moderated_at = datetime.now(timezone.utc).isoformat() if status == "saved" else None
        moderated_by = _moderated_by(ctx) if status == "saved" else None

        upsert_payload = {
            "club_id": club_id,
            "source_event_uid": source_event_uid,
            "name": str(event.get("name") or "JUPR Live Club Social"),
            "event_type": event_type,
            "result_mode": "social_unrated",
            "event_date": event_date,
            "status": status,
            "submission_mode": normalized_submission_mode,
            "submitted_by_name": submitted_by_name,
            "moderated_at": moderated_at,
            "moderated_by": moderated_by,
            "rejection_reason": None,
            "raw_event_json": {**event, "event_tags": normalized_event_tags},
            "summary_json": summary_json,
        }

        _upsert_live_event_row(supabase, upsert_payload)

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
            "status": status,
            "submission_mode": normalized_submission_mode,
            "submitted_by_name": submitted_by_name,
            "saved_rounds": _saved_rounds(event),
            "participant_count": len(participant_rows),
            "match_count": len(match_rows),
            "created_people_count": created_people_count,
            "linked_existing_players_count": linked_existing_players_count,
        }
    except Exception as exc:
        if is_missing_social_tables_error(exc):
            raise SocialTablesNotInstalledError(SOCIAL_TABLES_INSTALL_MESSAGE) from exc
        raise


def save_social_round_robin(ctx, event: dict, *, saved_by: str = "admin") -> dict:
    return save_social_live_event(
        ctx,
        event,
        target_club_id=str(ctx.club_id),
        submission_mode="admin" if str(saved_by).strip().lower() == "admin" else "public",
        host_name=str(saved_by or "").strip() or "admin",
    )


def list_social_submissions_for_review(
    ctx,
    *,
    status: str = "pending",
    limit: int = 100,
) -> list[dict]:
    club_id = str(ctx.club_id)
    base_select = (
        "id,name,event_type,event_date,status,submission_mode,"
        "summary_json,raw_event_json,created_at,updated_at,rejection_reason,moderated_at,moderated_by"
    )
    try:
        rows = (
            ctx.supabase.table("live_events")
            .select(f"{base_select},submitted_by_name")
            .eq("club_id", club_id)
            .eq("result_mode", "social_unrated")
            .eq("status", status)
            .order("updated_at", desc=True)
            .limit(int(limit))
            .execute()
            .data
            or []
        )
    except Exception as exc:
        payload = _error_payload_text(exc)
        if "submitted_by_name" not in payload or "schema cache" not in payload:
            raise
        rows = (
            ctx.supabase.table("live_events")
            .select(f"{base_select},submitted_by")
            .eq("club_id", club_id)
            .eq("result_mode", "social_unrated")
            .eq("status", status)
            .order("updated_at", desc=True)
            .limit(int(limit))
            .execute()
            .data
            or []
        )
    normalized_rows = []
    for row in rows:
        payload = dict(row)
        payload["submitted_by_name"] = submitted_by_name_from_row(payload)
        normalized_rows.append(payload)
    return normalized_rows


def moderate_social_submission(
    ctx,
    *,
    event_id: str,
    action: str,
    rejection_reason: str = "",
) -> dict:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"approve", "reject"}:
        raise ValueError("action must be 'approve' or 'reject'.")
    status = "saved" if normalized_action == "approve" else "rejected"
    payload = {
        "status": status,
        "moderated_at": datetime.now(timezone.utc).isoformat(),
        "moderated_by": _moderated_by(ctx),
        "rejection_reason": (
            normalize_name(rejection_reason) if normalized_action == "reject" else None
        ),
    }
    rows = (
        ctx.supabase.table("live_events")
        .update(payload)
        .eq("id", str(event_id))
        .eq("club_id", str(ctx.club_id))
        .eq("result_mode", "social_unrated")
        .execute()
        .data
        or []
    )
    if not rows:
        raise RuntimeError("Unable to moderate social submission for this club.")
    return dict(rows[0])


def list_social_match_log_rows(
    supabase,
    *,
    club_id: str,
    limit: int = 1000,
) -> pd.DataFrame:
    club_id = str(club_id)
    event_rows = (
        supabase.table("live_events")
        .select("id,name,event_date,status,submission_mode")
        .eq("club_id", club_id)
        .eq("result_mode", "social_unrated")
        .order("event_date", desc=True)
        .limit(max(int(limit), 1))
        .execute()
        .data
        or []
    )
    if not event_rows:
        return pd.DataFrame()

    event_map = {str(row["id"]): dict(row) for row in event_rows if row.get("id")}
    event_ids = list(event_map.keys())
    if not event_ids:
        return pd.DataFrame()

    participant_rows = (
        supabase.table("live_event_participants")
        .select("id,event_id,club_person_id,participant_key")
        .in_("event_id", event_ids)
        .execute()
        .data
        or []
    )
    participant_to_person_id: dict[str, str] = {}
    club_person_ids: set[str] = set()
    for row in participant_rows:
        participant_id = row.get("id")
        person_id = row.get("club_person_id")
        if participant_id is None or person_id is None:
            continue
        participant_to_person_id[str(participant_id)] = str(person_id)
        club_person_ids.add(str(person_id))

    person_name_map: dict[str, str] = {}
    if club_person_ids:
        people_rows = (
            supabase.table("club_people")
            .select("id,display_name")
            .in_("id", list(club_person_ids))
            .execute()
            .data
            or []
        )
        person_name_map = {
            str(row["id"]): normalize_name(row.get("display_name"))
            for row in people_rows
            if row.get("id") is not None
        }

    match_rows = (
        supabase.table("live_event_matches")
        .select(
            "id,event_id,match_key,played_on,round_number,court_number,mini_round_number,"
            "t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id,"
            "score_t1,score_t2"
        )
        .in_("event_id", event_ids)
        .order("played_on", desc=True)
        .limit(max(int(limit), 1))
        .execute()
        .data
        or []
    )

    def _participant_name(participant_id: object) -> str:
        if participant_id is None:
            return ""
        person_id = participant_to_person_id.get(str(participant_id))
        if person_id is None:
            return ""
        return person_name_map.get(person_id, "")

    rows: list[dict] = []
    for row in match_rows:
        event_id = str(row.get("event_id") or "")
        event = event_map.get(event_id, {})
        rows.append(
            {
                "source_type": "social",
                "source_label": "Club Social (Unrated)",
                "id": row.get("id"),
                "social_match_id": row.get("id"),
                "event_id": event_id,
                "event_name": normalize_name(event.get("name")),
                "date": row.get("played_on") or event.get("event_date"),
                "played_on": row.get("played_on") or event.get("event_date"),
                "round_number": row.get("round_number"),
                "court_number": row.get("court_number"),
                "mini_round_number": row.get("mini_round_number"),
                "status": event.get("status"),
                "submission_mode": event.get("submission_mode"),
                "match_key": row.get("match_key"),
                "t1_p1": _participant_name(row.get("t1_p1_participant_id")),
                "t1_p2": _participant_name(row.get("t1_p2_participant_id")),
                "t2_p1": _participant_name(row.get("t2_p1_participant_id")),
                "t2_p2": _participant_name(row.get("t2_p2_participant_id")),
                "score_t1": row.get("score_t1"),
                "score_t2": row.get("score_t2"),
            }
        )
    return pd.DataFrame(rows)


def update_social_match_row(
    supabase,
    *,
    club_id: str,
    social_match_id: str,
    patch: dict,
    expected_current: dict | None = None,
) -> dict:
    club_id = str(club_id)
    social_match_id = str(social_match_id)
    match_rows = (
        supabase.table("live_event_matches")
        .select(
            "id,event_id,played_on,score_t1,score_t2,round_number,court_number,mini_round_number"
        )
        .eq("id", social_match_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not match_rows:
        raise RuntimeError("Social match not found.")
    event_id = str(match_rows[0].get("event_id") or "")
    if not event_id:
        raise RuntimeError("Social match is missing event linkage.")

    event_rows = (
        supabase.table("live_events")
        .select("id,name")
        .eq("id", event_id)
        .eq("club_id", club_id)
        .eq("result_mode", "social_unrated")
        .limit(1)
        .execute()
        .data
        or []
    )
    if not event_rows:
        raise RuntimeError("Social match does not belong to this club or is not a Club Social match.")

    allowed_match_fields = (
        "played_on",
        "score_t1",
        "score_t2",
        "round_number",
        "court_number",
        "mini_round_number",
    )
    numeric_match_fields = {
        "score_t1",
        "score_t2",
        "round_number",
        "court_number",
        "mini_round_number",
    }

    def _normalized_match_value(field: str, value: object) -> object:
        if value is None:
            return None
        if field in numeric_match_fields:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be an integer.") from exc
        isoformat = getattr(value, "isoformat", None)
        return str(isoformat() if callable(isoformat) else value).strip()

    requested_patch = dict(patch or {})
    expected_patch = dict(expected_current or {}) if expected_current is not None else None
    current_match = dict(match_rows[0])
    current_event = dict(event_rows[0])
    match_payload: dict[str, object] = {}
    before: dict[str, object] = {}
    after: dict[str, object] = {}

    if expected_patch is not None:
        unsupported_expected = set(expected_patch) - {*allowed_match_fields, "event_name"}
        if unsupported_expected:
            raise ValueError(f"Unsupported Club Social compare fields: {sorted(unsupported_expected)}")
        for field, expected_value in expected_patch.items():
            if field == "event_name":
                current_value = normalize_name(current_event.get("name"))
                normalized_expected = normalize_name(expected_value)
            else:
                current_value = _normalized_match_value(field, current_match.get(field))
                normalized_expected = _normalized_match_value(field, expected_value)
            if current_value != normalized_expected:
                raise RuntimeError("Club Social row changed before rollback; newer data was preserved.")

    for field in allowed_match_fields:
        if field not in requested_patch:
            continue
        current_value = _normalized_match_value(field, current_match.get(field))
        requested_value = _normalized_match_value(field, requested_patch.get(field))
        if requested_value == current_value:
            continue
        match_payload[field] = requested_value
        before[field] = current_value
        after[field] = requested_value

    event_name_payload: str | None = None
    event_name = requested_patch.get("event_name")
    if event_name is not None:
        current_event_name = normalize_name(current_event.get("name"))
        requested_event_name = normalize_name(event_name)
        if not requested_event_name:
            raise ValueError("Club Social event name cannot be blank.")
        if requested_event_name != current_event_name:
            event_name_payload = requested_event_name
            before["event_name"] = current_event_name
            after["event_name"] = requested_event_name

    if not after:
        raise ValueError("No Club Social changes detected.")
    if match_payload and event_name_payload is not None:
        raise ValueError("Update the Club Social event name separately from match fields.")

    if match_payload:
        match_update_query = supabase.table("live_event_matches").update(match_payload).eq(
            "id", social_match_id
        ).eq("event_id", event_id)
        for field, expected_value in (expected_patch or {}).items():
            if field not in allowed_match_fields:
                continue
            normalized_expected = _normalized_match_value(field, expected_value)
            if normalized_expected is None:
                match_update_query = match_update_query.is_(field, None)
            else:
                match_update_query = match_update_query.eq(field, normalized_expected)
        match_update_rows = match_update_query.select("id").execute().data or []
        if not match_update_rows:
            raise RuntimeError("Social match update did not persist.")

    if event_name_payload is not None:
        event_update_query = (
            supabase.table("live_events")
            .update({"name": event_name_payload})
            .eq("id", event_id)
            .eq("club_id", club_id)
            .eq("result_mode", "social_unrated")
        )
        if expected_patch is not None and "event_name" in expected_patch:
            event_update_query = event_update_query.eq(
                "name", normalize_name(expected_patch.get("event_name"))
            )
        event_update_rows = event_update_query.select("id").execute().data or []
        if not event_update_rows:
            raise RuntimeError("Club Social event name update did not persist.")

    return {
        "social_match_id": social_match_id,
        "event_id": event_id,
        "updated_match_fields": sorted(match_payload.keys()),
        "updated_event_name": event_name_payload is not None,
        "patch": dict(after),
        "before": before,
        "after": after,
    }


def snapshot_social_matches_for_delete(
    supabase,
    *,
    club_id: str,
    social_match_ids: Iterable[str],
) -> list[dict]:
    club_id = str(club_id)
    ids = list(dict.fromkeys(str(v).strip() for v in (social_match_ids or []) if str(v).strip()))
    if not ids:
        return []

    rows = (
        supabase.table("live_event_matches")
        .select("*")
        .in_("id", ids)
        .execute()
        .data
        or []
    )
    if not rows:
        return []

    event_ids = [str(r.get("event_id")) for r in rows if r.get("event_id")]
    allowed_events = set()
    if event_ids:
        allowed_rows = (
            supabase.table("live_events")
            .select("id")
            .eq("club_id", club_id)
            .eq("result_mode", "social_unrated")
            .in_("id", event_ids)
            .execute()
            .data
            or []
        )
        allowed_events = {str(r["id"]) for r in allowed_rows if r.get("id")}

    return [dict(row) for row in rows if str(row.get("event_id")) in allowed_events]


def delete_social_match_snapshots(
    supabase,
    *,
    snapshots: Iterable[dict],
) -> tuple[int, list[dict]]:
    rows = [dict(row) for row in (snapshots or []) if isinstance(row, dict)]
    allowed_ids = [str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()]
    allowed_event_ids = list(
        dict.fromkeys(
            str(row.get("event_id") or "").strip()
            for row in rows
            if str(row.get("event_id") or "").strip()
        )
    )
    if not allowed_ids or not allowed_event_ids:
        return 0, []

    deleted_rows = (
        supabase.table("live_event_matches")
        .delete()
        .in_("id", allowed_ids)
        .in_("event_id", allowed_event_ids)
        .select("*")
        .execute()
        .data
        or []
    )
    snapshots = [dict(row) for row in deleted_rows]
    return len(snapshots), snapshots


def delete_social_matches_with_snapshot(
    supabase,
    *,
    club_id: str,
    social_match_ids: Iterable[str],
) -> tuple[int, list[dict]]:
    snapshots = snapshot_social_matches_for_delete(
        supabase,
        club_id=club_id,
        social_match_ids=social_match_ids,
    )
    return delete_social_match_snapshots(supabase, snapshots=snapshots)


def restore_social_matches(
    supabase,
    *,
    club_id: str,
    snapshots: Iterable[dict],
) -> int:
    club_id = str(club_id)
    rows = [dict(row) for row in (snapshots or []) if isinstance(row, dict)]
    if not rows:
        return 0

    expected_ids = {str(row.get("id") or "") for row in rows if str(row.get("id") or "").strip()}
    event_ids = {str(row.get("event_id") or "") for row in rows if str(row.get("event_id") or "").strip()}
    if len(expected_ids) != len(rows) or not event_ids:
        raise RuntimeError("Club Social delete snapshot is incomplete.")

    allowed_rows = (
        supabase.table("live_events")
        .select("id")
        .eq("club_id", club_id)
        .eq("result_mode", "social_unrated")
        .in_("id", list(event_ids))
        .execute()
        .data
        or []
    )
    allowed_event_ids = {str(row.get("id") or "") for row in allowed_rows}
    if allowed_event_ids != event_ids:
        raise RuntimeError("Club Social delete snapshot no longer belongs to this club.")

    restored_rows = (
        supabase.table("live_event_matches")
        .insert(rows)
        .select("id")
        .execute()
        .data
        or []
    )
    restored_ids = {str(row.get("id") or "") for row in restored_rows}
    if restored_ids != expected_ids:
        raise RuntimeError("Club Social deleted rows could not be fully restored.")
    return len(restored_ids)


def delete_social_matches(
    supabase,
    *,
    club_id: str,
    social_match_ids: Iterable[str],
) -> int:
    deleted_count, _snapshots = delete_social_matches_with_snapshot(
        supabase,
        club_id=club_id,
        social_match_ids=social_match_ids,
    )
    return deleted_count
