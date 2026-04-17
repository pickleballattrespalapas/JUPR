from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from jupr_app.domain.event_tags import (
    derive_default_date_tags,
    merge_event_tags,
)
from jupr_app.domain.live_beta_engine import match_is_scored, matches_for_round, normalize_name
from jupr_app.domain.live_social import (
    SOCIAL_TABLES_INSTALL_MESSAGE,
    SocialTablesNotInstalledError,
    build_social_event_summary,
    is_missing_social_tables_error,
    resolve_or_create_club_person,
    social_league_match_rows_from_event,
    social_round_robin_match_rows_from_event,
)
from jupr_app.domain.player_ops import safe_add_player

DEFAULT_PROVISIONAL_SEED_ELO = 1400.0


def _error_payload_text(exc: Exception) -> str:
    pieces = [str(exc)]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text:
            pieces.append(str(text))
    return " | ".join(pieces).lower()


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


def _players_frame(ctx) -> pd.DataFrame:
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())
    if not isinstance(df_players_all, pd.DataFrame) or df_players_all.empty:
        return pd.DataFrame()
    if "id" not in df_players_all.columns or "name" not in df_players_all.columns:
        return pd.DataFrame()
    frame = df_players_all.copy()
    frame["id"] = pd.to_numeric(frame.get("id"), errors="coerce")
    frame = frame.dropna(subset=["id"]).copy()
    frame["id"] = frame["id"].astype(int)
    frame["name"] = frame.get("name", "").fillna("").astype(str).map(normalize_name)
    if "rating" in frame.columns:
        frame["rating"] = pd.to_numeric(frame.get("rating"), errors="coerce")
    else:
        frame["rating"] = float(DEFAULT_PROVISIONAL_SEED_ELO)
    return frame


def _provisional_seed_elo(ctx, participants: list[dict]) -> float:
    explicit_ids: list[int] = []
    for participant in participants or []:
        raw_pid = participant.get("player_id")
        if raw_pid is None:
            continue
        text = str(raw_pid).strip()
        if not text:
            continue
        try:
            explicit_ids.append(int(text))
        except Exception:
            continue
    explicit_ids = sorted(set(explicit_ids))
    if not explicit_ids:
        return float(DEFAULT_PROVISIONAL_SEED_ELO)
    players_df = _players_frame(ctx)
    if players_df.empty:
        return float(DEFAULT_PROVISIONAL_SEED_ELO)
    seeded = players_df[players_df["id"].isin(explicit_ids)].dropna(subset=["rating"]).copy()
    if seeded.empty:
        return float(DEFAULT_PROVISIONAL_SEED_ELO)
    return float(seeded["rating"].mean())


def _fetch_player_by_exact_name(supabase, *, club_id: str, display_name: str) -> dict | None:
    rows = (
        supabase.table("players")
        .select("id,name,rating,starting_rating")
        .eq("club_id", str(club_id))
        .eq("name", normalize_name(display_name))
        .limit(2)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    return dict(rows[0])


def _ensure_rated_player_from_social(
    ctx,
    *,
    club_id: str,
    display_name: str,
    provisional_seed_elo: float,
) -> tuple[dict, bool]:
    existing = _fetch_player_by_exact_name(
        ctx.supabase,
        club_id=club_id,
        display_name=display_name,
    )
    if existing is not None:
        return existing, False
    ok, err = safe_add_player(
        supabase=ctx.supabase,
        club_id=club_id,
        name=normalize_name(display_name),
        rating_jupr=float(provisional_seed_elo) / 400.0,
    )
    if not ok:
        raise RuntimeError(err or f"Unable to create rated player for {display_name}.")
    created = _fetch_player_by_exact_name(
        ctx.supabase,
        club_id=club_id,
        display_name=display_name,
    )
    if created is None:
        raise RuntimeError(f"Unable to resolve created rated player for {display_name}.")
    return created, True


def _upsert_live_event_row(supabase, payload: dict) -> None:
    attempts = [dict(payload)]
    if "submitted_by_name" in payload and "submitted_by" in payload:
        attempts.append({k: v for k, v in payload.items() if k != "submitted_by"})
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


def save_resolved_social_live_event(
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
    event_date = str(event.get("eventDate") or "").strip() or datetime.now(timezone.utc).date().isoformat()
    source_event_uid = _event_uid(event)
    participants = list(event.get("participants") or [])
    admin_logged_in = bool(getattr(ctx, "admin_logged_in", False))
    provisional_seed_elo = _provisional_seed_elo(ctx, participants)

    created_people_count = 0
    linked_existing_players_count = 0
    created_rated_players_count = 0
    created_rated_player_names: list[str] = []
    participant_rows: list[dict] = []

    try:
        for participant in participants:
            display_name = str(participant.get("name") or participant.get("id") or "")
            match_status = str(participant.get("match_status") or "").strip().lower()
            explicit_player_id = participant.get("player_id")
            if explicit_player_id is not None and str(explicit_player_id).strip() != "":
                explicit_player_id = int(explicit_player_id)
            else:
                explicit_player_id = None

            if (
                explicit_player_id is None
                and admin_logged_in
                and match_status in {"new_social", "new social person", "create_rated", "create rated"}
            ):
                rated_player, created_rated = _ensure_rated_player_from_social(
                    ctx,
                    club_id=club_id,
                    display_name=display_name,
                    provisional_seed_elo=provisional_seed_elo,
                )
                explicit_player_id = int(rated_player["id"])
                participant["player_id"] = explicit_player_id
                participant["name"] = normalize_name(rated_player.get("name") or display_name)
                participant["match_status"] = "created_rated"
                display_name = str(participant.get("name") or display_name)
                if created_rated:
                    created_rated_players_count += 1
                    created_rated_player_names.append(display_name)

            auto_link_enabled = match_status not in {"new_social", "new social person"} or explicit_player_id is not None
            club_person, created_new, matched_player = resolve_or_create_club_person(
                ctx,
                display_name=display_name,
                event_date=event_date,
                club_id=club_id,
                explicit_linked_player_id=explicit_player_id,
                allow_name_auto_link=auto_link_enabled,
            )
            created_people_count += int(bool(created_new))
            linked_existing_players_count += int(bool(matched_player))
            participant_rows.append(
                {
                    "participant_key": str(participant.get("id")),
                    "club_person_id": club_person["id"],
                    "linked_player_id": club_person.get("linked_player_id"),
                    "display_name_snapshot": str(participant.get("name") or display_name),
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
            "submitted_by": submitted_by_name,
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
            "created_rated_players_count": created_rated_players_count,
            "created_rated_player_names": created_rated_player_names,
            "provisional_seed_elo": provisional_seed_elo,
        }
    except Exception as exc:
        if is_missing_social_tables_error(exc):
            raise SocialTablesNotInstalledError(SOCIAL_TABLES_INSTALL_MESSAGE) from exc
        raise
