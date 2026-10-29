"""Prepare one exhibition game without moving anyone out of their home group.

The existing hub transaction updates sessions but cannot create one.  We first
create an unattachable, deterministic draft, then return it in the caller's full
read set for one normal version-checked hub commit.  An interrupted preparation
can only recover the same draft; it never overwrites or creates a second game.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def _first(response: Any) -> dict[str, Any] | None:
    return next((dict(row) for row in (response.data or [])), None)


def _draft_row(db: Any, club_id: str, session_key: str) -> dict[str, Any] | None:
    return _first(db.table("live_sessions").select("*").eq("club_id", club_id)
        .eq("session_key", session_key).limit(1).execute())


def _selected_players(hub: dict[str, Any], events: dict[str, dict[str, Any]],
                      payload: dict[str, Any]) -> list[tuple[dict[str, str], dict[str, Any]]]:
    from jupr_app.domain.adaptive_play_engine import active_participant_ids
    from jupr_app.domain.round_robin_hub import (
        _claims, _complete, _event, _identity_data, _matches, player_identity,
    )

    if payload.get("standings_policy") != "exhibition":
        raise ValueError("Extra games are exhibitions and do not change division standings or ratings.")
    if any(not isinstance(payload.get(side), list) or len(payload[side]) != 2 for side in ("side_a", "side_b")):
        raise ValueError("Choose two players for each team.")
    identities = _identity_data(hub, events)
    claims = _claims(hub, events)
    selected, seen = [], set()
    for ref in [*payload["side_a"], *payload["side_b"]]:
        if not isinstance(ref, dict):
            raise ValueError("Choose existing players from this event.")
        key, pid = str(ref.get("session_key") or ""), str(ref.get("participant_id") or "")
        event = _event(hub, events, key)
        if event.get("status") != "active" or event.get("hubSupplemental"):
            raise ValueError("Choose players from active division round robins.")
        if event.get("eventFormat", "standard") != "standard" or event.get("playoff"):
            raise ValueError("Extra games use players from standard round robins before playoffs.")
        number = int(event.get("currentRoundNumber") or 1)
        player = next((p for p in event.get("participants") or [] if str(p.get("id")) == pid), None)
        if player is None or pid not in active_participant_ids(event, number):
            raise ValueError("Choose players who are still active in their division.")
        identity = player_identity(player, key, hub)
        if identity in seen:
            raise ValueError("Each person can appear only once in the extra game.")
        seen.add(identity)
        if identity in identities["pending"]:
            raise ValueError("Resolve matching player names before arranging an extra game.")
        if (hub.get("players", {}).get(identity) or {}).get("paused"):
            raise ValueError("Resume paused players before arranging an extra game.")
        if identity in claims:
            raise ValueError("Finish or release each player's court and reservation before arranging an extra game.")
        # A spare-court game may not jump ahead of a player's unfinished current
        # obligation in any attached division, including a shared guest identity.
        for source_key, source in events.items():
            if source_key not in {str(g["session_key"]) for g in hub.get("groups") or []}:
                continue
            by_id = {str(p["id"]): p for p in source.get("participants") or []}
            current = int(source.get("currentRoundNumber") or 1)
            for row in source.get("rounds") or []:
                if int(row["number"]) > current or row.get("status") in {"saved", "played", "skipped"}:
                    continue
                for match in _matches(row):
                    if _complete(match, source):
                        continue
                    ids = (match.get("sideA") or match.get("teamA") or []) + (match.get("sideB") or match.get("teamB") or [])
                    if any(player_identity(by_id[str(value)], source_key, hub) == identity for value in ids):
                        raise ValueError("Finish each player's current division game before arranging an extra game.")
        selected.append(({"session_key": key, "participant_id": pid}, deepcopy(player)))
    return selected


def prepare_extra_game(db: Any, *, club_id: str, hub_id: str,
                       hub: dict[str, Any], events: dict[str, dict[str, Any]],
                       payload: dict[str, Any], operation_key: str,
                       request_fingerprint: str, actor_email: str, now: str
                       ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (durable draft row, proposed hub, proposed events) for atomic attach.

    The caller must check the hub expected_version and operation receipt first,
    exclude submitted source sessions, and include the returned row in its CAS
    read set.  It must not expose a draft as an ordinarily attachable session.
    """
    from jupr_app.domain.adaptive_play_engine import create_generator_preview
    from jupr_app.domain.round_robin_hub import _claims, _ensure_players
    from jupr_app.services.admin_play_generator_service import _base_state, mark_generator_write_attempt

    state, updated = deepcopy(hub), deepcopy(events)
    _ensure_players(state, updated, now)
    selected = _selected_players(state, updated, payload)
    session_key = uuid5(NAMESPACE_URL, f"jupr:hub-extra-game:{club_id}:{hub_id}:{operation_key}").hex
    source_refs = [{**ref, "extra_participant_id": f"p-{index}"} for index, (ref, _) in enumerate(selected, 1)]
    metadata = {"standingsPolicy": "exhibition", "targetHubId": hub_id,
        "operationKey": operation_key, "requestFingerprint": request_fingerprint,
        "sourceParticipants": source_refs}
    existing = _draft_row(db, club_id, session_key)
    if existing is None:
        event = create_generator_preview(generator_kind="round_robin", play_format="doubles",
            title="Extra game · exhibition", participant_names=[f"Extra player {i}" for i in range(1, 5)],
            total_rounds=1, court_count=1, scoring_mode="scored", rating_mode="unrated")
        for participant, (_, original) in zip(event["participants"], selected):
            participant["name"] = original["name"]
            if original.get("player_id") is not None:
                participant["player_id"] = original["player_id"]
        match = {"id": f"extra-{session_key}-m1", "round": 1, "court": 1,
            "sideA": ["p-1", "p-2"], "sideB": ["p-3", "p-4"],
            "teamA": ["p-1", "p-2"], "teamB": ["p-3", "p-4"],
            "scoreA": None, "scoreB": None, "status": "scheduled", "hubPlanLocked": True}
        event.update(sourceEventUid=f"hub-extra-{session_key}", createdAt=now,
            hubSupplemental=metadata, hubSupplementalDraft=True,
            hubRegistrationTargetId=hub_id, hubFinishAfterRound=1,
            rounds=[{"number": 1, "status": "preview", "matches": [match], "byeParticipantIds": []}],
            rosterByRound={"1": ["p-1", "p-2", "p-3", "p-4"]})
        created = datetime.fromisoformat(now.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        draft = {"club_id": club_id, "session_key": session_key, "status": "active",
            "title": event["name"], "state": _base_state(club_id=club_id, event=event, source="hub_extra_game"),
            "source": "play_generator", "created_by_email": str(actor_email or "").strip().lower() or None,
            "created_at": now, "updated_at": now, "last_seen_at": now,
            "expires_at": (created + timedelta(days=30)).isoformat()}
        try:
            # Never upsert: a concurrent retry must not overwrite a session.
            mark_generator_write_attempt()
            db.table("live_sessions").insert(draft).execute()
        except Exception:
            # A lost insert response or concurrent exact retry can leave the
            # intended draft durable. Confirm it before deciding to fail.
            existing = _draft_row(db, club_id, session_key)
            if existing is None:
                raise
        if existing is None:
            existing = _draft_row(db, club_id, session_key)
        if existing is None:
            raise RuntimeError("The extra game draft could not be confirmed. Retry the same action.")

    stored_state = existing.get("state") or {}
    event = deepcopy(stored_state.get("page_state", {}).get("event") or {})
    rounds = event.get("rounds") or []
    matches = rounds[0].get("matches") or [] if len(rounds) == 1 else []
    match = matches[0] if len(matches) == 1 else {}
    expected_players = [(f"p-{index}", original["name"], original.get("player_id"))
        for index, (_, original) in enumerate(selected, 1)]
    stored_players = [(str(player.get("id")), player.get("name"), player.get("player_id"))
        for player in event.get("participants") or []]
    intact_draft = (len(rounds) == 1 and rounds[0].get("number") == 1 and rounds[0].get("status") == "preview"
        and match.get("id") == f"extra-{session_key}-m1" and match.get("sideA") == ["p-1", "p-2"]
        and match.get("sideB") == ["p-3", "p-4"] and match.get("scoreA") is None and match.get("scoreB") is None
        and match.get("status") == "scheduled" and stored_players == expected_players
        and event.get("ratingMode") == "unrated" and event.get("hubFinishAfterRound") == 1)
    if (event.get("hubSupplemental") != metadata or event.get("hubSupplementalDraft") is not True
            or event.get("status") != "preview" or stored_state.get("round_robin_hub_id")
            or existing.get("status") != "active" or not intact_draft):
        raise ValueError("This extra-game request already has a different or attached result. Refresh the hub.")
    if session_key in updated or any(str(g["session_key"]) == session_key for g in state.get("groups") or []):
        raise ValueError("This extra game is already attached. Refresh the hub.")

    event.update(status="active", startedAt=now, hubSupplementalDraft=False,
        roundRobinHubId=hub_id, hubRollingEnabled=True, hubPreparedThroughRound=1)
    event["rounds"][0]["status"] = "active"
    updated[session_key] = event
    state.setdefault("groups", []).append({"session_key": session_key, "joined_at": now})
    for ref in source_refs:
        state.setdefault("identity_decisions", []).append({
            "left": {"session_key": ref["session_key"], "participant_id": ref["participant_id"]},
            "right": {"session_key": session_key, "participant_id": ref["extra_participant_id"]},
            "resolution": "same", "at": now, "source": "extra_game_selection"})
    _ensure_players(state, updated, now)
    _claims(state, updated)
    return existing, state, updated
