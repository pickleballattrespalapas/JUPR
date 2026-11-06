"""Result-based starting estimates for profiles created during generator review.

The existing guarded player-create ledger stores provenance and the frozen
request. No game is published by creation, and unrated games remain unrated.
"""
from __future__ import annotations

import math
from typing import Any

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.generator_rating_estimation import estimate_generator_player_ratings
from jupr_app.domain.player_search import player_search_key
from jupr_app.domain.player_visibility import is_merged_player
from jupr_app.services.admin_guarded_write_service import get_guarded_operation
from jupr_app.services.admin_player_editor_service import create_admin_player_editor_player
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.services.starting_rating_review_service import get_starting_rating_review, confirm_starting_ratings

SOURCE = "generator_submission_estimate"


def generator_created_players(db: Any, *, club_id: str, session_keys: list[str]) -> dict[str, dict[str, dict]]:
    """Read authoritative creation receipts, including a reconciled create."""
    if not session_keys:
        return {}
    operations = read_all_rows(lambda: db.table("admin_guarded_operations")
        .select("before_json,result_json,status").eq("club_id", club_id)
        .eq("workflow", "player_editor_create").eq("source", SOURCE).eq("status", "completed")
        .in_("before_json->generator_estimate->>session_key", session_keys))
    result: dict[str, dict[str, dict]] = {}
    for operation in operations:
        evidence = (operation.get("before_json") or {}).get("generator_estimate") or {}
        player = (operation.get("result_json") or {}).get("player") or {}
        if evidence.get("session_key") and evidence.get("participant_id") and player.get("id"):
            result.setdefault(evidence["session_key"], {})[evidence["participant_id"]] = {
                "player_id": int(player["id"]), "ratings": evidence["ratings"],
            }
    return result


def _estimates(event: dict, players: list[dict], mapping: dict[str, int],
               estimated_ids: set[int]) -> dict[str, dict[str, dict]]:
    from jupr_app.services.admin_play_generator_service import _saved_matches
    by_id = {int(player["id"]): player for player in players if not is_merged_player(player)}
    participants = {str(p["id"]): p for p in event.get("participants", [])}
    matches = [{**match, "round_number": number} for number, match in _saved_matches(event)]
    result: dict[str, dict[str, dict]] = {pid: {} for pid in participants}
    formats = ("doubles", "singles") if event.get("playFormat") == "doubles_singles" else (str(event.get("playFormat") or "doubles"),)
    # Always seed doubles independently; a singles-only result never becomes
    # evidence of doubles ability. Its doubles seed uses the group's average.
    for fmt in dict.fromkeys(("doubles", *formats)):
        known = {}
        for participant_id, player_id in mapping.items():
            player = by_id.get(player_id)
            if not player or player_id in estimated_ids:
                continue
            value = player.get("singles_rating" if fmt == "singles" else "rating")
            if value is None and fmt == "singles":
                value = (player.get("singles_replay_baseline") or {}).get("rating")
            if value is not None and math.isfinite(float(value)) and 400 <= float(value) <= 2800:
                known[participant_id] = float(value) / 400
        targets = sorted(set(participants) - set(known))
        calculated = estimate_generator_player_ratings(matches, known, targets, match_format=fmt)
        for participant_id, estimate in calculated.items():
            result[participant_id][fmt] = estimate
    return result


def _mapping(event: dict, supplied: dict[str, int], players: list[dict], created: dict[str, dict]) -> dict[str, int]:
    participants = {str(p["id"]): p for p in event.get("participants", [])}
    if set(supplied) - set(participants):
        raise ValueError("Player mappings must belong to this session.")
    valid_ids = {int(p["id"]) for p in players if not is_merged_player(p)}
    mapping = {pid: int(p["player_id"]) for pid, p in participants.items() if p.get("player_id")}
    mapping.update({pid: int(item["player_id"]) for pid, item in created.items() if pid in participants})
    mapping.update(supplied)
    if any(type(value) is not int or value not in valid_ids for value in mapping.values()):
        raise ValueError("Choose current player profiles from this club.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Choose a different profile for each participant.")
    return mapping


def create_generator_submission_player(db: Any, *, club_id: str, session_key: str,
    participant_id: str, name: str, player_ids: dict[str, int], expected_version: int,
    idempotency_key: str, actor_email: str, actor_role: str) -> dict:
    from jupr_app.services.generator_submission_service import _row, _event, require_reviewer, RATING_PLAYER_COLUMNS
    require_reviewer(actor_role)
    if not actor_email:
        raise PermissionError("An authenticated administrator is required.")
    clean_name = " ".join(str(name or "").replace("<", "").replace(">", "").split())[:160]
    if not clean_name:
        raise ValueError("Enter the new player's name.")
    request = {"session_key": session_key, "participant_id": participant_id, "name": clean_name,
               "player_ids": player_ids, "expected_version": expected_version}
    previous = get_guarded_operation(db, club_id=club_id, workflow="player_editor_create", operation_key=idempotency_key)
    if previous is not None:
        frozen = (previous.get("before_json") or {}).get("generator_estimate") or {}
        if frozen.get("request") != request:
            raise PublicLiveConflictError("This creation key belongs to a different request. Check the previous creation.")
        evidence = frozen
    else:
        row = _row(db, club_id, session_key)
        submission = (row.get("state") or {}).get("generator_submission") or {}
        if row.get("status") != "completed" or submission.get("status") != "pending" or submission.get("review"):
            raise ValueError("Create a player only while reviewing completed, unpublished results.")
        if int(row.get("version") or 1) != expected_version or row.get("pending_operation_key"):
            raise ValueError("This submission changed. Refresh before creating a player.")
        event = _event(row)
        from jupr_app.services.admin_play_generator_service import _saved_matches
        played = {str(pid) for _, match in _saved_matches(event) for side in ("A", "B")
                  for pid in (match.get(f"side{side}") or match.get(f"team{side}") or [])}
        if participant_id not in played:
            raise ValueError("Choose a participant with saved games in this session.")
        players = read_all_rows(lambda: db.table("players").select(RATING_PLAYER_COLUMNS).eq("club_id", club_id))
        if any(not is_merged_player(p) and player_search_key(p.get("name")) == player_search_key(clean_name) for p in players):
            raise ValueError("A matching club profile already exists. Select the existing player.")
        created = generator_created_players(db, club_id=club_id, session_keys=[session_key]).get(session_key, {})
        mapping = _mapping(event, player_ids, players, created)
        if participant_id in mapping:
            raise ValueError("This participant already has a player profile. Select that existing player.")
        estimated_ids = {item["player_id"] for item in created.values()}
        ratings = _estimates(event, players, mapping, estimated_ids)[participant_id]
        evidence = {"session_key": session_key, "participant_id": participant_id,
                    "request": request, "ratings": ratings}
    result = create_admin_player_editor_player(db, club_id=club_id, name=clean_name,
        starting_jupr=evidence["ratings"]["doubles"]["rating_jupr"], actor_email=actor_email,
        actor_role=actor_role, idempotency_key=idempotency_key, source=SOURCE, generator_estimate=evidence)
    ratings = evidence["ratings"]
    primary = "singles" if set(ratings) == {"doubles", "singles"} and ratings["doubles"]["games_used"] == 0 else "doubles"
    return {**result, "rating_estimates": ratings, "rating_estimate": ratings[primary]}


def plan_generator_estimates(db: Any, *, club_id: str, session_key: str, event: dict,
    player_ids: dict[str, int]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Freeze a read-only estimate plan before reserving the submission."""
    from jupr_app.services.generator_submission_service import RATING_PLAYER_COLUMNS
    created = generator_created_players(db, club_id=club_id, session_keys=[session_key]).get(session_key, {})
    if not created:
        return {}, {}
    players = read_all_rows(lambda: db.table("players").select(RATING_PLAYER_COLUMNS).eq("club_id", club_id))
    mapping = _mapping(event, player_ids, players, created)
    selected = {pid: item for pid, item in created.items() if mapping.get(pid) == item["player_id"]}
    if not selected:
        return {}, {}
    snapshots = {}
    for fmt in ("doubles", "singles"):
        ids = sorted({item["player_id"] for item in selected.values() if fmt in item["ratings"]})
        if ids:
            preview = get_starting_rating_review(db, club_id=club_id, player_ids=ids, match_format=fmt)
            snapshots[fmt] = {p["player_id"]: p for p in preview["players"]}
    # Profiles with subsequent rated history or an intervening manual rating
    # edit are real anchors, not unknowns to be fitted again.
    estimated_ids = set()
    for item in selected.values():
        snapshot = snapshots.get("doubles", {}).get(item["player_id"])
        seed = item["ratings"]["doubles"]["rating_jupr"]
        if snapshot and not snapshot["has_rated_history"] and snapshot["rating_jupr"] is not None and snapshot["starting_jupr"] is not None:
            if abs(snapshot["rating_jupr"] - seed) < 0.000001 and abs(snapshot["starting_jupr"] - seed) < 0.000001:
                estimated_ids.add(item["player_id"])
    selected = {pid: item for pid, item in selected.items() if item["player_id"] in estimated_ids}
    calculated = _estimates(event, players, mapping, estimated_ids)
    plans: dict[str, list[dict]] = {}
    for fmt, by_id in snapshots.items():
        for participant_id, item in selected.items():
            snapshot = by_id.get(item["player_id"])
            if not snapshot or snapshot["has_rated_history"]:
                continue
            # Singles baselines are immutable; preserve an independently
            # inferred creation seed. Explicit later doubles edits also win.
            infer = fmt == "doubles" and item["player_id"] in estimated_ids
            value = calculated[participant_id][fmt]["rating_jupr"] if infer else snapshot["starting_jupr"]
            if value is None:
                continue
            plans.setdefault(fmt, []).append({"player_id": item["player_id"], "rating_jupr": value,
                "expected_state_fingerprint": snapshot["state_fingerprint"],
                "previous_rating_jupr": snapshot["rating_jupr"], "previous_starting_jupr": snapshot["starting_jupr"]})
            if infer:
                item["ratings"][fmt] = calculated[participant_id][fmt]
            else:
                item["ratings"][fmt] = {**item["ratings"][fmt], "rating_jupr": value}
    return selected, plans


def apply_generator_estimates(db: Any, *, club_id: str, plans: dict[str, list[dict]],
    actor_email: str, actor_role: str) -> None:
    """Apply only the reserved plan; exact retries never refit published games."""
    for fmt, reviews in plans.items():
        current = get_starting_rating_review(db, club_id=club_id,
            player_ids=[r["player_id"] for r in reviews], match_format=fmt)
        by_id = {row["player_id"]: row for row in current["players"]}
        pending = []
        for review in reviews:
            snapshot = by_id[review["player_id"]]
            # A prior batch may already have published rated games, or another
            # session may have established this profile. Never reset history.
            if snapshot["has_rated_history"]:
                continue
            value = review["rating_jupr"]
            changed = snapshot["state_fingerprint"] != review["expected_state_fingerprint"]
            if changed and (snapshot["rating_jupr"] != review.get("previous_rating_jupr")
                    or snapshot["starting_jupr"] != review.get("previous_starting_jupr")):
                # An intentional edit after reservation wins over the inferred
                # seed. Preserve current ability and make resume possible.
                value = (snapshot["rating_jupr"] if fmt == "doubles"
                         and snapshot["rating_jupr"] != review.get("previous_rating_jupr")
                         else snapshot["starting_jupr"])
            if value is None:
                raise ValueError("The player's starting estimate was removed during approval. Restore it before resuming.")
            if (not snapshot["review_required"] and snapshot["rating_jupr"] is not None
                    and snapshot["starting_jupr"] is not None
                    and abs(snapshot["rating_jupr"] - value) < 0.000001
                    and abs(snapshot["starting_jupr"] - value) < 0.000001):
                continue
            pending.append({**review, "rating_jupr": value,
                            "expected_state_fingerprint": snapshot["state_fingerprint"]})
        if pending:
            confirm_starting_ratings(db, club_id=club_id, match_format=fmt, reviews=pending,
                actor_email=actor_email, actor_role=actor_role, source=SOURCE)
