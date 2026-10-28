"""Edit saved generator labels and rating intent without rebuilding any games."""
from __future__ import annotations

import copy

from jupr_app.services import public_play_generator_service as public
from jupr_app.services.admin_play_generator_service import _session_payload
from jupr_app.services.admin_guarded_write_service import required_audit_event
from jupr_app.services.generator_submission_service import (
    _event, _now, _row, _rows, require_reviewer,
)
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


def _details(event: dict, *, title: str, rating_mode: str) -> dict:
    title = " ".join(str(title or "").replace("<", "").replace(">", "").split())
    if not title or len(title) > 160:
        raise ValueError("Enter a title between 1 and 160 characters.")
    if rating_mode not in {"rated", "unrated"}:
        raise ValueError("Choose Rated or Unrated.")
    if event.get("hubSupplementalDraft"):
        raise ValueError("Retry the saved extra-game action from its hub first.")
    if event.get("hubSupplemental") and rating_mode != "unrated":
        raise ValueError("Extra games are exhibitions and cannot be rated.")
    if rating_mode == "rated" and event.get("scoringMode") == "unscored":
        raise ValueError("An unscored session cannot be rated.")
    event = copy.deepcopy(event)
    event.update(name=title, ratingMode=rating_mode)
    # Moneyball's original setup label is part of its integrity digest. Only
    # rename the display label; preserve the complete frozen setup unchanged.
    return event


def update_public_generator_details(db, *, title: str, rating_mode: str, **kwargs) -> dict:
    def mutate(event):
        event = _details(event, title=title, rating_mode=rating_mode)
        return event, {"title": event["name"]}
    return public._run_mutation(db, **kwargs, action="details",
        request_payload={"title": title, "rating_mode": rating_mode},
        mutate=mutate, allow_completed=True)


def update_admin_generator_details(
    db, *, club_id: str, session_key: str, expected_version: str | int,
    title: str, rating_mode: str, actor_email: str, actor_role: str,
) -> dict:
    row = _row(db, club_id, session_key)
    state = copy.deepcopy(row["state"])
    submission = state.get("generator_submission") or {}
    # Public organizers own their unsubmitted sessions. Staff may intervene only
    # as club administrators; ordinary scorekeepers cannot edit another owner.
    if state["mode"] == "public_play_generator" or submission:
        require_reviewer(actor_role)
    elif actor_role not in {"operator", "administrator", "club_owner", "super_admin"} or not actor_email:
        raise PermissionError("Sign in as club staff to edit session details.")
    official = state.get("official_publish") or {}
    event = _event(row)
    if (row.get("status") not in {"active", "completed"}
            or submission.get("status") not in {None, "pending"}
            or official.get("published_at") or official.get("published_match_ids")
            or event.get("publishedMatchIds")):
        raise ValueError("Session details are locked once approval starts or results are published.")
    if row.get("pending_operation_key"):
        raise PublicLiveConflictError("A previous change is pending. Retry it first.")
    version = (int(row.get("version") or 1) if submission
               else str(row.get("updated_at") or ""))
    if str(version) != str(expected_version):
        raise PublicLiveConflictError("This session changed. Refresh before editing its details.")
    updated_event = _details(event, title=title, rating_mode=rating_mode)
    state = public._put_event(state, updated_event)
    if submission:
        state["generator_submission"]["rating_mode"] = rating_mode
    before = {"title": row.get("title"), "rating_mode": event.get("ratingMode"), "version": version}
    after = {"title": updated_event["name"], "rating_mode": rating_mode}
    audit = dict(club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        entity_type="play_generator_session", entity_id=session_key,
        source="generator_session_details", before=before, after=after)
    required_audit_event(db, **audit, action_type="update_generator_details_intent", intent=True)
    patch = {"title": updated_event["name"], "state": state, "updated_at": _now(),
             "version": int(row.get("version") or 1) + 1}
    if state.get("round_robin_hub_id"):
        from jupr_app.services.admin_round_robin_hub_service import persist_linked_generator_patch
        updated = persist_linked_generator_patch(db, before=row, patch=patch,
            expected_version=str(row.get("updated_at") or ""))
    else:
        # Staff scoring uses timestamps; public scoring and approval use integer
        # versions. Compare both so neither kind of concurrent edit is lost.
        rows = _rows(db.table("live_sessions").update(patch).eq("club_id", club_id)
            .eq("session_key", session_key).eq("version", int(row.get("version") or 1))
            .eq("updated_at", row.get("updated_at")))
        if not rows:
            raise PublicLiveConflictError("This session changed. Refresh before editing its details.")
        updated = rows[0]
    required_audit_event(db, **audit, action_type="update_generator_details", intent=False)
    return {"ok": True, "session": _session_payload(updated), **after, "version": int(updated["version"])}
