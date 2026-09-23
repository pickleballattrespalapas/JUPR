"""Private, pair-specific gender eligibility decisions for tournament entries."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from jupr_app.domain.tournament_registration_compiler import evaluate_selection_gender_eligibility

REVIEW_TABLE = "tournament_gender_eligibility_reviews"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gender_review_snapshots(
    *, tournament_id: str, events: list[dict[str, Any]],
    registrations: list[dict[str, Any]], selections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recompute from current identities; old approvals cannot follow a changed pair."""
    event_by_id = {str(row["id"]): row for row in events}
    registration_by_id = {str(row["id"]): row for row in registrations}
    registration_by_email = {str(row.get("email") or "").strip().lower(): row for row in registrations if row.get("email")}
    result = {}

    def identity(row: dict[str, Any]) -> dict[str, Any]:
        return {key: str(row.get(key) or "").strip().lower() for key in
                ("id", "player_id", "email", "display_name", "gender")}

    for selection in selections:
        registration = registration_by_id.get(str(selection.get("registration_id")), {})
        if str(registration.get("status") or registration.get("registration_status") or "").lower() in {"cancelled", "canceled", "withdrawn"}:
            continue
        event = event_by_id.get(str(selection.get("event_option_id")))
        if not event or not registration:
            continue
        linked = registration_by_id.get(str(selection.get("partner_registration_id")))
        mode = "HAS_PARTNER" if linked else str(selection.get("partner_mode") or "NONE").upper()
        partner = None
        if mode == "HAS_PARTNER":
            partner = linked or registration_by_email.get(str(selection.get("partner_email") or "").strip().lower()) or {
                "display_name": selection.get("partner_name"), "email": selection.get("partner_email"),
                "gender": selection.get("partner_gender"),
            }
        review = evaluate_selection_gender_eligibility(
            event=event, selection={**selection, "partner_mode": mode}, player=registration,
            partner=partner, allow_missing_partner_for_preview=True,
        )
        if review.get("issue_type") != "GENDER_NOT_ELIGIBLE":
            continue
        selection_ids = sorted({str(selection["id"]), *([str(selection["partner_selection_id"])] if linked and selection.get("partner_selection_id") else [])})
        participants = sorted([identity(registration), *([identity(partner)] if partner else [])], key=lambda row: json.dumps(row, sort_keys=True))
        snapshot = {
            "tournament_id": str(tournament_id), "event_option_id": str(event["id"]),
            "restriction": review["restriction"], "event_type": event.get("event_type"),
            "partner_required": bool(event.get("partner_required")), "partner_mode": mode,
            "selection_ids": selection_ids, "participants": participants,
        }
        result[str(selection["id"])] = {
            "fingerprint": _fingerprint(snapshot), "snapshot": snapshot,
            "player_gender": str(registration.get("gender") or ""),
            "partner_gender": str((partner or {}).get("gender") or ""),
            "reason": str(review.get("issue") or "Gender eligibility requires admin approval."),
        }
    return result


def load_gender_reviews(supabase: Any, *, tournament_id: str, snapshots: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not snapshots:
        return {}
    try:
        rows = supabase.table(REVIEW_TABLE).select("fingerprint,decision,reviewed_by,reviewed_at").eq("tournament_id", str(tournament_id)).execute().data or []
    except Exception as exc:
        raise RuntimeError("Gender eligibility approvals could not be loaded. Please retry before importing registrations.") from exc
    decisions = {row["fingerprint"]: row for row in rows}
    reviews = {}
    for selection_id, snapshot in snapshots.items():
        decision = decisions.get(snapshot["fingerprint"], {})
        status = decision.get("decision") or "PENDING"
        reviews[selection_id] = {
            **snapshot, "status": status, "reviewed_by": decision.get("reviewed_by"),
            "reviewed_at": decision.get("reviewed_at"),
            "review_version": _fingerprint([snapshot["fingerprint"], status, decision.get("reviewed_at")]),
        }
    return reviews
