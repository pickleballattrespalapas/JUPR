from __future__ import annotations

import os
from typing import Any

from jupr_app.services.staging_write_guard import staging_public_intake_writes_enabled
from scripts.staging_write_waves import wave_allows_request


SUPPORT_UNAVAILABLE_MESSAGE = (
    "Support requests are temporarily unavailable. Please contact your club organizer or PCS support."
)
REVIEW_COLUMNS = (
    "id,club_id,status,request_type,updated_at,admin_note,reviewed_by,reviewed_at,"
    "identity_status,fulfillment_status,resolution_action,resolution_evidence"
)


def support_review_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS", "").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _write_path_enabled(method: str, path: str) -> bool:
    """Mirror the deployment write boundary without weakening the route guard."""
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment in {"local", "test", "development", "dev"}:
        return True
    if environment == "production":
        return os.getenv("JUPR_PRODUCTION_WRITE_POLICY", "").strip().lower() == "enabled"
    if environment == "staging":
        return wave_allows_request(os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip(), method, path)
    return False


def support_review_writes_enabled() -> bool:
    return support_review_enabled() and _write_path_enabled(
        "PATCH", "/admin/clubs/club/support-requests/request"
    )


def support_intake_writes_enabled() -> bool:
    # A public intake wave alone is insufficient: staff must be able to complete
    # the same request while it is accepted, including under the global policy.
    return (
        support_review_writes_enabled()
        and staging_public_intake_writes_enabled()
        and _write_path_enabled("POST", "/clubs/club/support/intake")
    )


def support_queue_schema_ready(supabase: Any, *, club_id: str) -> bool:
    """Probe the review projection, not only the columns needed for insertion.

    Do not return queue data or counts on the public readiness route. An empty
    queue is healthy; missing columns or an unavailable database are not.
    """
    try:
        supabase.table("public_support_requests").select(REVIEW_COLUMNS).eq(
            "club_id", str(club_id)
        ).limit(1).execute()
        return True
    except Exception:
        return False
