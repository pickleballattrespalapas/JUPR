"""Club-scoped, read-only counts for actions available to the current staff member."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_PLAYERS,
    PERMISSION_MANAGE_SUBSCRIPTIONS,
    PERMISSION_MANAGE_TOURNAMENTS,
    PERMISSION_VIEW_AUDIT_LOG,
    ROLE_PERMISSION_MATRIX,
)
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from jupr_app.domain.notifications.player_profile_update_repo import REQUEST_STATUS_PENDING
from jupr_app.services.admin_support_requests_service import is_admin_support_requests_enabled
from jupr_app.services.admin_tools_service import is_admin_tools_enabled
from jupr_app.services.admin_verified_updates_service import is_admin_verified_updates_enabled
from jupr_app.services.admin_weekly_recap_service import is_admin_weekly_recap_enabled

logger = logging.getLogger(__name__)




def build_admin_dashboard(db: Any, *, club_id: str, assignments: list[dict[str, Any]]) -> dict[str, Any]:
    # Operators' destination routes enforce individual program scopes. None of
    # these club-wide review queues is available to an Operator, even when its
    # legacy permission matrix contains the corresponding manage_* permission.
    roles = {row["role"] for row in assignments if row.get("club_id") == club_id and row.get("role") != "operator"}
    permissions = set().union(*(ROLE_PERMISSION_MATRIX.get(role, frozenset()) for role in roles))
    administrator = bool(roles & ADMIN_ROLES)
    queues: list[dict[str, Any]] = []

    def count_query(table: str, columns: str = "id"):
        return db.table(table).select(columns, count="exact", head=True).eq("club_id", club_id)

    def add(key: str, label: str, description: str, href: str, query: Callable[[], Any],
            next_href: Callable[[dict[str, Any]], str] | None = None) -> None:
        entry = {"key": key, "label": label, "description": description, "href": href, "count": None, "status": "unavailable"}
        try:
            response = query().execute()
            count = response.count
            # Missing count metadata must never be rendered as an empty queue.
            if type(count) is not int or count < 0:
                raise ValueError("Exact queue count unavailable")
            if count and next_href is not None:
                # A limited representation supplies only the oldest item's IDs;
                # the exact Content-Range count still describes the full queue.
                rows = response.data
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                    raise ValueError("Next review item unavailable")
                entry["href"] = next_href(rows[0])
            entry.update(count=count, status="ready")
        except Exception:
            # Keep private database details out of the response and continue
            # checking independent queues when one source is unavailable.
            logger.warning("Admin dashboard source unavailable: %s", key)
        queues.append(entry)

    if administrator:
        # The review list remains available when generator creation is disabled.
        # Include interrupted approvals so their retry action stays discoverable.
        add("generator_submissions", "Generator results awaiting approval",
            "Review submitted round-robin and ladder results before they enter club records.",
            "/admin/play-generators/submissions",
            lambda: count_query("live_sessions", "session_key")
                .in_("state->>mode", ["public_play_generator", "admin_play_generator"])
                .in_("state->generator_submission->>status", ["pending", "processing"]))

    support_permissions = {PERMISSION_MANAGE_PLAYERS, PERMISSION_MANAGE_MATCHES,
                          PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_SUBSCRIPTIONS}
    if permissions & support_permissions and is_admin_support_requests_enabled():
        add("support_new", "New player requests",
            "Review new requests for help, record corrections, or profile privacy.",
            "/admin/support-requests?status=new",
            lambda: count_query("public_support_requests").eq("status", "new"))
        add("support_in_review", "Player requests in progress",
            "Follow up on requests already under review.",
            "/admin/support-requests?status=in_review",
            lambda: count_query("public_support_requests").eq("status", "in_review"))

    if PERMISSION_MANAGE_SUBSCRIPTIONS in permissions and is_admin_verified_updates_enabled():
        add("verified_updates", "Player update requests awaiting approval",
            "Verify requests to receive a player's rating and results updates.",
            "/admin/player-updates/verified-requests",
            lambda: count_query("player_profile_update_subscriptions").eq("request_status", REQUEST_STATUS_PENDING))

    if PERMISSION_MANAGE_MATCHES in permissions and is_admin_weekly_recap_enabled():
        add("weekly_recaps", "Weekly recap drafts",
            "Review saved recap drafts and publish them when ready.",
            "/admin/weekly-recap",
            lambda: count_query("weekly_recaps").eq("status", "draft"))

    if {PERMISSION_VIEW_AUDIT_LOG, PERMISSION_MANAGE_MATCHES} <= permissions and is_admin_tools_enabled():
        add("social_submissions", "Club Social results awaiting review",
            "Approve or reject submitted Club Social results.",
            "/admin/tools#social-submissions",
            lambda: count_query("live_events").eq("result_mode", "social_unrated").eq("status", "pending"))

    return {"club_id": club_id, "checked_at": datetime.now(timezone.utc).isoformat(), "queues": queues}
