"""Read-only, club-scoped source adapters for personal staff notifications.

Pending work is read from current records, not fabricated event history. Activity
uses explicit creation/acceptance/cancellation timestamps. Every source is bounded
independently and failures cannot turn an unknown queue into an empty one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Callable
from urllib.parse import quote, urlencode

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_MATCHES, PERMISSION_MANAGE_PLAYERS,
    PERMISSION_MANAGE_SUBSCRIPTIONS, PERMISSION_MANAGE_TOURNAMENTS,
    PERMISSION_VIEW_AUDIT_LOG, ROLE_PERMISSION_MATRIX,
)
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES, assignment_active
from jupr_app.domain.notifications.player_profile_update_repo import REQUEST_STATUS_PENDING
from jupr_app.services.admin_league_manager_service import is_admin_league_manager_enabled
from jupr_app.services.admin_support_requests_service import is_admin_support_requests_enabled
from jupr_app.services.admin_tools_service import is_admin_tools_enabled
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from jupr_app.services.admin_verified_updates_service import is_admin_verified_updates_enabled
from jupr_app.services.admin_weekly_recap_service import is_admin_weekly_recap_enabled
from services.api.team_league_feature import team_leagues_enabled

logger = logging.getLogger(__name__)
MAX_SOURCE_ITEMS = 200


@dataclass(frozen=True)
class _Source:
    category: dict[str, str]
    query: Callable[[str | None, int], Any]
    item: Callable[[dict[str, Any]], dict[str, Any]]
    rpc: bool = False


def _required(value: Any) -> str:
    if value is None or not str(value).strip():
        raise ValueError("Notification source identity is unavailable")
    return str(value).strip()


def _text(value: Any, fallback: str, limit: int = 180) -> str:
    return " ".join(str(value or fallback).split())[:limit]


def _timestamp(value: Any) -> str:
    parsed = datetime.fromisoformat(_required(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Notification timestamp must have a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _item(row: dict[str, Any], *, title: str, description: str, href: str,
          id_field: str = "id", time_field: str = "created_at", version_field: str | None = None,
          version: Any = None) -> dict[str, Any]:
    occurred_at = _timestamp(row.get(time_field))
    return {"source_id": _required(row.get(id_field)),
            "source_version": _required(version if version is not None else row.get(version_field or time_field)),
            "title": title, "description": description, "href": href, "occurred_at": occurred_at}


def _href(path: str, **params: Any) -> str:
    return path + "?" + urlencode({key: _required(value) for key, value in params.items()})


def _season_name(row: dict[str, Any]) -> str:
    season = row.get("season") or {}
    return _text((season.get("details") or {}).get("name"), "Interclub season")


def _sources(db: Any, *, club_id: str, assignments: list[dict[str, Any]], now: datetime,
             history_days: int) -> list[_Source]:
    # The current dashboard does not expose any club-wide collection to Operators.
    # Their legacy manage_* permissions must never bypass resource scopes here.
    roles = {row["role"] for row in assignments if row.get("club_id") == club_id
             and row.get("role") != "operator" and assignment_active(row, now)}
    permissions = set().union(*(ROLE_PERMISSION_MATRIX.get(role, frozenset()) for role in roles))
    administrator = bool(roles & ADMIN_ROLES)
    since = (now - timedelta(days=history_days)).isoformat()
    sources: list[_Source] = []

    def add(key: str, label: str, description: str, href: str, kind: str,
            table: str, columns: str, render: Callable[[dict[str, Any]], dict[str, Any]], *,
            filters: Callable[[Any], Any] = lambda query: query, id_field: str = "id",
            time_field: str = "created_at", club_scoped: bool = True) -> None:
        def query(source_id: str | None, limit: int):
            result = db.table(table).select(columns, count="exact")
            if club_scoped:
                result = result.eq("club_id", club_id)
            result = filters(result)
            if kind == "activity":
                result = result.gte(time_field, since).lte(time_field, now.isoformat())
            if source_id is not None:
                result = result.eq(id_field, source_id)
            return result.order(time_field, desc=True).order(id_field).limit(limit)
        sources.append(_Source({"key": key, "label": label, "description": description,
                                "href": href, "kind": kind}, query, render))

    if administrator:
        def generator(row):
            submission = row.get("generator_submission") or {}
            return _item({**row, "submitted_at": submission.get("submitted_at")},
                         id_field="session_key", time_field="submitted_at", version=_required(submission.get("id")),
                         title=_text(row.get("title"), "Generator results"),
                         description="Results awaiting approval." if submission.get("status") == "pending" else "Resume the interrupted results approval.",
                         href=_href("/admin/play-generators/submissions", session=row.get("session_key")))
        add("generator_submissions", "Generator result approvals", "Round-robin and ladder results awaiting approval.",
            "/admin/play-generators/submissions", "action", "live_sessions",
            "session_key,title,updated_at,generator_submission:state->generator_submission", generator,
            id_field="session_key", time_field="updated_at",
            filters=lambda q: q.in_("state->>mode", ["public_play_generator", "admin_play_generator"])
                .in_("state->generator_submission->>status", ["pending", "processing"]))

    support_permissions = {PERMISSION_MANAGE_PLAYERS, PERMISSION_MANAGE_MATCHES,
                           PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_SUBSCRIPTIONS}
    if permissions & support_permissions and is_admin_support_requests_enabled():
        for status, key, label in (("new", "support_new", "New player requests"),
                                   ("in_review", "support_in_review", "Player requests in progress")):
            href = _href("/admin/support-requests", status=status)
            add(key, label, "Requests for help, record corrections, or profile privacy.", href, "action",
                "public_support_requests", "id,request_type,created_at,updated_at",
                lambda row, status=status, label=label: _item(row, title=label,
                    description={"data_correction": "A player requested a record correction.",
                                 "profile_privacy": "A player requested a profile privacy review."}.get(
                                     row.get("request_type"), "A player requested help."),
                    href=_href("/admin/support-requests", status=status, request=row.get("id")), version_field="updated_at"),
                filters=lambda q, status=status: q.eq("status", status))

    if PERMISSION_MANAGE_SUBSCRIPTIONS in permissions and is_admin_verified_updates_enabled():
        add("verified_updates", "Player update approvals", "Requests to receive a player's rating and results updates.",
            "/admin/player-updates/verified-requests", "action", "player_profile_update_subscriptions",
            "id,created_at,row_version", lambda row: _item(row, title="Player update request",
                description="Verify permission to receive a player's updates.",
                href=_href("/admin/player-updates/verified-requests", request=row.get("id")),
                version_field="row_version"), filters=lambda q: q.eq("request_status", REQUEST_STATUS_PENDING))

    if PERMISSION_MANAGE_MATCHES in permissions and is_admin_weekly_recap_enabled():
        add("weekly_recaps", "Weekly recap drafts", "Saved recap drafts ready for review or deletion.",
            "/admin/weekly-recap", "action", "weekly_recaps", "id,week_start,week_end,created_at,row_version",
            lambda row: _item(row, title=f"Weekly recap: {_required(row.get('week_start'))}",
                description="Review, publish, or delete this saved draft.",
                href=_href("/admin/weekly-recap", week_start=row.get("week_start")), version_field="row_version"),
            filters=lambda q: q.eq("status", "draft"))

    if {PERMISSION_VIEW_AUDIT_LOG, PERMISSION_MANAGE_MATCHES} <= permissions and is_admin_tools_enabled():
        add("social_submissions", "Club Social result approvals", "Submitted Club Social results awaiting review.",
            "/admin/tools#social-submissions", "action", "live_events", "id,name,created_at,updated_at",
            lambda row: _item(row, title=_text(row.get("name"), "Club Social results"),
                description="Review submitted Club Social results.",
                href=_href("/admin/tools", submission=row.get("id")) + "#social-submissions", version_field="updated_at"),
            filters=lambda q: q.eq("result_mode", "social_unrated").eq("status", "pending"))

    if PERMISSION_MANAGE_TOURNAMENTS in permissions and is_admin_tournament_admin_enabled():
        for kind, key, label in (("registration", "tournament_registrations", "Tournament registrations"),
                                 ("cancellation", "tournament_cancellations", "Tournament cancellations")):
            def tournament_query(source_id, limit, kind=kind):
                return db.rpc("pcs_admin_tournament_notification_source", {
                    "p_club_id": club_id, "p_kind": kind, "p_since": since,
                    "p_limit": limit, "p_source_id": source_id})
            def tournament_item(row, kind=kind):
                path = "/admin/tournaments/registration/registrants"
                if kind == "registration":
                    path += "/" + quote(_required(row.get("id")), safe="")
                name = _text(row.get("display_name"), "A player")
                tournament = _text(row.get("tournament_name"), "Tournament")
                return _item(row, title=f"{name}: {'registered' if kind == 'registration' else 'registration cancelled'}",
                    description=tournament, href=_href(path, tournament=row.get("tournament_id")),
                    time_field="occurred_at", version_field="source_version")
            sources.append(_Source({"key": key, "label": label, "kind": "activity", "href": "/admin/tournaments",
                                    "description": f"{label} from the past {history_days} days."},
                                   tournament_query, tournament_item, rpc=True))

    if PERMISSION_MANAGE_MATCHES in permissions and is_admin_league_manager_enabled() and team_leagues_enabled():
        for table, key, label, columns in (
            ("team_league_teams", "team_league_registrations", "Team league registrations", "id,league_name,team_name,created_at"),
            ("team_league_solo_waitlist", "team_league_solo_signups", "Team league solo signups", "id,league_name,created_at"),
        ):
            add(key, label, f"{label} from the past {history_days} days.", "/admin/league-manager", "activity",
                table, columns, lambda row, key=key: _item(row,
                    title=_text(row.get("team_name"), "New team") + " registered" if key == "team_league_registrations" else "New solo league signup",
                    description=_text(row.get("league_name"), "Team league"),
                    href=_href("/admin/league-manager/teams", league_id=row.get("league_name"), league_name=row.get("league_name"), mode="Team")))

    return sources


def _read(source: _Source, source_id: str | None, limit: int) -> tuple[list[dict[str, Any]], int]:
    response = source.query(source_id, limit).execute()
    rows = response.data
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Notification source rows unavailable")
    total = rows[0].get("total_count") if source.rpc and rows else 0 if source.rpc else response.count
    if type(total) is not int or total < len(rows) or total > 0 and not rows:
        raise ValueError("Notification source count unavailable")
    return rows, total


def collect_admin_notification_sources(db: Any, *, club_id: str, assignments: list[dict[str, Any]],
                                       now: datetime | None = None, history_days: int = 30,
                                       item_limit: int = MAX_SOURCE_ITEMS) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    item_limit = max(1, min(MAX_SOURCE_ITEMS, int(item_limit)))
    history_days = max(1, min(30, int(history_days)))
    plans = _sources(db, club_id=club_id, assignments=assignments, now=now, history_days=history_days)
    items: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for source in plans:
        key = source.category["key"]
        report = {"key": key, "status": "unavailable", "total_count": None, "truncated": False}
        try:
            rows, total = _read(source, None, item_limit + 1)
            parsed = [{**source.item(row), "category": key, "kind": source.category["kind"]} for row in rows[:item_limit]]
            items.extend(parsed)
            report.update(status="ready", total_count=total, truncated=total > item_limit)
        except Exception:
            logger.warning("Admin notification source unavailable: %s", key)
        reports.append(report)
    items.sort(key=lambda row: (row["occurred_at"], row["category"], row["source_id"]), reverse=True)
    return {"categories": [source.category for source in plans], "items": items, "sources": reports,
            "generated_at": now.isoformat()}


def resolve_admin_notification_item(db: Any, *, club_id: str, assignments: list[dict[str, Any]],
                                    category: str, source_id: str, source_version: str,
                                    now: datetime | None = None) -> dict[str, Any] | None:
    """Resolve an exact saved version, including pending items beyond the page.

    A missing or changed item returns None. Database/shape failures propagate so
    callers preserve an unknown flagged action instead of marking it resolved.
    """
    plans = _sources(db, club_id=club_id, assignments=assignments,
                     now=now or datetime.now(timezone.utc), history_days=30)
    source = next((plan for plan in plans if plan.category["key"] == category), None)
    if source is None:
        return None
    rows, _ = _read(source, source_id, 2)
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("Notification source identity is not unique")
    item = {**source.item(rows[0]), "category": category, "kind": source.category["kind"]}
    if item["source_id"] != source_id or item["source_version"] != source_version:
        return None
    return item
