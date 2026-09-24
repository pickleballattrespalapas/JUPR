from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from jupr_app.services import admin_notification_sources as service


NOW = datetime(2026, 9, 21, 18, tzinfo=timezone.utc)
STAMP = "2026-09-20T12:00:00+00:00"
ASSIGNMENTS = [{"club_id": "club-a", "role": "administrator"}]


class Query:
    def __init__(self, db, table, params=None):
        self.db, self.table, self.params = db, table, params
        self.filters, self.orders = [], []
        self.row_limit = None
        self.columns = ""

    def select(self, columns, **options):
        self.columns = columns
        assert options == {"count": "exact"}
        return self

    def eq(self, key, value):
        self.filters.append((key, "eq", value)); return self

    def neq(self, key, value):
        self.filters.append((key, "neq", value)); return self

    def in_(self, key, value):
        self.filters.append((key, "in", value)); return self

    def gte(self, key, value):
        self.filters.append((key, "gte", value)); return self

    def lte(self, key, value):
        self.filters.append((key, "lte", value)); return self

    def order(self, key, desc=False):
        self.orders.append((key, desc)); return self

    def limit(self, value):
        self.row_limit = value; return self

    def execute(self):
        self.db.queries.append(self)
        if self.table in self.db.fail:
            raise RuntimeError("Do not expose private database details")
        rows = self.db.rows.get(self.table, [])
        if self.params:
            assert self.table == "pcs_admin_tournament_notification_source"
            assert set(self.params) == {"p_club_id", "p_kind", "p_since", "p_limit", "p_source_id"}
            rows = [row for row in rows if row["club_id"] == self.params["p_club_id"]
                    and row["kind"] == self.params["p_kind"] and row["occurred_at"] >= self.params["p_since"]
                    and (self.params["p_source_id"] is None or row["id"] == self.params["p_source_id"])]
            return SimpleNamespace(data=[{**row, "total_count": len(rows)} for row in rows[:self.params["p_limit"]]])

        def matches(row):
            for key, op, expected in self.filters:
                value = row
                for part in key.replace("->>", ".").replace("->", ".").split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if (op == "eq" and value != expected or op == "neq" and value == expected
                        or op == "in" and value not in expected):
                    return False
                if op in {"gte", "lte"} and (value is None or (value < expected if op == "gte" else value > expected)):
                    return False
            return True
        rows = [row for row in rows if matches(row)]
        for key, descending in reversed(self.orders):
            rows = sorted(rows, key=lambda row: row.get(key) or "", reverse=descending)
        total = None if self.table in self.db.no_count else len(rows)
        return SimpleNamespace(data=rows[:self.row_limit], count=total)


class Database:
    def __init__(self):
        self.rows, self.queries, self.fail, self.no_count = {}, [], set(), set()

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        return Query(self, name, params)


@pytest.fixture
def db(monkeypatch):
    for function in ("is_admin_support_requests_enabled", "is_admin_tools_enabled",
                     "is_admin_tournament_admin_enabled", "is_admin_verified_updates_enabled",
                     "is_admin_weekly_recap_enabled", "is_admin_league_manager_enabled", "team_leagues_enabled"):
        monkeypatch.setattr(service, function, lambda: True)
    return Database()


def collect(db, **kwargs):
    return service.collect_admin_notification_sources(db, club_id="club-a", assignments=ASSIGNMENTS, now=NOW, **kwargs)


def reports(payload):
    return {source["key"]: source for source in payload["sources"]}


def test_empty_sources_have_exact_zero_and_no_operator_or_other_club_leak(db):
    result = collect(db)
    assert len(result["categories"]) == 10
    assert all(source["status"] == "ready" and source["total_count"] == 0 for source in result["sources"])
    for assignments in ([{"club_id": "club-a", "role": "operator", "scopes": [{"kind": "club"}]}],
                        [{"club_id": "club-b", "role": "super_admin"}],
                        [{"club_id": "club-a", "role": "administrator", "expires_at": "2026-09-20T00:00:00Z"}],
                        [{"club_id": "club-a", "role": "administrator", "revoked_at": STAMP}]):
        db.queries.clear()
        result = service.collect_admin_notification_sources(db, club_id="club-a", assignments=assignments, now=NOW)
        assert result["categories"] == [] and result["items"] == [] and db.queries == []


def test_generator_is_individual_and_uses_submission_identity_not_count(db):
    def row(key, club="club-a", status="pending"):
        submission = {"id": "submit-" + key, "status": status, "submitted_at": STAMP, "submitted_by_email": "private@example.com"}
        return {"club_id": club, "session_key": key, "title": "Round robin", "updated_at": STAMP,
                "state": {"mode": "public_play_generator", "generator_submission": submission}, "generator_submission": submission}
    db.rows["live_sessions"] = [row("one"), row("two", status="processing"), row("other", club="club-b"), row("done", status="approved")]
    result = collect(db)
    assert reports(result)["generator_submissions"]["total_count"] == 2
    assert {item["source_id"] for item in result["items"]} == {"one", "two"}
    before = next(item for item in result["items"] if item["source_id"] == "one")
    assert urlsplit(before["href"]).path == "/admin/play-generators/submissions"
    assert parse_qs(urlsplit(before["href"]).query) == {"session": ["one"]}
    db.rows["live_sessions"][0] = row("new")
    after = collect(db)
    assert reports(after)["generator_submissions"]["total_count"] == 2
    assert all(item["source_version"] != before["source_version"] for item in after["items"])
    assert "private@example.com" not in json.dumps(result)


def test_old_pending_work_is_kept_and_activities_use_explicit_timestamp(db):
    old = "2025-01-01T00:00:00+00:00"
    db.rows["weekly_recaps"] = [{"club_id": "club-a", "id": "draft", "status": "draft", "week_start": "2025-01-01",
        "week_end": "2025-01-07", "created_at": old, "updated_at": STAMP, "row_version": 1, "final_json": {"secret": "hidden"}}]
    db.rows["pcs_interclub_pool_members"] = [
        {"club_id": club, "id": id, "season_id": "season/a", "name": "Alice", "created_at": stamp, "updated_at": STAMP}
        for id, club, stamp in (("recent", "club-a", STAMP), ("old", "club-a", old), ("other", "club-b", STAMP))]
    result = collect(db)
    assert {item["source_id"] for item in result["items"]} == {"draft"}
    draft = next(item for item in result["items"] if item["category"] == "weekly_recaps")
    assert draft["occurred_at"] == old
    assert parse_qs(urlsplit(draft["href"]).query) == {"week_start": ["2025-01-01"]}
    assert not any(q.table.startswith("pcs_interclub") for q in db.queries)
    assert "hidden" not in json.dumps(result)


def test_interclub_approvals_filter_by_organizer_and_closed_registration(db):
    db.rows["pcs_interclub_competition_batches"] = [
        {"id": id, "season_id": "season", "meet_id": "meet", "revision": 2, "state": "submitted", "updated_at": STAMP,
         "season": {"organizer_club_id": club, "registration_closes_at": closes, "details": {"name": "Season"}}}
        for id, club, closes in (("mine", "club-a", STAMP), ("theirs", "club-b", STAMP),
                                ("open", "club-a", "2027-01-01T00:00:00+00:00"))]
    db.rows["pcs_interclub_pool_members"] = [
        {"id": id, "club_id": "participant", "season_id": "season", "name": "Player", "revision": 3, "created_at": STAMP,
         "status": "active", "approval_status": "pending", "late_join": True,
         "pool_settings": {"participation": {"season": {"organizer_club_id": club}}}}
        for id, club in (("late", "club-a"), ("wrong", "club-b"))]
    result = collect(db)
    assert result["items"] == []
    assert not any(key.startswith("interclub") for key in reports(result))


def test_action_links_identify_the_specific_request_and_review_controls(db):
    db.rows["public_support_requests"] = [
        {"id": status + " /&?", "club_id": "club-a", "status": status, "request_type": "data_correction",
         "created_at": STAMP, "updated_at": STAMP} for status in ("new", "in_review")]
    db.rows["player_profile_update_subscriptions"] = [
        {"id": "verified /&?", "club_id": "club-a", "request_status": service.REQUEST_STATUS_PENDING,
         "created_at": STAMP, "row_version": 1}]
    db.rows["live_events"] = [
        {"id": "social /&?", "club_id": "club-a", "name": "Social", "result_mode": "social_unrated", "status": "pending",
         "created_at": STAMP, "updated_at": STAMP}]
    db.rows["pcs_interclub_participations"] = [
        {"season_id": "invitation /&?", "club_id": "club-a", "status": "invited", "revision": 1, "updated_at": STAMP,
         "season": {"organizer_club_id": "club-b", "details": {"name": "Season"}}}]
    links = {item["category"]: urlsplit(item["href"]) for item in collect(db)["items"]}
    for status in ("new", "in_review"):
        link = links["support_" + status]
        assert link.path == "/admin/support-requests"
        assert parse_qs(link.query) == {"status": [status], "request": [status + " /&?"]}
    assert links["verified_updates"].path == "/admin/player-updates/verified-requests"
    assert parse_qs(links["verified_updates"].query) == {"request": ["verified /&?"]}
    assert links["social_submissions"].path == "/admin/tools"
    assert links["social_submissions"].fragment == "social-submissions"
    assert parse_qs(links["social_submissions"].query) == {"submission": ["social /&?"]}


def test_tournament_sources_use_safe_rpc_original_timestamps_and_surviving_destination(db):
    db.rows["pcs_admin_tournament_notification_source"] = [
        {"id": "reg /one" if kind == "registration" else "cancel-id", "club_id": "club-a", "kind": kind,
         "tournament_id": "tournament/a", "tournament_name": "Tournament", "display_name": "Alice",
         "occurred_at": STAMP, "source_version": STAMP,
         "actor_email": "private@example.com", "before_json": {"phone": "private-number"}, "href": "https://attacker.invalid"}
        for kind in ("registration", "cancellation")]
    result = collect(db)
    registration = next(item for item in result["items"] if item["category"] == "tournament_registrations")
    cancellation = next(item for item in result["items"] if item["category"] == "tournament_cancellations")
    assert urlsplit(registration["href"]).path.endswith("/reg%20%2Fone")
    assert urlsplit(cancellation["href"]).path == "/admin/tournaments/registration/registrants"
    assert parse_qs(urlsplit(cancellation["href"]).query) == {"tournament": ["tournament/a"]}
    assert all(secret not in json.dumps(result) for secret in ("private@example.com", "private-number", "attacker.invalid"))


def test_truncation_and_direct_resolution_find_older_pending_item(db):
    db.rows["weekly_recaps"] = [{"id": f"recap-{i:03d}", "club_id": "club-a", "status": "draft",
        "week_start": "2026-09-14", "created_at": STAMP, "updated_at": STAMP, "row_version": 1} for i in range(205)]
    result = collect(db)
    assert len(result["items"]) == 200
    assert reports(result)["weekly_recaps"] == {"key": "weekly_recaps", "status": "ready", "total_count": 205, "truncated": True}
    item = service.resolve_admin_notification_item(db, club_id="club-a", assignments=ASSIGNMENTS,
        category="weekly_recaps", source_id="recap-204", source_version="1", now=NOW)
    assert item["source_id"] == "recap-204"
    db.rows["weekly_recaps"][-1]["row_version"] = 2
    assert service.resolve_admin_notification_item(db, club_id="club-a", assignments=ASSIGNMENTS,
        category="weekly_recaps", source_id="recap-204", source_version="1", now=NOW) is None
    db.rows["weekly_recaps"][-1]["status"] = "published"
    assert service.resolve_admin_notification_item(db, club_id="club-a", assignments=ASSIGNMENTS,
        category="weekly_recaps", source_id="recap-204", source_version="2", now=NOW) is None


def test_source_failures_are_independent_and_resolver_does_not_hide_failure(db):
    db.fail.add("weekly_recaps")
    db.no_count.add("live_events")
    result = collect(db)
    assert reports(result)["weekly_recaps"]["status"] == "unavailable"
    assert reports(result)["social_submissions"]["status"] == "unavailable"
    assert reports(result)["support_new"]["status"] == "ready"
    assert "private database" not in json.dumps(result)
    with pytest.raises(RuntimeError):
        service.resolve_admin_notification_item(db, club_id="club-a", assignments=ASSIGNMENTS,
            category="weekly_recaps", source_id="draft", source_version=STAMP, now=NOW)


def test_preferences_catalog_follows_role_and_feature_gates(db, monkeypatch):
    for name in ("is_admin_weekly_recap_enabled", "is_admin_tournament_admin_enabled", "team_leagues_enabled"):
        monkeypatch.setattr(service, name, lambda: False)
    keys = {category["key"] for category in collect(db)["categories"]}
    assert not keys & {"weekly_recaps", "tournament_registrations", "tournament_cancellations",
                       "team_league_registrations", "team_league_solo_signups"}
    result = service.collect_admin_notification_sources(db, club_id="club-a",
        assignments=[{"club_id": "club-a", "role": "organizer"}], now=NOW)
    assert {category["key"] for category in result["categories"]} == {"support_new", "support_in_review"}
