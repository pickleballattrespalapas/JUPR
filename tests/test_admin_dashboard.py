from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from postgrest import SyncPostgrestClient

from jupr_app.services import admin_dashboard_service as service
from services.api.admin_dashboard_routes import install_admin_dashboard_routes


FLAGS = (
    "JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
    "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
    "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    "JUPR_ENABLE_NEXT_ADMIN_TOOLS",
)


class Query:
    def __init__(self, db, table):
        self.db, self.table_name = db, table
        self.filters = []
        self.count_options = None
        self.orders = []
        self.row_limit = None

    def select(self, columns, **options):
        self.columns, self.count_options = columns, options
        return self

    def eq(self, column, value):
        self.filters.append((column, "eq", value))
        return self

    def neq(self, column, value):
        self.filters.append((column, "neq", value))
        return self

    def in_(self, column, values):
        self.filters.append((column, "in", values))
        return self

    def lte(self, column, value):
        self.filters.append((column, "lte", value))
        return self

    def order(self, column):
        self.orders.append(column)
        return self

    def limit(self, limit):
        self.row_limit = limit
        return self

    def execute(self):
        self.db.queries.append(self)
        if self.table_name in self.db.fail_tables:
            raise RuntimeError("Private database detail")

        def matches(row):
            for column, op, expected in self.filters:
                value = row
                for key in column.replace("->>", ".").replace("->", ".").split("."):
                    value = value.get(key) if isinstance(value, dict) else None
                if op == "eq" and value != expected or op == "neq" and value == expected or op == "in" and value not in expected:
                    return False
                if op == "lte" and (value is None or value > expected):
                    return False
            return True

        rows = [row for row in self.db.rows.get(self.table_name, []) if matches(row)]
        if self.table_name == "admin_role_assignments":
            return SimpleNamespace(data=rows)
        organizer_source = self.table_name in {"pcs_interclub_competition_batches", "pcs_interclub_pool_members"}
        if organizer_source:
            assert self.count_options == {"count": "exact"}
            owner_path = "season.organizer_club_id" if self.table_name == "pcs_interclub_competition_batches" else "pool_settings.participation.season.organizer_club_id"
            assert (owner_path, "eq", "club-a") in self.filters
            assert self.row_limit == 1
        else:
            assert self.count_options == {"count": "exact", "head": True}
            assert ("club_id", "eq", "club-a") in self.filters
        count = None if self.table_name in self.db.missing_counts else len(rows)
        if organizer_source:
            rows.sort(key=lambda row: tuple(row.get(key, "") for key in self.orders))
            return SimpleNamespace(data=rows[:self.row_limit], count=count)
        # HEAD has no response rows. Dashboard must use the count metadata.
        return SimpleNamespace(data=[], count=count)


class Database:
    def __init__(self):
        self.rows = {}
        self.queries = []
        self.fail_tables = set()
        self.missing_counts = set()

    def table(self, name):
        return Query(self, name)


@pytest.fixture
def db(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, "true")
    return Database()


def dashboard(db, role="administrator"):
    return service.build_admin_dashboard(db, club_id="club-a", assignments=[{"club_id": "club-a", "role": role}])


def by_key(payload):
    return {queue["key"]: queue for queue in payload["queues"]}


def test_generator_notifications_include_interrupted_reviews_and_exclude_other_clubs_and_modes(db):
    def row(status, *, club="club-a", mode="public_play_generator"):
        return {"club_id": club, "state": {"mode": mode, "generator_submission": {"status": status}}}

    db.rows["live_sessions"] = [row("pending"), row("processing", mode="admin_play_generator"),
        row("approved"), row("rejected"), row("pending", club="club-b"), row("pending", mode="other_live_session")]
    queue = by_key(dashboard(db))["generator_submissions"]
    assert queue["count"] == 2
    assert queue["status"] == "ready"
    assert queue["href"] == "/admin/play-generators/submissions"


def test_queue_filters_count_only_outstanding_actions(db):
    db.rows["public_support_requests"] = [{"club_id": "club-a", "status": s} for s in ("new", "in_review", "resolved", "dismissed")]
    db.rows["player_profile_update_subscriptions"] = [{"club_id": "club-a", "request_status": s} for s in ("pending_admin_review", "active", "rejected", "unsubscribed")]
    db.rows["weekly_recaps"] = [{"club_id": "club-a", "status": s} for s in ("draft", "published")]
    db.rows["live_events"] = [
        {"club_id": "club-a", "status": "pending", "result_mode": "social_unrated"},
        {"club_id": "club-a", "status": "saved", "result_mode": "social_unrated"},
        {"club_id": "club-a", "status": "pending", "result_mode": "rated"},
        {"club_id": "club-b", "status": "pending", "result_mode": "social_unrated"},
    ]
    queues = by_key(dashboard(db))
    assert {key: queues[key]["count"] for key in ("support_new", "support_in_review", "verified_updates", "weekly_recaps", "social_submissions")} == {
        "support_new": 1, "support_in_review": 1, "verified_updates": 1, "weekly_recaps": 1, "social_submissions": 1}
    assert queues["support_new"]["href"].endswith("?status=new")
    assert queues["support_in_review"]["href"].endswith("?status=in_review")
    assert queues["social_submissions"]["href"] == "/admin/tools#social-submissions"




def meet_result(id, *, state="submitted", organizer="club-a", updated_at="2026-09-21", closes_at="2000-01-01T00:00:00+00:00", **fields):
    return {"id": id, "season_id": f"season-{id}", "meet_id": f"meet-{id}", "state": state,
            "updated_at": updated_at, "season": {"organizer_club_id": organizer, "registration_closes_at": closes_at}, **fields}


def pool_member(id, *, status="active", approval="pending", late=True, organizer="club-a", created_at="2026-09-21", **fields):
    return {"id": id, "season_id": f"season-{id}", "status": status, "approval_status": approval,
            "late_join": late, "created_at": created_at,
            "pool_settings": {"participation": {"season": {"organizer_club_id": organizer}}}, **fields}












def test_counts_are_exact_beyond_any_list_page_limit(db):
    db.rows["weekly_recaps"] = [{"club_id": "club-a", "status": "draft"}] * 1532
    assert by_key(dashboard(db))["weekly_recaps"]["count"] == 1532


@pytest.mark.parametrize("failure", ["fail_tables", "missing_counts"])
def test_independent_source_failure_never_becomes_zero(db, failure):
    getattr(db, failure).add("live_sessions")
    queues = by_key(dashboard(db))
    assert queues["generator_submissions"]["count"] is None
    assert queues["generator_submissions"]["status"] == "unavailable"
    assert queues["support_new"]["count"] == 0
    assert queues["support_new"]["status"] == "ready"
    assert "Private database detail" not in str(queues)


@pytest.mark.parametrize("role", ["operator", "scorekeeper", "read_only"])
def test_staff_without_destination_action_access_get_no_club_wide_queues(db, role):
    assert dashboard(db, role)["queues"] == []
    assert db.queries == []


def test_legacy_organizer_can_follow_up_support_but_cannot_review_results(db):
    assert set(by_key(dashboard(db, "organizer"))) == {"support_new", "support_in_review"}


def test_assignments_to_other_clubs_do_not_grant_queue_access(db):
    payload = service.build_admin_dashboard(db, club_id="club-a", assignments=[{"club_id": "club-b", "role": "super_admin"}])
    assert payload["queues"] == []
    assert db.queries == []


def test_disabled_features_are_not_queried(db, monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, "false")
    queues = by_key(dashboard(db))
    assert set(queues) == {"generator_submissions"}
    assert {q.table_name for q in db.queries} == {"live_sessions"}


def route_client(db, monkeypatch, assignment=None):
    db.rows["admin_role_assignments"] = [assignment or {
        "club_id": "club-a", "role": "administrator", "email": "admin@example.invalid", "user_id": "user-1"}]

    def authenticate(authorization):
        if authorization != "Bearer test-session":
            raise HTTPException(401, "invalid bearer token")
        return SimpleNamespace(email="admin@example.invalid", user_id="user-1")

    monkeypatch.setattr("services.api.admin_auth_routes.authenticate_bearer", authenticate)
    app = FastAPI()
    install_admin_dashboard_routes(app, get_supabase_client=lambda: db)
    return TestClient(app)


def test_authenticated_route_returns_private_selected_club_counts(db, monkeypatch):
    client = route_client(db, monkeypatch)
    response = client.get("/admin/clubs/club-a/dashboard", headers={"Authorization": "Bearer test-session"})
    assert response.status_code == 200
    assert response.json()["club_id"] == "club-a"
    assert response.json()["checked_at"]
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vary"] == "Authorization"
    assert all(queue["count"] == 0 and queue["status"] == "ready" for queue in response.json()["queues"])


@pytest.mark.parametrize("assignment", [
    {"club_id": "club-b"}, {"user_id": "different-user"},
    {"expires_at": "2000-01-01T00:00:00+00:00"}, {"revoked_at": "2026-01-01T00:00:00+00:00"},
])
def test_route_rejects_wrong_club_expired_revoked_or_other_user_assignment(db, monkeypatch, assignment):
    client = route_client(db, monkeypatch, {"club_id": "club-a", "role": "super_admin", "email": "admin@example.invalid", "user_id": "user-1", **assignment})
    response = client.get("/admin/clubs/club-a/dashboard", headers={"Authorization": "Bearer test-session"})
    assert response.status_code == 403
    assert response.json() == {"detail": "admin access denied"}
    assert {q.table_name for q in db.queries} == {"admin_role_assignments"}


def test_route_requires_authentication_and_auth_backend_failure_is_not_an_empty_dashboard(db, monkeypatch):
    client = route_client(db, monkeypatch)
    assert client.get("/admin/clubs/club-a/dashboard").status_code == 401
    assert not db.queries
    db.fail_tables.add("admin_role_assignments")
    response = client.get("/admin/clubs/club-a/dashboard", headers={"Authorization": "Bearer test-session"})
    assert response.status_code == 503
    assert response.json() == {"detail": "admin access check unavailable"}


def test_installed_postgrest_client_uses_exact_content_range_for_head_and_one_row_reviews(db):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["prefer"] == "count=exact"
        if request.url.path.endswith(("/pcs_interclub_competition_batches", "/pcs_interclub_pool_members")):
            assert request.method == "GET"
            assert request.url.params["limit"] == "1"
            assert "club_id" not in request.url.params
            if request.url.path.endswith("/pcs_interclub_competition_batches"):
                assert request.url.params["season.organizer_club_id"] == "eq.club-a"
                assert request.url.params["select"] == "id,season_id,meet_id,season:pcs_interclub_seasons!inner(id)"
                assert request.url.params["state"] == "eq.submitted"
                assert request.url.params["season.registration_closes_at"].startswith("lte.")
                assert request.url.params["order"] == "updated_at.asc,id.asc"
            else:
                assert request.url.params["pool_settings.participation.season.organizer_club_id"] == "eq.club-a"
                assert request.url.params["select"] == "id,season_id,pool_settings:pcs_interclub_pool_settings!inner(participation:pcs_interclub_participations!inner(season:pcs_interclub_seasons!inner(id)))"
                assert request.url.params["approval_status"] == "eq.pending"
                assert request.url.params["status"] == "eq.active"
                assert request.url.params["late_join"].lower() == "eq.true"
                assert request.url.params["order"] == "created_at.asc,id.asc"
            return httpx.Response(206, headers={"Content-Range": "0-0/1532"}, json=[{"id": "first", "season_id": "season-first", "meet_id": "meet-first"}])
        assert request.method == "HEAD"
        assert request.url.params["club_id"] == "eq.club-a"
        assert "limit" not in request.url.params
        return httpx.Response(200, headers={"Content-Range": "*/1532"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        postgrest = SyncPostgrestClient("https://database.example.invalid/rest/v1", http_client=client)
        payload = dashboard(postgrest)
    assert all(queue["count"] == 1532 for queue in payload["queues"])
    generator = next(request for request in requests if request.url.path.endswith("/live_sessions"))
    assert generator.url.params["state->generator_submission->>status"] == "in.(pending,processing)"
    assert generator.url.params["state->>mode"] == "in.(public_play_generator,admin_play_generator)"
    verified = next(request for request in requests if request.url.path.endswith("/player_profile_update_subscriptions"))
    assert verified.url.params["request_status"] == "eq.pending_admin_review"


def test_dashboard_is_registered_in_application():
    from services.api.main import app

    assert any(route.path == "/admin/clubs/{club_id}/dashboard" for route in app.routes)


def test_deferred_interclub_sources_are_not_queried(db):
    result = dashboard(db)
    assert not any(q.table_name.startswith("pcs_interclub") for q in db.queries)
    assert not any(q["key"].startswith("interclub") for q in result["queues"])
