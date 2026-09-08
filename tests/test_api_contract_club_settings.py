from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, club_settings_routes as routes


class Query:
    def __init__(self, rows):
        self.rows, self.filters, self.columns = rows, [], "*"

    def select(self, columns):
        self.columns = columns
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, _):
        return self

    def execute(self):
        rows = [r for r in self.rows if all(r.get(k) == v for k, v in self.filters)]
        if self.columns != "*":
            rows = [{k: r.get(k) for k in self.columns.split(",")} for r in rows]
        return SimpleNamespace(data=rows)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", lambda _: SimpleNamespace(user_id="verified", email="admin@example.test"))
    assignment = dict(email="admin@example.test", user_id="verified", club_id="alpha", role="administrator")
    club = dict(id="alpha", slug="alpha-club", name="Alpha Club", tagline="Local play", support_email="info@example.test",
                is_active=False, onboarding_status="draft", updated_at="2026-09-08T05:00:00.123456+00:00", features_json={"private":True})
    state = {"calls": [], "assignment": assignment, "club": club, "error": ""}

    def table(name):
        return Query([assignment] if name == "admin_role_assignments" else [club])

    def rpc(name, params):
        state["calls"].append((name, params))
        def execute():
            if state["error"]:
                e = RuntimeError("secret database error")
                e.code = state["error"]
                raise e
            return SimpleNamespace(data=club)
        return SimpleNamespace(execute=execute)

    app = FastAPI()
    routes.install_club_settings_routes(app, get_supabase_client=lambda: SimpleNamespace(table=table, rpc=rpc))
    return TestClient(app), state


def payload(**overrides):
    return dict(name="  Alpha Club  ", tagline=" Local play ", support_email=" INFO@EXAMPLE.TEST ",
                expected_updated_at="2026-09-08T05:00:00.123456Z", **overrides)


def test_club_admin_reads_only_own_club_and_allowlisted_fields(setup):
    c, _ = setup
    response = c.get("/admin/clubs/alpha/settings")
    assert response.status_code == 200
    assert response.json()["club"]["id"] == "alpha"
    assert response.json()["setup"] == {"missing": [], "can_submit": True}
    assert "features_json" not in response.text and "admin@example.test" not in response.text
    assert c.get("/admin/clubs/beta/settings").status_code == 403


@pytest.mark.parametrize("change", [dict(role="operator"), dict(role="read_only"), dict(revoked_at="2026-01-01"),
                                   dict(expires_at="2020-01-01T00:00:00Z"), dict(user_id="other"), dict(club_id="beta")])
def test_disallowed_staff_cannot_read_or_write_settings(setup, change):
    c, state = setup
    state["assignment"].update(change)
    assert c.get("/admin/clubs/alpha/settings").status_code == 403
    assert c.put("/admin/clubs/alpha/settings", json=payload()).status_code == 403
    assert not state["calls"]


def test_save_uses_verified_identity_route_club_and_exact_revision(setup):
    c, state = setup
    r = c.put("/admin/clubs/alpha/settings", json=payload(submit_for_review=True))
    assert r.status_code == 200
    name, args = state["calls"][0]
    assert name == "pcs_save_club_settings"
    assert args == dict(p_actor_id="verified", p_actor_email="admin@example.test", p_club_id="alpha",
                        p_expected_updated_at="2026-09-08T05:00:00.123456+00:00", p_name="Alpha Club",
                        p_tagline="Local play", p_support_email="info@example.test", p_submit=True)
    assert "features_json" not in r.text


@pytest.mark.parametrize("field,value", [("is_active",True),("plan_status","top"),("p_actor_id","forged"),("club_id","beta"),
                                       ("name","  "),("tagline","a"*241),("support_email","bad@email"),
                                       ("expected_updated_at", "2026-09-08T05:00:00")])
def test_invalid_or_privileged_fields_are_rejected(setup, field, value):
    c, state = setup
    assert c.put("/admin/clubs/alpha/settings", json={**payload(), field: value}).status_code == 422
    assert not state["calls"]


def test_submission_requires_email_but_draft_can_be_saved_without_it(setup):
    c, state = setup
    data = {**payload(), "support_email":"", "submit_for_review":True}
    assert c.put("/admin/clubs/alpha/settings", json=data).status_code == 422
    data["submit_for_review"] = False
    assert c.put("/admin/clubs/alpha/settings", json=data).status_code == 200


@pytest.mark.parametrize("code,status", [("40001",409),("42501",403),("22023",422),("P0002",404),("unknown",503)])
def test_database_failures_are_actionable_and_do_not_disclose_internal_details(setup, code, status):
    c, state = setup
    state["error"] = code
    r = c.put("/admin/clubs/alpha/settings", json=payload())
    assert r.status_code == status and "secret" not in r.text


def test_active_club_never_offers_setup_submission(setup):
    c, state = setup
    state["club"].update(is_active=True, onboarding_status="ready")
    assert not c.get("/admin/clubs/alpha/settings").json()["setup"]["can_submit"]
