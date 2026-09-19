from copy import deepcopy
from types import ModuleType, SimpleNamespace
from uuid import uuid4
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.services import interclub_player_email_service as service
from tests.test_admin_match_log_service import FakeSupabase, FakeQuery

SEASON = "11111111-1111-4111-8111-111111111111"
MEET = "22222222-2222-4222-8222-222222222222"
MEMBER = "33333333-3333-4333-8333-333333333333"
USER = SimpleNamespace(user_id="admin", email="admin@example.invalid")


class Query(FakeQuery):
    offset = None
    end = None
    def range(self, offset, end):
        self.offset, self.end = offset, end
        return self

    def execute(self):
        if self.table_name == service.TABLE and self.insert_payload:
            key = self.insert_payload["operation_key"]
            if any(row["operation_key"] == key for row in self.storage.get(service.TABLE, [])):
                raise RuntimeError("unique violation")
        result = super().execute()
        if self.table_name == service.TABLE and self.insert_payload and self.storage.get("__lost_claim_response__") and self.insert_payload["operation_type"].startswith("interclub_player_email_recipient:"):
            raise RuntimeError("claim response lost")
        if self.offset is not None:
            result.data = result.data[self.offset:self.end + 1]
        return result


class DB(FakeSupabase):
    def table(self, name):
        return Query(self.tables, name)


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setattr(service, "write_admin_activity_log", lambda *a: SimpleNamespace(ok=True))
    helpers = ModuleType("services.api.interclub_player_pool_routes")
    helpers.pool_signup_url = lambda share_id: "https://staging.invalid/interclub/signup/" + str(share_id)
    helpers.pool_response_url = lambda row, season, meet: "https://staging.invalid/interclub/respond#token=" + row["member_id"]
    def prepare(db, user, club_id, season_id, meet_id, member_ids):
        rows = [{"id": "response-"+member_id, "member_id": member_id} for member_id in member_ids]
        db.tables.setdefault("__prepared__", []).extend(rows)
        return {"meet": db.tables["pcs_interclub_meets"][0], "settings": {}, "responses": rows}
    helpers.prepare_meet_invitations = prepare
    def admin_context(get_db, authorization, club_id, season_id):
        if authorization != "Bearer valid-admin" or club_id != "club":
            raise HTTPException(403, "Club administrator required")
        db = get_db()
        club, season = service._authorized_context(db, club_id, season_id)
        return db, USER, club, season
    helpers.pool_admin_context = admin_context
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    return DB({
        "clubs": [{"id": "club", "name": "Cabo"}, {"id": "other", "name": "Other Club"}],
        "pcs_interclub_participations": [{"club_id": "club", "season_id": SEASON, "status": "accepted"}, {"club_id": "other", "season_id": SEASON, "status": "accepted"}],
        "pcs_interclub_seasons": [{"id": SEASON, "details": {"name": "Winter", "timezone": "America/Mazatlan", "end_date": "2099-12-31"}}],
        "pcs_interclub_pool_settings": [{"club_id": "club", "season_id": SEASON, "open": True, "share_id": "share", "revision": 1}],
        "players": [{"id": 1, "club_id": "club", "name": "Alex", "active": True}, {"id": 2, "club_id": "club", "name": "Beth", "active": True}, {"id": 3, "club_id": "other", "name": "Foreign", "active": True}],
        "player_profile_update_subscriptions": [{"id": "contact1", "club_id": "club", "player_id": 1, "email": "alex@example.invalid", "request_status": "active"}, {"id": "contact2", "club_id": "club", "player_id": 2, "email": "beth@example.invalid", "request_status": "active"}, {"id": "contact3", "club_id": "other", "player_id": 3, "email": "foreign@example.invalid", "request_status": "active"}],
        "pcs_interclub_meets": [{"id": MEET, "season_id": SEASON, "club_ids": ["club", "other"], "host_club_id": "other", "starts_at": "2099-12-01T15:00:00Z"}],
        "pcs_interclub_availability_settings": [{"club_id": "club", "season_id": SEASON, "meet_id": MEET, "open": True, "deadline": "2099-11-30T12:00:00Z", "revision": 1}],
        "pcs_interclub_pool_members": [{"id": MEMBER, "club_id": "club", "season_id": SEASON, "name": "Alex", "email": "alex@example.invalid", "status": "active", "revision": 1}],
    })


def payload(kind="season", **overrides):
    return {"kind": kind, "meet_id": MEET if kind == "meet" else None, "recipient_ids": [MEMBER] if kind == "meet" else ["1"], "subject": "Join us", "message": "Please join the pool.", **overrides}


def prepare(db, **options):
    body = payload(**options)
    preview = service.preview_email(db, club_id="club", season_id=SEASON, **body)
    return {**body, "operation_key": str(uuid4()), "preview_fingerprint": preview["preview_fingerprint"]}


def create(db, **options):
    params = prepare(db, **options)
    return service.create_email(db, club_id="club", season_id=SEASON, user=USER, **params), params


def send(db, key, index=0):
    return service.send_recipient(db, club_id="club", season_id=SEASON, user=USER, operation_key=key, recipient_index=index)


def test_audience_is_own_active_players_with_usable_contacts_only(fixture):
    fixture.tables["players"].extend([{"id": 4, "club_id": "club", "name": "No email", "active": True}, {"id": 5, "club_id": "club", "name": "Inactive", "active": False}])
    fixture.tables["player_profile_update_subscriptions"][1]["request_status"] = "unsubscribed"
    result = service.build_audience(fixture, club_id="club", season_id=SEASON, kind="season")
    assert {r["id"] for r in result["candidates"]} == {"1", "2", "4"}
    assert [r["id"] for r in result["candidates"] if r["available"]] == ["1"]
    assert "foreign@example.invalid" not in str(result)
    assert result["send_available"] is True  # dry-run does not require SMTP


def test_ambiguous_or_global_opt_out_contacts_are_not_selectable(fixture):
    fixture.tables["player_profile_update_subscriptions"].append({"id": "new", "club_id": "club", "player_id": 1, "email": "different@example.invalid", "request_status": "active"})
    fixture.tables["player_profile_update_subscriptions"][1]["preferences_json"] = {"optional_emails_enabled": False}
    with pytest.raises(ValueError, match="unavailable"):
        prepare(fixture)
    assert not any(r["available"] for r in service.build_audience(fixture, club_id="club", season_id=SEASON, kind="season")["candidates"])


def test_foreign_selection_cannot_expand_audience(fixture):
    with pytest.raises(ValueError, match="unavailable"):
        prepare(fixture, recipient_ids=["3"])
    assert not fixture.tables.get(service.TABLE)


def test_no_invitation_or_email_created_by_preview(fixture):
    result = service.preview_email(fixture, club_id="club", season_id=SEASON, **payload("meet"))
    assert "#personal-meet-response-link" in result["preview"]["html"]
    assert not fixture.tables.get(service.TABLE)
    assert not fixture.tables.get("__prepared__")


def test_dry_run_prepares_test_link_once_and_never_touches_smtp(fixture, monkeypatch):
    monkeypatch.setattr(service, "send_email_with_inline_chart", lambda **_: pytest.fail("SMTP must not run"))
    batch, params = create(fixture, kind="meet")
    first = send(fixture, batch["operation_key"])
    second = send(fixture, batch["operation_key"])
    assert first == second
    assert first["status"] == "dry_run"
    assert "#token=" in first["links"][0]["url"]
    assert len(fixture.tables["__prepared__"]) == 1
    replay = service.create_email(fixture, club_id="club", season_id=SEASON, user=USER, **params)
    assert replay["pending_count"] == 0
    assert replay["recipients"][0]["links"] == first["links"]


def test_shared_inbox_gets_one_email_for_selected_members(fixture):
    second = {**fixture.tables["pcs_interclub_pool_members"][0], "id": str(uuid4()), "name": "Beth"}
    fixture.tables["pcs_interclub_pool_members"].append(second)
    batch, _ = create(fixture, kind="meet", recipient_ids=[MEMBER, second["id"]])
    assert batch["recipient_count"] == 1
    result = send(fixture, batch["operation_key"])
    assert {link["name"] for link in result["links"]} == {"Alex", "Beth"}
    assert len(fixture.tables["__prepared__"]) == 2


def test_season_signup_link_is_available_from_dry_run(fixture):
    batch, _ = create(fixture)
    assert send(fixture, batch["operation_key"])["links"] == [{"name": "Season signup", "url": "https://staging.invalid/interclub/signup/share"}]


@pytest.mark.parametrize("field,value", [("email", "changed@example.invalid"), ("request_status", "unsubscribed")])
def test_changed_contact_invalidates_review_and_remaining_send(fixture, field, value):
    params = prepare(fixture)
    fixture.tables["player_profile_update_subscriptions"][0][field] = value
    with pytest.raises(ValueError):
        service.create_email(fixture, club_id="club", season_id=SEASON, user=USER, **params)
    assert not fixture.tables.get(service.TABLE)


def test_changed_member_and_deadline_stop_remaining_send(fixture):
    batch, _ = create(fixture, kind="meet")
    fixture.tables["pcs_interclub_availability_settings"][0]["deadline"] = "2099-11-29T12:00:00Z"
    with pytest.raises(ValueError, match="changed"):
        send(fixture, batch["operation_key"])
    assert not fixture.tables.get("__prepared__")


def test_withdrawn_member_is_no_longer_invitable(fixture):
    batch, _ = create(fixture, kind="meet")
    fixture.tables["pcs_interclub_pool_members"][0]["status"] = "withdrawn"
    with pytest.raises(ValueError):
        send(fixture, batch["operation_key"])


def test_cancelled_club_cannot_resume_email(fixture):
    batch, _ = create(fixture)
    fixture.tables["pcs_interclub_participations"][0]["status"] = "cancelled"
    with pytest.raises(PermissionError):
        send(fixture, batch["operation_key"])


def test_operation_key_reuse_with_changed_input_rejected(fixture):
    _, params = create(fixture)
    with pytest.raises(ValueError, match="different communications request"):
        service.create_email(fixture, club_id="club", season_id=SEASON, user=USER, **{**params, "subject": "Changed"})


def test_durable_claim_response_loss_never_enters_delivery(fixture, monkeypatch):
    batch, _ = create(fixture)
    fixture.tables["__lost_claim_response__"] = True
    monkeypatch.setattr(service, "_deliver", lambda **_: pytest.fail("Cannot send after unknown claim"))
    assert send(fixture, batch["operation_key"])["status"] == "uncertain"
    assert send(fixture, batch["operation_key"])["status"] == "uncertain"


def test_uncertain_provider_result_not_retried_or_leaked(fixture, monkeypatch):
    calls = []
    def uncertain(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("SMTP secret diagnostics")
    monkeypatch.setattr(service, "_deliver", uncertain)
    batch, _ = create(fixture)
    result = send(fixture, batch["operation_key"])
    assert result["status"] == "uncertain"
    assert "secret" not in str(result)
    send(fixture, batch["operation_key"])
    assert len(calls) == 1


def test_cross_club_cannot_read_batch(fixture):
    batch, _ = create(fixture)
    with pytest.raises(ValueError, match="not found"):
        service.get_email(fixture, club_id="other", season_id=SEASON, operation_key=batch["operation_key"])


@pytest.mark.parametrize("subject", ["Bad\r\nBcc: another@example.com", "", "x" * 201])
def test_rejects_invalid_subject(fixture, subject):
    with pytest.raises(ValueError):
        prepare(fixture, subject=subject)


def test_email_escapes_user_content(fixture):
    result = service.preview_email(fixture, club_id="club", season_id=SEASON, **payload(subject="<script>", message='<img src=x onerror="alert(1)">'))
    assert "<script>" not in result["preview"]["html"]
    assert "<img" not in result["preview"]["html"]


def test_email_routes_require_club_admin_and_do_not_cache(fixture):
    from services.api.interclub_player_email_routes import install_interclub_player_email_routes
    app = FastAPI()
    install_interclub_player_email_routes(app, get_supabase_client=lambda: fixture)
    client = TestClient(app)
    base = f"/admin/clubs/club/interclub/player-pools/{SEASON}/emails"
    assert client.get(base + "/audience?kind=season").status_code == 403
    headers = {"Authorization": "Bearer valid-admin"}
    response = client.get(base + "/audience?kind=season", headers=headers)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert not any(k.startswith("_") for k in response.json())
    assert client.post(base + "/preview", headers=headers, json=payload()).status_code == 200
    assert client.post(base + "/preview", headers=headers, json={**payload(), "club_id": "other"}).status_code == 422


def test_real_pool_helper_creates_scoped_signed_meet_link(fixture, monkeypatch):
    import importlib
    monkeypatch.delitem(sys.modules, "services.api.interclub_player_pool_routes")
    pool = importlib.import_module("services.api.interclub_player_pool_routes")
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "email-integration-test-secret-32-bytes-long")
    monkeypatch.setenv("JUPR_NEXT_WEB_BASE_URL", "https://staging.invalid")
    row = {"id": str(uuid4()), "club_id": "club", "season_id": SEASON, "meet_id": MEET,
        "member_id": MEMBER, "token_nonce": str(uuid4()), "status": "invited", "revision": 1}
    def rpc(name, args):
        assert name == "pcs_interclub_pool_action"
        assert args["p_club_id"] == "club"
        assert args["p_season_id"] == SEASON
        assert args["p_action"] == "invite"
        assert args["p_payload"]["member_ids"] == [MEMBER]
        return SimpleNamespace(execute=lambda: SimpleNamespace(data={"responses": [row], "settings": {}, "meet": fixture.tables["pcs_interclub_meets"][0]}))
    fixture.rpc = rpc
    batch, _ = create(fixture, kind="meet")
    result = send(fixture, batch["operation_key"])
    assert result["status"] == "dry_run"
    link = result["links"][0]["url"]
    assert link.startswith("https://staging.invalid/interclub/respond#token=")
    claims = pool._verify_token(link.split("#token=")[1])
    assert (claims["club_id"], claims["season_id"], claims["id"], claims["kind"]) == ("club", SEASON, row["id"], "meet")


@pytest.mark.parametrize("moment,expected", [("2026-09-20T06:59:59+00:00", True), ("2026-09-20T07:00:00+00:00", False)])
def test_season_email_closes_at_local_midnight_after_end_date(fixture, monkeypatch, moment, expected):
    from datetime import datetime as real_datetime
    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromisoformat(moment).astimezone(tz)
    monkeypatch.setattr(service, "datetime", Clock)
    fixture.tables["pcs_interclub_seasons"][0]["details"]["end_date"] = "2026-09-19"
    result = service.build_audience(fixture, club_id="club", season_id=SEASON, kind="season")
    assert result["send_available"] is expected
    if not expected:
        assert "season has ended" in result["send_unavailable_reason"]
        with pytest.raises(ValueError, match="season has ended"):
            create(fixture)
        assert not fixture.tables.get(service.TABLE)


def test_missing_email_batch_returns_404_for_safe_client_recovery(fixture):
    from services.api.interclub_player_email_routes import install_interclub_player_email_routes
    app = FastAPI()
    install_interclub_player_email_routes(app, get_supabase_client=lambda: fixture)
    client = TestClient(app)
    url = f"/admin/clubs/club/interclub/player-pools/{SEASON}/emails/{uuid4()}"
    result = client.get(url, headers={"Authorization": "Bearer valid-admin"})
    assert result.status_code == 404
