from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from jupr_app.services.tournament_registration_attempt_service import (
    TABLE, audited_registration_call, list_registration_attempts, record_registration_attempt,
)
from tests.test_public_tournament_registration_service import FakeQuery, FakeSupabase, fake_storage


class AuditQuery(FakeQuery):
    def order(self, key, desc=False):
        self.audit_orders = [*getattr(self, "audit_orders", []), (key, desc)]
        return self

    def gte(self, key, value):
        self.filters.append(("gte", key, value))
        return self

    def ilike(self, key, value):
        self.filters.append(("ilike", key, value))
        return self

    def _apply_filters(self, rows):
        rows = super()._apply_filters(rows)
        for op, key, value in self.filters:
            if op == "gte":
                rows = [r for r in rows if str(r.get(key) or "") >= str(value)]
            elif op == "ilike":
                rows = [r for r in rows if str(value).strip("%").lower() in str(r.get(key) or "").lower()]
        for key, desc in reversed(getattr(self, "audit_orders", [])):
            rows.sort(key=lambda row: str(row.get(key) or ""), reverse=desc)
        return rows


class AuditSupabase(FakeSupabase):
    def table(self, name):
        return AuditQuery(self.storage, name)

    def rpc(self, name, params):
        if name != "record_tournament_registration_attempt":
            return super().rpc(name, params)
        row = dict(params["p_event"])
        row["created_at"] = datetime.now(timezone.utc).isoformat()
        self.storage.setdefault(TABLE, []).append(row)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=True))


def payload():
    return {"attempt_id": str(uuid4()), "tournament_id": "t1", "display_name": "Terri Fixture",
            "email": "terri@example.invalid", "stage": "events", "event": "CLIENT_REJECTED",
            "message": "A player cannot register themselves as their own partner.",
            "selections": [{"event_option_id": "event1", "partner_name": "Brent Fixture", "same_email_as_player": True}]}


def test_client_rejection_has_safe_support_data_and_cannot_claim_success():
    storage = fake_storage()
    db = AuditSupabase(storage)
    data = {**payload(), "edit_token": "private-token", "phone": "private-phone", "notes": "private-notes"}
    data["selections"][0].update({"partner_email": "private-partner", "partner_phone": "private-phone"})
    record_registration_attempt(db, club_id="club-1", payload=data)
    row = storage[TABLE][0]
    assert row["error_code"] == "SELF_PARTNER"
    assert row["selections"][0]["same_email_as_player"] is True
    assert "private" not in json.dumps(row)
    assert row["source"] == "BROWSER" and row["registration_id"] is None
    record_registration_attempt(db, club_id="club-1", payload={**data, "event": "SAVED", "registration_id": "fake"})
    assert len(storage[TABLE]) == 1
    assert not storage["tournament_registrations"]


def test_scope_rejection_and_storage_failure_do_not_change_registration_results():
    storage = fake_storage()
    db = AuditSupabase(storage)
    record_registration_attempt(db, club_id="another-club", payload=payload())
    assert not storage.get(TABLE)
    db.rpc = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("secret-database-detail"))
    result = audited_registration_call(db, club_id="club-1", payload=payload(), stage="submit",
                                      operation=lambda: {"registration_id": "saved-real"})
    assert result == {"registration_id": "saved-real"}


def test_server_records_rejection_then_success_with_same_attempt():
    storage = fake_storage()
    db = AuditSupabase(storage)
    data = payload()
    def reject():
        raise ValueError("A player cannot register themselves as their own partner.")
    with pytest.raises(ValueError, match="own partner"):
        audited_registration_call(db, club_id="club-1", payload=data, stage="submit", operation=reject)
    assert storage[TABLE][-1]["event"] == "SERVER_REJECTED"
    assert storage[TABLE][-1]["error_code"] == "SELF_PARTNER"
    audited_registration_call(db, club_id="club-1", payload=data, stage="submit", operation=lambda: {"registration_id": "reg-real"})
    assert storage[TABLE][-1]["event"] == "SAVED"
    assert storage[TABLE][-1]["registration_id"] == "reg-real"


def test_history_does_not_hide_later_failures_or_call_link_loads_failed_edits():
    db = AuditSupabase(fake_storage())
    data = payload()
    audited_registration_call(db, club_id="club-1", payload=data, stage="edit", operation=lambda: {"registration_id": "reg-real"})
    record_registration_attempt(db, club_id="club-1", payload=data)
    assert list_registration_attempts(db, club_id="club-1", tournament_id="t1")["attempts"][0]["outcome"] == "Change not saved — earlier save succeeded"
    audited_registration_call(db, club_id="club-1", payload=data, stage="edit", operation=lambda: {"registration_id": "reg-real", "withdrawn": True})
    assert list_registration_attempts(db, club_id="club-1", tournament_id="t1")["attempts"][0]["outcome"] == "Withdrawn"
    second = {**payload(), "email": "load@example.invalid"}
    record_registration_attempt(db, club_id="club-1", payload=second, source="SERVER", event="PROGRESS", stage="edit", message="Registration edit page loaded.")
    assert list_registration_attempts(db, club_id="club-1", tournament_id="t1", search="load@example.invalid")["attempts"][0]["outcome"] == "Editor loaded — no save recorded"


def test_email_search_includes_earlier_anonymous_timeline_and_filters_other_clubs():
    storage = fake_storage()
    db = AuditSupabase(storage)
    data = payload()
    record_registration_attempt(db, club_id="club-1", payload={**data, "email": "", "display_name": "", "event": "PROGRESS", "stage": "contact"})
    record_registration_attempt(db, club_id="club-1", payload=data)
    result = list_registration_attempts(db, club_id="club-1", tournament_id="t1", search="terri@example.invalid")
    assert len(result["attempts"]) == 1
    assert len(result["attempts"][0]["events"]) == 2
    assert result["attempts"][0]["outcome"] == "In progress / incomplete"
    with pytest.raises(ValueError, match="Tournament not found"):
        list_registration_attempts(db, club_id="another-club", tournament_id="t1")


def test_public_capture_has_no_read_or_registration_side_effect_and_staff_requires_auth(monkeypatch):
    from fastapi.testclient import TestClient
    from services.api.main import app
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "fixture", "name": "Fixture"}]
    db = AuditSupabase(storage)
    monkeypatch.setattr("services.api.main.create_client", lambda *args: db)
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    api = TestClient(app)
    response = api.post("/clubs/fixture/tournament-registration/attempts", json=payload())
    assert response.status_code == 202 and response.json() == {"accepted": True}
    assert "no-store" in response.headers["cache-control"]
    assert len(storage[TABLE]) == 1 and not storage["tournament_registrations"]
    assert api.get("/clubs/fixture/tournament-registration/attempts").status_code in (404, 405)
    rejected = api.post("/clubs/fixture/tournament-registration", json={
        **payload(), "first_name": "Terri", "last_name": "Fixture", "gender": "Women", "age": 50,
        "doubles_skill": 3.5, "terms_accepted": True,
        "selections": [{"event_option_id": "event1", "partner_mode": "HAS_PARTNER",
                        "partner_name": "Brent Fixture", "partner_email": "terri@example.invalid",
                        "partner_age": 50, "partner_gender": "Men", "partner_skill": 3.5}],
    })
    assert rejected.status_code == 400
    assert storage[TABLE][-1]["event"] == "SERVER_REJECTED"
    assert storage[TABLE][-1]["error_code"] == "SELF_PARTNER"
    assert not storage["tournament_registrations"]
    private = api.get("/admin/clubs/club-1/tournaments/admin/tournaments/t1/registration-attempts")
    assert private.status_code in (401, 403)
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert api.post("/clubs/fixture/tournament-registration/attempts", json=payload()).status_code == 403


def test_support_errors_strip_links_tokens_and_unknown_exception_details():
    storage = fake_storage()
    db = AuditSupabase(storage)
    data = payload()
    data["message"] = "Try https://example.invalid?edit_token=secret token=private eyJprivate.jwt.signature"
    record_registration_attempt(db, club_id="club-1", payload=data)
    assert "private" not in storage[TABLE][-1]["message"]
    def fail():
        raise RuntimeError("SQL failure with secret details")
    with pytest.raises(RuntimeError):
        audited_registration_call(db, club_id="club-1", payload=data, stage="submit", operation=fail)
    assert storage[TABLE][-1]["message"] == "Registration service could not complete this request."
