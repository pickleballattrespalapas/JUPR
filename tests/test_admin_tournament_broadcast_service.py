from copy import deepcopy
from uuid import uuid4

import pytest

from jupr_app.services import admin_tournament_broadcast_service as service
from jupr_app.services.admin_tournament_registration_reporting_service import build_admin_tournament_broadcast_preview
from tests.test_admin_match_log_service import FakeSupabase, FakeQuery
from tests.test_api_contract_admin_tournament import tournament_tables


class LedgerQuery(FakeQuery):
    def execute(self):
        if self.table_name == service.TABLE and self.insert_payload:
            key = self.insert_payload["operation_key"]
            if any(row["operation_key"] == key for row in self.storage.get(service.TABLE, [])):
                raise RuntimeError("duplicate key 23505")
        result = super().execute()
        if self.table_name == service.TABLE and self.insert_payload and self.insert_payload["operation_type"].startswith("tournament_broadcast_recipient:") and self.storage.get("__lost_claim_response__"):
            raise RuntimeError("Claim response lost")
        return result


class LedgerSupabase(FakeSupabase):
    def table(self, name):
        return LedgerQuery(self.tables, name)


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = tournament_tables()
    first = tables["tournament_registrations"][0]
    tables["tournament_registrations"].extend([
        {**first, "id": "reg_b", "display_name": "Beth", "email": "beth@example.com"},
        {**first, "id": "reg_shared", "display_name": "Sam", "email": "ALEX@example.com"},
    ])
    return LedgerSupabase(tables)


def prepare(db, ids=None, **overrides):
    data = dict(club_id="club", tournament_id="tour_1", registration_ids=ids or ["registration_1"],
        subject="Update", message="Hi Alex,\nSee you tomorrow.", include_cancelled=False)
    data.update(overrides)
    preview = build_admin_tournament_broadcast_preview(db, **data)
    return {**data, "operation_key": str(uuid4()), "preview_fingerprint": preview["preview_fingerprint"],
        "confirmation_text": service.CONFIRM_SEND, "actor_email": "admin@example.com", "actor_role": "club_owner"}


def attempt(db, key, index=0):
    return service.send_tournament_broadcast_recipient(db, club_id="club", tournament_id="tour_1",
        operation_key=key, recipient_index=index, confirmation_text=service.CONFIRM_SEND,
        actor_email="admin@example.com", actor_role="club_owner")


def test_single_and_bulk_deduplicate_only_selected_participants(fixture, monkeypatch):
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw) or {"status": "sent"})
    payload = prepare(fixture, ["registration_1", "reg_shared"])
    root = service.create_tournament_broadcast(fixture, **payload)
    assert root["recipient_count"] == 1
    assert sent == []  # Confirmed plan is durable before any provider call.
    attempt(fixture, root["operation_key"])
    attempt(fixture, root["operation_key"])
    assert [row["recipient_email"] for row in sent] == ["alex@example.com"]
    assert sent[0]["personalize_greeting"] is False
    assert sent[0]["message_id"]
    replay = service.create_tournament_broadcast(fixture, **payload)
    assert replay["pending_count"] == 0
    assert replay["recipients"][0]["status"] == "sent"


def test_bulk_progress_survives_reload_and_does_not_resend(fixture, monkeypatch):
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw["recipient_email"]) or {"status": "sent"})
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_b"]))
    attempt(fixture, root["operation_key"], 0)
    result = service.get_tournament_broadcast(fixture, club_id="club", tournament_id="tour_1", operation_key=root["operation_key"])
    assert [row["status"] for row in result["recipients"]] == ["sent", "pending"]
    attempt(fixture, root["operation_key"], 0)
    attempt(fixture, root["operation_key"], 1)
    assert sent == ["alex@example.com", "beth@example.com"]
    assert service.list_tournament_broadcasts(fixture, club_id="club", tournament_id="tour_1")["broadcasts"][0]["operation_key"] == root["operation_key"]


@pytest.mark.parametrize("change", ["email", "status", "display_name"])
def test_changed_recipient_invalidates_preview_before_claim_or_send(fixture, change):
    payload = prepare(fixture)
    fixture.tables["tournament_registrations"][0][change] = {"email": "new@example.com", "status": "cancelled", "display_name": "Different Person"}[change]
    with pytest.raises(ValueError):
        service.create_tournament_broadcast(fixture, **payload)
    assert not fixture.tables.get(service.TABLE)


def test_changed_recipient_after_confirmation_stops_remaining_send(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    fixture.tables["tournament_registrations"][0]["email"] = "changed@example.com"
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **_: pytest.fail("Must not send"))
    with pytest.raises(ValueError, match="Participant details changed"):
        attempt(fixture, root["operation_key"])


@pytest.mark.parametrize("field,value", [("message", "New body"), ("registration_ids", ["reg_b"]), ("actor_email", "other@example.com")])
def test_reused_operation_id_cannot_change_scope(fixture, field, value):
    payload = prepare(fixture)
    service.create_tournament_broadcast(fixture, **payload)
    with pytest.raises(ValueError, match="different communications request"):
        service.create_tournament_broadcast(fixture, **{**payload, field: value})


def test_uncertain_smtp_attempt_is_saved_and_never_retried(fixture, monkeypatch):
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise RuntimeError("SMTP password-sensitive-provider-diagnostic")
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", fail)
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    result = attempt(fixture, root["operation_key"])
    assert result["status"] == "uncertain"
    assert "password-sensitive" not in str(result)
    attempt(fixture, root["operation_key"])
    assert len(calls) == 1


def test_claim_response_loss_does_not_enter_smtp(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    fixture.tables["__lost_claim_response__"] = True
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **_: pytest.fail("Must not send"))
    assert attempt(fixture, root["operation_key"])["status"] == "uncertain"
    assert attempt(fixture, root["operation_key"])["status"] == "uncertain"


def test_concurrent_claim_loser_never_sends(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    original = service.get_communications_admin_operation
    reads = []
    def concurrent(db, *, operation_key):
        row = original(db, operation_key=operation_key)
        if operation_key != root["operation_key"] and not reads:
            reads.append(1)
            db.tables[service.TABLE].append({"operation_key": operation_key, "club_id": "club",
                "operation_type": service._recipient_type(root["operation_key"]), "status": "started",
                "request_json": {"broadcast_id": root["operation_key"], "recipient_index": 0,
                    "preview_fingerprint": db.tables[service.TABLE][0]["request_json"]["preview_fingerprint"]}})
            return None
        return row
    monkeypatch.setattr(service, "get_communications_admin_operation", concurrent)
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **_: pytest.fail("Must not send"))
    assert attempt(fixture, root["operation_key"])["status"] == "uncertain"


def test_audit_failure_blocks_delivery(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    fixture.tables["__failed_insert_tables__"] = {"admin_activity_log"}
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **_: pytest.fail("Must not send"))
    with pytest.raises(RuntimeError, match="audit"):
        attempt(fixture, root["operation_key"])


def test_result_save_failure_leaves_uncertain_claim(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw) or {"status": "sent"})
    fixture.tables["__empty_update_tables__"] = {service.TABLE}
    assert attempt(fixture, root["operation_key"])["status"] == "uncertain"
    assert attempt(fixture, root["operation_key"])["status"] == "uncertain"
    assert len(sent) == 1


def test_staging_dry_run_never_calls_smtp(fixture, monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "1")
    monkeypatch.setattr("jupr_app.domain.notifications.tournament_registrant_broadcast_email.send_email_with_inline_chart", lambda **_: pytest.fail("No live email"))
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    assert attempt(fixture, root["operation_key"])["status"] == "dry_run"


def test_closed_gate_blocks_even_an_existing_broadcast(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    with pytest.raises(PermissionError):
        attempt(fixture, root["operation_key"])


def test_mode_change_cannot_turn_reviewed_test_into_live_send(fixture, monkeypatch):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    monkeypatch.setenv("JUPR_EMAIL_MODE", "live")
    with pytest.raises(ValueError, match="settings changed"):
        attempt(fixture, root["operation_key"])


@pytest.mark.parametrize("club,tournament", [("other", "tour_1"), ("club", "missing")])
def test_cross_club_and_tournament_recovery_is_rejected(fixture, club, tournament):
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    with pytest.raises(ValueError):
        service.get_tournament_broadcast(fixture, club_id=club, tournament_id=tournament, operation_key=root["operation_key"])


@pytest.mark.parametrize("patch", [{"registration_ids": []}, {"registration_ids": ["missing"]}, {"confirmation_text": ""}, {"subject": "Header\nBcc: hidden@example.com"}])
def test_invalid_send_requests_fail_without_records(fixture, patch):
    with pytest.raises(ValueError):
        service.create_tournament_broadcast(fixture, **{**prepare(fixture), **patch})
    assert not fixture.tables.get(service.TABLE)


def test_records_and_selection_data_are_never_changed(fixture):
    before = deepcopy(fixture.tables["tournament_registrations"])
    root = service.create_tournament_broadcast(fixture, **prepare(fixture))
    attempt(fixture, root["operation_key"])
    assert fixture.tables["tournament_registrations"] == before
