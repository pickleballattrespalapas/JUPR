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


def _published_sponsor(fixture):
    sponsor = {"name": "Title Sponsor", "tier": "presenting", "public_description": "Public description", "notes": "Private contract"}
    fixture.tables["tournament_registration_settings"] = [{"tournament_id": "tour_1", "sponsors_json": [sponsor]}]
    return sponsor


def test_published_sponsor_changes_invalidate_review_before_any_email(fixture):
    sponsor = _published_sponsor(fixture)
    payload = prepare(fixture)
    sponsor["public_description"] = "Changed description"
    with pytest.raises(ValueError, match="Preview the email again"):
        service.create_tournament_broadcast(fixture, **payload)
    assert not fixture.tables.get(service.TABLE)


def test_saved_sponsors_are_used_for_delivery_and_shared_address_still_sends_once(fixture, monkeypatch):
    from jupr_app.services import admin_tournament_registration_reporting_service as reporting
    _published_sponsor(fixture)
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw) or {"status": "sent"})
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_shared"]))
    review = fixture.tables[service.TABLE][0]["request_json"]["review"]
    assert "Title Sponsor" in review["preview"]["html"]
    assert "Public description" in review["preview"]["text"]
    assert "Private contract" not in str(review)
    # Delivery uses the reviewed logo copies, not new storage URLs or downloads.
    monkeypatch.setattr(reporting, "prepare_tournament_email_sponsors", lambda *a: pytest.fail("Do not fetch logo copies again after confirmation"))
    attempt(fixture, root["operation_key"])
    attempt(fixture, root["operation_key"])
    assert len(sent) == 1
    assert sent[0]["email_sponsors"] == review["email_sponsors"]


def test_sponsor_change_after_confirmation_stops_only_remaining_recipients(fixture, monkeypatch):
    sponsor = _published_sponsor(fixture)
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw) or {"status": "sent"})
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_b"]))
    attempt(fixture, root["operation_key"], 0)
    sponsor["name"] = "New sponsor"
    assert attempt(fixture, root["operation_key"], 0)["status"] == "sent"
    with pytest.raises(ValueError, match="sponsor details changed"):
        attempt(fixture, root["operation_key"], 1)
    assert len(sent) == 1


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



def _event_preview(db, ids=None, **overrides):
    return build_admin_tournament_broadcast_preview(db, club_id="club", tournament_id="tour_1",
        registration_ids=ids or ["registration_1"], subject="Schedule", message="See you soon.",
        include_registration_events=True, **overrides)


def test_events_are_optional_and_only_selected_registrations_are_included(fixture):
    ordinary = build_admin_tournament_broadcast_preview(fixture, club_id="club", tournament_id="tour_1",
        registration_ids=["registration_1"], subject="Schedule", message="See you soon.")
    assert ordinary["include_registration_events"] is False
    assert "Your registration events" not in ordinary["preview"]["text"]
    assert "registration_events" not in ordinary["recipients"][0]
    preview = _event_preview(fixture)
    assert preview["preview_fingerprint"] != ordinary["preview_fingerprint"]
    assert "Gender Doubles / 3.5" in preview["preview"]["text"]
    assert "Apr 10, 2026" in preview["preview"]["html"]
    assert "Looking for a partner" in preview["preview"]["html"]
    assert "Sam" not in preview["preview"]["html"]  # Unselected, same inbox.
    assert "Original note" not in str(preview["recipients"])
    assert "555-0100" not in str(preview["recipients"])


def test_each_recipient_preview_and_actual_smtp_body_contains_only_their_events(fixture, monkeypatch):
    from jupr_app.domain.notifications import tournament_registrant_broadcast_email as email
    fixture.tables["tournament_registration_selections"].append({
        **fixture.tables["tournament_registration_selections"][0], "id": "selection_b", "registration_id": "reg_b",
        "event_option_id": "event_2", "partner_mode": "KNOWN", "partner_name": "Beth's Partner",
        "partner_email": "private@example.com", "partner_phone": "private-phone"})
    first = _event_preview(fixture, ["registration_1", "reg_b"])
    second = _event_preview(fixture, ["registration_1", "reg_b"], preview_recipient_email="beth@example.com")
    assert first["preview_fingerprint"] == second["preview_fingerprint"]
    assert second["preview"]["to_email"] == "beth@example.com"
    assert "Gender Doubles / 4.0" in second["preview"]["text"]
    assert "Gender Doubles / 3.5" not in second["preview"]["text"]
    assert "Beth's Partner" in second["preview"]["text"]
    assert "private@example.com" not in str(second["recipients"])
    with pytest.raises(ValueError, match="selected participants"):
        _event_preview(fixture, preview_recipient_email="beth@example.com")
    delivered = []
    monkeypatch.setenv("JUPR_EMAIL_MODE", "live")
    monkeypatch.setattr("jupr_app.services.admin_tournament_registration_reporting_service.broadcast_delivery_settings",
        lambda: {"enabled": True, "delivery_mode": "live", "sender": {}})
    monkeypatch.setattr(service, "broadcast_delivery_settings", lambda: {"enabled": True, "delivery_mode": "live", "sender": {}})
    monkeypatch.setattr(email, "send_email_with_inline_chart", lambda **kw: delivered.append(kw) or "fake-id")
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_b"], include_registration_events=True))
    for index in (0, 1, 0):
        attempt(fixture, root["operation_key"], index)
    assert len(delivered) == 2
    assert delivered[0]["to_email"] == "alex@example.com"
    assert "Gender Doubles / 3.5" in delivered[0]["html_body"]
    assert "Gender Doubles / 4.0" not in delivered[0]["text_body"]
    assert "Gender Doubles / 4.0" in delivered[1]["html_body"]
    assert "Alex Example" not in delivered[1]["text_body"]
    assert "private@example.com" not in str(delivered)


def test_shared_mailbox_has_named_sections_for_selected_players_and_sends_once(fixture, monkeypatch):
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(kw) or {"status": "sent"})
    preview = _event_preview(fixture, ["registration_1", "reg_shared"])
    assert preview["recipient_count"] == 1
    assert "Alex Example" in preview["preview"]["html"] and "Sam" in preview["preview"]["html"]
    assert "No registration events are currently listed." in preview["preview"]["text"]
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_shared"], include_registration_events=True))
    attempt(fixture, root["operation_key"])
    attempt(fixture, root["operation_key"])
    assert len(sent) == 1
    assert [row["name"] for row in sent[0]["registration_events"]] == ["Alex Example", "Sam"]


def test_audience_event_filter_does_not_remove_other_events_from_the_email(fixture):
    fixture.tables["tournament_registration_selections"].append({
        **fixture.tables["tournament_registration_selections"][0], "id": "selection_second", "event_option_id": "event_2"})
    preview = _event_preview(fixture, event_option_id="event_1")
    assert "Gender Doubles / 3.5" in preview["preview"]["text"]
    assert "Gender Doubles / 4.0" in preview["preview"]["text"]


@pytest.mark.parametrize("change", ["partner", "date", "division", "removed"])
def test_changed_event_details_require_a_fresh_review_before_confirmation(fixture, change):
    payload = prepare(fixture, include_registration_events=True)
    if change == "partner":
        fixture.tables["tournament_registration_selections"][0]["partner_name"] = "New partner"
    elif change == "date":
        fixture.tables["tournament_registration_days"][0]["event_date"] = "2026-04-11"
    elif change == "division":
        fixture.tables["tournament_event_options"][0]["division_name"] = "4.5"
    else:
        fixture.tables["tournament_registration_selections"] = []
    with pytest.raises(ValueError, match="Preview the email again"):
        service.create_tournament_broadcast(fixture, **payload)
    assert not fixture.tables.get(service.TABLE)


def test_event_change_after_confirmation_pauses_remaining_and_keeps_reviewed_snapshot(fixture, monkeypatch):
    sent = []
    monkeypatch.setattr(service, "send_tournament_registrant_broadcast_email", lambda **kw: sent.append(deepcopy(kw)) or {"status": "sent"})
    root = service.create_tournament_broadcast(fixture, **prepare(fixture, ["registration_1", "reg_b"], include_registration_events=True))
    attempt(fixture, root["operation_key"], 0)
    fixture.tables["tournament_registration_days"][0]["event_date"] = "2026-04-11"
    assert attempt(fixture, root["operation_key"], 0)["status"] == "sent"
    with pytest.raises(ValueError, match="registration events changed"):
        attempt(fixture, root["operation_key"], 1)
    assert len(sent) == 1
    assert sent[0]["registration_events"][0]["events"][0]["event_date"] == "2026-04-10"


def test_event_option_cannot_be_changed_using_an_existing_preview_or_operation(fixture):
    payload = prepare(fixture, include_registration_events=True)
    with pytest.raises(ValueError, match="Preview the email again"):
        service.create_tournament_broadcast(fixture, **{**payload, "include_registration_events": False})
    service.create_tournament_broadcast(fixture, **payload)
    with pytest.raises(ValueError, match="different communications request"):
        service.create_tournament_broadcast(fixture, **{**payload, "include_registration_events": False})


def test_cancelled_registrations_are_labelled_when_explicitly_included(fixture):
    fixture.tables["tournament_registrations"][0]["status"] = "cancelled"
    assert _event_preview(fixture)["recipient_count"] == 0
    preview = _event_preview(fixture, include_cancelled=True)
    assert "Registration cancelled" in preview["preview"]["html"]
