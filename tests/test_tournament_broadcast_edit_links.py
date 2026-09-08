import re
from urllib.parse import parse_qs, urlsplit

import pytest

from jupr_app.domain.notifications import tournament_registrant_broadcast_email as email
from jupr_app.domain import tournament_registration_edit_tokens as tokens
from jupr_app.services import admin_tournament_broadcast_service as service
from jupr_app.services import public_tournament_registration_edit_service as public_edit
from jupr_app.services.admin_tournament_registration_reporting_service import build_admin_tournament_broadcast_preview
from tests.test_admin_tournament_broadcast_service import fixture, prepare, attempt

SECRET = "edit-link-test-secret-not-real-credentials-1234"


@pytest.fixture
def edit_db(fixture, monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", SECRET)
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://registration.example.test")
    fixture.tables["clubs"] = [{"id": "club", "slug": "tres-palapas"}]
    return fixture


def preview(db, **kwargs):
    return build_admin_tournament_broadcast_preview(db, club_id="club", tournament_id="tour_1",
        registration_ids=["registration_1", "reg_shared", "reg_b"], subject="Please update", message="Please change your partner.",
        include_registration_edit_links=True, **kwargs)


def test_preview_groups_only_selected_registrations_without_issuing_links(edit_db, monkeypatch):
    monkeypatch.setattr("jupr_app.services.tournament_broadcast_edit_link_service.build_registration_edit_token",
        lambda **_: pytest.fail("A preview must not issue bearer tokens"))
    first = preview(edit_db)
    second = preview(edit_db, preview_recipient_email="beth@example.com")
    assert first["preview_fingerprint"] == second["preview_fingerprint"]
    assert first["include_registration_edit_links"] is True
    assert "Alex Example" in first["preview"]["html"] and "Sam" in first["preview"]["html"]
    assert "Beth" not in first["preview"]["html"]
    assert "Sam" not in second["preview"]["html"]
    assert first["preview"]["html"].count('aria-disabled="true"') == 2
    assert "edit_token=" not in str(first)
    assert "registration_edit_links" not in first["recipient_csv"]
    assert "Your registration events" not in first["preview"]["text"]
    assert "Your registration events" in preview(edit_db, include_registration_events=True)["preview"]["text"]
    payload = prepare(edit_db, include_registration_edit_links=True)
    created = service.create_tournament_broadcast(edit_db, **payload)
    assert created["include_registration_edit_links"] is True
    assert attempt(edit_db, created["operation_key"])["status"] == "dry_run"
    assert "edit_token=" not in str(edit_db.tables[service.TABLE])
    review = edit_db.tables[service.TABLE][0]["request_json"]["review"]
    assert review["recipients"][0]["registration_edit_links"] == [{"registration_id": "registration_1", "name": "Alex Example"}]


def test_delivered_buttons_open_only_the_selected_registration_and_start_expiry_at_send(edit_db, monkeypatch):
    delivered = []
    clock = [1_800_000_000]
    monkeypatch.setattr(tokens.time, "time", lambda: clock[0])
    monkeypatch.setenv("JUPR_EMAIL_MODE", "live")
    settings = lambda: {"enabled": True, "delivery_mode": "live", "sender": {}}
    monkeypatch.setattr("jupr_app.services.admin_tournament_registration_reporting_service.broadcast_delivery_settings", settings)
    monkeypatch.setattr(service, "broadcast_delivery_settings", settings)
    monkeypatch.setattr(email, "send_email_with_inline_chart", lambda **kw: delivered.append(kw) or "fake-message-id")
    payload = prepare(edit_db, ["registration_1", "reg_shared", "reg_b"], include_registration_edit_links=True, include_registration_events=True)
    clock[0] += 300  # A delayed confirmation must not invalidate the preview.
    created = service.create_tournament_broadcast(edit_db, **payload)
    for index in (0, 1):
        clock[0] += 5 * 86400  # A resumed batch still sends a fresh link.
        assert attempt(edit_db, created["operation_key"], index)["status"] == "sent"
        mail = delivered[-1]
        links = re.findall(r'Edit Registration: (https://\S+)', mail["text_body"])
        expected_ids = ["registration_1", "reg_shared"] if index == 0 else ["reg_b"]
        assert len(links) == len(expected_ids)
        assert mail["html_body"].count(">Edit Registration</a>") == len(expected_ids)
        assert mail["to_email"] == ("alex@example.com" if index == 0 else "beth@example.com")
        for link, registration_id in zip(links, expected_ids):
            url = urlsplit(link)
            query = parse_qs(url.query)
            token = query["edit_token"][0]
            assert url.netloc == "registration.example.test"
            assert url.path == "/clubs/tres-palapas/tournament-registration/edit"
            assert query["tournament_id"] == ["tour_1"]
            verified = tokens.verify_registration_edit_token(token, expected_tournament_id="tour_1",
                expected_registration_id=registration_id, expected_email=mail["to_email"], now=clock[0], secret=SECRET)
            assert int(verified["exp"]) == clock[0] + 48 * 3600
            with pytest.raises(ValueError, match="expired"):
                tokens.verify_registration_edit_token(token, now=clock[0] + 48 * 3600 + 1, secret=SECRET)
            with pytest.raises(ValueError, match="different registration"):
                tokens.verify_registration_edit_token(token, expected_registration_id="not-selected", now=clock[0], secret=SECRET)
            with pytest.raises(ValueError, match="different email"):
                tokens.verify_registration_edit_token(token, expected_email="other@example.com", now=clock[0], secret=SECRET)
            registration = next(row for row in edit_db.tables["tournament_registrations"] if row["id"] == registration_id)
            monkeypatch.setattr(public_edit, "get_registration_confirmation_bundle", lambda *a: {
                "registration": registration, "tournament": {"id": "tour_1", "club_id": "club"}})
            # Exercise the existing public edit endpoint's token/bundle verifier.
            _, bundle = public_edit._verified_bundle(edit_db, club_id="club", edit_token=token, tournament_id="tour_1")
            assert bundle["registration"]["id"] == registration_id
            with pytest.raises(ValueError, match="different club"):
                public_edit._verified_bundle(edit_db, club_id="another-club", edit_token=token, tournament_id="tour_1")
        if index == 1:
            assert "Alex Example" not in mail["text_body"] and "Sam" not in mail["text_body"]
    assert attempt(edit_db, created["operation_key"], 0)["status"] == "sent"
    assert len(delivered) == 2
    assert "edit_token=" not in str(edit_db.tables[service.TABLE])
    assert "edit_token=" not in str(service.get_tournament_broadcast(edit_db, club_id="club", tournament_id="tour_1", operation_key=created["operation_key"]))


@pytest.mark.parametrize("change", ["email", "club_slug", "web_base", "option"])
def test_changed_link_scope_invalidates_confirmation(edit_db, monkeypatch, change):
    payload = prepare(edit_db, include_registration_edit_links=True)
    if change == "email":
        edit_db.tables["tournament_registrations"][0]["email"] = "new@example.com"
    elif change == "club_slug":
        edit_db.tables["clubs"][0]["slug"] = "changed-club"
    elif change == "web_base":
        monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://changed.example.test")
    else:
        payload["include_registration_edit_links"] = False
    with pytest.raises(ValueError, match="Preview the email again"):
        service.create_tournament_broadcast(edit_db, **payload)
    assert not edit_db.tables.get(service.TABLE)


def test_changed_email_after_confirmation_stops_link_delivery(edit_db):
    created = service.create_tournament_broadcast(edit_db, **prepare(edit_db, include_registration_edit_links=True))
    edit_db.tables["tournament_registrations"][0]["email"] = "changed@example.com"
    with pytest.raises(ValueError, match="Participant details changed"):
        attempt(edit_db, created["operation_key"])
    assert len(edit_db.tables[service.TABLE]) == 1


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://user:password@example.test", "https://example.test/?redirect=elsewhere", "http://example.test"])
def test_unsafe_link_destinations_fail_before_confirmation(edit_db, monkeypatch, url):
    monkeypatch.setenv("JUPR_WEB_BASE_URL", url)
    with pytest.raises(ValueError, match="website is not configured"):
        preview(edit_db)


def test_missing_explicit_signing_secret_fails_only_when_buttons_requested(edit_db, monkeypatch):
    def missing():
        raise ValueError("Missing explicit secret")
    monkeypatch.setattr(public_edit, "get_explicit_registration_edit_token_secret", missing)
    with pytest.raises(public_edit.PublicRegistrationEditUnavailableError):
        preview(edit_db)
    assert prepare(edit_db)  # Ordinary Communications remains available.
