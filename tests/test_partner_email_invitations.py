from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.domain import tournament_partner_invitation_tokens as tokens
from jupr_app.domain.notifications import tournament_partner_invitation_email as mail
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token
from jupr_app.services import public_tournament_partner_invitation_service as service
from services.api import public_tournament_pairing_routes as routes
from tests.test_tournament_partner_requests import _FakeSupabase


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "local-test-partner-secret-only")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    row = dict(id="pinv_test", club_id="club", tournament_id="tournament", event_option_id="doubles",
        target_selection_id="target-selection", requester_name="Casey Guest", requester_email="casey@example.com",
        message="Would you like to play?", status="PENDING", verified_at="2026-09-01T00:00:00+00:00",
        expires_at=(datetime.now(timezone.utc) + timedelta(days=14)).isoformat())
    target = dict(id="target", tournament_id="tournament", email="private-target@example.com", display_name="Alex Player", wants_partner_board_contact=True, status="CONFIRMED")
    selection = dict(id="target-selection", registration_id="target", tournament_id="tournament", event_option_id="doubles", partner_mode="NEEDS_PARTNER", show_on_partner_board=True)
    ctx = dict(tournament={"id": "tournament", "name": "Fixture Tournament"}, settings={"partner_board_enabled": True, "registration_slug": "fixture"},
        event={"id": "doubles", "label": "Mixed doubles", "partner_board_enabled": True, "status": "open", "enabled": True}, target=selection, target_registration=target)
    db = _FakeSupabase({service.TABLE: [row], "tournament_registration_selections": [selection], "tournament_registrations": [target]})
    monkeypatch.setattr(service, "get_public_tournament_bundle", lambda *a, **k: (ctx["tournament"], ctx["settings"], [], [ctx["event"]]))
    monkeypatch.setattr(service, "_public_web_base_url", lambda: "https://fixture.invalid")
    return db, row, ctx


def test_token_is_scoped_to_invitation_role_club_and_email_and_expires():
    kw = dict(invitation_id="one", club_id="club", tournament_id="tournament", role="target", email="private@example.com", expires_at=200, secret="test")
    token = tokens.build_partner_invitation_token(**kw)
    verified = tokens.verify_partner_invitation_token(token, club_id="club", now=100, secret="test")
    assert verified["id"] == "one" and verified["role"] == "target"
    assert "private@example.com" not in str(verified)
    for bad, club, now in [(token + "a", "club", 100), (token, "other", 100), (token, "club", 200)]:
        with pytest.raises(ValueError): tokens.verify_partner_invitation_token(bad, club_id=club, now=now, secret="test")
    with pytest.raises(ValueError): verify_registration_edit_token(token, secret="test", now=100)
    edit = build_registration_edit_token(tournament_id="tournament", registration_id="one", email="private@example.com", secret="test", now=100)
    with pytest.raises(ValueError): tokens.verify_partner_invitation_token(edit, club_id="club", now=100, secret="test")


def test_review_is_read_only_and_target_never_receives_requester_registration_capability(context):
    db, row, ctx = context
    before = deepcopy(db.storage)
    result = service.review_invitation(db, club_id="club", club_slug="fixture", token=service._token(row, ctx, "target"))
    assert result["actions"] == ["accept", "decline"]
    assert "registration_url" not in result and "registration_prefill" not in result
    assert "private-target@example.com" not in json.dumps(result)
    assert "casey@example.com" not in json.dumps(result)
    assert db.storage == before


@pytest.mark.parametrize("role,action", [("requester", "accept"), ("requester", "decline"), ("target", "verify"), ("target", "complete"), ("target", "cancel")])
def test_wrong_participant_cannot_respond(context, role, action):
    db, row, ctx = context
    with pytest.raises(ValueError, match="does not allow"):
        service.act_on_invitation(db, club_id="club", club_slug="fixture", token=service._token(row, ctx, role), action=action)


def test_target_email_change_invalidates_old_email_button(context):
    db, row, ctx = context
    token = service._token(row, ctx, "target")
    ctx["target_registration"]["email"] = "changed@example.com"
    with pytest.raises(ValueError, match="no longer available"):
        service.review_invitation(db, club_id="club", club_slug="fixture", token=token)


def test_requester_gets_only_their_own_prefill_and_fragment_capability(context):
    db, row, ctx = context
    result = service.review_invitation(db, club_id="club", club_slug="fixture", token=service._token(row, ctx, "requester"))
    assert result["registration_prefill"] == {"name": "Casey Guest", "email": "casey@example.com", "event_option_id": "doubles"}
    assert "#partner_invitation=" in result["registration_url"]
    assert "private-target@example.com" not in json.dumps(result)


def test_registration_completion_requires_sender_token_same_email_and_division(context):
    db, row, ctx = context
    token = service._token(row, ctx, "requester")
    payload = {"email": "casey@example.com", "selections": [{"event_option_id": "doubles", "partner_mode": "NEEDS_PARTNER"}]}
    service.validate_invitation_registration(db, club_id="club", tournament_id="tournament", token=token, payload=payload)
    for patch in [{"email": "other@example.com"}, {"selections": []}, {"selections": [{"event_option_id": "singles", "partner_mode": "NONE"}]}]:
        with pytest.raises(ValueError): service.validate_invitation_registration(db, club_id="club", tournament_id="tournament", token=token, payload={**payload, **patch})
    with pytest.raises(ValueError): service.validate_invitation_registration(db, club_id="club", tournament_id="other", token=token, payload=payload)
    with pytest.raises(ValueError): service.validate_invitation_registration(db, club_id="club", tournament_id="tournament", token=service._token(row, ctx, "target"), payload=payload)


def test_saved_registration_is_not_reported_failed_when_pairing_fails(context, monkeypatch):
    db, row, ctx = context
    row["status"] = "RESERVED"
    monkeypatch.setattr(service, "_pairing_candidate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")))
    result = service.finish_invitation_registration(db, club_id="club", club_slug="fixture", token=service._token(row, ctx, "requester"), registration_id="saved")
    assert result["status"] == "RETRY_REQUIRED"
    assert "registration is saved" in result["message"]


def test_shared_email_does_not_select_another_person(context):
    db, row, ctx = context
    db.storage["tournament_registrations"].append(dict(id="someone-else", email=row["requester_email"], display_name="Another Person", status="CONFIRMED", tournament_id="tournament"))
    db.storage["tournament_registrations"].append(dict(id="second-person", email=row["requester_email"], display_name="Second Person", status="CONFIRMED", tournament_id="tournament"))
    with pytest.raises(ValueError, match="More than one"):
        service._requester_registration(db, row)


def test_pairing_resolves_verified_email_and_enforces_registration_eligibility(context, monkeypatch):
    db, row, ctx = context
    stamp = "2026-09-09T12:00:00+00:00"
    requester = dict(id="requester", email=row["requester_email"], display_name="Casey Registered",
        tournament_id="tournament", status="CONFIRMED", gender="Men", age=40, doubles_skill=3.5, updated_at=stamp)
    db.storage["tournament_registrations"].append(requester)
    db.storage["tournament_registration_selections"].append(dict(id="requester-selection", registration_id="requester",
        tournament_id="tournament", event_option_id="doubles", partner_mode="NEEDS_PARTNER", updated_at=stamp))
    ctx["target_registration"].update(gender="Women", age=41, doubles_skill=3.5, updated_at=stamp)
    ctx["target"]["updated_at"] = stamp
    ctx["event"].update(event_type="MIXED_DOUBLES", gender_restriction="MIXED", partner_required=True,
        skill_mode="OPEN", age_mode="OPEN", registration_day_id="day", updated_at=stamp)
    monkeypatch.setattr(service, "registration_has_imported_draw_selection", lambda *a, **kw: False)
    selection, versions = service._pairing_candidate(db, row, ctx)
    assert selection == "requester-selection"
    assert set(versions.values()) == {stamp}
    assert service._requester_registration(db, row)["display_name"] == "Casey Registered"
    requester["gender"] = "Women"
    with pytest.raises(ValueError, match="[Mm]ixed"):
        service._pairing_candidate(db, row, ctx)


def test_accept_uses_atomic_rpc_and_returns_updated_state(context, monkeypatch):
    db, row, ctx = context
    calls = []
    monkeypatch.setattr(service, "_pairing_candidate", lambda *a, **k: ("requester-selection", {"version": "checked"}))
    def rpc(db, name, args):
        calls.append((name, args)); row["status"] = "COMPLETED"; return row
    monkeypatch.setattr(service, "_rpc", rpc)
    monkeypatch.setattr(service, "_notify", lambda *a: {"completed_target": "dry_run", "completed_requester": "dry_run"})
    result = service.act_on_invitation(db, club_id="club", club_slug="fixture", token=service._token(row, ctx, "target"), action="accept")
    assert result["status"] == "COMPLETED"
    assert calls == [("transition_tournament_partner_invitation", {"p_invitation_id": "pinv_test", "p_action": "accept", "p_requester_selection_id": "requester-selection", "p_versions": {"version": "checked"}})]


def test_email_message_is_escaped_and_dry_run_never_sends(context, monkeypatch):
    copy = dict(title="Partner request", description="Confirm below", tournament_name="Fixture", division_name="Mixed", requester_name="<Casey>", target_name="Alex", message="<script>oops</script>\nHi!", action_url="https://fixture.invalid/request#token=private", action_label="Accept partnership")
    html, plain = mail.invitation_email(**copy)
    assert "<script>" not in html and "&lt;script&gt;" in html and "Accept partnership" in html
    assert "<script>oops</script>" in plain
    monkeypatch.setattr(mail, "send_email_with_inline_chart", lambda **k: pytest.fail("Dry-run sent email"))
    assert mail.send_partner_invitation_email(to_email="fixture@example.com", **copy) == "dry_run"


def test_public_form_has_no_token_and_review_never_mutates(context, monkeypatch):
    db, row, ctx = context
    app = FastAPI()
    routes.install_public_tournament_pairing_routes(app, get_club=lambda _: {"id": "club"}, get_supabase_client=lambda: db, public_club_payload=lambda *a: {})
    monkeypatch.setattr(routes, "require_public_intake_or_403", lambda: None)
    monkeypatch.setattr(routes, "create_invitation", lambda *a, **k: {"ok": True, "status": "PENDING", "notification_status": {"request_target": "dry_run"}})
    with TestClient(app) as client:
        response = client.post("/clubs/fixture/tournament-registration/partner-invitations", json={"tournament_id": "tournament", "board_entry_key": "opaque", "name": "Casey Guest", "email": "casey@example.com", "message": "Hello", "request_key": "12345678901234567890"})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert "token" not in response.text and "@" not in response.text
        review = client.post("/clubs/fixture/tournament-registration/partner-invitations/review", json={"token": service._token(row, ctx, "target")})
        assert review.status_code == 200 and row["status"] == "PENDING"
        assert client.get("/clubs/fixture/tournament-registration/partner-invitations/respond").status_code == 405


def test_anonymous_send_delivers_to_target_immediately_without_verifying_sender(context, monkeypatch):
    db, row, ctx = context
    row["verified_at"] = None
    payload = dict(tournament_id="tournament", board_entry_key="opaque", name=row["requester_name"],
        email=row["requester_email"], message=row["message"], request_key="same-browser-submission")
    monkeypatch.setattr(service, "_selection_by_public_entry_key", lambda *a, **k: ctx["target"])
    monkeypatch.setattr(service, "load_tournament_email_sponsors", lambda *a, **k: [])
    sent = []
    def send(**kwargs):
        sent.append(kwargs)
        return "dry_run"
    monkeypatch.setattr(service, "send_partner_invitation_email", send)
    def rpc(_db, name, args):
        if name == "create_tournament_partner_invitation":
            assert args["p_invitation"]["send_directly"] is True
            assert args["p_invitation"]["verified"] is False
            return dict(row)
        assert name == "claim_partner_invitation_delivery"
        assert args["p_kind"] == "request_target", "The sender must never receive a verification email"
        deliveries = db.storage.setdefault("tournament_partner_invitation_deliveries", [])
        if deliveries:
            return False
        deliveries.append(dict(invitation_id=row["id"], kind="request_target", status="sending", attempt_id=args["p_attempt_id"]))
        return True
    monkeypatch.setattr(service, "_rpc", rpc)
    for _ in range(2):
        result = service.create_invitation(db, club_id="club", club_slug="fixture", payload=payload)
        assert result == {"ok": True, "status": "PENDING", "notification_status": {"request_target": "dry_run"}}
        assert "@" not in json.dumps(result) and "token" not in json.dumps(result)
    assert len(sent) == 1
    assert sent[0]["to_email"] == ctx["target_registration"]["email"]
    assert sent[0]["message"] == payload["message"]
    assert sent[0]["action_label"] == "Accept partnership"
    assert row["verified_at"] is None


def test_direct_sender_private_email_link_can_complete_registration(context):
    db, row, ctx = context
    row.update(verified_at=None, status="RESERVED")
    token = service._token(row, ctx, "requester")
    result = service.review_invitation(db, club_id="club", club_slug="fixture", token=token)
    assert result["registration_prefill"]["email"] == row["requester_email"]
    service.validate_invitation_registration(db, club_id="club", tournament_id="tournament", token=token,
        payload={"email": row["requester_email"], "selections": [{"event_option_id": "doubles", "partner_mode": "NEEDS_PARTNER"}]})
    assert row["verified_at"] is None, "Review stays read-only"


def test_direct_sender_registration_requires_matching_name_and_email(context):
    db, row, ctx = context
    row["verified_at"] = None
    registration = dict(id="requester", email=row["requester_email"], display_name="Different Person",
        tournament_id="tournament", status="CONFIRMED")
    db.storage["tournament_registrations"].append(registration)
    with pytest.raises(ValueError, match="does not match"):
        service._requester_registration(db, row)
    registration["display_name"] = "  CASEY   GUEST  "
    assert service._requester_registration(db, row)["id"] == "requester"


def test_honeypot_does_not_send_email_or_save_invitation(context, monkeypatch):
    db, _, _ = context
    monkeypatch.setattr(service, "_rpc", lambda *a, **k: pytest.fail("Honeypot persisted data"))
    monkeypatch.setattr(service, "_notify", lambda *a, **k: pytest.fail("Honeypot sent email"))
    assert service.create_invitation(db, club_id="club", club_slug="fixture", payload={"website": "spam"})["status"] == "PENDING"
