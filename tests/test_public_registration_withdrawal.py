from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, registration_edit_email_hash
from jupr_app.domain.tournament_registration_repo import TournamentRegistrationEditConflictError, TournamentRegistrationRelationshipLockedError
from jupr_app.services.public_tournament_registration_edit_service import withdraw_public_tournament_registration, PublicRegistrationEditUnavailableError


class WithdrawalDB:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def rpc(self, name, params):
        self.calls.append((name, params))
        return self

    def execute(self):
        if self.error:
            raise RuntimeError(self.error)
        return SimpleNamespace(data={"ok": True, "withdrawn": True, "registration_id": "owner"})


@pytest.fixture
def withdrawal(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "withdrawal-test-secret-at-least-32-bytes")
    token = build_registration_edit_token(tournament_id="tournament", registration_id="owner", email="owner@example.invalid", secret="withdrawal-test-secret-at-least-32-bytes")
    payload = {"tournament_id": "tournament", "confirmed": True, "expected_updated_at": "2026-10-30T00:00:00Z",
               "expected_selection_versions": [{"id": "entry", "updated_at": "2026-10-30T00:00:00Z"}]}
    return token, payload


def test_withdrawal_uses_only_token_bound_identity_and_versioned_transaction(withdrawal):
    token, payload = withdrawal
    db = WithdrawalDB()
    result = withdraw_public_tournament_registration(db, club_id="club", edit_token=token,
        payload={**payload, "registration_id": "someone-else", "email": "someone-else@example.invalid"})
    assert result["withdrawn"] is True
    name, params = db.calls[0]
    assert name == "server_withdraw_public_tournament_registration"
    assert params["p_registration_id"] == "owner"
    assert params["p_email_hash"] == registration_edit_email_hash("owner@example.invalid")
    assert params["p_expected_selection_versions"] == payload["expected_selection_versions"]
    assert params["p_club_id"] == "club"


@pytest.mark.parametrize("case", ["tampered", "expired", "other_tournament", "unconfirmed"])
def test_withdrawal_rejects_unverified_requests_before_rpc(withdrawal, case):
    token, payload = withdrawal
    db = WithdrawalDB()
    if case == "tampered": token += "x"
    if case == "expired": token = build_registration_edit_token(tournament_id="tournament", registration_id="owner", email="owner@example.invalid", now=0, secret="withdrawal-test-secret-at-least-32-bytes")
    if case == "other_tournament": payload["tournament_id"] = "other"
    if case == "unconfirmed": payload["confirmed"] = False
    with pytest.raises(ValueError):
        withdraw_public_tournament_registration(db, club_id="club", edit_token=token, payload=payload)
    assert not db.calls


@pytest.mark.parametrize("marker,kind,message", [
    ("JUPR_WITHDRAW_IDENTITY", ValueError, "Invalid registration edit link"),
    ("JUPR_CANCEL_CONFLICT", TournamentRegistrationEditConflictError, "changed"),
    ("JUPR_CANCEL_IMPORTED", TournamentRegistrationRelationshipLockedError, "draw"),
    ("JUPR_CANCEL_CAPTAIN", TournamentRegistrationRelationshipLockedError, "captain"),
    ("JUPR_TOURNAMENT_COMMERCE_FULFILLED_REGISTRATION_CANCEL_LOCKED", TournamentRegistrationRelationshipLockedError, "fulfilled extras"),
    ("database secret failure", PublicRegistrationEditUnavailableError, "temporarily unavailable"),
])
def test_withdrawal_failures_have_safe_actionable_messages(withdrawal, marker, kind, message):
    token, payload = withdrawal
    with pytest.raises(kind, match=message):
        withdraw_public_tournament_registration(WithdrawalDB(marker), club_id="club", edit_token=token, payload=payload)
