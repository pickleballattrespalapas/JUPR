"""Original generator write paths must keep the shared court transaction authoritative."""
from copy import deepcopy

import pytest

from jupr_app.services import admin_play_generator_service as admin
from jupr_app.services import public_play_generator_service as public
from jupr_app.services import generator_sharing_service as sharing
from jupr_app.services import admin_round_robin_hub_service as hub
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


class NoDirectWrites:
    def table(self, _name):
        raise AssertionError("A linked generator must not write around the hub transaction")


@pytest.fixture
def linked_row():
    event = {"name": "Social RR", "generatorKind": "round_robin", "playFormat": "doubles",
             "status": "active", "currentRoundNumber": 1, "totalRounds": 1,
             "participants": [], "rounds": [], "roundRobinHubId": "hub"}
    return {"club_id": "club", "session_key": "rr", "updated_at": "2026-10-20T02:00:00+00:00",
            "created_at": "2026-10-20T01:00:00+00:00", "status": "active", "version": 7,
            "state": {"mode": "admin_play_generator", "round_robin_hub_id": "hub",
                      "page_state": {"event": event}}}


def intercept(monkeypatch):
    calls = []
    def persist(db, *, before, patch, expected_version):
        calls.append({"before": deepcopy(before), "patch": deepcopy(patch), "version": expected_version})
        return {**deepcopy(before), **deepcopy(patch), "hub_committed": True}
    monkeypatch.setattr(hub, "persist_linked_generator_patch", persist)
    return calls


def test_admin_round_changes_route_through_atomic_hub_commit(monkeypatch, linked_row):
    calls = intercept(monkeypatch)
    event = deepcopy(linked_row["state"]["page_state"]["event"])
    event["status"] = "completed"
    updated = admin._persist_event(NoDirectWrites(), before=linked_row, event=event,
                                  expected_version=linked_row["updated_at"], status="completed")
    assert updated["hub_committed"]
    assert calls[0]["version"] == linked_row["updated_at"]
    assert calls[0]["patch"]["completed_at"]
    assert calls[0]["patch"]["state"]["page_state"]["event"]["status"] == "completed"
    assert linked_row["state"]["page_state"]["event"]["status"] == "active"


def test_public_changes_retain_integer_lease_and_recovery_receipt(monkeypatch, linked_row):
    calls = intercept(monkeypatch)
    operation = {"operation_key": "public-score-retry", "request_fingerprint": "stable"}
    updated = public._update_cas(NoDirectWrites(), row=linked_row,
        patch={"state": linked_row["state"], "expires_at": "2026-10-22T00:00:00Z"}, operation=operation)
    assert updated["hub_committed"]
    patch = calls[0]["patch"]
    assert patch["version"] == 8
    assert patch["last_operation_key"] == "public-score-retry"
    assert patch["last_request_fingerprint"] == "stable"
    assert patch["expires_at"] == "2026-10-22T00:00:00Z"


def test_shared_operator_changes_use_same_commit_and_keep_receipt(monkeypatch, linked_row):
    calls = intercept(monkeypatch)
    updated = sharing._cas(NoDirectWrites(), linked_row,
        {"state": linked_row["state"], "last_operation_key": "handoff-score-retry",
         "last_request_fingerprint": "shared-stable"})
    assert updated["hub_committed"]
    assert calls[0]["patch"]["version"] == 8
    assert calls[0]["patch"]["last_operation_key"] == "handoff-score-retry"
    assert calls[0]["version"] == linked_row["updated_at"]


@pytest.mark.parametrize("failure", [PublicLiveConflictError("stale hub"), hub.HubCommitRejectedError("players are on court")])
def test_hub_conflicts_remain_rejections_without_unsafe_fallback_reads(monkeypatch, linked_row, failure):
    def reject(*args, **kwargs):
        raise failure
    monkeypatch.setattr(hub, "persist_linked_generator_patch", reject)
    with pytest.raises(type(failure), match=str(failure)):
        public._update_cas(NoDirectWrites(), row=linked_row, patch={"state": linked_row["state"]},
                           operation={"operation_key": "retry", "request_fingerprint": "fingerprint"})


def test_all_payloads_expose_hub_context_without_private_state(linked_row):
    for payload in [admin._session_payload(linked_row), public.public_play_generator_session_payload(linked_row)]:
        assert payload["round_robin_hub_id"] == "hub"
        assert "state" not in payload


def test_publication_rejects_active_hub_before_loading_or_writing_matches(monkeypatch, linked_row):
    monkeypatch.setattr(admin, "_live_row", lambda *args, **kwargs: linked_row)
    with pytest.raises(ValueError, match="finish it"):
        admin.publish_play_generator_matches(NoDirectWrites(), club_id="club", session_key="rr",
            match_date="2026-10-19", expected_version=linked_row["updated_at"],
            idempotency_key="publish-hub", operation_key="publish-hub", actor_email="admin@test.invalid",
            actor_role="administrator", source="test")


def test_publication_requires_released_assignments_even_on_completed_session(linked_row):
    linked_row["status"] = "completed"
    event = linked_row["state"]["page_state"]["event"]
    event["status"] = "completed"
    event["hubAssignedMatchIds"] = ["match-on-court"]
    with pytest.raises(ValueError, match="Resolve"):
        admin._require_hub_ready_for_publication(linked_row)
    event["hubAssignedMatchIds"] = []
    admin._require_hub_ready_for_publication(linked_row)


def test_staff_can_manage_only_hub_attached_public_round_robins(linked_row):
    linked_row["state"]["mode"] = "public_play_generator"
    linked_row["state"]["generator_kind"] = "round_robin"
    assert admin._is_generator_row(linked_row, "round_robin")
    assert not admin._is_generator_row(linked_row, "ladder")
    del linked_row["state"]["round_robin_hub_id"]
    assert not admin._is_generator_row(linked_row)
    linked_row["state"]["round_robin_hub_id"] = "hub"
    linked_row["state"]["page_state"]["event"]["generatorKind"] = "ladder"
    assert not admin._is_generator_row(linked_row)


def test_real_hub_preflight_valueerror_has_verified_rejection_type(linked_row):
    before = deepcopy(linked_row)
    del before['state']['round_robin_hub_id']
    with pytest.raises(hub.HubCommitRejectedError, match='not connected'):
        hub.persist_linked_generator_patch(NoDirectWrites(), before=before,
            patch={'state': before['state']}, expected_version=before['updated_at'])


def test_hub_decoder_valueerror_uses_receipt_proof(monkeypatch, linked_row):
    calls = []
    def lost_response(*args, **kwargs):
        raise ValueError('simulated response decoding failure')
    def recovered(*args, **kwargs):
        calls.append(True)
        return {**deepcopy(linked_row), 'receipt_confirmed': True}
    monkeypatch.setattr(hub, 'persist_linked_generator_patch', lost_response)
    monkeypatch.setattr(public, '_recovered_mutation_row', recovered)
    result = public._update_cas(NoDirectWrites(), row=linked_row,
        patch={'state': linked_row['state']}, operation={'operation_key': 'retry', 'request_fingerprint': 'fingerprint'})
    assert calls == [True]
    assert result['receipt_confirmed'] is True


def test_committed_hub_replay_read_decode_failure_stays_uncertain(monkeypatch, linked_row):
    import json
    def unavailable_replay(*args, **kwargs):
        raise json.JSONDecodeError('simulated committed replay decoding failure', '', 0)
    monkeypatch.setattr(hub, '_replayed', lambda *args, **kwargs: True)
    monkeypatch.setattr(hub, '_session_rows', unavailable_replay)
    with pytest.raises(json.JSONDecodeError):
        hub.persist_linked_generator_patch(NoDirectWrites(), before=linked_row,
            patch={'state': linked_row['state']}, expected_version=linked_row['updated_at'])
    # The public caller must consult its immutable proof instead of labeling the
    # already-committed replay a terminal validation rejection.
    monkeypatch.setattr(public, '_recovered_mutation_row', lambda *args, **kwargs: deepcopy(linked_row))
    recovered = public._update_cas(NoDirectWrites(), row=linked_row,
        patch={'state': linked_row['state']}, operation={'operation_key': 'retry', 'request_fingerprint': 'fingerprint'})
    assert recovered == linked_row
