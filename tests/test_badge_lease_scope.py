from types import SimpleNamespace

import pytest

from jupr_app.domain.gamification import badge_lease, badge_queue, badge_worker
from jupr_app.domain.matches import side_effects


@pytest.fixture
def scoped(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    monkeypatch.setenv("JUPR_BADGE_WORKER_CLUB_ID", "tres_palapas")
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")


class QueueDB:
    def __init__(self):
        self.calls = []

    def rpc(self, name, args):
        self.calls.append((name, args))
        if name == "claim_badge_eval_queue_job":
            data = [{"id": "other-job", "club_id": "other"}]
        elif name == "claim_badge_eval_queue_job_leased":
            data = [{"id": "pilot-job", "club_id": "tres_palapas", "claim_token": "owner"}]
        else:
            data = True
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=data))

    def table(self, name):
        assert name == "badge_eval_queue"
        return self

    def update(self, payload):
        self.calls.append(("legacy_ack", payload))
        return self

    def eq(self, column, value):
        assert (column, value) == ("id", "other-job")
        return self

    def execute(self):
        return SimpleNamespace(data=[])


def test_pilot_scope_selects_lease_claim_without_changing_other_clubs(scoped):
    db = QueueDB()
    assert badge_lease.leases_enabled()  # Entry-point activation flag.
    assert badge_lease.leases_enabled("tres_palapas")
    assert not badge_lease.leases_enabled("other")
    assert badge_queue.dequeue_badge_eval(db, club_id="tres_palapas")["claim_token"] == "owner"
    assert "claim_token" not in badge_queue.dequeue_badge_eval(db, club_id="other")
    assert [name for name, _ in db.calls] == [
        "claim_badge_eval_queue_job_leased", "claim_badge_eval_queue_job",
    ]


def test_missing_token_fails_closed_for_pilot_or_unknown_club(scoped):
    db = QueueDB()
    for kwargs in ({"club_id": "tres_palapas"}, {}):
        with pytest.raises(badge_lease.BadgeLeaseLost):
            badge_queue.ack_badge_eval(db, job_id="pilot-job", status="done", **kwargs)
    assert db.calls == []


def test_existing_token_always_uses_fenced_completion_even_after_flags_change(scoped, monkeypatch):
    db = QueueDB()
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "0")
    badge_queue.ack_badge_eval(db, job_id="pilot-job", club_id="tres_palapas",
                              status="done", claim_token="owner")
    assert db.calls == [("finish_badge_eval_queue_job", {
        "p_job_id": "pilot-job", "p_claim_token": "owner",
        "p_status": "done", "p_error": None,
    })]


def test_scoped_global_flag_does_not_put_other_club_fallback_on_a_leased_path(scoped, monkeypatch):
    db = QueueDB()
    monkeypatch.setattr(side_effects, "enqueue_badge_eval", lambda *a, **k: {"queued": True})
    monkeypatch.setattr(badge_worker, "_process_job_with_retry", lambda *a, **k: None)
    monkeypatch.setattr(side_effects, "run_live_badge_awards", lambda *a, **k: pytest.fail("Unexpected fallback"))
    result = side_effects.run_badge_side_effects(
        supabase=db, club_id="other", has_badge_eligible_match=True,
        affected_players={1}, db_matches=[{}], match_payloads=[], dedupe_match_id="saved",
    )
    assert result["mode"] == "queue" and result["processed"] == 1
    assert [name for name, _ in db.calls] == ["claim_badge_eval_queue_job", "legacy_ack"]


@pytest.mark.parametrize("leases", ["0", "1"])
@pytest.mark.parametrize("queued", [False, True])
def test_pilot_api_never_falls_back_with_worker_only_or_shared_lease_flag(scoped, monkeypatch, leases, queued):
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", leases)
    monkeypatch.setattr(side_effects, "enqueue_badge_eval", lambda *a, **k: {"queued": queued})
    monkeypatch.setattr(side_effects, "process_badge_eval_queue", lambda *a, **k: pytest.fail("Pilot API drain"))
    monkeypatch.setattr(side_effects, "run_live_badge_awards", lambda *a, **k: pytest.fail("Pilot inline fallback"))
    result = side_effects.run_badge_side_effects(
        supabase=object(), club_id="tres_palapas", has_badge_eligible_match=True,
        affected_players={1}, db_matches=[{}], match_payloads=[], dedupe_match_id="saved",
    )
    assert result["mode"] == ("queue" if queued else "queue_error")
    if not queued:
        assert result["recovery_required"] and result["recovery_match_refs"] == ["saved"]
