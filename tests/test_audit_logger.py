from __future__ import annotations

from jupr_app.domain import audit_logger


def test_log_event_swallows_insert_errors(monkeypatch):
    monkeypatch.setattr(audit_logger, "sb_insert", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no table")))

    audit_logger.log_event(
        supabase=object(),
        club_id="club-1",
        actor="tester",
        action_type="record_match",
        payload={"id": 1},
    )


def test_log_event_includes_required_fields(monkeypatch):
    inserted = {}

    def _fake_insert(_supabase, table, payload):
        inserted["table"] = table
        inserted["payload"] = payload

    monkeypatch.setattr(audit_logger, "sb_insert", _fake_insert)

    audit_logger.log_event(
        supabase=object(),
        club_id="club-1",
        actor="tester",
        action_type="update_match",
        payload={"id": 2},
    )

    assert inserted["table"] == "admin_audit_events"
    assert inserted["payload"]["club_id"] == "club-1"
    assert inserted["payload"]["actor"] == "tester"
    assert inserted["payload"]["action_type"] == "update_match"
    assert inserted["payload"]["payload_json"] == {"id": 2}
    assert inserted["payload"]["created_at"]
