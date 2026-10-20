from __future__ import annotations

from types import SimpleNamespace
import json
import smtplib
from uuid import NAMESPACE_URL, uuid5

import pytest

from jupr_app.config import SMTPConfig
from jupr_app.domain.notifications import smtp_mailer
from jupr_app.services.staging_write_guard import staging_communications_mutations_enabled
from jupr_app.workers import player_update_email_scheduler as scheduler
from jupr_app.workers import player_update_email_worker as worker
from scripts import production_email_probe as probe


def _enable(monkeypatch):
    for name, value in {
        "JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "server-only-test-key",
        "JUPR_PRODUCTION_WRITE_POLICY": "enabled", "JUPR_STAGING_WRITE_WAVE": "none",
        "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS": "1",
        "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL": "1",
        "JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS": "1",
        "JUPR_REQUIRE_WORKER_RUN_LOG": "1", "JUPR_EMAIL_MODE": "live",
        "SMTP_HOST": "smtp.example.org", "SMTP_PORT": "587",
        "SMTP_USERNAME": "test-user", "SMTP_PASSWORD": "do-not-print-this",
        "SMTP_FROM_EMAIL": "notifications@example.org", "SMTP_USE_TLS": "1",
    }.items():
        monkeypatch.setenv(name, value)


def test_production_communication_controls_require_both_gates(monkeypatch):
    _enable(monkeypatch)
    assert staging_communications_mutations_enabled()
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    assert not staging_communications_mutations_enabled()
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "enabled")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "0")
    assert not staging_communications_mutations_enabled()


def test_api_starts_worker_and_stops_it_on_shutdown(monkeypatch):
    _enable(monkeypatch)
    from fastapi.testclient import TestClient
    from services.api.main import app
    with TestClient(app) as client:
        assert client.get("/health").json()["write_prerequisites"]["player_update_worker_running"]
        task = app.state.player_email_task
    assert task.cancelled()


def test_tournament_handoff_queues_without_delivering_admin_drafts(monkeypatch):
    _enable(monkeypatch)
    from jupr_app.services import admin_player_updates_service as service
    monkeypatch.setattr(service, "_build_ctx", lambda *_a, **_kw: pytest.fail("must not send inline"))
    result = service.auto_send_player_updates_for_match_payloads(None, club_id="club", match_payloads=[{"date": "2026-09-06"}])
    assert result["mode"] == "queued"


def test_selected_outbox_rows_are_filtered_before_page_limit(monkeypatch):
    from tests.test_admin_match_log_service import FakeQuery, FakeSupabase
    monkeypatch.setattr(FakeQuery, "range", lambda self, start, end: self.limit(end - start + 1), raising=False)
    from jupr_app.domain.notifications.player_profile_update_repo import list_outbox_rows
    db = FakeSupabase({"player_profile_update_outbox": [
        {"id": "newer", "club_id": "club", "send_status": "pending", "created_at": "2026-09-06"},
        {"id": "older", "club_id": "club", "send_status": "pending", "created_at": "2026-09-05"},
    ]})
    assert [row["id"] for row in list_outbox_rows(db, "club", status="pending", limit=1, outbox_ids=["older"])] == ["older"]


def test_sender_passes_selected_ids_to_the_database_query(monkeypatch):
    from jupr_app.domain.notifications import player_update_sender as sender
    captured = {}
    monkeypatch.setattr(sender, "list_outbox_rows", lambda *_a, **kw: captured.update(kw) or [])
    sender.send_pending_player_update_emails(
        SimpleNamespace(supabase=None, club_id="club"), limit=1,
        outbox_items=[{"id": "older", "expected_row_version": 1}],
    )
    assert captured["outbox_ids"] == ["older"]


@pytest.mark.parametrize("name,value", [
    ("JUPR_ENV", "staging"), ("FLY_APP_NAME", "juprleagues-api-staging"),
    ("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co"),
    ("JUPR_PRODUCTION_WRITE_POLICY", "read_only"), ("JUPR_EMAIL_MODE", "dry_run"),
    ("JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS", "0"),
    ("JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL", "0"),
    ("JUPR_REQUIRE_WORKER_RUN_LOG", "0"), ("SMTP_PASSWORD", ""),
])
def test_scheduler_cannot_send_outside_approved_production(monkeypatch, name, value):
    _enable(monkeypatch)
    assert scheduler.scheduler_enabled()
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_: pytest.fail("must not access DB"))
    assert not scheduler.scheduler_enabled()
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}


def test_scheduler_only_delivers_pending_clubs_once_per_poll(monkeypatch):
    _enable(monkeypatch)
    filters = []
    class Query:
        def table(self, name):
            assert name == "player_profile_update_outbox"
            return self
        def select(self, fields):
            assert "queue_operation_key" in fields
            return self
        def eq(self, name, value):
            filters.append((name, value))
            return self
        def order(self, name):
            return self
        def limit(self, count):
            return self
        def execute(self):
            rows = []
            for i, club in enumerate(("a", "a", "b")):
                rows.append({"id": str(i), "club_id": club, "subscription_id": "s",
                    "week_start": "2026-09-06", "week_end": "2026-09-06", "row_version": 1,
                    "queue_operation_key": str(uuid5(NAMESPACE_URL, f"jupr:auto-player-update:{club}:s:2026-09-06:2026-09-06"))})
            rows.append({"id": "manual", "club_id": "c", "queue_operation_key": "manual-operation"})
            return SimpleNamespace(data=rows)
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_: Query())
    calls = []
    monkeypatch.setattr(scheduler, "run_player_update_email_worker", lambda club, **kw: calls.append((club, kw)) or {"sent": 2})
    assert scheduler.deliver_pending_updates() == {"clubs": 2, "sent": 4, "errors": 0}
    assert filters == [("send_status", "pending"), ("digest_snapshot_json", "{}")]
    assert calls == [
        ("a", {"limit": 25, "outbox_items": [{"id": "0", "expected_row_version": 1}, {"id": "1", "expected_row_version": 1}]}),
        ("b", {"limit": 25, "outbox_items": [{"id": "2", "expected_row_version": 1}]}),
    ]


def test_missing_worker_marker_stops_before_delivery(monkeypatch):
    _enable(monkeypatch)
    class Query:
        def table(self, _): return self
        def insert(self, _): return self
        def execute(self): return SimpleNamespace(data=[{}])
    monkeypatch.setattr(worker, "make_supabase", lambda *_: Query())
    monkeypatch.setattr(worker, "send_pending_player_update_emails", lambda *_a, **_kw: pytest.fail("must not send"))
    with pytest.raises(RuntimeError, match="marker"):
        worker.run_player_update_email_worker("club")


class FakeSMTP:
    calls = []
    def __init__(self, host, port, **kwargs):
        self.calls.append(("connect", port, "context" in kwargs))
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def ehlo(self): self.calls.append(("ehlo",))
    def starttls(self, *, context): self.calls.append(("starttls", context.check_hostname))
    def login(self, username, password): self.calls.append(("login",))
    def sendmail(self, *args): self.calls.append(("sendmail",))


@pytest.mark.parametrize("port", [465, 587])
def test_smtp_probe_authenticates_without_delivering(monkeypatch, port):
    _enable(monkeypatch)
    monkeypatch.setenv("SMTP_PORT", str(port))
    FakeSMTP.calls = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    result = probe.probe_production_email()
    assert result["ok"] and result["messages_sent"] == 0
    assert FakeSMTP.calls[0] == ("connect", port, port == 465)
    assert (("starttls", True) in FakeSMTP.calls) == (port == 587)
    assert ("login",) in FakeSMTP.calls
    assert ("sendmail",) not in FakeSMTP.calls


def test_smtp_probe_hides_provider_error_details(monkeypatch):
    _enable(monkeypatch)
    def fail(*_a, **_kw):
        raise smtplib.SMTPAuthenticationError(535, b"rejected do-not-print-this")
    monkeypatch.setattr(smtplib, "SMTP", fail)
    result = probe.probe_production_email()
    assert result["ok"] is False and result["smtp_code"] == 535
    assert "do-not-print-this" not in json.dumps(result)


def test_smtp_probe_rejects_missing_configuration_without_connecting(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.delenv("SMTP_PASSWORD")
    monkeypatch.setattr(smtplib, "SMTP", lambda *_a, **_kw: pytest.fail("must not connect"))
    assert probe.probe_production_email()["missing"] == ["SMTP_PASSWORD"]


def test_port_465_uses_implicit_tls_and_does_not_starttls_again(monkeypatch):
    cfg = SMTPConfig("smtp.example.org", 465, "user", "secret", "notify@example.org", "PCS", "reply@example.org", True)
    FakeSMTP.calls = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP", lambda *_a, **_kw: pytest.fail("wrong transport"))
    message_id = smtp_mailer.send_email_with_inline_chart(to_email="test@example.org", subject="Test", html_body="Test", text_body="Test", smtp_config=cfg)
    assert message_id.endswith("@example.org>")
    assert FakeSMTP.calls == [("connect", 465, True), ("ehlo",), ("login",), ("sendmail",)]
