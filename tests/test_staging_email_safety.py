from __future__ import annotations

from jupr_app import config
from jupr_app.domain.notifications import player_update_sender as sender


class _FakeTable:
    def update(self, *_args, **_kwargs):
        return self
    def eq(self, *_args, **_kwargs):
        return self
    def insert(self, *_args, **_kwargs):
        return self
    def execute(self):
        class R: data = []
        return R()


class _FakeSupabase:
    def table(self, _name):
        return _FakeTable()


def test_staging_defaults_to_dry_run(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.delenv("JUPR_EMAIL_MODE", raising=False)
    assert config.get_email_mode() == "dry_run"


def test_production_defaults_to_live(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.delenv("JUPR_EMAIL_MODE", raising=False)
    assert config.get_email_mode() == "live"


def test_staging_live_requires_explicit_allow(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "live")
    monkeypatch.delenv("JUPR_ALLOW_STAGING_LIVE_EMAIL", raising=False)
    try:
        config.get_email_mode()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "blocked" in str(exc).lower()


def test_dry_run_does_not_call_smtp_send(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.delenv("JUPR_EMAIL_MODE", raising=False)

    calls = {"smtp": 0, "status": []}

    monkeypatch.setattr(sender, "list_outbox_rows", lambda *a, **k: [{"id": "o1", "subscription_id": "s1", "week_start": "2026-05-01", "week_end": "2026-05-07", "player_id": 9, "email": "real@example.com"}])
    monkeypatch.setattr(sender, "_safe_subscription", lambda *a, **k: {"id": "s1", "request_status": "active", "preferences_json": {"send_only_if_changed": False}})
    monkeypatch.setattr(sender, "_safe_digest_for_week", lambda *a, **k: {"final_json": {"summary": {"matches_played": 1}, "links": {}}})
    monkeypatch.setattr(sender, "_merge_links_for_send", lambda **k: {"summary": {"matches_played": 1}, "links": {"unsubscribe": "https://x/u"}})
    monkeypatch.setattr(sender, "ensure_unsubscribe_token", lambda *a, **k: "tok")
    monkeypatch.setattr(sender, "render_player_digest_chart_png", lambda *a, **k: None)
    monkeypatch.setattr(sender, "build_player_update_email_subject", lambda *a, **k: "Subject")
    monkeypatch.setattr(sender, "build_player_update_email_html", lambda *a, **k: "<p>Hi</p>")
    monkeypatch.setattr(sender, "build_player_update_email_text", lambda *a, **k: "Hi")
    monkeypatch.setattr(sender, "update_outbox_status", lambda *a, **k: calls["status"].append(k))

    def _smtp_should_not_be_called(**kwargs):
        calls["smtp"] += 1
        return "smtp"

    monkeypatch.setattr(sender, "send_email_with_inline_chart", _smtp_should_not_be_called)

    class Ctx:
        supabase = _FakeSupabase()
        club_id = "club"

    result = sender.send_pending_player_update_emails(Ctx(), limit=10)
    assert result["sent"] == 1
    assert result["email_mode"] == "dry_run"
    assert calls["smtp"] == 0


def test_staging_redirect_sends_to_redirect_only(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "staging_redirect")
    monkeypatch.setenv("JUPR_STAGING_EMAIL_REDIRECT_TO", "safe@example.com")

    captured = {"to": None}
    monkeypatch.setattr(sender, "list_outbox_rows", lambda *a, **k: [{"id": "o1", "subscription_id": "s1", "week_start": "2026-05-01", "week_end": "2026-05-07", "player_id": 9, "email": "real@example.com"}])
    monkeypatch.setattr(sender, "_safe_subscription", lambda *a, **k: {"id": "s1", "request_status": "active", "preferences_json": {"send_only_if_changed": False}})
    monkeypatch.setattr(sender, "_safe_digest_for_week", lambda *a, **k: {"final_json": {"summary": {"matches_played": 1}, "links": {}}})
    monkeypatch.setattr(sender, "_merge_links_for_send", lambda **k: {"summary": {"matches_played": 1}, "links": {"unsubscribe": "https://x/u"}})
    monkeypatch.setattr(sender, "ensure_unsubscribe_token", lambda *a, **k: "tok")
    monkeypatch.setattr(sender, "render_player_digest_chart_png", lambda *a, **k: None)
    monkeypatch.setattr(sender, "build_player_update_email_subject", lambda *a, **k: "Subject")
    monkeypatch.setattr(sender, "build_player_update_email_html", lambda *a, **k: "<p>Hi</p>")
    monkeypatch.setattr(sender, "build_player_update_email_text", lambda *a, **k: "Hi")
    monkeypatch.setattr(sender, "update_outbox_status", lambda *a, **k: None)

    def _smtp(**kwargs):
        captured["to"] = kwargs["to_email"]
        return "smtp"

    monkeypatch.setattr(sender, "send_email_with_inline_chart", _smtp)

    class Ctx:
        supabase = _FakeSupabase()
        club_id = "club"

    sender.send_pending_player_update_emails(Ctx(), limit=10)
    assert captured["to"] == "safe@example.com"


def test_summaries_do_not_print_secrets(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret")

    from jupr_app.workers import player_update_email_worker as worker

    monkeypatch.setattr(worker, "make_supabase", lambda u, k: _FakeSupabase())
    monkeypatch.setattr(worker, "send_pending_player_update_emails", lambda *a, **k: {"attempted": 0, "sent": 0, "skipped": 0, "errors": 0, "email_mode": "dry_run"})

    summary = worker.run_player_update_email_worker("club")
    assert "super-secret" not in str(summary)
