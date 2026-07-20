from __future__ import annotations

import importlib
import sys

import pytest

from jupr_app.config import SMTPConfig, get_next_web_base_url, get_public_base_url, get_smtp_config


def test_smtp_mailer_imports_without_streamlit(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", None)
    module = importlib.import_module("jupr_app.domain.notifications.smtp_mailer")
    assert module is not None


def test_get_smtp_config_missing_is_clear(monkeypatch):
    for key in ["SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL"]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError) as exc:
        get_smtp_config()

    assert "Missing SMTP configuration" in str(exc.value)
    assert "SMTP_HOST" in str(exc.value)


def test_get_public_base_url_from_env(monkeypatch):
    monkeypatch.setenv("JUPR_PUBLIC_BASE_URL", "https://example.org")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://fallback.example.org")
    assert get_public_base_url() == "https://example.org"


def test_get_next_web_base_url_prefers_next_origin_over_legacy_streamlit_origin(monkeypatch):
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.org/")
    monkeypatch.setenv("JUPR_PUBLIC_BASE_URL", "https://legacy.streamlit.app")

    assert get_next_web_base_url() == "https://next.example.org"


def test_player_update_sender_imports_without_streamlit(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", None)
    module = importlib.import_module("jupr_app.domain.notifications.player_update_sender")
    assert module is not None


def test_send_email_uses_provided_smtp_config(monkeypatch):
    from jupr_app.domain.notifications import smtp_mailer

    class FakeSMTP:
        payloads = []
        def __init__(self, host, port, timeout=30):
            self.host = host
            self.port = port
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def ehlo(self):
            return None

        def starttls(self):
            return None

        def login(self, username, password):
            self.username = username
            self.password = password

        def sendmail(self, from_email, to_emails, payload):
            self.from_email = from_email
            self.to_emails = to_emails
            self.payload = payload
            self.__class__.payloads.append(payload)

    monkeypatch.setattr(smtp_mailer.smtplib, "SMTP", FakeSMTP)

    def _raise_if_called():
        raise AssertionError("get_smtp_config should not be called when smtp_config is provided")

    monkeypatch.setattr(smtp_mailer, "get_smtp_config", _raise_if_called)

    cfg = SMTPConfig(
        host="smtp.example.org",
        port=2525,
        username="user",
        password="pass",
        from_email="noreply@example.org",
        from_name="JUPR Notifications",
        reply_to="reply@example.org",
        use_tls=True,
    )

    provider = smtp_mailer.send_email_with_inline_chart(
        to_email="to@example.org",
        subject="Subject",
        html_body="<p>hi</p>",
        text_body="hi",
        smtp_config=cfg,
    )
    assert provider == "smtp"

    attempt_id = "11111111-1111-1111-1111-111111111111"
    provider = smtp_mailer.send_email_with_inline_chart(
        to_email="to@example.org",
        subject="Subject",
        html_body="<p>hi</p>",
        text_body="hi",
        smtp_config=cfg,
        message_id=attempt_id,
    )
    assert provider == f"<{attempt_id}@notifications.juprleagues.com>"
    assert f"Message-ID: <{attempt_id}@notifications.juprleagues.com>" in FakeSMTP.payloads[-1]
