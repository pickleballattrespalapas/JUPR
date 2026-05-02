from __future__ import annotations

import importlib
import sys

import pytest

from jupr_app.config import get_public_base_url, get_smtp_config


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
