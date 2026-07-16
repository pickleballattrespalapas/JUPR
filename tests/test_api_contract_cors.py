from __future__ import annotations

import re

import pytest

from services.api.main import get_cors_allowed_origin_regex, get_cors_allowed_origins


def test_cors_origin_regex_is_optional(monkeypatch):
    monkeypatch.delenv("JUPR_ALLOWED_ORIGIN_REGEX", raising=False)
    assert get_cors_allowed_origin_regex() is None


def test_cors_origin_regex_supports_vercel_preview_pattern(monkeypatch):
    pattern = r"^https://jupr(?:-[a-z0-9-]+)?-pickleballattrespalapas1\.vercel\.app$"
    monkeypatch.setenv("JUPR_ALLOWED_ORIGIN_REGEX", pattern)
    assert get_cors_allowed_origin_regex() == pattern
    assert re.fullmatch(pattern, "https://jupr-qctngqniw-pickleballattrespalapas1.vercel.app")
    assert re.fullmatch(
        pattern,
        "https://jupr-git-agent-restore-full-sta-dad76d-pickleballattrespalapas1.vercel.app",
    )
    assert not re.fullmatch(pattern, "https://jupr-preview-attacker.vercel.app")


def test_invalid_cors_origin_regex_fails_fast(monkeypatch):
    monkeypatch.setenv("JUPR_ALLOWED_ORIGIN_REGEX", "[")
    with pytest.raises(re.error):
        get_cors_allowed_origin_regex()


def test_explicit_cors_origins_are_normalized(monkeypatch):
    monkeypatch.setenv(
        "JUPR_ALLOWED_ORIGINS",
        "https://pickleballclubsandwich.com/, https://juprleagues.com/",
    )
    assert get_cors_allowed_origins() == [
        "https://pickleballclubsandwich.com",
        "https://juprleagues.com",
    ]
