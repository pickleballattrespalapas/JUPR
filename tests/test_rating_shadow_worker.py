from __future__ import annotations

import pytest

from jupr_app.workers.rating_shadow_worker import (
    _fetch_matches,
    _require_shadow_worker_enabled,
)


class _Response:
    def __init__(self, data: list[dict]) -> None:
        self.data = data


class _Query:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.page_index = 0

    def select(self, _columns: str) -> "_Query":
        return self

    def eq(self, _column: str, _value: str) -> "_Query":
        return self

    def order(self, _column: str) -> "_Query":
        return self

    def range(self, _start: int, _end: int) -> "_Query":
        return self

    def execute(self) -> _Response:
        page = self.pages[self.page_index]
        self.page_index += 1
        return _Response(page)


class _Supabase:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.query = _Query(pages)

    def table(self, name: str) -> _Query:
        assert name == "matches"
        return self.query


def test_worker_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_RATING_SHADOW_WORKER", raising=False)

    with pytest.raises(ValueError, match="disabled"):
        _require_shadow_worker_enabled()


def test_production_requires_separate_enablement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_RATING_SHADOW_WORKER", "1")
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.delenv("JUPR_ENABLE_PRODUCTION_RATING_SHADOW", raising=False)

    with pytest.raises(ValueError, match="separate approval"):
        _require_shadow_worker_enabled()


def test_worker_can_be_enabled_in_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_RATING_SHADOW_WORKER", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")

    _require_shadow_worker_enabled()


def test_match_fetch_paginates() -> None:
    supabase = _Supabase([[{"id": 1}, {"id": 2}], [{"id": 3}]])

    rows = _fetch_matches(supabase, club_id="club-a", page_size=2)

    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
