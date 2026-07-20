from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._limit: int | None = None
        self._order_key: str | None = None
        self._order_desc = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def order(self, key, desc=False):
        self._order_key = key
        self._order_desc = bool(desc)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self._order_key) or ""), reverse=self._order_desc)
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


@pytest.fixture
def client(monkeypatch):
    tables = {
        "clubs": [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}],
        "weekly_recaps": [
            {
                "club_id": "club-1",
                "week_start": "2026-02-01",
                "week_end": "2026-02-07",
                "status": "draft",
                "final_json": {"highlights": ["draft should not appear"]},
            },
            {
                "club_id": "club-1",
                "week_start": "2026-02-08",
                "week_end": "2026-02-14",
                "status": "published",
                "updated_at": "2026-02-15T00:00:00Z",
                "generated_json": {"private": True},
                "edits_json": {"private": True},
                "final_json": {
                    "numbers": {"matches": 12, "players": 18},
                    "spotlight": [
                        {
                            "key": "TOP_PERFORMER_WEEK",
                            "label": "Top Performer",
                            "players": ["Alex"],
                            "description": "Best week",
                            "order": 1,
                            "include": True,
                            "candidate_ids": [1],
                        }
                    ],
                    "around_club": {"leagues": [{"league_name": "Open", "highlights": [{"display": "Open had 8 matches"}]}]},
                    "looking_ahead": ["More matches next week"],
                    "internal_notes": "private",
                },
            },
        ],
    }
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    return TestClient(app)


def test_public_weekly_recaps_contract(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert len(payload["recaps"]) == 1
    assert payload["pagination"] == {"page": 1, "page_size": 8, "has_previous": False, "has_next": False}
    assert payload["recaps"][0]["week_start"] == "2026-02-08"
    assert payload["selected_recap"]["recap"]["numbers"] == {"matches": 12, "players": 18}
    assert len(payload["selected_recap"]["recap"]["numbers_cards"]) == 6
    assert payload["selected_recap"]["recap"]["spotlight"][0]["players"] == ["Alex"]
    assert "candidate_ids" not in payload["selected_recap"]["recap"]["spotlight"][0]
    assert "internal_notes" not in payload["selected_recap"]["recap"]
    assert "admin_notes" not in payload["club"]
    serialized = json.dumps(payload)
    assert "draft should not appear" not in serialized
    assert "generated_json" not in serialized
    assert "edits_json" not in serialized
    assert "private" not in serialized


def test_public_weekly_recap_detail_contract(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps/2026-02-08")

    assert response.status_code == 200
    assert response.json()["selected_recap"]["week_start"] == "2026-02-08"


def test_public_weekly_recap_pdf_contract(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps/2026-02-08/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-1.4")


def test_public_weekly_recap_missing_detail_returns_404(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps/2026-01-01")

    assert response.status_code == 404


def test_public_weekly_recap_draft_detail_is_not_public(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps/2026-02-01")

    assert response.status_code == 404


def test_public_weekly_recap_paging_contract(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps", params={"page": 2, "page_size": 1})

    assert response.status_code == 200
    assert response.json()["recaps"] == []
    assert response.json()["selected_recap"] is None
    assert response.json()["pagination"] == {"page": 2, "page_size": 1, "has_previous": True, "has_next": False}


def test_public_weekly_recap_page_size_is_bounded(client):
    response = client.get("/clubs/tres-palapas/weekly-recaps", params={"page_size": 13})

    assert response.status_code == 422
