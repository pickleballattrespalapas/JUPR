from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.clubs import (
    create_club_config,
    get_club_by_slug,
    get_club_config,
    list_clubs,
    validate_club_slug,
)


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters = []

    def select(self, _cols):
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def limit(self, _n):
        return self

    def insert(self, payload):
        self._insert_payload = payload
        self._rows.append(payload)
        return self

    def execute(self):
        rows = self._rows
        for key, value in self._filters:
            rows = [r for r in rows if r.get(key) == value]
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _Query(self._rows)


class _MissingTableSupabase:
    def table(self, _name):
        raise RuntimeError('relation "clubs" does not exist')


def test_validate_slug_accepts_valid_value():
    assert validate_club_slug("  New-Club-01 ") == "new-club-01"


def test_validate_slug_rejects_invalid_value():
    with pytest.raises(ValueError, match="only lowercase letters"):
        validate_club_slug("Bad Slug!")


def test_duplicate_slug_is_rejected():
    supabase = _Supabase([{"id": "c1", "slug": "pilot-club", "name": "Pilot", "is_active": True}])
    with pytest.raises(ValueError, match="already exists"):
        create_club_config(supabase, club_id="c2", slug="pilot-club", name="Another")


def test_inactive_club_is_hidden_by_default():
    supabase = _Supabase(
        [
            {"id": "a", "slug": "active", "name": "Active", "is_active": True},
            {"id": "i", "slug": "inactive", "name": "Inactive", "is_active": False},
        ]
    )
    visible = list_clubs(supabase)
    all_rows = list_clubs(supabase, include_inactive=True)
    assert [r["slug"] for r in visible] == ["active"]
    assert len(all_rows) == 2


def test_get_club_by_slug_honors_include_inactive():
    supabase = _Supabase([{"id": "i", "slug": "inactive", "name": "Inactive", "is_active": False}])
    assert get_club_by_slug(supabase, "inactive") is None
    assert get_club_by_slug(supabase, "inactive", include_inactive=True)["id"] == "i"


def test_tres_palapas_fallback_when_table_missing():
    config = get_club_config(_MissingTableSupabase(), "tres_palapas")
    assert config["slug"] == "tres-palapas"
    assert config["name"] == "Tres Palapas"


def test_create_club_config_without_production_only_fields():
    supabase = _Supabase([])
    created = create_club_config(
        supabase,
        club_id="staging-club-2",
        slug="staging-club-2",
        name="Staging Club 2",
    )
    assert created["id"] == "staging-club-2"
    assert created["slug"] == "staging-club-2"
    assert created["features_json"] == {}
