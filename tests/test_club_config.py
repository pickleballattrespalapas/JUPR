from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.clubs import get_club_config, get_default_club_id


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, _cols):
        return self

    def eq(self, _key, _value):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeSupabase:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeQuery(self._rows)


class _MissingTableSupabase:
    def table(self, _name):
        raise RuntimeError("relation \"clubs\" does not exist")


def test_default_club_id_can_come_from_env(monkeypatch):
    monkeypatch.setenv("JUPR_DEFAULT_CLUB_ID", "demo_club")
    assert get_default_club_id() == "demo_club"


def test_get_club_config_falls_back_when_table_missing():
    config = get_club_config(_MissingTableSupabase(), "tres_palapas")

    assert config["id"] == "tres_palapas"
    assert config["name"] == "Tres Palapas"
    assert config["tagline"] == "Official player ratings and events for Tres Palapas"


def test_get_club_config_uses_database_row():
    supabase = _FakeSupabase(
        [
            {
                "id": "tres_palapas",
                "slug": "tres-palapas",
                "name": "Tres Palapas Custom",
                "tagline": "DB tagline",
                "support_email": "support@example.org",
                "public_base_url": "https://example.org",
                "is_active": True,
            }
        ]
    )

    config = get_club_config(supabase, "tres_palapas")

    assert config["name"] == "Tres Palapas Custom"
    assert config["tagline"] == "DB tagline"
    assert config["support_email"] == "support@example.org"
