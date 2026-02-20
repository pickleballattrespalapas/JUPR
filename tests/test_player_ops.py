from __future__ import annotations

from types import SimpleNamespace

from postgrest.exceptions import APIError

from jupr_app.domain.player_ops import safe_add_player


class _Query:
    def __init__(self, supabase, table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self._filters: list[tuple[str, object]] = []

    def upsert(self, payload, on_conflict=None):
        self.supabase.last_payload = dict(payload)
        self.supabase.last_on_conflict = on_conflict
        return self

    def select(self, _fields: str):
        return self

    def eq(self, col: str, value):
        self._filters.append((col, value))
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        if self.supabase.raise_api_error is not None:
            raise self.supabase.raise_api_error
        if self.supabase.upsert_data is not None:
            data = self.supabase.upsert_data
            self.supabase.upsert_data = None
            return SimpleNamespace(data=data)

        rows = self.supabase.rows.get(self.table_name, [])
        for col, value in self._filters:
            rows = [row for row in rows if str(row.get(col)) == str(value)]
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self):
        self.rows = {"players": []}
        self.upsert_data = None
        self.raise_api_error = None
        self.last_payload = None
        self.last_on_conflict = None

    def table(self, name: str):
        return _Query(self, name)


def test_safe_add_player_sets_normalized_name_and_conflict_target():
    supabase = _Supabase()
    supabase.upsert_data = [{"id": 17}]

    ok, player_id = safe_add_player(
        supabase=supabase,
        club_id="club-1",
        name="  Alice   Smith ",
        rating_jupr=3.5,
    )

    assert ok is True
    assert player_id == 17
    assert supabase.last_on_conflict == "club_id,normalized_name"
    assert supabase.last_payload["normalized_name"] == "alice smith"


def test_safe_add_player_falls_back_to_lookup_when_upsert_returns_empty_data():
    supabase = _Supabase()
    supabase.upsert_data = []
    supabase.rows["players"].append({"id": 99, "club_id": "club-1", "normalized_name": "alice smith"})

    ok, player_id = safe_add_player(
        supabase=supabase,
        club_id="club-1",
        name="Alice Smith",
        rating_jupr=3.5,
    )

    assert ok is True
    assert player_id == 99


def test_safe_add_player_surfaces_schema_mismatch_for_on_conflict():
    supabase = _Supabase()
    supabase.raise_api_error = APIError(
        {
            "code": "42P10",
            "message": "there is no unique or exclusion constraint matching the ON CONFLICT specification",
            "hint": None,
            "details": None,
        }
    )

    ok, err = safe_add_player(
        supabase=supabase,
        club_id="club-1",
        name="Alice Smith",
        rating_jupr=3.5,
    )

    assert ok is False
    assert "Schema mismatch" in err
