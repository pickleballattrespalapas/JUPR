import pytest

from jupr_app.domain.tournament_podium import upsert_tournament_podium
from jupr_app.ui.pages.tournaments import _ensure_tournament_write_payload_has_club_id


class DummyTable:
    def __init__(self) -> None:
        self.upsert_calls = []

    def upsert(self, payload, on_conflict=None):
        if isinstance(payload, list):
            for row in payload:
                if "club_id" not in row:
                    raise RuntimeError("schema mismatch: missing club_id")
        self.upsert_calls.append({"payload": payload, "on_conflict": on_conflict})
        return self

    def execute(self):
        return self


class DummySupabase:
    def __init__(self) -> None:
        self.last_table = None
        self.table_obj = DummyTable()

    def table(self, name: str):
        self.last_table = name
        return self.table_obj


def test_upsert_tournament_podium_is_idempotent():
    supabase = DummySupabase()
    payload = [
        {"tournament_id": "tour1", "placement": 1, "team_id": "t1", "source": "ROUND_ROBIN"},
        {"tournament_id": "tour1", "placement": 2, "team_id": "t2", "source": "ROUND_ROBIN"},
    ]

    upsert_tournament_podium(supabase, "club-1", "tour1", payload)
    upsert_tournament_podium(supabase, "club-1", "tour1", payload)

    assert supabase.last_table == "tournament_podium"
    assert len(supabase.table_obj.upsert_calls) == 2
    for call in supabase.table_obj.upsert_calls:
        assert call["on_conflict"] == "club_id,tournament_id,placement"
        assert all(row["club_id"] == "club-1" for row in call["payload"])


def test_tournament_team_payload_requires_club_id_before_insert():
    with pytest.raises(RuntimeError, match="Missing club_id"):
        _ensure_tournament_write_payload_has_club_id({"team_number": 1}, "")


def test_tournament_team_payload_with_club_id_is_valid():
    payload = {"tournament_id": "tour1", "team_number": 1, "player1_id": 1, "player2_id": 2}
    enriched = _ensure_tournament_write_payload_has_club_id(payload, "club-1")

    assert enriched["club_id"] == "club-1"
