from jupr_app.domain.tournament_podium import upsert_tournament_podium


class DummyTable:
    def __init__(self) -> None:
        self.upsert_calls = []

    def upsert(self, payload, on_conflict=None):
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

    upsert_tournament_podium(supabase, "tour1", payload)
    upsert_tournament_podium(supabase, "tour1", payload)

    assert supabase.last_table == "tournament_podium"
    assert len(supabase.table_obj.upsert_calls) == 2
    for call in supabase.table_obj.upsert_calls:
        assert call["payload"] == payload
        assert call["on_conflict"] == "tournament_id,placement"
