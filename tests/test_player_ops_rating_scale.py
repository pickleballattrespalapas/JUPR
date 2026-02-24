from jupr_app.domain.player_ops import _coerce_rating_to_elo, get_or_create_player


class _Query:
    def __init__(self, sink):
        self.sink = sink

    def upsert(self, payload, on_conflict=None, returning=None):
        self.sink["payload"] = payload
        self.sink["on_conflict"] = on_conflict
        self.sink["op"] = "upsert"
        return self

    def insert(self, payload, returning=None):
        self.sink["payload"] = payload
        self.sink["op"] = "insert"
        return self

    def update(self, payload):
        self.sink["payload"] = payload
        self.sink["op"] = "update"
        return self

    def delete(self):
        self.sink["op"] = "delete"
        return self

    def execute(self):
        return type("Resp", (), {"data": [{"id": 1, **self.sink["payload"]}]})


class _FakeSupabase:
    def __init__(self):
        self.sink = {}

    def table(self, name):
        assert name == "players"
        return _Query(self.sink)


def test_coerce_rating_to_elo_converts_jupr_scale_values():
    assert _coerce_rating_to_elo(3.5) == 1400.0
    assert _coerce_rating_to_elo("3.0") == 1200.0


def test_coerce_rating_to_elo_preserves_elo_scale_values():
    assert _coerce_rating_to_elo(1400) == 1400.0


def test_get_or_create_player_normalizes_rating_payload_before_write():
    supabase = _FakeSupabase()
    ok, row, err = get_or_create_player(
        supabase=supabase,
        club_id="club-1",
        normalized_name="new_player",
        payload={
            "club_id": "club-1",
            "name": "New Player",
            "normalized_name": "new_player",
            "rating": 3.5,
            "starting_rating": 3.5,
        },
    )

    assert ok is True
    assert err is None
    assert row is not None
    assert supabase.sink["payload"]["rating"] == 1400.0
    assert supabase.sink["payload"]["starting_rating"] == 1400.0
