from types import SimpleNamespace

from copy import deepcopy

from jupr_app.services.admin_moneyball_service import (
    build_moneyball_preview,
    build_moneyball_settlement_preview,
    compute_moneyball_settlement,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def execute(self):
        rows = list(self.storage.get(self.table_name, []))
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [{"club_id": "club", "id": i, "name": f"Player {i}", "rating": 1200 + i * 10, "active": True} for i in range(1, 9)],
            "league_ratings": [],
            "leagues_metadata": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_moneyball_preview_requires_exactly_8_unique_players(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "1")
    try:
        build_moneyball_preview(FakeSupabase(), club_id="club", player_ids=[1, 2, 3])
    except ValueError as exc:
        assert "exactly 8" in str(exc)
    else:
        raise AssertionError("expected exact player count error")


def test_moneyball_preview_generates_expected_schedule(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "1")
    payload = build_moneyball_preview(FakeSupabase(), club_id="club", player_ids=list(range(1, 9)))
    assert payload["ok"] is True
    assert payload["matches"]
    assert payload["matches"][0]["row_id"].startswith("moneyball-")
    assert "expected_win_pct_t1" in payload["matches"][0]


def test_moneyball_settlement_balances_net(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "1")
    preview = build_moneyball_preview(FakeSupabase(), club_id="club", player_ids=list(range(1, 9)))
    scores = [{"row_id": match["row_id"], "score_t1": 11, "score_t2": 8} for match in preview["matches"][:4]]
    settlement = compute_moneyball_settlement(matches=preview["matches"], scores=scores, win_rate=5, point_rate=2)
    assert settlement["standings"]
    assert round(sum(float(row["net"]) for row in settlement["standings"]), 2) == 0.0


def test_moneyball_settlement_preview_is_read_only_named_and_fingerprinted(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "1")
    supabase = FakeSupabase()
    preview = build_moneyball_preview(supabase, club_id="club", player_ids=list(range(1, 9)))
    scores = [{"row_id": preview["matches"][0]["row_id"], "score_t1": 11, "score_t2": 7}]
    before = deepcopy(supabase.storage)

    result = build_moneyball_settlement_preview(
        supabase,
        club_id="club",
        player_ids=list(range(1, 9)),
        scores=scores,
        win_rate=5,
        point_rate=2,
    )

    assert result["authority"] == "python_fastapi"
    assert len(result["settlement_fingerprint"]) == 64
    assert result["would_publish_count"] == 1
    assert {row["settlement_direction"] for row in result["settlement"]["standings"]} <= {"owes", "receives", "even"}
    assert all(row["player_name"].startswith("Player ") for row in result["settlement"]["standings"])
    assert supabase.storage == before


def test_moneyball_settlement_does_not_count_unknown_score_rows(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MONEYBALL", "1")
    result = build_moneyball_settlement_preview(
        FakeSupabase(),
        club_id="club",
        player_ids=list(range(1, 9)),
        scores=[{"row_id": "not-in-python-schedule", "score_t1": 11, "score_t2": 7}],
    )

    assert result["would_publish_count"] == 0
