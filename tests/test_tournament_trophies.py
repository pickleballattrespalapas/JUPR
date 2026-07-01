from types import SimpleNamespace

from jupr_app.domain.gamification.trophies import (
    get_player_tournament_trophies,
    parse_tournament_podium_context,
)


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.ordering = None

    def select(self, _cols, *args, **kwargs):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def like(self, column, pattern):
        self.filters.append(("like", column, pattern))
        return self

    def order(self, column, desc=False):
        self.ordering = (column, desc)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                normalized_values = {str(item) for item in value}
                data = [row for row in data if str(row.get(column)) in normalized_values]
            elif op == "like":
                needle = str(value).replace("%", "")
                data = [row for row in data if needle in str(row.get(column) or "")]
        if self.ordering:
            column, desc = self.ordering
            data = sorted(data, key=lambda row: row.get(column) or "", reverse=bool(desc))
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_parse_tournament_podium_context():
    tournament_id, placement = parse_tournament_podium_context("abc123:podium:2")

    assert tournament_id == "abc123"
    assert placement == 2


def test_parse_tournament_podium_context_fallbacks_to_value_num():
    tournament_id, placement = parse_tournament_podium_context("abc123:podium:2", value_num=3)

    assert tournament_id == "abc123"
    assert placement == 2


def test_get_player_tournament_trophies_falls_back_to_archived_podium_rows():
    storage = {
        "player_badges": [],
        "tournaments": [
            {"id": "tour1", "club_id": "club1", "name": "Archived Open", "status": "ARCHIVED"},
        ],
        "tournament_teams": [
            {
                "id": "team1",
                "tournament_id": "tour1",
                "team_number": 1,
                "player1_id": 101,
                "player2_id": 102,
            },
        ],
        "tournament_podium": [
            {"tournament_id": "tour1", "placement": 1, "team_id": "team1", "source": "PLAYOFF"},
        ],
        "players": [
            {"id": 101, "club_id": "club1", "name": "Player One"},
            {"id": 102, "club_id": "club1", "name": "Partner Two"},
        ],
    }
    supabase = FakeSupabase(storage)

    trophies = get_player_tournament_trophies(supabase, "club1", 101)

    assert len(trophies) == 1
    assert trophies[0]["placement"] == 1
    assert trophies[0]["tournament_id"] == "tour1"
    assert trophies[0]["tournament_name"] == "Archived Open"
    assert trophies[0]["team_id"] == "team1"
    assert trophies[0]["teammate_names"] == "Partner Two"


def test_get_player_tournament_trophies_prefers_existing_badge_rows_over_podium_fallback():
    storage = {
        "player_badges": [
            {
                "club_id": "club1",
                "player_id": 101,
                "context_type": "tournament",
                "context_id": "tour1:podium:1",
                "earned_at": "2026-02-15T00:00:00+00:00",
                "value_num": 1,
                "value_json": {
                    "tournament_id": "tour1",
                    "tournament_name": "Archived Open",
                    "placement": 1,
                    "team_id": "team1",
                },
            }
        ],
        "tournaments": [
            {"id": "tour1", "club_id": "club1", "name": "Archived Open", "status": "ARCHIVED"},
        ],
        "tournament_teams": [
            {
                "id": "team1",
                "tournament_id": "tour1",
                "team_number": 1,
                "player1_id": 101,
                "player2_id": 102,
            },
        ],
        "tournament_podium": [
            {"tournament_id": "tour1", "placement": 1, "team_id": "team1", "source": "PLAYOFF"},
        ],
        "players": [
            {"id": 101, "club_id": "club1", "name": "Player One"},
            {"id": 102, "club_id": "club1", "name": "Partner Two"},
        ],
    }
    supabase = FakeSupabase(storage)

    trophies = get_player_tournament_trophies(supabase, "club1", 101)

    assert len(trophies) == 1
    assert trophies[0]["placement"] == 1
    assert trophies[0]["earned_at"] == "2026-02-15T00:00:00+00:00"
    assert trophies[0]["teammate_names"] == "Partner Two"
