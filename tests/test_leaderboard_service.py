from jupr_app.services.leaderboard_service import get_public_leaderboard


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, sb, table_name):
        self.sb = sb
        self.table_name = table_name
        self.filters = {}

    def select(self, _cols):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table_name == "public_leaderboards" and self.sb.raise_on_view:
            raise RuntimeError("relation public_leaderboards does not exist")
        rows = [dict(r) for r in self.sb.store.get(self.table_name, [])]
        for key, value in self.filters.items():
            rows = [r for r in rows if r.get(key) == value]
        return _Resp(rows)


class _Supabase:
    def __init__(self, store, raise_on_view=False):
        self.store = store
        self.raise_on_view = raise_on_view

    def table(self, name):
        return _Query(self, name)


def test_view_backed_path_returns_normalized_safe_rows():
    sb = _Supabase(
        {
            "public_leaderboards": [
                {
                    "club_id": "c1",
                    "league_name": "A",
                    "player_id": 10,
                    "player_name": "Ava",
                    "rating": 1501.2,
                    "rating_jupr": 1501.2,
                    "wins": 8,
                    "losses": 2,
                    "matches_played": 10,
                    "is_active": True,
                    "rank_position": 1,
                    "updated_at": None,
                    "email": "hidden@example.com",
                }
            ]
        }
    )

    rows = get_public_leaderboard(sb, "c1", "A")

    assert len(rows) == 1
    assert rows[0]["player_name"] == "Ava"
    assert rows[0]["rank_position"] == 1
    assert "email" not in rows[0]


def test_missing_view_falls_back_to_tables_and_stays_public_only():
    sb = _Supabase(
        {
            "league_ratings": [
                {
                    "club_id": "c1",
                    "league_name": "A",
                    "player_id": 2,
                    "rating": 1499.0,
                    "wins": 7,
                    "losses": 3,
                    "matches_played": 10,
                    "is_active": True,
                },
                {
                    "club_id": "c1",
                    "league_name": "A",
                    "player_id": 1,
                    "rating": 1600.0,
                    "wins": 9,
                    "losses": 1,
                    "matches_played": 10,
                    "is_active": True,
                },
            ],
            "players": [
                {"club_id": "c1", "id": 1, "name": "Zoe", "email": "zoe@example.com"},
                {"club_id": "c1", "id": 2, "name": "Ana", "email": "ana@example.com"},
            ],
        },
        raise_on_view=True,
    )

    rows = get_public_leaderboard(sb, "c1", "A")

    assert [r["player_name"] for r in rows] == ["Zoe", "Ana"]
    assert [r["rank_position"] for r in rows] == [1, 2]
    for row in rows:
        assert "email" not in row
        assert set(row.keys()).issubset(
            {
                "club_id",
                "league_name",
                "player_id",
                "player_name",
                "rating",
                "rating_jupr",
                "wins",
                "losses",
                "matches_played",
                "is_active",
                "rank_position",
                "updated_at",
            }
        )
from pathlib import Path


def test_public_leaderboards_migration_is_guarded_and_has_no_metadata_dependency():
    sql = Path("supabase/migrations/20260502133000_public_leaderboards_view.sql").read_text(encoding="utf-8")

    assert "to_regclass('public.league_ratings')" in sql
    assert "to_regclass('public.players')" in sql
    assert "public.leagues_metadata" not in sql
