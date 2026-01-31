from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.top_performer_awards import TOP_PERFORMER_BADGE_IDS
from jupr_app.domain.leagues import end_league_and_award_top_performers, mint_top_performer_badges


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.pending_update = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def update(self, payload):
        self.pending_update = payload
        return self

    def upsert(self, rows, on_conflict=None):
        existing = self.storage.setdefault(self.name, [])
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        for row in rows:
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            existing.append(row)
            if key is not None:
                existing_keys.add(key)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if str(row.get(column)) in {str(v) for v in value}]
        if self.pending_update is not None:
            for row in data:
                row.update(self.pending_update)
            self.storage[self.name] = self.storage.get(self.name, [])
            return SimpleNamespace(data=data)
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage or {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _build_league_context(is_active: bool) -> SimpleNamespace:
    df_leagues = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "player_id": 1,
                "rating": 1600,
                "starting_rating": 1400,
                "wins": 9,
                "losses": 3,
                "matches_played": 12,
            },
            {
                "league_name": "Spring 2024 Ladder",
                "player_id": 2,
                "rating": 1700,
                "starting_rating": 1500,
                "wins": 10,
                "losses": 2,
                "matches_played": 12,
            },
        ]
    )
    df_meta = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "min_games": 6,
                "is_active": is_active,
                "status": "completed" if not is_active else "active",
            }
        ]
    )
    return SimpleNamespace(
        df_leagues=df_leagues,
        df_meta=df_meta,
        id_to_name={1: "Ace", 2: "Blaze"},
        club_id="club",
    )


def test_top_performer_candidates_require_ended_league():
    active_ctx = _build_league_context(is_active=True)
    active_candidates = list(
        compute_candidates_for_club(
            "club",
            league_id="Spring 2024 Ladder",
            ctx=active_ctx,
            status="seasonal",
            award_timing="on_league_close",
        )
    )
    top_performer_ids = set(TOP_PERFORMER_BADGE_IDS.values())
    assert not [c for c in active_candidates if c.badge_id in top_performer_ids]

    ended_ctx = _build_league_context(is_active=False)
    ended_candidates = list(
        compute_candidates_for_club(
            "club",
            league_id="Spring 2024 Ladder",
            ctx=ended_ctx,
            status="seasonal",
            award_timing="on_league_close",
        )
    )
    assert [c for c in ended_candidates if c.badge_id in top_performer_ids]


def test_end_league_award_is_idempotent():
    storage = {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Spring 2024 Ladder",
                "is_active": True,
                "status": "active",
                "min_games": 6,
            }
        ]
    }
    supabase = FakeSupabase(storage)
    ctx = _build_league_context(is_active=True)
    ctx.supabase = supabase
    ctx.public_mode = False

    end_league_and_award_top_performers(ctx, "Spring 2024 Ladder", admin_id="admin")
    end_league_and_award_top_performers(ctx, "Spring 2024 Ladder", admin_id="admin")

    inserted = storage.get("player_badges", [])
    unique_keys = {
        (row.get("club_id"), row.get("player_id"), row.get("badge_id"), row.get("context_id"))
        for row in inserted
    }
    assert len(unique_keys) == len(inserted)


def test_mint_top_performer_badges_is_idempotent_and_sets_context():
    storage = {"player_badges": []}
    supabase = FakeSupabase(storage)
    awards = [
        {
            "category_key": "most_wins",
            "category_label": "Most Wins",
            "player_id": 7,
            "metric_value": 12,
            "metric_display": "12",
            "rank": 1,
        },
        {
            "category_key": "best_win_pct",
            "category_label": "Best Win %",
            "player_id": 8,
            "metric_value": 72.5,
            "metric_display": "72.5%",
            "rank": 2,
        },
    ]
    created = mint_top_performer_badges(
        supabase,
        club_id="club",
        league_id="Spring 2024 Ladder",
        awards=awards,
        ended_at="2026-01-01T00:00:00Z",
        override_notes={},
    )
    created_again = mint_top_performer_badges(
        supabase,
        club_id="club",
        league_id="Spring 2024 Ladder",
        awards=awards,
        ended_at="2026-01-01T00:00:00Z",
        override_notes={},
    )
    assert len(created) == 2
    assert len(created_again) == 0
    inserted = storage.get("player_badges", [])
    assert len(inserted) == 2
    context_ids = {row.get("context_id") for row in inserted}
    assert "Spring 2024 Ladder:top_performer:most_wins:1" in context_ids
    assert "Spring 2024 Ladder:top_performer:best_win_pct:2" in context_ids
    value_json = inserted[0].get("value_json") or {}
    assert value_json.get("league_id") == "Spring 2024 Ladder"
    assert value_json.get("category_label")
    assert value_json.get("rank")
