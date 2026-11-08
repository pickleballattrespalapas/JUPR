from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.public_badge_codex_service import (
    build_public_badge_codex,
    get_public_badge_earners,
)


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._in_filters: dict[str, list] = {}

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def in_(self, key, values):
        self._in_filters[key] = list(values)
        return self

    def range(self, start, end):
        self.page_bounds = (start, end)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        for key, values in self._in_filters.items():
            rows = [row for row in rows if row.get(key) in values]
        if hasattr(self, "page_bounds"):
            rows = rows[self.page_bounds[0]:self.page_bounds[1] + 1]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "badges": [
                {
                    "badge_id": "participant",
                    "name": "Participant",
                    "category": "Participation",
                    "prestige": 10,
                    "requirements": "Requirements: Play a recorded match",
                    "description_md": "Earn your first reel.",
                    "state": "live",
                    "is_active": True,
                    "admin_notes": "private",
                },
                {
                    "badge_id": "giant_slayer",
                    "name": "Giant Slayer",
                    "category": "Upsets",
                    "prestige": 50,
                    "requirements": "Beat a much stronger team",
                    "description_md": "<b>HTML should be stripped</b>",
                    "state": "frozen",
                    "is_active": True,
                },
                {
                    "badge_id": "old_unused",
                    "name": "Old Unused",
                    "category": "Old",
                    "prestige": 1,
                    "state": "deprecated",
                    "is_active": False,
                },
            ],
            "player_badges": [
                {"club_id": "club", "player_id": 1, "badge_id": "participant", "earned_at": "2026-01-03T00:00:00Z"},
                {"club_id": "club", "player_id": 2, "badge_id": "participant", "earned_at": "2026-01-02T00:00:00Z"},
                {"club_id": "club", "player_id": 1, "badge_id": "participant", "earned_at": "2026-01-01T00:00:00Z"},
                {"club_id": "club", "player_id": 2, "badge_id": "giant_slayer", "earned_at": "2026-01-04T00:00:00Z"},
                {"club_id": "other", "player_id": 3, "badge_id": "participant", "earned_at": "2026-01-05T00:00:00Z"},
            ],
            "players": [
                {"id": 1, "club_id": "club", "name": "Alex", "active": True, "private_email": "hidden"},
                {"id": 2, "club_id": "club", "name": "Blair", "active": True},
                {"id": 3, "club_id": "other", "name": "Casey", "active": True},
            ],
        }
    )


def test_public_badge_codex_groups_badges_and_counts_unique_earners() -> None:
    payload = build_public_badge_codex(fake_supabase(), club_id="club")

    assert payload["summary"]["badge_count"] == 2
    assert payload["summary"]["earned_badge_count"] == 2
    assert payload["summary"]["complete_definition_count"] == 2
    assert {section["name"] for section in payload["sections"]} == {"Participation", "Match Achievements"}
    assert [bucket["name"] for bucket in payload["catalog_buckets"]] == [
        "Live Now",
        "Seasonal / League Close",
        "Manual / Curated",
        "Tracked / Disabled",
    ]

    all_badges = [badge for section in payload["sections"] for badge in section["badges"]]
    participant = next(badge for badge in all_badges if badge["badge_id"] == "participant")
    assert participant["earners_count"] == 2
    assert participant["requirements"] == "Play 1 recorded match (lifetime)."
    assert participant["badge_status"] == "live"
    assert participant["badge_award_timing"] == "live"
    assert participant["badge_scope"] == "lifetime"
    assert participant["catalog_bucket"] == "Live Now"
    assert "obtainable" in participant["availability"].lower()
    assert participant["recent_earners"][0] == {"player_id": 1, "player_name": "Alex", "earned_at": "2026-01-03T00:00:00Z"}
    assert "admin_notes" not in participant
    assert "private_email" not in participant["recent_earners"][0]

    giant_slayer = next(badge for badge in all_badges if badge["badge_id"] == "giant_slayer")
    assert giant_slayer["description"] == giant_slayer["requirements"]
    assert giant_slayer["state"] == "frozen"
    assert "old_unused" not in {badge["badge_id"] for badge in all_badges}
    assert payload["trophy_room"][0]["player_name"] == "Blair"
    assert payload["trophy_room"][0]["prestige_total"] == 60
    assert "private_email" not in str(payload["trophy_room"])


def test_public_badge_earners_paginates_public_safe_rows() -> None:
    payload = get_public_badge_earners(fake_supabase(), club_id="club", badge_id="participant", offset=1, limit=1)

    assert payload["badge_id"] == "participant"
    assert payload["total"] == 2
    assert payload["has_more"] is False
    assert payload["earners"] == [{"player_id": 2, "player_name": "Blair", "earned_at": "2026-01-02T00:00:00Z"}]
    assert payload["badge"]["catalog_bucket"] == "Live Now"
    assert payload["badge"]["recent_earners"] == []


def test_public_badge_codex_uses_authoritative_timing_buckets() -> None:
    supabase = FakeSupabase(
        {
            "badges": [
                {"badge_id": "participant", "name": "Participant", "category": "Activity", "prestige": 10, "state": "live", "is_active": True},
                {"badge_id": "league_champion", "name": "League Champion", "category": "Awards", "prestige": 100, "state": "live", "is_active": False},
                {"badge_id": "tournament_champion", "name": "Tournament Champion", "category": "Awards", "prestige": 100, "state": "live", "is_active": True},
                {"badge_id": "breakthrough", "name": "Breakthrough", "category": "Momentum", "prestige": 55, "state": "live", "is_active": True},
            ],
            "player_badges": [
                {"club_id": "club", "player_id": 1, "badge_id": "league_champion", "earned_at": "2026-01-01T00:00:00Z"},
            ],
            "players": [{"id": 1, "club_id": "club", "name": "Alex", "active": True}],
        }
    )

    payload = build_public_badge_codex(supabase, club_id="club")
    buckets = {
        bucket["name"]: {
            badge["badge_id"]
            for section in bucket["sections"]
            for badge in section["badges"]
        }
        for bucket in payload["catalog_buckets"]
    }

    assert "participant" in buckets["Live Now"]
    assert all("league_champion" not in ids for ids in buckets.values())
    assert supabase._tables["player_badges"][0]["badge_id"] == "league_champion"
    assert "tournament_champion" in buckets["Manual / Curated"]
    assert "breakthrough" in buckets["Live Now"]
    assert set(payload["filters"]["scopes"]) >= {"lifetime", "season"}


def test_inactive_players_keep_their_earned_badges() -> None:
    supabase = fake_supabase()
    supabase._tables["players"].append({"id": 4, "club_id": "club", "name": "Returning Pat", "active": False})
    supabase._tables["player_badges"].append(
        {"club_id": "club", "player_id": 4, "badge_id": "participant", "earned_at": "2026-02-01T00:00:00Z"}
    )

    payload = build_public_badge_codex(supabase, club_id="club")
    participant = next(
        badge
        for section in payload["sections"]
        for badge in section["badges"]
        if badge["badge_id"] == "participant"
    )

    assert participant["earners_count"] == 3
    assert "Returning Pat" in str(payload)
    assert "Player 4" not in str(payload)


def test_badge_earners_read_only_selected_badge_and_its_identities():
    db = fake_supabase()
    db._tables["players"].extend([
        {"id": 4, "club_id": "club", "name": "Merged (merged into 1)"},
        {"id": 5, "club_id": "club", "name": "Revoked"},
        {"id": 6, "club_id": "club", "name": "Different badge"},
        {"id": 7, "club_id": "club", "name": "Inactive earner", "active": False},
    ])
    db._tables["player_badges"].extend([
        {"club_id": "club", "player_id": 4, "badge_id": "participant", "earned_at": "2026-02-04"},
        {"club_id": "club", "player_id": 5, "badge_id": "participant", "earned_at": "2026-02-05", "revoked_at": "2026-02-06"},
        {"club_id": "club", "player_id": 6, "badge_id": "giant_slayer", "earned_at": "2026-02-06"},
        {"club_id": "club", "player_id": 7, "badge_id": "participant", "earned_at": "2026-02-07"},
        {"club_id": "club", "player_id": 999, "badge_id": "participant", "earned_at": "2026-02-08"},
    ])
    queries = []
    original_table = db.table

    def table(name):
        query = original_table(name)
        queries.append((name, query))
        return query

    db.table = table
    payload = get_public_badge_earners(db, club_id="club", badge_id="participant", offset=1, limit=1)
    assert payload["total"] == 3
    assert payload["has_more"] is True
    assert payload["earners"] == [{"player_id": 1, "player_name": "Alex", "earned_at": "2026-01-03T00:00:00Z"}]
    assert all(query._filters["badge_id"] == "participant" for name, query in queries if name in {"badges", "player_badges"})
    assert all(query._filters["club_id"] == "club" for name, query in queries if name != "badges")
    assert all(set(query._in_filters["id"]) == {1, 2, 4, 7, 999} for name, query in queries if name == "players")


def test_badge_earners_keep_exact_counts_across_bounded_identity_batches():
    db = fake_supabase()
    db._tables["players"] = [{"id": i, "club_id": "club", "name": f"Player {i}"} for i in range(1, 206)]
    db._tables["player_badges"] = [
        {"id": i, "club_id": "club", "player_id": i, "badge_id": "participant", "earned_at": f"2026-01-01T00:{i:03}"}
        for i in range(1, 206)
    ]
    payload = get_public_badge_earners(db, club_id="club", badge_id="participant", offset=200, limit=100)
    assert payload["total"] == 205
    assert len(payload["earners"]) == 5
    assert payload["has_more"] is False
    assert [row["player_id"] for row in payload["earners"]] == [5, 4, 3, 2, 1]


def test_missing_badge_does_not_read_awards_or_players():
    db = fake_supabase()
    original_table = db.table
    names = []

    def table(name):
        names.append(name)
        return original_table(name)

    db.table = table
    with pytest.raises(ValueError, match="badge not found"):
        get_public_badge_earners(db, club_id="club", badge_id="missing")
    assert set(names) == {"badges"}


def test_codex_visits_only_each_badges_own_awards_for_recent_earners(monkeypatch):
    from jupr_app.services import public_badge_codex_service as service

    db = fake_supabase()
    original = service._recent_earners_for_badge
    visited = []

    def recent(rows, **kwargs):
        visited.extend(rows)
        assert all(row["badge_id"] == kwargs["badge_id"] for row in rows)
        return original(rows, **kwargs)

    monkeypatch.setattr(service, "_recent_earners_for_badge", recent)
    payload = build_public_badge_codex(db, club_id="club")
    assert payload["summary"]["total_unique_earners_by_badge"] == 3
    assert len(visited) == 4  # Each local award once, not every award for every badge.
