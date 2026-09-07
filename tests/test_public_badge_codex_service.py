from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_badge_codex_service import (
    build_public_badge_codex,
    get_public_badge_earners,
)


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
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


def test_inactive_or_missing_players_are_not_projected_as_public_earners() -> None:
    supabase = fake_supabase()
    supabase._tables["players"].append({"id": 4, "club_id": "club", "name": "Private Pat", "active": False})
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

    assert participant["earners_count"] == 2
    assert "Private Pat" not in str(payload)
    assert "Player 4" not in str(payload)
