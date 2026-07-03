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

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
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
    assert {section["name"] for section in payload["sections"]} == {"Participation", "Upsets"}

    all_badges = [badge for section in payload["sections"] for badge in section["badges"]]
    participant = next(badge for badge in all_badges if badge["badge_id"] == "participant")
    assert participant["earners_count"] == 2
    assert participant["requirements"] == "Play a recorded match"
    assert participant["recent_earners"][0] == {"player_id": 1, "player_name": "Alex", "earned_at": "2026-01-03T00:00:00Z"}
    assert "admin_notes" not in participant
    assert "private_email" not in participant["recent_earners"][0]

    giant_slayer = next(badge for badge in all_badges if badge["badge_id"] == "giant_slayer")
    assert giant_slayer["description"] == "HTML should be stripped"
    assert giant_slayer["state"] == "frozen"
    assert "old_unused" not in {badge["badge_id"] for badge in all_badges}


def test_public_badge_earners_paginates_public_safe_rows() -> None:
    payload = get_public_badge_earners(fake_supabase(), club_id="club", badge_id="participant", offset=1, limit=1)

    assert payload["badge_id"] == "participant"
    assert payload["total"] == 2
    assert payload["has_more"] is False
    assert payload["earners"] == [{"player_id": 2, "player_name": "Blair", "earned_at": "2026-01-02T00:00:00Z"}]
