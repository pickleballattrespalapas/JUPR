from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._limit: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "ladder_settings": [
                {
                    "club_id": "club",
                    "challenge_range": 7,
                    "accept_window_hours": 48,
                    "play_window_days": 7,
                    "cooldown_hours": 72,
                    "protected_hours": 72,
                    "pass_hold_hours": 72,
                    "internal_notes": "private",
                }
            ],
            "players": [
                {"id": 1, "club_id": "club", "name": "Alex", "rating": 1700, "active": True, "private_email": "hidden"},
                {"id": 2, "club_id": "club", "name": "Blair", "rating": 1600, "active": True},
                {"id": 3, "club_id": "club", "name": "Casey", "rating": 1500, "active": True},
                {"id": 4, "club_id": "club", "name": "Devon", "rating": 1400, "active": False},
            ],
            "ladder_roster": [
                {"id": 10, "club_id": "club", "player_id": 1, "tier_id": "PREM", "rank": 1, "is_active": True, "notes": "private"},
                {"id": 11, "club_id": "club", "player_id": 2, "tier_id": "PREM", "rank": 2, "is_active": True},
                {"id": 12, "club_id": "club", "player_id": 3, "tier_id": "ADV", "rank": 1, "is_active": True},
                {"id": 13, "club_id": "club", "player_id": 4, "tier_id": "ADV", "rank": 2, "is_active": True},
            ],
            "ladder_player_flags": [
                {"club_id": "club", "player_id": 3, "reinstate_required": True, "reinstate_notes": "Needs admin review", "private_reason": "hidden"},
            ],
            "ladder_challenges": [
                {
                    "id": 20,
                    "club_id": "club",
                    "challenger_id": 2,
                    "defender_id": 1,
                    "tier_id": "PREM",
                    "status": "PENDING_ACCEPTANCE",
                    "created_at": "2099-01-01T00:00:00Z",
                    "accept_by": "2099-01-03T00:00:00Z",
                    "accepted_at": None,
                    "play_by": None,
                    "completed_at": None,
                    "winner_id": None,
                    "ledger_ref": "private ledger",
                    "challenger_contact": "private@example.com",
                },
                {
                    "id": 21,
                    "club_id": "club",
                    "challenger_id": 3,
                    "defender_id": 1,
                    "tier_id": "ADV",
                    "status": "COMPLETED",
                    "created_at": "2026-01-01T00:00:00Z",
                    "accept_by": None,
                    "accepted_at": None,
                    "play_by": None,
                    "completed_at": "2026-01-05T00:00:00Z",
                    "winner_id": 3,
                    "resolution_notes": "private",
                },
            ],
            "ladder_pass_usage": [],
        }
    )


def test_public_challenge_ladder_builds_tiers_status_and_challenge_buckets() -> None:
    payload = build_public_challenge_ladder(fake_supabase(), club_id="club")

    assert payload["settings"]["challenge_range"] == 7
    assert payload["summary"]["active_player_count"] == 3
    assert payload["summary"]["active_challenge_count"] == 1

    prem = next(tier for tier in payload["tiers"] if tier["tier_id"] == "PREM")
    assert [player["player_name"] for player in prem["players"]] == ["Alex", "Blair"]
    locked = next(player for player in prem["players"] if player["player_name"] == "Alex")
    assert locked["status"] == "Locked"
    assert locked["challenge_id"] == 20
    assert "private" not in str(locked).lower()

    adv = next(tier for tier in payload["tiers"] if tier["tier_id"] == "ADV")
    assert [player["player_name"] for player in adv["players"]] == ["Casey"]
    assert adv["players"][0]["status"] == "Reinstate Required"

    pending = next(section for section in payload["challenge_sections"] if section["name"] == "Pending Acceptance")
    assert pending["challenges"][0]["challenger"] == {"player_id": 2, "player_name": "Blair"}
    assert "ledger_ref" not in pending["challenges"][0]
    assert "challenger_contact" not in pending["challenges"][0]

    recent = next(section for section in payload["challenge_sections"] if section["name"] == "Recently Completed")
    assert recent["challenges"][0]["winner"] == {"player_id": 3, "player_name": "Casey"}
    assert "resolution_notes" not in recent["challenges"][0]


def test_public_challenge_ladder_degrades_to_empty_when_tables_missing() -> None:
    payload = build_public_challenge_ladder(FakeSupabase({}), club_id="club")

    assert payload["summary"]["active_player_count"] == 0
    assert payload["tiers"][0]["players"] == []
    assert payload["quick_rules"]
