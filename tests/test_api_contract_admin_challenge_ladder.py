from types import SimpleNamespace

from jupr_app.services.admin_challenge_ladder_service import get_admin_challenge_ladder_dashboard, update_admin_challenge_ladder_challenge


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.insert_payload = None
        self.update_payload = None
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

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            row = {"id": f"row-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[row])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=scoped)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [
                {"club_id": "club", "id": 1, "name": "Alex", "rating": 1600, "active": True},
                {"club_id": "club", "id": 2, "name": "Blair", "rating": 1500, "active": True},
                {"club_id": "club", "id": 3, "name": "Casey", "rating": 1450, "active": True},
            ],
            "ladder_settings": [{"club_id": "club", "challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7, "cooldown_hours": 72, "protected_hours": 72, "pass_hold_hours": 72}],
            "ladder_roster": [
                {"club_id": "club", "id": 10, "player_id": 1, "tier_id": "ADV", "rank": 1, "is_active": True},
                {"club_id": "club", "id": 11, "player_id": 2, "tier_id": "ADV", "rank": 2, "is_active": True},
            ],
            "ladder_player_flags": [
                {"club_id": "club", "player_id": 2, "vacation_until": None, "reinstate_required": True, "reinstate_notes": "Contact director"},
            ],
            "ladder_pass_usage": [],
            "ladder_challenges": [{"club_id": "club", "id": 100, "challenger_id": 2, "defender_id": 1, "tier_id": "ADV", "status": "PENDING_ACCEPTANCE", "created_at": "2026-01-01T00:00:00Z"}],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_challenge_ladder_dashboard(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    payload = get_admin_challenge_ladder_dashboard(FakeSupabase(), club_id="club")
    assert payload["ok"] is True
    assert payload["summary"]["active_player_count"] == 2
    assert payload["challenges"][0]["challenger_name"] == "Blair"
    assert payload["player_options"] == [
        {"player_id": 1, "player_name": "Alex"},
        {"player_id": 2, "player_name": "Blair"},
        {"player_id": 3, "player_name": "Casey"},
    ]
    assert [(row["player_name"], row["tier_id"], row["rank"], row["is_active"]) for row in payload["roster_rows"]] == [
        ("Alex", "ADV", 1, True),
        ("Blair", "ADV", 2, True),
    ]
    assert payload["player_flags"] == [
        {
            "player_id": 2,
            "player_name": "Blair",
            "vacation_until": None,
            "reinstate_required": True,
            "reinstate_notes": "Contact director",
            "tier_move_flag": False,
            "tier_move_dest_tier": None,
            "tier_move_count": 0,
        }
    ]


def test_challenge_ladder_update_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    try:
        update_admin_challenge_ladder_challenge(FakeSupabase(), club_id="club", challenge_id=100, status="CANCELLED", admin_note="", actor_email="admin@example.com", actor_role="club_owner", confirmation_text="SAVE")
    except ValueError as exc:
        assert "SAVE LADDER" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_challenge_ladder_update_writes_audit(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    updated = update_admin_challenge_ladder_challenge(supabase, club_id="club", challenge_id=100, status="CANCELLED", admin_note="operator cancel", actor_email="admin@example.com", actor_role="club_owner", confirmation_text="SAVE LADDER")
    assert updated["challenge"]["status"] == "CANCELLED"
    assert supabase.storage["admin_activity_log"]
