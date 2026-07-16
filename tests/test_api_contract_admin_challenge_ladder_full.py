from types import SimpleNamespace

from jupr_app.services.admin_challenge_ladder_service import (
    accept_admin_challenge_ladder_challenge,
    create_admin_challenge_ladder_challenge,
    preview_admin_challenge_ladder_result,
    record_admin_challenge_ladder_forfeit,
    record_admin_challenge_ladder_result,
)


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
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        if self.insert_payload is not None:
            row = {"id": len(rows) + 100, **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[dict(row)])
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in scoped])


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [
                {"club_id": "club", "id": 1, "name": "Defender"},
                {"club_id": "club", "id": 2, "name": "Challenger"},
                {"club_id": "club", "id": 3, "name": "Partner A"},
                {"club_id": "club", "id": 4, "name": "Partner B"},
                {"club_id": "club", "id": 5, "name": "Partner C"},
                {"club_id": "club", "id": 6, "name": "Partner D"},
            ],
            "ladder_settings": [{"club_id": "club", "challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7}],
            "ladder_roster": [
                {"club_id": "club", "id": 10, "player_id": 1, "tier_id": "ADV", "rank": 1, "is_active": True},
                {"club_id": "club", "id": 11, "player_id": 2, "tier_id": "ADV", "rank": 2, "is_active": True},
                {"club_id": "club", "id": 12, "player_id": 3, "tier_id": "ADV", "rank": 3, "is_active": True},
                {"club_id": "club", "id": 13, "player_id": 4, "tier_id": "ADV", "rank": 4, "is_active": True},
                {"club_id": "club", "id": 14, "player_id": 5, "tier_id": "ADV", "rank": 5, "is_active": True},
                {"club_id": "club", "id": 15, "player_id": 6, "tier_id": "ADV", "rank": 6, "is_active": True},
            ],
            "ladder_challenges": [{"club_id": "club", "id": 100, "challenger_id": 2, "defender_id": 1, "tier_id": "ADV", "status": "PENDING_ACCEPTANCE", "created_at": "2026-01-01T00:00:00Z"}],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_create_challenge_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    try:
        create_admin_challenge_ladder_challenge(FakeSupabase(), club_id="club", challenger_id=2, defender_id=1, tier_id="ADV", ledger_ref=None, override=False, start_clock=False, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="CREATE")
    except ValueError as exc:
        assert "CREATE LADDER CHALLENGE" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_create_and_accept_challenge(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    created = create_admin_challenge_ladder_challenge(supabase, club_id="club", challenger_id=2, defender_id=1, tier_id="ADV", ledger_ref="ledger", override=False, start_clock=True, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="CREATE LADDER CHALLENGE")
    assert created["challenge"]["status"] == "PENDING_ACCEPTANCE"
    accepted = accept_admin_challenge_ladder_challenge(supabase, club_id="club", challenge_id=created["challenge"]["id"], actor_email="admin@example.com", actor_role="club_owner", confirmation_text="ACCEPT LADDER CHALLENGE")
    assert accepted["challenge"]["status"] == "ACCEPTED_SCHEDULING"


def test_preview_defender_holds_exact_tie():
    preview = preview_admin_challenge_ladder_result(
        challenger_id=2,
        defender_id=1,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=5,
        partner_b_defender_id=6,
        match_a_games=[[11, 9], [7, 11]],
        match_b_games=[[9, 11], [11, 7]],
    )
    assert preview["final_winner_id"] == 1


def test_forfeit_swaps_rank_when_defender_forfeits(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    result = record_admin_challenge_ladder_forfeit(supabase, club_id="club", challenge_id=100, forfeited_by_id=1, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="RECORD LADDER FORFEIT")
    assert result["challenge"]["winner_id"] == 2
    ranks = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}
    assert ranks[2] == 1
    assert ranks[1] == 2


def test_played_result_publishes_matches_and_swaps(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    monkeypatch.setattr("jupr_app.services.admin_challenge_ladder_service.load_data", lambda *_args, **_kwargs: ([], [], [], [], [], [], [], {}, {}, False, None))
    monkeypatch.setattr("jupr_app.services.admin_challenge_ladder_service.submit_match_batch", lambda *_args, **_kwargs: SimpleNamespace(ok=True, data={"inserted": 2}, errors=[]))
    result = record_admin_challenge_ladder_result(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=5,
        partner_b_defender_id=6,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=True,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH LADDER RESULT",
    )
    assert result["official_matches"]["inserted"] == 2
    assert result["challenge"]["winner_id"] == 2
    ranks = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}
    assert ranks[2] == 1
