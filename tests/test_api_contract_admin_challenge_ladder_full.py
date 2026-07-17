from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services.admin_challenge_ladder_service import (
    accept_admin_challenge_ladder_challenge,
    add_admin_challenge_ladder_roster_player,
    create_admin_challenge_ladder_challenge,
    get_admin_challenge_ladder_tier_movement_review,
    move_admin_challenge_ladder_roster_player,
    preview_admin_challenge_ladder_result,
    preview_admin_challenge_ladder_result_for_challenge,
    preview_admin_challenge_ladder_tier_roster_replacement,
    record_admin_challenge_ladder_forfeit,
    record_admin_challenge_ladder_pass,
    record_admin_challenge_ladder_result,
    replace_admin_challenge_ladder_tier_roster,
    save_admin_challenge_ladder_player_overrides,
)
from jupr_app.domain.challenge_ladder import month_key_utc
from services.api.admin_challenge_ladder_routes import install_admin_challenge_ladder_routes


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.insert_payload = None
        self.upsert_payload = None
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

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = deepcopy(payload)
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
        if self.upsert_payload is not None:
            payloads = self.upsert_payload if isinstance(self.upsert_payload, list) else [self.upsert_payload]
            saved = []
            for payload in payloads:
                existing = next(
                    (
                        row
                        for row in rows
                        if str(row.get("club_id")) == str(payload.get("club_id"))
                        and str(row.get("player_id")) == str(payload.get("player_id"))
                    ),
                    None,
                )
                if existing is None:
                    existing = {"id": len(rows) + 100, **payload}
                    rows.append(existing)
                else:
                    existing.update(payload)
                saved.append(dict(existing))
            return SimpleNamespace(data=saved)
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
                {"club_id": "club", "id": 7, "name": "Roster Candidate"},
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
            "ladder_pass_usage": [],
            "ladder_player_flags": [],
            "matches": [],
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
    supabase.storage["ladder_challenges"] = []
    created = create_admin_challenge_ladder_challenge(supabase, club_id="club", challenger_id=2, defender_id=1, tier_id="ADV", ledger_ref="ledger", override=False, start_clock=True, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="CREATE LADDER CHALLENGE")
    assert created["challenge"]["status"] == "PENDING_ACCEPTANCE"
    accepted = accept_admin_challenge_ladder_challenge(supabase, club_id="club", challenge_id=created["challenge"]["id"], actor_email="admin@example.com", actor_role="club_owner", confirmation_text="ACCEPT LADDER CHALLENGE")
    assert accepted["challenge"]["status"] == "ACCEPTED_SCHEDULING"


def test_create_challenge_rejects_locked_players_without_override(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    try:
        create_admin_challenge_ladder_challenge(
            supabase,
            club_id="club",
            challenger_id=2,
            defender_id=1,
            tier_id="ADV",
            ledger_ref=None,
            override=False,
            start_clock=False,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="CREATE LADDER CHALLENGE",
        )
    except ValueError as exc:
        assert "status: Locked" in str(exc)
    else:
        raise AssertionError("expected locked-player eligibility rejection")
    assert len(supabase.storage["ladder_challenges"]) == 1


def test_create_challenge_override_returns_copyable_notice(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    monkeypatch.setenv("LADDER_ADMIN_NAME", "Director Dana")
    monkeypatch.setenv("LADDER_ADMIN_CONTACT", "dana@example.com")
    supabase = FakeSupabase()

    created = create_admin_challenge_ladder_challenge(
        supabase,
        club_id="club",
        challenger_id=2,
        defender_id=1,
        tier_id="ADV",
        challenger_contact="challenger@example.com",
        ledger_ref="front desk 42",
        override=True,
        start_clock=False,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="CREATE LADDER CHALLENGE",
    )

    assert created["challenge"]["id"] == 101
    assert "challenger@example.com" in created["notice"]["email_full"]
    assert "Director Dana" in created["notice"]["email_full"]
    assert "dana@example.com" in created["notice"]["sms"]
    assert "front desk 42" in created["notice"]["email_full"]
    assert "Challenge ID: #101" in created["notice"]["email_full"]


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


def test_challenge_result_preview_is_read_only_and_reports_rank_effect(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    before = deepcopy(supabase.storage)

    result = preview_admin_challenge_ladder_result_for_challenge(
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
    )

    assert result["mode"] == "challenge_ladder_result_preview"
    assert result["preview"]["final_winner_id"] == 2
    assert result["rank_result"] == {"would_swap": True, "reason": "challenger win"}
    assert result["partner_names"]["a_chal"] == "Partner A"
    assert result["would_publish_official_matches"] is True
    assert supabase.storage == before


def test_challenge_result_preview_rejects_player_from_another_club(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"

    try:
        preview_admin_challenge_ladder_result_for_challenge(
            supabase,
            club_id="club",
            challenge_id=100,
            partner_a_challenger_id=3,
            partner_a_defender_id=4,
            partner_b_challenger_id=5,
            partner_b_defender_id=999,
            match_a_games=[[11, 5], [11, 6]],
            match_b_games=[[11, 4], [11, 6]],
            match_date="2026-01-02T00:00:00Z",
            winner_override="computed",
            publish_official_matches=True,
        )
    except ValueError as exc:
        assert "players in this club" in str(exc)
    else:
        raise AssertionError("expected club-scoped partner validation")


def test_challenge_result_preview_route_requires_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    app = FastAPI()
    install_admin_challenge_ladder_routes(app, get_supabase_client=lambda: supabase)

    contract_path = "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview"
    path = "/admin/clubs/club/challenge-ladder/challenges/100/result/preview"
    assert "post" in app.openapi()["paths"][contract_path]
    response = TestClient(app).post(
        path,
        json={
            "partner_a_challenger_id": 3,
            "partner_a_defender_id": 4,
            "partner_b_challenger_id": 5,
            "partner_b_defender_id": 6,
            "match_a_games": [[11, 5], [11, 6]],
            "match_b_games": [[11, 4], [11, 6]],
            "match_date": "2026-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 401
    assert supabase.storage["admin_activity_log"] == []


def test_forfeit_swaps_rank_when_defender_forfeits(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    result = record_admin_challenge_ladder_forfeit(supabase, club_id="club", challenge_id=100, forfeited_by_id=1, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="RECORD LADDER FORFEIT")
    assert result["challenge"]["winner_id"] == 2
    ranks = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}
    assert ranks[2] == 1
    assert ranks[1] == 2


def test_monthly_pass_closes_challenge_without_rank_change(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    ranks_before = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}

    result = record_admin_challenge_ladder_pass(
        supabase,
        club_id="club",
        challenge_id=100,
        player_id=2,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="RECORD LADDER PASS",
    )

    assert result["challenge"]["status"] == "CANCELED"
    assert result["pass_usage"]["player_id"] == 2
    assert result["pass_usage"]["month_key"] == month_key_utc(datetime.now(timezone.utc))
    assert {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]} == ranks_before
    assert supabase.storage["admin_activity_log"][-1]["action_type"] == "challenge_pass_used"


def test_monthly_pass_rejects_duplicate_usage_in_utc_month(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_pass_usage"].append(
        {
            "club_id": "club",
            "player_id": 2,
            "month_key": month_key_utc(datetime.now(timezone.utc)),
            "used_at": "2026-07-01T00:00:00Z",
            "challenge_id": 99,
        }
    )

    try:
        record_admin_challenge_ladder_pass(
            supabase,
            club_id="club",
            challenge_id=100,
            player_id=2,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="RECORD LADDER PASS",
        )
    except ValueError as exc:
        assert "already used" in str(exc)
    else:
        raise AssertionError("expected duplicate monthly-pass rejection")
    assert supabase.storage["ladder_challenges"][0]["status"] == "PENDING_ACCEPTANCE"


def test_roster_add_appends_new_player_to_selected_tier(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = add_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=7,
        tier_id="INT",
        admin_note="approved placement",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="ADD LADDER PLAYER",
    )

    assert result["reactivated"] is False
    assert result["roster"]["player_name"] == "Roster Candidate"
    assert result["roster"]["tier_id"] == "INT"
    assert result["roster"]["rank"] == 1
    assert supabase.storage["admin_activity_log"][-1]["entity_type"] == "ladder_roster"


def test_roster_add_reactivates_at_bottom(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    inactive = supabase.storage["ladder_roster"][-1]
    inactive.update({"is_active": False, "left_at": "2026-06-01T00:00:00Z"})

    result = add_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=6,
        tier_id="ADV",
        admin_note=None,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="ADD LADDER PLAYER",
    )

    assert result["reactivated"] is True
    assert result["roster"]["rank"] == 6
    assert result["roster"]["is_active"] is True


def test_tier_roster_replacement_preview_is_read_only_and_blocks_open_challenges(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    result = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Challenger", "Defender", "Roster Candidate"],
    )

    assert result["can_apply"] is False
    assert result["summary"]["current_count"] == 6
    assert result["summary"]["proposed_count"] == 3
    assert result["summary"]["removed_count"] == 4
    assert [row["player_id"] for row in result["proposed_roster"]] == [2, 1, 7]
    assert result["open_challenge_blockers"][0]["challenge_id"] == 100
    assert supabase.storage == before


def test_tier_roster_replacement_preview_rejects_duplicate_and_missing_names(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    for ranked_names, expected in [
        (["Defender", "Defender"], "Duplicate names"),
        (["Defender", "Not A Club Player"], "Create these club players"),
    ]:
        try:
            preview_admin_challenge_ladder_tier_roster_replacement(
                supabase,
                club_id="club",
                tier_id="ADV",
                ranked_names=ranked_names,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("expected replacement roster name validation error")


def test_tier_roster_replacement_rejects_stale_preview(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = []
    preview = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Defender", "Challenger", "Partner A", "Partner B", "Partner C", "Partner D"],
    )
    supabase.storage["ladder_roster"][0]["rank"] = 2

    try:
        replace_admin_challenge_ladder_tier_roster(
            supabase,
            club_id="club",
            tier_id="ADV",
            ranked_player_ids=[row["player_id"] for row in preview["proposed_roster"]],
            preview_fingerprint=preview["preview_fingerprint"],
            admin_note=None,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="REPLACE LADDER TIER",
        )
    except ValueError as exc:
        assert "changed after preview" in str(exc)
    else:
        raise AssertionError("expected stale tier-roster preview rejection")
    assert supabase.storage["admin_activity_log"] == []


def test_tier_roster_replacement_applies_reviewed_order_and_recompresses_source(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = []
    supabase.storage["ladder_roster"][-1].update({"tier_id": "INT", "rank": 2})
    supabase.storage["ladder_roster"].append(
        {"club_id": "club", "id": 16, "player_id": 7, "tier_id": "INT", "rank": 1, "is_active": True}
    )
    preview = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Challenger", "Defender", "Roster Candidate"],
    )

    result = replace_admin_challenge_ladder_tier_roster(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_player_ids=[row["player_id"] for row in preview["proposed_roster"]],
        preview_fingerprint=preview["preview_fingerprint"],
        admin_note="annual roster reset",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="REPLACE LADDER TIER",
    )

    assert [(row["player_id"], row["rank"]) for row in result["roster"]] == [(2, 1), (1, 2), (7, 3)]
    assert result["summary"]["moved_from_other_tier_count"] == 1
    assert result["recompressed_player_ids"] == [6]
    inactive_ids = {
        row["player_id"]
        for row in supabase.storage["ladder_roster"]
        if row.get("is_active") is False
    }
    assert {3, 4, 5}.issubset(inactive_ids)
    source_player = next(row for row in supabase.storage["ladder_roster"] if row["player_id"] == 6)
    assert source_player["tier_id"] == "INT"
    assert source_player["rank"] == 1
    audit = supabase.storage["admin_activity_log"][-1]
    assert audit["action_type"] == "roster_replace_tier"
    assert audit["before_json"]["rows"]
    assert audit["after_json"]["preview_fingerprint"] == preview["preview_fingerprint"]


def test_roster_move_appends_and_recompresses_previous_tier(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = move_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=4,
        destination_tier="INT",
        recompress_old=True,
        admin_note="reviewed movement",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="MOVE LADDER PLAYER",
    )

    assert result["previous_tier"] == "ADV"
    assert result["roster"]["tier_id"] == "INT"
    assert result["roster"]["rank"] == 1
    assert result["recompressed_count"] == 2
    advanced = sorted(
        (row for row in supabase.storage["ladder_roster"] if row["tier_id"] == "ADV" and row["is_active"]),
        key=lambda row: row["rank"],
    )
    assert [row["rank"] for row in advanced] == [1, 2, 3, 4, 5]


def test_player_overrides_insert_normalizes_utc_and_writes_audit(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = save_admin_challenge_ladder_player_overrides(
        supabase,
        club_id="club",
        player_id=2,
        vacation_until="2026-08-01T12:00:00-05:00",
        reinstate_required=True,
        reinstate_notes="Contact ladder director",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE LADDER OVERRIDES",
    )

    assert result["player_flags"]["vacation_until"] == "2026-08-01T17:00:00+00:00"
    assert result["player_flags"]["reinstate_required"] is True
    assert supabase.storage["ladder_player_flags"][0]["player_id"] == 2
    assert supabase.storage["admin_activity_log"][-1]["entity_type"] == "ladder_player_flags"


def test_player_overrides_update_can_clear_values_without_touching_tier_flags(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_player_flags"].append(
        {
            "club_id": "club",
            "player_id": 2,
            "vacation_until": "2026-08-01T17:00:00+00:00",
            "reinstate_required": True,
            "reinstate_notes": "Old note",
            "tier_move_flag": True,
            "tier_move_dest_tier": "INT",
            "tier_move_count": 10,
        }
    )

    result = save_admin_challenge_ladder_player_overrides(
        supabase,
        club_id="club",
        player_id=2,
        vacation_until=None,
        reinstate_required=False,
        reinstate_notes="",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE LADDER OVERRIDES",
    )

    stored = supabase.storage["ladder_player_flags"][0]
    assert stored["vacation_until"] is None
    assert stored["reinstate_required"] is False
    assert stored["reinstate_notes"] is None
    assert stored["tier_move_flag"] is True
    assert result["player_flags"]["tier_move_dest_tier"] == "INT"


def test_player_overrides_reject_naive_date_time(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    try:
        save_admin_challenge_ladder_player_overrides(
            supabase,
            club_id="club",
            player_id=2,
            vacation_until="2026-08-01T12:00:00",
            reinstate_required=False,
            reinstate_notes=None,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="SAVE LADDER OVERRIDES",
        )
    except ValueError as exc:
        assert "UTC offset or Z" in str(exc)
    else:
        raise AssertionError("expected timezone validation error")
    assert supabase.storage["ladder_player_flags"] == []


def _tier_movement_match(match_id, date, end_rating):
    return {
        "club_id": "club",
        "id": match_id,
        "date": date,
        "t1_p1": 2,
        "t1_p2": 3,
        "t2_p1": 4,
        "t2_p2": 5,
        "t1_p1_r_end": end_rating,
        "t1_p2_r_end": 1600,
        "t2_p1_r_end": 1600,
        "t2_p2_r_end": 1600,
        "deleted_at": None,
    }


def test_tier_movement_review_reports_ten_match_trigger(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["matches"] = [
        _tier_movement_match(index, f"2026-07-{index:02d}T12:00:00Z", 1400)
        for index in range(1, 11)
    ]

    result = get_admin_challenge_ladder_tier_movement_review(supabase, club_id="club")

    assert result["summary"]["trigger_count"] == 1
    assert result["summary"]["required_consecutive_matches"] == 10
    assert result["triggers"] == [
        {
            "player_id": 2,
            "player_name": "Challenger",
            "current_tier": "ADV",
            "destination_tier": "INT",
            "consecutive_match_count": 10,
            "latest_match_at": "2026-07-10T12:00:00+00:00",
        }
    ]


def test_tier_movement_review_resets_after_current_tier_match(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["matches"] = [
        *[_tier_movement_match(index, f"2026-07-{index:02d}T12:00:00Z", 1400) for index in range(1, 11)],
        _tier_movement_match(11, "2026-07-11T12:00:00Z", 1600),
    ]

    result = get_admin_challenge_ladder_tier_movement_review(supabase, club_id="club")

    assert result["summary"]["trigger_count"] == 0
    assert result["triggers"] == []


def test_new_ladder_operation_routes_require_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    app = FastAPI()
    install_admin_challenge_ladder_routes(app, get_supabase_client=lambda: supabase)
    openapi = app.openapi()["paths"]
    before = deepcopy(supabase.storage)

    requests = [
        (
            "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/pass",
            "/admin/clubs/club/challenge-ladder/challenges/100/pass",
            {"player_id": 2, "confirmation_text": "RECORD LADDER PASS"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster",
            "/admin/clubs/club/challenge-ladder/roster",
            {"player_id": 7, "tier_id": "INT", "confirmation_text": "ADD LADDER PLAYER"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/move",
            "/admin/clubs/club/challenge-ladder/roster/4/move",
            {"destination_tier": "INT", "confirmation_text": "MOVE LADDER PLAYER"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier/preview",
            "/admin/clubs/club/challenge-ladder/roster/replace-tier/preview",
            {"tier_id": "ADV", "ranked_names": ["Defender"]},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier",
            "/admin/clubs/club/challenge-ladder/roster/replace-tier",
            {
                "tier_id": "ADV",
                "ranked_player_ids": [1],
                "preview_fingerprint": "not-a-real-preview",
                "confirmation_text": "REPLACE LADDER TIER",
            },
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides",
            "/admin/clubs/club/challenge-ladder/roster/2/overrides",
            {"vacation_until": None, "reinstate_required": False, "confirmation_text": "SAVE LADDER OVERRIDES"},
            "put",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/tier-movement-review",
            "/admin/clubs/club/challenge-ladder/tier-movement-review",
            None,
            "get",
        ),
    ]
    for contract_path, path, payload, method in requests:
        assert method in openapi[contract_path]
        response = TestClient(app).request(method.upper(), path, json=payload)
        assert response.status_code == 401
    assert supabase.storage == before


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
