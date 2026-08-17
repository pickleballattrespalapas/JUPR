from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import hmac
from itertools import combinations
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from jupr_app.services.staging_write_guard import (
    staging_admin_team_league_writes_enabled,
    staging_public_team_league_writes_enabled,
)
from jupr_app.services.team_league_service import (
    COMMIT_PLAYOFFS_CONFIRMATION,
    COMMIT_SCHEDULE_CONFIRMATION,
    COMPENSATE_OPERATION_CONFIRMATION,
    FINALIZE_OPERATION_CONFIRMATION,
    FORFEIT_FIXTURE_CONFIRMATION,
    PAIR_WAITLIST_CONFIRMATION,
    RECONCILE_FIXTURE_CONFIRMATION,
    SAVE_SETTINGS_CONFIRMATION,
    SCORE_FIXTURE_CONFIRMATION,
    SOLO_SIGNUP_CONFIRMATION,
    TEAM_SIGNUP_CONFIRMATION,
    WITHDRAW_WAITLIST_CONFIRMATION,
    TeamLeagueConflictError,
    _fixture_match_date,
    _validate_fixture_players,
    commit_admin_team_league_schedule,
    confirm_public_team_league_partner,
    generate_playoff_fixtures,
    generate_round_robin_fixtures,
    get_public_team_league,
    list_public_team_leagues,
    partner_token_hash,
    register_public_team_league,
    save_admin_team_league_settings,
)


def _played_pairs(fixtures: list[dict]) -> Counter[tuple[str, str]]:
    return Counter(
        tuple(sorted((str(row["team_a_id"]), str(row["team_b_id"]))))
        for row in fixtures
        if row["status"] == "scheduled"
    )


class _Execute:
    def __init__(self, callback):
        self.callback = callback

    def execute(self):
        return SimpleNamespace(data=self.callback())


class _RowsQuery:
    def __init__(self, rows: list[dict]):
        self.rows = [dict(row) for row in rows]
        self.filters: dict[str, object] = {}
        self.limit_count: int | None = None

    def select(self, _columns: str):
        return self

    def eq(self, field: str, value: object):
        self.filters[field] = value
        return self

    def order(self, _field: str):
        return self

    def limit(self, count: int):
        self.limit_count = count
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(row.get(field) == value for field, value in self.filters.items())
        ]
        if self.limit_count is not None:
            rows = rows[: self.limit_count]
        return SimpleNamespace(data=rows)


class _RegistrationRecoverySupabase:
    def __init__(
        self,
        *,
        lose_first_register_response: bool = False,
        manager_status: str = "active",
    ):
        self.lose_first_register_response = lose_first_register_response
        self.manager_status = manager_status
        self.active_fingerprint: str | None = None
        self.result: dict | None = None
        self.register_calls = 0

    def table(self, name: str):
        rows = {
            "team_league_settings": [
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "status": "registration_open",
                    "registration_open": True,
                    "registration_closes_at": None,
                }
            ],
            "leagues_metadata": [
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "status": self.manager_status,
                    "is_active": self.manager_status == "active",
                }
            ],
        }
        return _RowsQuery(rows.get(name, []))

    def rpc(self, name: str, params: dict):
        return _Execute(lambda: self._execute_rpc(name, params))

    def _execute_rpc(self, name: str, params: dict):
        if name == "team_league_recover_public_registration_v1":
            if self.active_fingerprint is None:
                return {"ok": True, "found": False}
            if params["p_request_fingerprint"] != self.active_fingerprint:
                raise RuntimeError(
                    "TEAM_LEAGUE_REGISTRATION_IDENTITY_CONFLICT"
                )
            return {
                **dict(self.result or {}),
                "found": True,
                "idempotent": True,
                "recovered_by_business_identity": True,
            }
        if name == "team_league_register_public_v1":
            self.register_calls += 1
            self.active_fingerprint = params["p_request_fingerprint"]
            self.result = {
                "ok": True,
                "committed": True,
                "operation_id": params["p_operation_id"],
                "signup_type": "solo",
                "waitlist_id": "waitlist-1",
                "status": "waiting",
                "message": "You are on the partner waitlist.",
                "idempotent": False,
            }
            if self.lose_first_register_response:
                self.lose_first_register_response = False
                raise RuntimeError("network response lost after commit")
            return self.result
        raise AssertionError(f"unexpected RPC {name}")


class _RecoveredPendingTeamSupabase:
    def __init__(self):
        self.invite_token_hash = ""
        self.rpc_names: list[str] = []

    def table(self, name: str):
        rows = {
            "team_league_settings": [
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "status": "registration_open",
                    "registration_open": True,
                    "registration_closes_at": None,
                }
            ],
            "leagues_metadata": [
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "status": "active",
                    "is_active": True,
                }
            ],
        }
        return _RowsQuery(rows.get(name, []))

    def rpc(self, name: str, params: dict):
        self.rpc_names.append(name)
        return _Execute(lambda: self._execute_rpc(name, params))

    def _execute_rpc(self, name: str, params: dict):
        if name == "team_league_recover_public_registration_v1":
            self.invite_token_hash = params["p_invite_token_hash"]
            return {
                "ok": True,
                "found": True,
                "committed": True,
                "operation_id": "original-operation",
                "signup_type": "team",
                "team_id": "team-1",
                "status": "pending_partner",
                "message": "Team saved.",
                "idempotent": True,
                "recovered_by_business_identity": True,
                "invitation_send_required": True,
            }
        if name == "team_league_claim_partner_invitation_v1":
            assert params["p_token_hash"] == self.invite_token_hash
            return {
                "ok": True,
                "send_required": True,
                "status": "claimed",
            }
        if name == "team_league_finish_partner_invitation_v1":
            assert params["p_delivery_status"] == "dry_run"
            return {"ok": True, "status": "dry_run"}
        raise AssertionError(f"unexpected RPC {name}")


class _PartnerConfirmationSupabase:
    def __init__(self):
        self.filters: dict[str, object] = {}
        self.rpc_names: list[str] = []

    def table(self, name: str):
        assert name == "team_league_teams"
        self.filters = {}
        return self

    def select(self, _columns: str):
        return self

    def eq(self, field: str, value: object):
        self.filters[field] = value
        return self

    def limit(self, _count: int):
        return self

    def execute(self):
        matches = (
            self.filters.get("id") == "team-1"
            and self.filters.get("club_id") == "club-a"
        )
        return SimpleNamespace(
            data=[{"id": "team-1", "club_id": "club-a"}] if matches else []
        )

    def rpc(self, name: str, _params: dict):
        self.rpc_names.append(name)
        raise AssertionError("cross-club confirmation must not reach the RPC")


def _register_solo(
    supabase: _RegistrationRecoverySupabase,
    *,
    idempotency_key: str,
    note: str = "",
) -> dict:
    return register_public_team_league(
        supabase,
        club_id="club",
        league_name="Open",
        signup_type="solo",
        player_id=1,
        contact_email="alex@example.com",
        note=note,
        idempotency_key=idempotency_key,
        confirmation_text=SOLO_SIGNUP_CONFIRMATION,
        public_base_url="https://staging-web.example.test/clubs/club",
        club_name="Test Club",
    )


def test_registration_response_loss_recovers_committed_business_request(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _RegistrationRecoverySupabase(
        lose_first_register_response=True
    )

    result = _register_solo(
        supabase, idempotency_key="register:lost-response"
    )

    assert result["waitlist_id"] == "waitlist-1"
    assert result["recovered_by_business_identity"] is True
    assert supabase.register_calls == 1


def test_draft_manager_league_is_hidden_and_rejects_direct_registration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _RegistrationRecoverySupabase(manager_status="draft")

    public_list = list_public_team_leagues(supabase, club_id="club")

    assert public_list == {"ok": True, "leagues": [], "league_count": 0}
    with pytest.raises(ValueError, match="Registration is not open"):
        _register_solo(supabase, idempotency_key="register:draft-direct")
    with pytest.raises(ValueError, match="not found"):
        get_public_team_league(
            supabase,
            club_id="club",
            league_name="Open",
        )
    assert supabase.register_calls == 0


def test_active_manager_league_is_listed_for_public_registration() -> None:
    supabase = _RegistrationRecoverySupabase(manager_status="active")

    public_list = list_public_team_leagues(supabase, club_id="club")

    assert public_list["league_count"] == 1
    assert public_list["leagues"][0]["league_name"] == "Open"
    assert public_list["leagues"][0]["registration_open"] is True


def test_registration_refresh_with_new_key_recovers_exact_prior_success(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _RegistrationRecoverySupabase()
    original = _register_solo(
        supabase, idempotency_key="register:first-browser-key"
    )

    recovered = _register_solo(
        supabase, idempotency_key="register:new-browser-key"
    )

    assert recovered["waitlist_id"] == original["waitlist_id"]
    assert recovered["operation_id"] == original["operation_id"]
    assert recovered["recovered_by_business_identity"] is True
    assert recovered["idempotent"] is True
    assert supabase.register_calls == 1


def test_registration_refresh_with_changed_request_conflicts(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _RegistrationRecoverySupabase()
    _register_solo(
        supabase, idempotency_key="register:original-request"
    )

    with pytest.raises(TeamLeagueConflictError, match="changed"):
        _register_solo(
            supabase,
            idempotency_key="register:changed-request",
            note="Please pair me with Jordan.",
        )
    assert supabase.register_calls == 1


def test_recovered_pending_team_rotates_and_delivers_a_usable_invitation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET", "s" * 40)
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._fetch_rows",
        lambda *_args, **_kwargs: [
            {"id": 1, "name": "Alex"},
            {"id": 2, "name": "Blair"},
        ],
    )

    def fake_send(**kwargs):
        captured["confirmation_url"] = kwargs["confirmation_url"]
        return {"status": "dry_run", "provider_message_id": None}

    monkeypatch.setattr(
        "jupr_app.services.team_league_service."
        "send_team_league_partner_invitation_email",
        fake_send,
    )
    supabase = _RecoveredPendingTeamSupabase()

    result = register_public_team_league(
        supabase,
        club_id="club",
        league_name="Open",
        signup_type="team",
        player_id=1,
        partner_player_id=2,
        contact_email="alex@example.com",
        partner_email="blair@example.com",
        team_name="Aces",
        idempotency_key="register:recovered-team",
        confirmation_text=TEAM_SIGNUP_CONFIRMATION,
        public_base_url="https://staging-web.example.test/clubs/club",
        club_name="Test Club",
    )

    assert result["invitation_delivery_status"] == "dry_run"
    assert "#token=" in captured["confirmation_url"]
    assert "?token=" not in captured["confirmation_url"]
    assert supabase.rpc_names == [
        "team_league_recover_public_registration_v1",
        "team_league_claim_partner_invitation_v1",
        "team_league_finish_partner_invitation_v1",
    ]


@pytest.mark.parametrize("team_count,expected_weeks", [(4, 3), (5, 5)])
def test_round_robin_meets_each_opponent_once_and_at_most_once_per_week(
    team_count: int, expected_weeks: int
) -> None:
    team_ids = [f"team-{index}" for index in range(1, team_count + 1)]

    fixtures = generate_round_robin_fixtures(
        team_ids,
        start_date="2026-08-03",
        start_time="18:00",
        timezone_name="America/Chicago",
    )

    assert {row["week_number"] for row in fixtures} == set(
        range(1, expected_weeks + 1)
    )
    assert _played_pairs(fixtures) == Counter(
        tuple(sorted(pair)) for pair in combinations(team_ids, 2)
    )
    appearances: defaultdict[tuple[int, str], int] = defaultdict(int)
    for row in fixtures:
        for team_id in (row.get("team_a_id"), row.get("team_b_id")):
            if team_id:
                appearances[(int(row["week_number"]), str(team_id))] += 1
    assert appearances
    assert max(appearances.values()) == 1
    if team_count % 2:
        byes = [row for row in fixtures if row["status"] == "bye"]
        assert len(byes) == team_count
        assert Counter(str(row["team_a_id"]) for row in byes) == Counter(
            {team_id: 1 for team_id in team_ids}
        )


def test_round_robin_preserves_local_clock_across_daylight_saving() -> None:
    fixtures = generate_round_robin_fixtures(
        ["a", "b", "c", "d"],
        start_date="2026-10-26",
        start_time="18:00",
        timezone_name="America/Chicago",
    )
    first_each_week = {
        int(row["week_number"]): datetime.fromisoformat(row["scheduled_at"])
        for row in fixtures
        if int(row["bracket_slot"]) == 1
    }
    local = [
        value.astimezone(ZoneInfo("America/Chicago"))
        for _, value in sorted(first_each_week.items())
    ]

    assert [value.hour for value in local] == [18, 18, 18]
    assert local[0].utcoffset() != local[-1].utcoffset()


@pytest.mark.parametrize(
    "team_count,expected_byes,expected_fixtures",
    [(6, 2, 7), (8, 0, 7), (9, 7, 15), (16, 0, 15)],
)
def test_seeded_playoffs_put_top_seeds_in_opposite_halves(
    team_count: int, expected_byes: int, expected_fixtures: int
) -> None:
    standings = [
        {"team_id": f"team-{seed}", "team_name": f"Team {seed}"}
        for seed in range(1, team_count + 1)
    ]

    fixtures = generate_playoff_fixtures(
        standings,
        playoff_format="all_team_single_elimination",
        playoff_team_count=team_count,
    )

    first_round = [row for row in fixtures if row["round_number"] == 1]
    assert sum(row["status"] == "bye" for row in first_round) == expected_byes
    seed_one_slot = next(
        int(row["bracket_slot"])
        for row in first_round
        if row.get("team_a_id") == "team-1" or row.get("team_b_id") == "team-1"
    )
    seed_two_slot = next(
        int(row["bracket_slot"])
        for row in first_round
        if row.get("team_a_id") == "team-2" or row.get("team_b_id") == "team-2"
    )
    half_size = len(first_round) // 2
    assert (seed_one_slot <= half_size) != (seed_two_slot <= half_size)
    assert len(fixtures) == expected_fixtures
    assert Counter(
        str(team_id)
        for row in first_round
        for team_id in (row.get("team_a_id"), row.get("team_b_id"))
        if team_id
    ) == Counter({f"team-{seed}": 1 for seed in range(1, team_count + 1)})


def test_fixture_match_date_uses_league_local_calendar_date() -> None:
    assert (
        _fixture_match_date(
            "2026-12-08T00:00:00+00:00",
            timezone_name="America/Chicago",
        )
        == "2026-12-07"
    )


def test_partner_confirmation_is_bound_to_the_route_club(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _PartnerConfirmationSupabase()

    with pytest.raises(ValueError, match="not found"):
        confirm_public_team_league_partner(
            supabase,
            club_id="club-b",
            team_id="team-1",
            token="token-with-at-least-twenty-four-characters",
            accept=True,
            idempotency_key="confirm:cross-club",
        )

    assert supabase.rpc_names == []


def test_stale_schedule_preview_is_rejected_before_rpc() -> None:
    with pytest.raises(TeamLeagueConflictError, match="stale"):
        commit_admin_team_league_schedule(
            object(),
            club_id="club",
            league_name="Open",
            phase="regular",
            fixtures=[
                {
                    "round_number": 1,
                    "week_number": 1,
                    "bracket_slot": 1,
                    "team_a_id": "a",
                    "team_b_id": "b",
                    "status": "scheduled",
                }
            ],
            expected_schedule_version=0,
            expected_standings_version=0,
            expected_roster_version=0,
            confirmed_roster_fingerprint_value="0" * 64,
            preview_fingerprint="f" * 64,
            idempotency_key="schedule:test-1",
            confirmation_text=COMMIT_SCHEDULE_CONFIRMATION,
            actor_email="owner@example.com",
            actor_role="club_owner",
            source="test",
        )


def test_substitutes_follow_the_saved_setup_and_cannot_be_on_another_team() -> None:
    teams = [
        {
            "id": "a",
            "status": "confirmed",
            "captain_player_id": 1,
            "partner_player_id": 2,
        },
        {
            "id": "b",
            "status": "confirmed",
            "captain_player_id": 3,
            "partner_player_id": 4,
        },
        {
            "id": "c",
            "status": "confirmed",
            "captain_player_id": 5,
            "partner_player_id": 6,
        },
    ]
    fixture = {"team_a_id": "a", "team_b_id": "b"}
    active_players = {player_id: {"id": player_id} for player_id in range(1, 8)}

    with pytest.raises(ValueError, match="disabled"):
        _validate_fixture_players(
            players=active_players,
            teams=teams,
            fixture=fixture,
            team_a_player_ids=[1, 7],
            team_b_player_ids=[3, 4],
            allow_substitutes=False,
        )

    assert _validate_fixture_players(
        players=active_players,
        teams=teams,
        fixture=fixture,
        team_a_player_ids=[1, 7],
        team_b_player_ids=[3, 4],
        allow_substitutes=True,
    ) == [{"incoming_player_id": 7, "outgoing_player_id": 2}]

    with pytest.raises(ValueError, match="another team"):
        _validate_fixture_players(
            players=active_players,
            teams=teams,
            fixture=fixture,
            team_a_player_ids=[1, 5],
            team_b_player_ids=[3, 4],
            allow_substitutes=True,
        )
    with pytest.raises(ValueError, match="active club player"):
        _validate_fixture_players(
            players={key: value for key, value in active_players.items() if key != 7},
            teams=teams,
            fixture=fixture,
            team_a_player_ids=[1, 7],
            team_b_player_ids=[3, 4],
            allow_substitutes=True,
        )


def test_partner_token_uses_dedicated_purpose_and_legacy_secret_fallback(
    monkeypatch
) -> None:
    token = "token-with-at-least-twenty-four-characters"
    secret = "s" * 40
    monkeypatch.delenv("JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("JUPR_PUBLIC_REGISTRATION_TOKEN_SECRET", secret)

    expected = hmac.new(
        secret.encode(),
        f"team-league-partner:v1:{token}".encode(),
        hashlib.sha256,
    ).hexdigest()

    assert partner_token_hash(token) == expected


def test_partner_token_uses_staged_registration_edit_secret_as_final_fallback(
    monkeypatch,
) -> None:
    token = "token-with-at-least-twenty-four-characters"
    secret = "registration-edit-secret-" + "x" * 24
    monkeypatch.delenv("JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("JUPR_PUBLIC_REGISTRATION_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", secret)

    assert partner_token_hash(token) == hmac.new(
        secret.encode(),
        f"team-league-partner:v1:{token}".encode(),
        hashlib.sha256,
    ).hexdigest()


def test_partner_token_rejects_short_fallback_secret(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("JUPR_PUBLIC_REGISTRATION_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "too-short")

    with pytest.raises(PermissionError, match="at least 32"):
        partner_token_hash("token-with-at-least-twenty-four-characters")


def test_team_league_confirmations_are_distinct() -> None:
    confirmations = {
        TEAM_SIGNUP_CONFIRMATION,
        SOLO_SIGNUP_CONFIRMATION,
        SAVE_SETTINGS_CONFIRMATION,
        PAIR_WAITLIST_CONFIRMATION,
        WITHDRAW_WAITLIST_CONFIRMATION,
        COMMIT_SCHEDULE_CONFIRMATION,
        COMMIT_PLAYOFFS_CONFIRMATION,
        SCORE_FIXTURE_CONFIRMATION,
        FORFEIT_FIXTURE_CONFIRMATION,
        RECONCILE_FIXTURE_CONFIRMATION,
        FINALIZE_OPERATION_CONFIRMATION,
        COMPENSATE_OPERATION_CONFIRMATION,
    }
    assert len(confirmations) == 12


def test_team_league_runtime_gates_deny_production(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "1"
    )
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-manager")

    assert staging_admin_team_league_writes_enabled() is False
    assert staging_public_team_league_writes_enabled() is False
    with pytest.raises(PermissionError, match="staging-only"):
        save_admin_team_league_settings(
            object(),
            club_id="club",
            league_name="Open",
            settings={},
            expected_settings_version=0,
            idempotency_key="settings:test-1",
            confirmation_text=SAVE_SETTINGS_CONFIRMATION,
            actor_email="owner@example.com",
            actor_role="club_owner",
            source="test",
        )
    with pytest.raises(PermissionError, match="staging-only"):
        confirm_public_team_league_partner(
            object(),
            club_id="club",
            team_id="team",
            token="token-with-at-least-twenty-four-characters",
            accept=True,
            idempotency_key="confirm:test-1",
        )


def test_team_league_staging_gates_require_the_matching_wave_and_flag(
    monkeypatch
) -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "1"
    )
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert staging_admin_team_league_writes_enabled() is False
    assert staging_public_team_league_writes_enabled() is False

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-manager")
    assert staging_admin_team_league_writes_enabled() is True
    assert staging_public_team_league_writes_enabled() is False

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    assert staging_admin_team_league_writes_enabled() is False
    assert staging_public_team_league_writes_enabled() is True

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    assert staging_admin_team_league_writes_enabled() is True
    assert staging_public_team_league_writes_enabled() is True

    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "0"
    )
    assert staging_admin_team_league_writes_enabled() is False
    assert staging_public_team_league_writes_enabled() is True
