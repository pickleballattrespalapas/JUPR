from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.team_league_service import (
    CREATE_TEAM_CONFIRMATION,
    PAIR_WAITLIST_CONFIRMATION,
    SAVE_SETTINGS_CONFIRMATION,
    TEAM_SIGNUP_CONFIRMATION,
    _enforce_team_category,
    admin_team_league_roster_action,
    admin_team_league_waitlist_action,
    create_admin_team_league_team,
    get_public_team_league,
    list_public_team_leagues,
    register_public_team_league,
    save_admin_team_league_settings,
)


class _RowsQuery:
    def __init__(self, rows: list[dict]):
        self.rows = [dict(row) for row in rows]
        self.filters: dict[str, object] = {}
        self.limit_count: int | None = None
        self.range_start: int | None = None
        self.range_end: int | None = None

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

    def range(self, start: int, end: int):
        self.range_start = start
        self.range_end = end
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(row.get(field) == value for field, value in self.filters.items())
        ]
        if self.range_start is not None and self.range_end is not None:
            rows = rows[self.range_start : self.range_end + 1]
        if self.limit_count is not None:
            rows = rows[: self.limit_count]
        return SimpleNamespace(data=rows)


class _Execute:
    def __init__(self, callback):
        self.callback = callback

    def execute(self):
        return SimpleNamespace(data=self.callback())


class _CompositionSupabase:
    def __init__(
        self,
        *,
        settings: dict | None = None,
        players: list[dict] | None = None,
        waitlist: list[dict] | None = None,
    ):
        base_settings = {
            "club_id": "club",
            "league_name": "Open",
            "status": "registration_open",
            "registration_open": True,
            "registration_closes_at": None,
            "team_size": 2,
            "team_category": "open",
            "allow_substitutes": False,
            "settings_version": 0,
        }
        if settings:
            base_settings.update(settings)
        self.tables = {
            "team_league_settings": [base_settings],
            "leagues_metadata": [
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "status": "active",
                    "is_active": True,
                }
            ],
            "team_league_teams": [],
            "team_league_fixtures": [],
            "team_league_solo_waitlist": waitlist or [],
            "players": players or [],
        }
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, name: str):
        return _RowsQuery(self.tables.get(name, []))

    def rpc(self, name: str, params: dict):
        self.rpc_calls.append((name, dict(params)))
        return _Execute(
            lambda: {
                "ok": True,
                "committed": True,
                "operation_id": params.get("p_operation_id"),
            }
        )


def _save_settings(monkeypatch, settings: dict) -> tuple[dict, _CompositionSupabase]:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase()
    result = save_admin_team_league_settings(
        supabase,
        club_id="club",
        league_name="Open",
        settings=settings,
        expected_settings_version=0,
        idempotency_key="settings:composition:test",
        confirmation_text=SAVE_SETTINGS_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="admin",
        source="composition_test",
    )
    return result, supabase


def test_admin_settings_persist_normalized_roster_policy(monkeypatch) -> None:
    result, supabase = _save_settings(
        monkeypatch,
        {
            "registration_open": True,
            "team_size": 3,
            "team_category": "Mixed",
            "max_alternates": 2,
            "substitute_pool_enabled": True,
            "mixed_required_men": 1,
            "mixed_required_women": 2,
            "allow_substitutes": True,
            "timezone": "America/Chicago",
        },
    )

    assert result["committed"] is True
    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == "team_league_save_settings_v2"
    assert params["p_settings"]["team_size"] == 3
    assert params["p_settings"]["team_category"] == "mixed"
    assert params["p_settings"]["max_alternates"] == 2
    assert params["p_settings"]["substitute_pool_enabled"] is True
    assert params["p_settings"]["mixed_required_men"] == 1
    assert params["p_settings"]["mixed_required_women"] == 2
    assert params["p_settings"]["allow_substitutes"] is True
    assert "allow_alternates" not in params["p_settings"]


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"team_size": 5}, "Team size"),
        ({"team_category": "juniors"}, "Choose Open"),
        (
            {
                "team_size": 3,
                "team_category": "mixed",
                "mixed_required_men": 1,
                "mixed_required_women": 1,
            },
            "Mixed roster counts",
        ),
    ],
)
def test_admin_settings_reject_unsupported_composition_policy(
    monkeypatch,
    settings: dict,
    message: str,
) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase()

    with pytest.raises(ValueError, match=message):
        save_admin_team_league_settings(
            supabase,
            club_id="club",
            league_name="Open",
            settings=settings,
            expected_settings_version=0,
            idempotency_key="settings:invalid:test",
            confirmation_text=SAVE_SETTINGS_CONFIRMATION,
            actor_email="admin@example.com",
            actor_role="admin",
            source="composition_test",
        )

    assert supabase.rpc_calls == []


def test_public_legacy_settings_receive_safe_fixed_pair_defaults() -> None:
    supabase = _CompositionSupabase()

    payload = list_public_team_leagues(supabase, club_id="club")
    detail = get_public_team_league(supabase, club_id="club", league_name="Open")

    assert payload["league_count"] == 1
    league = payload["leagues"][0]
    assert league["team_size"] == 2
    assert league["team_category"] == "open"
    assert league["online_team_registration_supported"] is True
    assert league["registration_open"] is True
    assert league["max_alternates"] == 0
    assert league["substitute_pool_enabled"] is False
    assert league["mixed_required_men"] == 1
    assert league["mixed_required_women"] == 1
    assert "allow_alternates" not in league
    assert "alternate_management_supported" not in detail["registration"]
    assert "substitute_pool_registration_supported" not in detail["registration"]


def test_public_multi_player_registration_remains_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_public_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={
            "team_size": 3,
            "max_alternates": 1,
            "substitute_pool_enabled": True,
            "allow_substitutes": True,
        },
        players=[
            {"id": 1, "club_id": "club", "gender": "Woman"},
            {"id": 2, "club_id": "club", "gender": "Man"},
        ],
    )

    listing = list_public_team_leagues(supabase, club_id="club")["leagues"][0]
    detail = get_public_team_league(supabase, club_id="club", league_name="Open")

    assert listing["registration_configured_open"] is True
    assert listing["registration_open"] is False
    assert listing["online_team_registration_supported"] is False
    assert detail["registration"]["open"] is False
    assert detail["registration"]["signup_types"] == []
    assert "Contact league staff" in detail["registration"]["unavailable_reason"]
    with pytest.raises(ValueError, match="not available yet"):
        register_public_team_league(
            supabase,
            club_id="club",
            league_name="Open",
            signup_type="team",
            player_id=1,
            partner_player_id=2,
            contact_email="alex@example.com",
            partner_email="partner@example.com",
            team_name="Partial Team",
            idempotency_key="register:multi:closed",
            confirmation_text=TEAM_SIGNUP_CONFIRMATION,
            public_base_url="https://example.test/clubs/club",
            club_name="Test Club",
        )
    assert supabase.rpc_calls == []


def test_admin_can_create_a_forming_three_player_team(monkeypatch) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={
            "team_size": 3,
            "team_category": "mixed",
            "mixed_required_men": 1,
            "mixed_required_women": 2,
            "max_alternates": 1,
            "roster_version": 7,
        },
        players=[
            {"id": 1, "club_id": "club", "name": "Captain", "gender": "Man"},
            {"id": 2, "club_id": "club", "name": "Primary", "gender": "Woman"},
        ],
    )

    result = create_admin_team_league_team(
        supabase,
        club_id="club",
        league_name="Open",
        team_name="Forming Aces",
        captain_player_id=1,
        captain_contact_email="captain@example.com",
        initial_primary_player_id=2,
        initial_primary_contact_email="primary@example.com",
        expected_roster_version=7,
        idempotency_key="create:forming:team",
        confirmation_text=CREATE_TEAM_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="admin",
    )

    assert result["committed"] is True
    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == "team_league_create_team_v1"
    assert params["p_team_name"] == "Forming Aces"
    assert params["p_captain_player_id"] == 1
    assert params["p_initial_primary_player_id"] == 2
    assert params["p_expected_roster_version"] == 7


def test_create_team_retry_reaches_database_receipt_after_team_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={
            "team_size": 3,
            "team_category": "mixed",
            "mixed_required_men": 1,
            "mixed_required_women": 2,
            "max_alternates": 1,
            "roster_version": 9,
        },
        players=[
            {"id": 1, "club_id": "club", "name": "Captain", "gender": "Man"},
            {"id": 2, "club_id": "club", "name": "Primary", "gender": "Woman"},
        ],
    )
    team_id = "00000000-0000-0000-0000-000000000002"
    supabase.tables["team_league_teams"] = [
        {
            "id": team_id,
            "club_id": "club",
            "league_name": "Open",
            "team_name": "Forming Aces",
            "status": "pending_partner",
            "captain_player_id": 1,
            "partner_player_id": 2,
        }
    ]
    supabase.tables["team_league_team_members"] = [
        {
            "id": "member-1",
            "team_id": team_id,
            "club_id": "club",
            "league_name": "Open",
            "player_id": 1,
            "role": "captain",
            "status": "active",
        },
        {
            "id": "member-2",
            "team_id": team_id,
            "club_id": "club",
            "league_name": "Open",
            "player_id": 2,
            "role": "primary",
            "status": "active",
        },
    ]

    create_admin_team_league_team(
        supabase,
        club_id="club",
        league_name="Open",
        team_name="Forming Aces",
        captain_player_id=1,
        captain_contact_email="captain@example.com",
        initial_primary_player_id=2,
        initial_primary_contact_email="primary@example.com",
        expected_roster_version=7,
        idempotency_key="create:forming:team",
        confirmation_text=CREATE_TEAM_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="admin",
    )

    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == "team_league_create_team_v1"
    assert params["p_expected_roster_version"] == 7
    assert params["p_idempotency_key"] == "create:forming:team"


def test_roster_retry_forwards_stale_version_to_database_receipt(monkeypatch) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={"max_alternates": 1, "roster_version": 9},
        players=[
            {"id": 1, "club_id": "club", "name": "Captain", "gender": "Man"},
            {"id": 2, "club_id": "club", "name": "Primary", "gender": "Woman"},
            {"id": 3, "club_id": "club", "name": "Alternate", "gender": "Woman"},
        ],
    )
    supabase.tables["team_league_teams"] = [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "club_id": "club",
            "league_name": "Open",
            "team_name": "Aces",
            "status": "confirmed",
            "captain_player_id": 1,
            "partner_player_id": 2,
        }
    ]
    supabase.tables["team_league_team_members"] = [
        {
            "id": "member-1",
            "team_id": "00000000-0000-0000-0000-000000000001",
            "club_id": "club",
            "league_name": "Open",
            "player_id": 1,
            "role": "captain",
            "status": "active",
        },
        {
            "id": "member-2",
            "team_id": "00000000-0000-0000-0000-000000000001",
            "club_id": "club",
            "league_name": "Open",
            "player_id": 2,
            "role": "primary",
            "status": "active",
        },
    ]

    admin_team_league_roster_action(
        supabase,
        club_id="club",
        league_name="Open",
        action="add_member",
        team_id="00000000-0000-0000-0000-000000000001",
        player_id=3,
        member_role="alternate",
        member_status="active",
        expected_roster_version=2,
        idempotency_key="roster:receipt:retry",
        confirmation_text="UPDATE TEAM ROSTER",
        actor_email="admin@example.com",
        actor_role="admin",
    )

    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == "team_league_apply_roster_action_v1"
    assert params["p_expected_roster_version"] == 2

    with pytest.raises(ValueError, match="captain role"):
        admin_team_league_roster_action(
            supabase,
            club_id="club",
            league_name="Open",
            action="add_member",
            team_id="00000000-0000-0000-0000-000000000001",
            player_id=1,
            member_role="alternate",
            member_status="active",
            expected_roster_version=9,
            idempotency_key="roster:captain:demote",
            confirmation_text="UPDATE TEAM ROSTER",
            actor_email="admin@example.com",
            actor_role="admin",
        )
    assert len(supabase.rpc_calls) == 1


@pytest.mark.parametrize(
    "category,genders,message",
    [
        ("mixed", ("Men", "Male"), "one man and one woman"),
        ("womens", ("Women", "Prefer not to say"), "cannot verify"),
    ],
)
def test_public_registration_rejects_ineligible_or_unknown_gender_before_write(
    monkeypatch,
    category: str,
    genders: tuple[str, str],
    message: str,
) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_public_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={"team_category": category},
        players=[
            {"id": 1, "club_id": "club", "gender": genders[0]},
            {"id": 2, "club_id": "club", "gender": genders[1]},
        ],
    )

    with pytest.raises(ValueError, match=message):
        register_public_team_league(
            supabase,
            club_id="club",
            league_name="Open",
            signup_type="team",
            player_id=1,
            partner_player_id=2,
            contact_email="alex@example.com",
            partner_email="partner@example.com",
            team_name="The Pair",
            idempotency_key="register:category:test",
            confirmation_text=TEAM_SIGNUP_CONFIRMATION,
            public_base_url="https://example.test/clubs/club",
            club_name="Test Club",
        )

    assert supabase.rpc_calls == []


@pytest.mark.parametrize(
    "category,genders",
    [
        ("mens", ("Men", "M")),
        ("womens", ("Women", "F")),
        ("mixed", ("Female", "male")),
    ],
)
def test_gender_aliases_are_normalized_for_fixed_pair_categories(
    category: str,
    genders: tuple[str, str],
) -> None:
    _enforce_team_category(
        category,
        [{"gender": genders[0]}, {"gender": genders[1]}],
    )


@pytest.mark.parametrize(
    "genders,error",
    [
        (("Woman", "MAN"), None),
        (("Men", "Male"), "one man and one woman"),
        (("Women", "Unknown"), "cannot verify"),
    ],
)
def test_admin_waitlist_pairing_enforces_category_before_write(
    monkeypatch,
    genders: tuple[str, str],
    error: str | None,
) -> None:
    monkeypatch.setattr(
        "jupr_app.services.team_league_service._assert_admin_write_enabled",
        lambda: None,
    )
    supabase = _CompositionSupabase(
        settings={"team_category": "mixed"},
        players=[
            {"id": 1, "club_id": "club", "gender": genders[0]},
            {"id": 2, "club_id": "club", "gender": genders[1]},
        ],
        waitlist=[
            {
                "id": "wait-1",
                "club_id": "club",
                "league_name": "Open",
                "player_id": 1,
                "status": "waiting",
            },
            {
                "id": "wait-2",
                "club_id": "club",
                "league_name": "Open",
                "player_id": 2,
                "status": "waiting",
            },
        ],
    )

    kwargs = dict(
        supabase=supabase,
        club_id="club",
        league_name="Open",
        action="pair",
        waitlist_ids=["wait-1", "wait-2"],
        team_name="The Pair",
        idempotency_key="waitlist:category:test",
        confirmation_text=PAIR_WAITLIST_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="admin",
        source="composition_test",
    )
    if error:
        with pytest.raises(ValueError, match=error):
            admin_team_league_waitlist_action(**kwargs)
        assert supabase.rpc_calls == []
    else:
        result = admin_team_league_waitlist_action(**kwargs)
        assert result["committed"] is True
        assert len(supabase.rpc_calls) == 1
