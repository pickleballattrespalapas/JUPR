from __future__ import annotations

import math
from typing import Any

import pytest

from jupr_app.services import admin_tournament_setup_service as setup_service


class FakeSupabase:
    def __init__(self) -> None:
        self.table_calls: list[str] = []

    def table(self, name: str) -> Any:
        self.table_calls.append(str(name))
        raise AssertionError(f"Unexpected database access: {name}")


@pytest.fixture(autouse=True)
def _enable_tournament_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")


def _event(mode: str, **overrides: object) -> dict[str, object]:
    skill_label = {
        "MINIMUM": "3.5+",
        "OPEN": "Open",
        "COMBINED_RATING_CAP": "Combined < 8",
        "CUSTOM": "Custom",
    }.get(mode, "3.5")
    event: dict[str, object] = {
        "id": f"event-{mode.lower()}",
        "registration_day_id": "day-1",
        "division_name": f"{mode} division",
        "event_type": "SINGLES",
        "competition_format": "STANDARD",
        "eligibility_mode": mode,
        "skill_label": skill_label,
        "skill_min_rating": None,
        "skill_max_rating": None,
        "combined_rating_cap": None,
    }
    if mode == "MINIMUM":
        event["skill_min_rating"] = 3.5
    elif mode == "COMBINED_RATING_CAP":
        event["event_type"] = "GENDER_DOUBLES"
        event["combined_rating_cap"] = 8.0
    elif mode == "CUSTOM":
        event["skill_min_rating"] = 3.0
        event["skill_max_rating"] = 4.0
    event.update(overrides)
    return event


VALID_ELIGIBILITY_CASES = [
    pytest.param(_event("STANDARD"), "STANDARD", id="standard"),
    pytest.param(_event("MINIMUM"), "MINIMUM", id="minimum"),
    pytest.param(_event("OPEN"), "OPEN", id="open"),
    pytest.param(
        _event("COMBINED_RATING_CAP"),
        "COMBINED_RATING_CAP",
        id="combined-standard-team",
    ),
    pytest.param(
        _event("CUSTOM", skill_min_rating=3.0, skill_max_rating=None),
        "CUSTOM",
        id="custom-lower-only",
    ),
    pytest.param(
        _event("CUSTOM", skill_min_rating=None, skill_max_rating=4.0),
        "CUSTOM",
        id="custom-upper-only",
    ),
    pytest.param(
        _event("CUSTOM", skill_min_rating=3.0, skill_max_rating=4.0),
        "CUSTOM",
        id="custom-both",
    ),
    pytest.param(
        _event("STANDARD", competition_format="FOUR_PLAYER_TEAM"),
        "STANDARD",
        id="four-player-standard",
    ),
]


INVALID_ELIGIBILITY_CASES = [
    pytest.param(_event("LEGACY"), "eligibility_mode", id="unknown-mode"),
    pytest.param(_event("STANDARD", eligibility_mode=True), "eligibility_mode", id="boolean-mode"),
    pytest.param(_event("STANDARD", eligibility_mode=0), "eligibility_mode", id="numeric-zero-mode"),
    pytest.param(_event("STANDARD", skill_min_rating=3.0), "STANDARD", id="standard-minimum"),
    pytest.param(_event("STANDARD", skill_max_rating=math.nan), "finite", id="standard-nan"),
    pytest.param(_event("MINIMUM", skill_min_rating=None), "requires skill_min_rating", id="minimum-missing"),
    pytest.param(_event("MINIMUM", skill_min_rating=True), "not a boolean", id="minimum-boolean"),
    pytest.param(_event("MINIMUM", skill_min_rating=0.5), "between 1 and 7", id="minimum-low"),
    pytest.param(_event("MINIMUM", skill_max_rating=4.0), "cannot include a maximum", id="minimum-maximum"),
    pytest.param(_event("OPEN", combined_rating_cap=8.0), "OPEN", id="open-cap"),
    pytest.param(_event("COMBINED_RATING_CAP", combined_rating_cap=None), "requires combined_rating_cap", id="combined-missing"),
    pytest.param(_event("COMBINED_RATING_CAP", combined_rating_cap=math.inf), "finite", id="combined-infinite"),
    pytest.param(_event("COMBINED_RATING_CAP", combined_rating_cap=14.01), "no more than 14", id="combined-high"),
    pytest.param(_event("COMBINED_RATING_CAP", event_type="SINGLES"), "standard doubles/team", id="combined-singles"),
    pytest.param(
        _event("COMBINED_RATING_CAP", competition_format="FOUR_PLAYER_TEAM"),
        "four-player team divisions must keep STANDARD",
        id="combined-four-player",
    ),
    pytest.param(_event("CUSTOM", skill_min_rating=None, skill_max_rating=None), "requires a minimum", id="custom-empty"),
    pytest.param(_event("CUSTOM", skill_min_rating=0.5, skill_max_rating=None), "minimum rating", id="custom-low-minimum"),
    pytest.param(_event("CUSTOM", skill_min_rating=None, skill_max_rating=1.0), "maximum rating", id="custom-low-maximum"),
    pytest.param(_event("CUSTOM", skill_min_rating=4.0, skill_max_rating=4.0), "greater than the minimum", id="custom-equal"),
    pytest.param(_event("CUSTOM", combined_rating_cap=8.0), "cannot include a combined cap", id="custom-cap"),
    pytest.param(
        _event("OPEN", competition_format="FOUR_PLAYER_TEAM"),
        "four-player team divisions must keep STANDARD",
        id="four-player-open",
    ),
]


def _install_valid_read_stubs(
    monkeypatch: pytest.MonkeyPatch,
    analyze_calls: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(
        setup_service,
        "get_admin_tournament_setup_detail",
        lambda *_args, **_kwargs: {"state_fingerprint": "state-1"},
    )
    monkeypatch.setattr(
        setup_service,
        "_get_tournament_for_club",
        lambda *_args, **_kwargs: {"id": "tournament-1", "club_id": "club-1"},
    )

    def analyze(_supabase: Any, **kwargs: Any) -> dict[str, Any]:
        analyze_calls.append(dict(kwargs))
        return {}

    monkeypatch.setattr(setup_service, "analyze_registration_publish_impact", analyze)


def test_standard_event_cannot_enable_check_in_substitutes_before_database_access() -> None:
    supabase = FakeSupabase()

    with pytest.raises(ValueError, match="standard tournament events cannot enable substitutes"):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[
                _event("STANDARD", team_allow_substitutes=True)
            ],
            expected_state_fingerprint="state-1",
        )

    assert supabase.table_calls == []


def test_four_player_between_match_roster_replacement_remains_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)
    event = _event(
        "STANDARD",
        competition_format="FOUR_PLAYER_TEAM",
        team_allow_substitutes=True,
    )

    setup_service.review_admin_tournament_setup_impact(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        days=[],
        event_options=[event],
        expected_state_fingerprint="state-1",
    )

    assert analyze_calls[0]["event_options"][0]["team_allow_substitutes"] is True


@pytest.mark.parametrize("capacity", [4, 9, 16])
def test_setup_accepts_every_executable_capacity_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capacity: int,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)
    event = _event("OPEN", capacity_teams=capacity)

    setup_service.review_admin_tournament_setup_impact(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        days=[],
        event_options=[event],
        expected_state_fingerprint="state-1",
    )

    assert analyze_calls[0]["event_options"][0]["capacity_teams"] == capacity


@pytest.mark.parametrize("capacity", [3, 17, 4.5, True])
def test_setup_rejects_capacity_outside_executable_round_robin_contract(
    monkeypatch: pytest.MonkeyPatch,
    capacity: object,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)

    with pytest.raises(ValueError, match="capacity_teams"):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[_event("OPEN", capacity_teams=capacity)],
            expected_state_fingerprint="state-1",
        )
    assert analyze_calls == []


@pytest.mark.parametrize(
    ("division_name", "gender_restriction", "expected"),
    [
        ("Women's 3.0", "MEN", "WOMEN"),
        ("Women’s 3.5", "OPEN", "WOMEN"),
        ("Men's 4.0", "WOMEN", "MEN"),
    ],
)
def test_setup_rejects_gender_restriction_that_contradicts_division_label(
    monkeypatch: pytest.MonkeyPatch,
    division_name: str,
    gender_restriction: str,
    expected: str,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)

    with pytest.raises(ValueError, match=expected):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[
                _event(
                    "OPEN",
                    division_name=division_name,
                    gender_restriction=gender_restriction,
                )
            ],
            expected_state_fingerprint="state-1",
        )
    assert analyze_calls == []


@pytest.mark.parametrize("legacy_mode", [None, "STANDARD"], ids=["missing-mode", "standard-mode"])
@pytest.mark.parametrize(
    ("skill_label", "expected_minimum"),
    [("3.5+", 3.5), ("4.0+", 4.0)],
)
@pytest.mark.parametrize("boundary", ["review", "publish_dry_run"])
def test_legacy_minimum_rows_are_canonicalized_in_place_before_analysis(
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str | None,
    skill_label: str,
    expected_minimum: float,
    boundary: str,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)
    event = _event(
        "STANDARD",
        eligibility_mode=legacy_mode,
        skill_mode="minimum",
        skill_label=skill_label,
        skill_min_rating=None,
        skill_max_rating=6.0,
        combined_rating_cap=9.0,
    )

    if boundary == "review":
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            expected_state_fingerprint="state-1",
        )
    else:
        setup_service.publish_admin_tournament_setup(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH SETUP",
            dry_run=True,
        )

    assert event["eligibility_mode"] == "MINIMUM"
    assert event["skill_mode"] == "MINIMUM"
    assert event["skill_min_rating"] == expected_minimum
    assert event["skill_max_rating"] is None
    assert event["combined_rating_cap"] is None
    assert analyze_calls[0]["event_options"][0] is event


@pytest.mark.parametrize("legacy_mode", [None, "STANDARD"], ids=["missing-mode", "standard-mode"])
def test_legacy_open_labels_are_canonicalized_and_clear_stale_bounds(
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str | None,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)
    event = _event(
        "STANDARD",
        eligibility_mode=legacy_mode,
        skill_mode="OPEN",
        skill_label="Open",
        skill_min_rating=True,
        skill_max_rating=math.inf,
        combined_rating_cap=8.0,
    )

    setup_service.review_admin_tournament_setup_impact(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        days=[],
        event_options=[event],
        expected_state_fingerprint="state-1",
    )

    assert event["eligibility_mode"] == "OPEN"
    assert event["skill_mode"] == "OPEN"
    assert event["skill_label"] == "Open"
    assert event["skill_min_rating"] is None
    assert event["skill_max_rating"] is None
    assert event["combined_rating_cap"] is None
    assert analyze_calls[0]["event_options"][0] is event


@pytest.mark.parametrize(
    "skill_label",
    [None, "", "Open", "3.5+", "0.5", "7.5", "Beginner"],
)
def test_explicit_standard_skill_mode_requires_numeric_non_plus_label_before_downstream(
    monkeypatch: pytest.MonkeyPatch,
    skill_label: object,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)
    event = _event(
        "STANDARD",
        skill_mode="STANDARD",
        skill_label=skill_label,
    )

    with pytest.raises(ValueError, match="numeric skill_label between 1 and 7"):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            expected_state_fingerprint="state-1",
        )

    assert event["eligibility_mode"] == "STANDARD"
    assert analyze_calls == []
    assert supabase.table_calls == []


@pytest.mark.parametrize(
    "event",
    [
        _event("MINIMUM", skill_min_rating=None, skill_label="3.5+"),
        _event("OPEN", skill_min_rating=3.5),
        _event(
            "CUSTOM",
            skill_min_rating=None,
            skill_max_rating=None,
            skill_label="3.0–4.0",
        ),
    ],
    ids=["explicit-minimum", "explicit-open", "custom-does-not-infer-bounds"],
)
def test_explicit_new_modes_remain_strict_without_legacy_repair(
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, object],
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)

    with pytest.raises(ValueError):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            expected_state_fingerprint="state-1",
        )

    assert analyze_calls == []
    assert supabase.table_calls == []


@pytest.mark.parametrize(("event", "expected_mode"), VALID_ELIGIBILITY_CASES)
@pytest.mark.parametrize("boundary", ["review", "publish_dry_run"])
def test_valid_modes_and_custom_shapes_reach_impact_analysis(
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, object],
    expected_mode: str,
    boundary: str,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)

    if boundary == "review":
        result = setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            expected_state_fingerprint="state-1",
        )
    else:
        result = setup_service.publish_admin_tournament_setup(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[event],
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH SETUP",
            dry_run=True,
        )

    assert result["dry_run"] is True
    assert len(analyze_calls) == 1
    assert analyze_calls[0]["event_options"][0]["eligibility_mode"] == expected_mode
    assert supabase.table_calls == []


@pytest.mark.parametrize(("event", "message"), INVALID_ELIGIBILITY_CASES)
@pytest.mark.parametrize("boundary", ["review", "publish_dry_run", "publish_write"])
def test_invalid_eligibility_fails_before_analyze_or_write(
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, object],
    message: str,
    boundary: str,
) -> None:
    supabase = FakeSupabase()
    downstream_calls: list[str] = []

    def forbidden(name: str):
        def call(*_args: Any, **_kwargs: Any) -> Any:
            downstream_calls.append(name)
            raise AssertionError(f"Invalid eligibility reached {name}")

        return call

    monkeypatch.setattr(
        setup_service,
        "get_admin_tournament_setup_detail",
        forbidden("detail"),
    )
    monkeypatch.setattr(
        setup_service,
        "_get_tournament_for_club",
        forbidden("tournament_read"),
    )
    monkeypatch.setattr(
        setup_service,
        "analyze_registration_publish_impact",
        forbidden("analyze"),
    )
    monkeypatch.setattr(
        setup_service,
        "publish_registration_configuration",
        forbidden("publish_write"),
    )
    monkeypatch.setattr(
        setup_service,
        "save_builder_draft",
        forbidden("draft_write"),
    )

    with pytest.raises(ValueError, match=message):
        if boundary == "review":
            setup_service.review_admin_tournament_setup_impact(
                supabase,
                club_id="club-1",
                tournament_id="tournament-1",
                days=[],
                event_options=[event],
                expected_state_fingerprint="state-1",
            )
        else:
            setup_service.publish_admin_tournament_setup(
                supabase,
                club_id="club-1",
                tournament_id="tournament-1",
                days=[],
                event_options=[event],
                actor_email="admin@example.com",
                actor_role="club_owner",
                confirmation_text="PUBLISH SETUP",
                dry_run=boundary == "publish_dry_run",
            )

    assert downstream_calls == []
    assert supabase.table_calls == []


def test_invalid_builder_event_fails_review_before_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = FakeSupabase()
    analyze_calls: list[dict[str, Any]] = []
    _install_valid_read_stubs(monkeypatch, analyze_calls)

    with pytest.raises(ValueError, match="CUSTOM eligibility requires"):
        setup_service.review_admin_tournament_setup_impact(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            days=[],
            event_options=[_event("OPEN")],
            builder_event_options=[
                _event("CUSTOM", skill_min_rating=None, skill_max_rating=None)
            ],
            expected_state_fingerprint="state-1",
        )

    assert analyze_calls == []
    assert supabase.table_calls == []


def test_save_draft_remains_permissive_for_incomplete_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supabase = FakeSupabase()
    saved: list[dict[str, Any]] = []
    invalid_draft_event = _event(
        "CUSTOM",
        skill_min_rating=None,
        skill_max_rating=None,
        combined_rating_cap=math.nan,
    )
    monkeypatch.setattr(
        setup_service,
        "_get_tournament_for_club",
        lambda *_args, **_kwargs: {"id": "tournament-1", "club_id": "club-1"},
    )
    monkeypatch.setattr(setup_service, "get_builder_draft", lambda *_args, **_kwargs: None)

    def save_draft(_supabase: Any, **kwargs: Any) -> dict[str, Any]:
        saved.append(dict(kwargs))
        return {"event_options": kwargs["divisions"]}

    monkeypatch.setattr(setup_service, "save_builder_draft", save_draft)
    monkeypatch.setattr(setup_service, "_audit", lambda *_args, **_kwargs: [])

    result = setup_service.save_admin_tournament_setup_draft(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        days=[],
        event_families=[],
        event_options=[invalid_draft_event],
        saved_step="eligibility",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE SETUP DRAFT",
    )

    assert result["ok"] is True
    assert saved[0]["divisions"] == [invalid_draft_event]
    assert supabase.table_calls == []
