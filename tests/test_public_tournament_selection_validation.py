from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.public_tournament_registration_service import (
    build_tournament_registration_player_profile,
    validate_and_clean_tournament_selection,
)


def _event() -> dict[str, object]:
    return {
        "id": "event-1",
        "registration_day_id": "canonical-day",
        "division_name": "Open Doubles",
        "event_type": "DOUBLES",
        "gender_restriction": "ANY",
        "skill_label": "Open",
        "partner_required": False,
        "partner_board_enabled": True,
    }


def _stale_partner_selection(*, partner_mode: str) -> dict[str, object]:
    return {
        "event_option_id": "event-1",
        "registration_day_id": "stale-day",
        "partner_mode": partner_mode,
        "partner_name": "Old Partner",
        "partner_email": "old@example.com",
        "partner_phone": "555-0100",
        "partner_dupr_id": "old-dupr",
        "partner_skill": 4.25,
        "partner_age": 44,
        "partner_gender": "Women",
        "partner_note": "Still useful for a partner-board request.",
        "show_on_partner_board": True,
    }


@pytest.mark.parametrize(
    ("partner_mode", "expected_board_visibility"),
    [("NONE", False), ("NEEDS_PARTNER", True)],
)
def test_non_confirmed_partner_modes_clear_stale_partner_identity_rating_and_age(
    partner_mode: str,
    expected_board_visibility: bool,
) -> None:
    cleaned = validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=_event(),
        raw_selection=_stale_partner_selection(partner_mode=partner_mode),
        player_profile={
            "email": "player@example.com",
            "player_id": None,
            "doubles_skill": 3.5,
            "singles_skill": 3.5,
            "gender": "Men",
            "age": 40,
        },
        settings={"partner_board_enabled": True},
    )

    assert cleaned["registration_day_id"] == "canonical-day"
    assert cleaned["partner_name"] == ""
    assert cleaned["partner_email"] == ""
    assert cleaned["partner_phone"] == ""
    assert cleaned["partner_dupr_id"] == ""
    assert cleaned["partner_skill"] is None
    assert cleaned["partner_age"] is None
    assert cleaned["partner_gender"] == ""
    assert cleaned["show_on_partner_board"] is expected_board_visibility


class _PlayerQuery:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.filters: list[tuple[str, object]] = []

    def select(self, *_args: object, **_kwargs: object) -> _PlayerQuery:
        return self

    def eq(self, key: str, value: object) -> _PlayerQuery:
        self.filters.append((key, value))
        return self

    def limit(self, _value: int) -> _PlayerQuery:
        return self

    def execute(self) -> SimpleNamespace:
        rows = [
            row
            for row in self.rows
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        return SimpleNamespace(data=rows)


class _PlayerSupabase:
    def __init__(self, players: list[dict[str, object]]) -> None:
        self.players = players

    def table(self, name: str) -> _PlayerQuery:
        assert name == "players"
        return _PlayerQuery(self.players)


def test_existing_registration_profile_preserves_self_rating_when_official_singles_is_missing() -> None:
    profile = build_tournament_registration_player_profile(
        _PlayerSupabase(
            [
                {
                    "id": "player-1",
                    "club_id": "club-1",
                    "rating": 1600,
                    "gender": "Women",
                    "age": 42,
                    "active": True,
                    "inactive_at": None,
                }
            ]
        ),
        club_id="club-1",
        registration={
            "player_id": "player-1",
            "email": "PLAYER@EXAMPLE.COM",
            "doubles_skill": 2.5,
            "singles_skill": 2.5,
            "gender": "Men",
            "age": 30,
        },
    )

    assert profile == {
        "email": "player@example.com",
        "player_id": "player-1",
        "doubles_skill": 4.0,
        "singles_skill": 2.5,
        "gender": "Women",
        "age": 42,
    }


def test_existing_registration_profile_prefers_official_singles_rating() -> None:
    profile = build_tournament_registration_player_profile(
        _PlayerSupabase(
            [
                {
                    "id": "player-1",
                    "club_id": "club-1",
                    "rating": 1600,
                    "singles_rating": 1400,
                    "singles_matches_played": 1,
                    "gender": "Women",
                    "age": 42,
                    "active": True,
                    "inactive_at": None,
                }
            ]
        ),
        club_id="club-1",
        registration={
            "player_id": "player-1",
            "email": "PLAYER@EXAMPLE.COM",
            "doubles_skill": 2.5,
            "singles_skill": 2.5,
            "gender": "Men",
            "age": 30,
        },
    )

    assert profile["doubles_skill"] == 4.0
    assert profile["singles_skill"] == 3.5


def _age_event(
    *,
    mode: str,
    label: str,
    rules: dict[str, object],
    event_type: str = "SINGLES",
) -> dict[str, object]:
    return {
        "id": "age-event",
        "registration_day_id": "canonical-day",
        "division_name": label,
        "event_type": event_type,
        "gender_restriction": "ANY",
        "skill_label": "Open",
        "partner_required": event_type != "SINGLES",
        "partner_board_enabled": True,
        "age_mode": mode,
        "age_label": label,
        "age_rules": rules,
    }


def _profile(*, age: int, rating: float = 3.5) -> dict[str, object]:
    return {
        "email": "player@example.com",
        "player_id": None,
        "doubles_skill": rating,
        "singles_skill": rating,
        "gender": "Women",
        "age": age,
    }


def test_public_fixed_age_group_allows_older_players_to_play_down() -> None:
    event = _age_event(
        mode="FIXED_AGE_BRACKET",
        label="50–64",
        rules={
            "mode": "FIXED_AGE_BRACKET",
            "label": "50–64",
            "min_age": 50,
            "max_age": 64,
        },
    )

    cleaned = validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=event,
        raw_selection={"event_option_id": "age-event", "partner_mode": "NONE"},
        player_profile=_profile(age=67),
        settings={"partner_board_enabled": True},
    )

    assert cleaned["event_option_id"] == "age-event"


def test_public_fixed_age_group_blocks_younger_player_from_playing_up() -> None:
    event = _age_event(
        mode="FIXED_AGE_BRACKET",
        label="50+",
        rules={
            "mode": "FIXED_AGE_BRACKET",
            "label": "50+",
            "min_age": 50,
        },
    )

    with pytest.raises(ValueError, match="does not meet minimum age 50"):
        validate_and_clean_tournament_selection(
            object(),
            club_id="club-1",
            tournament_id="tournament-1",
            event=event,
            raw_selection={"event_option_id": "age-event", "partner_mode": "NONE"},
            player_profile=_profile(age=45),
            settings={"partner_board_enabled": True},
        )


def test_public_under_50_group_is_open_to_older_players() -> None:
    event = _age_event(
        mode="FIXED_AGE_BRACKET",
        label="Under 50",
        rules={
            "mode": "FIXED_AGE_BRACKET",
            "label": "Under 50",
            "min_age": None,
            "max_age": 49,
        },
    )

    cleaned = validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=event,
        raw_selection={"event_option_id": "age-event", "partner_mode": "NONE"},
        player_profile=_profile(age=72),
        settings={"partner_board_enabled": True},
    )

    assert cleaned["event_option_id"] == "age-event"


def test_public_exhaustive_age_groups_do_not_block_needs_partner_registration() -> None:
    event = _age_event(
        mode="AUTO_AGE_SPLIT",
        label="Age groups",
        event_type="DOUBLES",
        rules={
            "mode": "AUTO_AGE_SPLIT",
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "min_teams_per_age_group": 1,
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
    )

    cleaned = validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=event,
        raw_selection={
            "event_option_id": "age-event",
            "partner_mode": "NEEDS_PARTNER",
            "show_on_partner_board": True,
        },
        player_profile=_profile(age=67),
        settings={"partner_board_enabled": True},
    )

    assert cleaned["partner_mode"] == "NEEDS_PARTNER"
    assert cleaned["partner_age"] is None


def test_public_older_only_group_blocks_known_younger_player_even_when_partner_is_missing() -> None:
    event = _age_event(
        mode="FIXED_AGE_BRACKET",
        label="50+",
        event_type="DOUBLES",
        rules={
            "mode": "FIXED_AGE_BRACKET",
            "label": "50+",
            "min_age": 50,
            "team_age_rule": "YOUNGER",
        },
    )

    with pytest.raises(ValueError, match="older partner cannot make this team eligible"):
        validate_and_clean_tournament_selection(
            object(),
            club_id="club-1",
            tournament_id="tournament-1",
            event=event,
            raw_selection={
                "event_option_id": "age-event",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
            },
            player_profile=_profile(age=45),
            settings={"partner_board_enabled": True},
        )
