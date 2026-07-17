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
    assert "partner_gender" not in cleaned
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


def test_existing_registration_profile_prefers_canonical_linked_player_values() -> None:
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
        "singles_skill": 4.0,
        "gender": "Women",
        "age": 42,
    }
