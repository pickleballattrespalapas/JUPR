from __future__ import annotations

import pytest

from jupr_app.domain.tournament_registration_compiler import validate_selection_against_skill


def _event(skill_label: str, *, doubles: bool = True) -> dict[str, object]:
    return {
        "skill_label": skill_label,
        "event_type": "DOUBLES" if doubles else "SINGLES",
        "partner_required": doubles,
    }


def _player(rating: float | None) -> dict[str, object]:
    return {"doubles_skill": rating, "singles_skill": rating}


def _selection(partner_mode: str = "NONE") -> dict[str, object]:
    return {"partner_mode": partner_mode}


def _is_eligible(*, event, selection, player, partner=None, allow_missing_partner_for_preview=False):
    eligible, message = validate_selection_against_skill(
        event=event,
        selection=selection,
        player=player,
        partner=partner,
        allow_missing_partner_for_preview=allow_missing_partner_for_preview,
    )
    return eligible, message


@pytest.mark.parametrize("allow_missing_partner_for_preview", [True, False])
def test_doubles_player_can_play_up_with_needs_partner_in_preview_and_submit(allow_missing_partner_for_preview):
    eligible, message = _is_eligible(
        event=_event("3.5"),
        selection=_selection("NEEDS_PARTNER"),
        player=_player(3.47),
        partner=None,
        allow_missing_partner_for_preview=allow_missing_partner_for_preview,
    )

    assert eligible is True
    assert message is None


def test_singles_player_can_play_up_to_higher_skill_label():
    eligible, message = _is_eligible(
        event=_event("3.5", doubles=False),
        selection=_selection(),
        player=_player(3.47),
    )

    assert eligible is True
    assert message is None


def test_doubles_player_at_next_ceiling_cannot_register_down():
    eligible, message = _is_eligible(
        event=_event("3.0"),
        selection=_selection("NEEDS_PARTNER"),
        player=_player(3.50),
    )

    assert eligible is False
    assert "above the 3 division cap" in str(message)
    assert "register for a 3.5 or higher division" in str(message)


def test_doubles_known_partner_at_next_ceiling_cannot_register_down():
    eligible, message = _is_eligible(
        event=_event("3.0"),
        selection=_selection("HAS_PARTNER"),
        player=_player(3.2),
        partner=_player(3.50),
    )

    assert eligible is False
    assert "above the 3 division cap" in str(message)
    assert "register for a 3.5 or higher division" in str(message)


def test_doubles_missing_partner_skill_does_not_block_registration():
    eligible, message = _is_eligible(
        event=_event("3.0"),
        selection=_selection("HAS_PARTNER"),
        player=_player(3.2),
        partner={"doubles_skill": None, "singles_skill": None},
    )

    assert eligible is True
    assert message is None


def test_doubles_missing_player_and_partner_ratings_are_eligible():
    eligible, message = _is_eligible(
        event=_event("3.0"),
        selection=_selection("NEEDS_PARTNER"),
        player=_player(None),
        partner=None,
    )

    assert eligible is True
    assert message is None


@pytest.mark.parametrize("skill_label", ["Open", "Beginner"])
def test_open_or_non_controlled_skill_label_remains_eligible(skill_label):
    eligible, message = _is_eligible(
        event=_event(skill_label),
        selection=_selection("HAS_PARTNER"),
        player=_player(5.5),
        partner=_player(5.8),
    )

    assert eligible is True
    assert message is None


def test_custom_numeric_skill_label_is_directional_ceiling_band():
    eligible, message = _is_eligible(
        event=_event("3.25"),
        selection=_selection("HAS_PARTNER"),
        player=_player(3.6),
        partner=_player(3.7),
    )
    assert eligible is True
    assert message is None

    eligible, message = _is_eligible(
        event=_event("3.25"),
        selection=_selection("HAS_PARTNER"),
        player=_player(3.6),
        partner=_player(3.75),
    )
    assert eligible is False
    assert "above the 3.25 division cap" in str(message)


def test_lower_rated_player_can_play_far_up() -> None:
    eligible, message = _is_eligible(
        event=_event("4.5", doubles=False),
        selection=_selection(),
        player=_player(2.5),
    )

    assert eligible is True
    assert message is None


def test_trailing_plus_is_upward_open_and_does_not_create_a_minimum() -> None:
    for partner_rating in (3.0, 3.9, 4.0, 5.5):
        eligible, message = _is_eligible(
            event=_event("3.5+"),
            selection=_selection("HAS_PARTNER"),
            player=_player(2.9),
            partner=_player(partner_rating),
        )
        assert eligible is True
        assert message is None


def test_minimum_skill_mode_is_upward_open_even_without_plus_label() -> None:
    event = {**_event("3.5"), "skill_mode": "minimum"}
    eligible, message = _is_eligible(
        event=event,
        selection=_selection("HAS_PARTNER"),
        player=_player(2.9),
        partner=_player(5.5),
    )
    assert eligible is True
    assert message is None


def test_combined_rating_cap_is_strict_and_uses_both_partners() -> None:
    event = {
        **_event("Open"),
        "eligibility_mode": "COMBINED_RATING_CAP",
        "combined_rating_cap": 8.0,
    }
    eligible, message = _is_eligible(
        event=event,
        selection=_selection("HAS_PARTNER"),
        player=_player(3.9),
        partner=_player(4.0),
    )
    assert eligible is True
    assert message is None

    eligible, message = _is_eligible(
        event=event,
        selection=_selection("HAS_PARTNER"),
        player=_player(4.0),
        partner=_player(4.0),
    )
    assert eligible is False
    assert "not strictly below" in str(message)
