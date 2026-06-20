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


@pytest.mark.parametrize("skill_label", ["Open", "Beginner", "3.25"])
def test_open_or_non_controlled_skill_label_remains_eligible(skill_label):
    eligible, message = _is_eligible(
        event=_event(skill_label),
        selection=_selection("HAS_PARTNER"),
        player=_player(5.5),
        partner=_player(5.8),
    )

    assert eligible is True
    assert message is None
