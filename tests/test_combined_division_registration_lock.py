from copy import deepcopy

import pytest

from jupr_app.domain.tournament_registration_repo import registration_is_imported_to_draw
from tests.test_public_tournament_registration_service import FakeSupabase


def _storage():
    return {
        "tournament_registration_selections": [
            {"id": "selection-a", "tournament_id": "t1", "registration_id": "registration-a", "registration_day_id": "day1", "event_option_id": "division-45"},
            {"id": "selection-b", "tournament_id": "t1", "registration_id": "registration-b", "registration_day_id": "day1", "event_option_id": "division-45"},
        ],
        "tournament_registrations": [
            {"id": "registration-a", "tournament_id": "t1", "player_id": 101},
            {"id": "registration-b", "tournament_id": "t1", "player_id": 102},
        ],
        "tournament_event_draws": [{"id": "combined", "tournament_id": "t1", "registration_day_id": "day1", "event_option_id": "division-40", "source_event_option_ids": ["division-40", "division-45"]}],
        "tournament_teams": [{"id": "team", "tournament_id": "t1", "draw_id": "combined", "registration_day_id": "day1", "event_option_id": "division-40", "source_event_option_id": "division-45", "source_selection_id": "selection-a", "source": "REGISTRATION", "player1_id": 101, "player2_id": 102}],
    }


@pytest.mark.parametrize("selection_id", ["selection-a", "selection-b"])
@pytest.mark.parametrize("with_provenance", [True, False])
def test_combined_draw_locks_both_original_division_registrations(selection_id, with_provenance):
    storage = _storage()
    if not with_provenance:
        storage["tournament_teams"][0].pop("source_event_option_id")
    before = deepcopy(storage)
    assert registration_is_imported_to_draw(FakeSupabase(storage), tournament_id="t1", selection_id=selection_id)
    assert storage == before


@pytest.mark.parametrize("difference", ["player", "day", "event", "tournament"])
def test_combined_draw_does_not_lock_unrelated_selection(difference):
    storage = _storage()
    selection = storage["tournament_registration_selections"][1]
    if difference == "player":
        storage["tournament_registrations"][1]["player_id"] = 999
    elif difference == "day":
        selection["registration_day_id"] = "day2"
    elif difference == "event":
        selection["event_option_id"] = "division-50"
    else:
        storage["tournament_teams"][0]["tournament_id"] = "t2"
    assert not registration_is_imported_to_draw(FakeSupabase(storage), tournament_id="t1", selection_id="selection-b")


def test_explicit_team_source_does_not_lock_same_player_in_other_combined_division():
    storage = _storage()
    storage["tournament_registration_selections"][1]["event_option_id"] = "division-40"
    assert not registration_is_imported_to_draw(FakeSupabase(storage), tournament_id="t1", selection_id="selection-b")


def test_registration_lock_refuses_unavailable_team_evidence():
    class UnavailableTeams(FakeSupabase):
        def table(self, name):
            if name == "tournament_teams":
                raise RuntimeError("team evidence unavailable")
            return super().table(name)

    with pytest.raises(RuntimeError, match="team evidence unavailable"):
        registration_is_imported_to_draw(UnavailableTeams(_storage()), tournament_id="t1", selection_id="selection-b")
