from copy import deepcopy

from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event, generator_event_standings
from jupr_app.domain.hub_standings import hub_standings_view


def fixture():
    event = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="Evening", participant_names=[f"Player {i}" for i in range(8)], court_count=2, total_rounds=2))
    first, second = event["rounds"]
    first["status"] = "saved"
    for match in first["matches"]:
        match.update(scoreA=11, scoreB=7, status="scored")
    second["matches"][0].update(scoreA=11, scoreB=4, status="scored")
    event["currentRoundNumber"] = 2
    return event


def test_partial_results_have_an_explicit_view_without_changing_completed_round_standings():
    event = fixture()
    before = deepcopy(event)
    view = hub_standings_view(event)
    assert event == before
    assert sum(row["matches"] for row in generator_event_standings(event)) == 8
    assert sum(row["matches"] for row in view["provisional_standings"]) == 12
    assert view["standings_view"]["completed_through_round"] == 1
    assert view["standings_view"]["scored_matches_in_partial_rounds"] == 1
    assert len(view["standings_view"]["pending_matches"]) == 1
    assert view["standings_view"]["pending_matches"][0]["round_number"] == 2
    assert view["standings_view"]["final"] is False


def test_departed_players_keep_results_and_unplayed_games_do_not_create_results():
    event = fixture()
    person = event["participants"][0]
    person["inactive_from_round"] = 2
    event["rounds"][1]["matches"][1].update(status="not_played", scoreA=None, scoreB=None)
    view = hub_standings_view(event)
    assert next(row for row in view["provisional_standings"] if row["participantId"] == person["id"])["matches"] >= 1
    assert sum(row["matches"] for row in view["provisional_standings"]) == 12
    assert view["standings_view"]["pending_matches"] == []
    assert view["standings_view"]["final"] is False


def test_final_boundary_excludes_unused_previews_and_requires_all_scores():
    event = fixture()
    event.update(hubFinishAfterRound=1, status="completed")
    event["rounds"][1]["status"] = "skipped"
    view = hub_standings_view(event)
    assert sum(row["matches"] for row in view["provisional_standings"]) == 8
    assert view["standings_view"]["final"] is True
    event["rounds"][0]["matches"][0].update(scoreA=None, scoreB=None, status="scheduled")
    assert hub_standings_view(event)["standings_view"]["final"] is False


def test_fixed_formats_do_not_offer_a_different_settlement_view():
    event = fixture()
    for update in ({"eventFormat": "moneyball_expectations"}, {"scoringMode": "unscored"}, {"playoff": {"stage": "semifinal"}}):
        assert hub_standings_view({**event, **update}) == {}
