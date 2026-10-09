from copy import deepcopy

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event, create_generator_preview, generator_event_standings,
    mutate_generator_roster, reopen_generator_round, save_generator_round, skip_generator_round, start_generator_event,
)
from jupr_app.domain.generator_playoffs import generator_playoff_options, start_generator_playoff
from jupr_app.services import public_play_generator_service as public
from jupr_app.services import admin_play_generator_service as admin
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from services.api.public_play_generator_routes import install_public_play_generator_routes
from test_public_play_generator_service import FakeSupabase, matches, requester, token_secret
from test_admin_play_generator_service import FakeSupabase as AdminDatabase


def scored_event(count=8, **kwargs):
    event = start_generator_event(create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Playoffs",
        participant_names=[f"Player {i}" for i in range(1, count + 1)],
        total_rounds=5, court_count=0, **kwargs,
    ))
    for number in (1, 2):
        event = save_generator_round(event, round_number=number, scores=[
            {"match_id": match["id"], "score_a": 11, "score_b": 3 + index}
            for index, match in enumerate(matches(event["rounds"][number - 1]))
        ])
        if number == 1:
            event = advance_generator_event(event)
    return event


@pytest.mark.parametrize("count", [4, 5, 8, 12, 13, 16])
def test_groups_use_displayed_ranking_and_preserve_history_and_leftovers(count):
    event = scored_event(count)
    before = deepcopy(event)
    standings = generator_event_standings(event)
    playoff = start_generator_playoff(event, playoff_format="groups_of_four")
    ids = [row["participantId"] for row in playoff["playoff"]["seeds"]]
    assert ids == [row["participantId"] for row in standings if row["matches"]]
    games = [m for r in playoff["rounds"] if r.get("stage") == "playoff" for m in r["matches"]]
    assert len(games) == len(ids) // 4
    for index, game in enumerate(games):
        offset = 4 * index
        assert game["sideA"] == [ids[offset], ids[offset + 3]]
        assert game["sideB"] == [ids[offset + 1], ids[offset + 2]]
    assert playoff["playoff"]["sitOutParticipantIds"] == ids[len(games) * 4:]
    assert playoff["rounds"][:2] == before["rounds"][:2]
    assert generator_event_standings(playoff) == standings
    assert event == before, "Creating a playoff must not mutate its input"


@pytest.mark.parametrize("court_count", [0, 1, 2])
@pytest.mark.parametrize("winners", [(11, 5), (5, 11)])
def test_top_eight_carries_fixed_winning_teams_into_final(court_count, winners):
    event = scored_event(13)
    event["courtCount"] = court_count
    playoff = start_generator_playoff(event, playoff_format="top_eight")
    ids = [row["participantId"] for row in playoff["playoff"]["seeds"]]
    rounds = [r for r in playoff["rounds"] if r.get("stage") == "playoff"]
    semis = [m for r in rounds[:-1] for m in r["matches"]]
    assert [(m["sideA"], m["sideB"]) for m in semis] == [
        ([ids[0], ids[7]], [ids[3], ids[4]]),
        ([ids[1], ids[6]], [ids[2], ids[5]]),
    ]
    if court_count:
        assert all(len(r["matches"]) <= court_count for r in rounds)
    final_number = rounds[-1]["number"]
    with pytest.raises(ValueError, match="Only the current playoff"):
        save_generator_round(playoff, round_number=final_number, scores=[])
    while playoff["currentRoundNumber"] < final_number:
        number = playoff["currentRoundNumber"]
        row = next(r for r in playoff["rounds"] if r["number"] == number)
        playoff = save_generator_round(playoff, round_number=number, scores=[
            {"match_id": m["id"], "score_a": winners[0], "score_b": winners[1]} for m in row["matches"]
        ])
        playoff = advance_generator_event(playoff)
    final = playoff["rounds"][-1]["matches"][0]
    winning_side = "sideA" if winners[0] > winners[1] else "sideB"
    assert final["sideA"] == semis[0][winning_side]
    assert final["sideB"] == semis[1][winning_side]
    playoff = save_generator_round(playoff, round_number=final_number, scores=[
        {"match_id": final["id"], "score_a": 11, "score_b": 9},
    ])
    playoff = advance_generator_event(playoff)
    assert playoff["status"] == "completed"
    assert generator_event_standings(playoff) == generator_event_standings(event)
    assert len(admin._saved_matches(playoff)) == len(admin._saved_matches(event)) + 3


def test_departed_and_unplayed_players_do_not_displace_playoff_seeds():
    event = scored_event(12)
    departed = generator_event_standings(event)[0]["participantId"]
    event = mutate_generator_roster(event, action="remove", participant_id=departed)
    event = mutate_generator_roster(event, action="add", name="Just arrived")
    seeds = generator_playoff_options(event)["seeds"]
    assert departed not in {row["participantId"] for row in seeds}
    assert "Just arrived" not in {row["name"] for row in seeds}
    assert [row["seed"] for row in seeds] == list(range(1, len(seeds) + 1))


def test_reopened_history_must_be_resolved_before_seeding_and_is_locked_afterward():
    event = scored_event()
    event["rounds"][0]["status"] = "skipped"
    for match in event["rounds"][0]["matches"]:
        match["scoreA"] = match["scoreB"] = None
    reopened = reopen_generator_round(event, round_number=1)
    with pytest.raises(ValueError, match="every open round"):
        start_generator_playoff(reopened, playoff_format="top_eight")
    playoff = start_generator_playoff(event, playoff_format="top_eight")
    before = deepcopy(playoff)
    with pytest.raises(ValueError, match="fixed once the playoff"):
        reopen_generator_round(playoff, round_number=1)
    assert playoff == before


@pytest.mark.parametrize("sort", ["wins", "points", "differential"])
def test_seeds_follow_selected_sort_and_keep_starting_order_for_ties(sort):
    event = scored_event(8, standings_sort=sort)
    assert [row["participantId"] for row in generator_playoff_options(event)["seeds"]] == [
        row["participantId"] for row in generator_event_standings(event)
    ]


def test_playoff_requires_completed_scores_and_prevents_reseeding_or_skips():
    event = scored_event()
    active = advance_generator_event(event)
    with pytest.raises(ValueError, match="Save or skip"):
        start_generator_playoff(active, playoff_format="top_eight")
    with pytest.raises(ValueError, match="At least eight"):
        start_generator_playoff(scored_event(4), playoff_format="top_eight")
    with pytest.raises(ValueError, match="scored standings"):
        start_generator_playoff({**event, "scoringMode": "unscored"}, playoff_format="top_eight")
    with pytest.raises(ValueError, match="doubles round robins"):
        start_generator_playoff({**event, "playFormat": "singles"}, playoff_format="top_eight")
    playoff = start_generator_playoff(event, playoff_format="top_eight")
    with pytest.raises(ValueError, match="already has a playoff"):
        start_generator_playoff(playoff, playoff_format="groups_of_four")
    with pytest.raises(ValueError, match="teams are fixed"):
        mutate_generator_roster(playoff, action="add", name="Late player")
    with pytest.raises(ValueError, match="cannot be skipped"):
        skip_generator_round(playoff, round_number=playoff["currentRoundNumber"])


def public_session(db):
    created = public.create_public_play_generator_session(
        db, club_id="club", generator_kind="round_robin", play_format="doubles",
        title="Playoffs", participant_names=[f"Player {i}" for i in range(1, 9)],
        participant_player_ids={}, total_rounds=1, court_count=2,
        preview_fingerprint=None,
        idempotency_key="playoff-create-0001", requester_hash=requester(), token_secret=token_secret(),
    )
    session = public.save_public_play_generator_round(
        db, club_id="club", session_key=created["session"]["session_key"], round_number=1,
        scores=[{"match_id": m["id"], "score_a": 11, "score_b": 5} for m in matches(created["session"]["event"]["rounds"][0])],
        edit_token=created["edit_token"], expected_version=created["session"]["version"],
        idempotency_key="playoff-score-0001", requester_hash=requester(),
    )["session"]
    return created["edit_token"], session


def test_public_playoff_reopens_unsubmitted_completed_session_and_replays_safely():
    db = FakeSupabase()
    token, session = public_session(db)
    common = dict(club_id="club", session_key=session["session_key"], edit_token=token, requester_hash=requester())
    session = public.complete_public_play_generator_session(db, **common, expected_version=session["version"],
        idempotency_key="playoff-complete-0001")["session"]
    assert session["status"] == "completed"
    request = dict(**common, expected_version=session["version"], playoff_format="top_eight", idempotency_key="playoff-start-0001")
    with pytest.raises(PermissionError):
        public.start_public_play_generator_playoff(db, **{**request, "edit_token": "wrong", "idempotency_key": "playoff-wrong-0001"})
    result = public.start_public_play_generator_playoff(db, **request)
    replay = public.start_public_play_generator_playoff(db, **request)
    assert replay["idempotent_replay"]
    assert replay["session"]["version"] == result["session"]["version"]
    assert result["session"]["status"] == "active"
    assert db.db["live_sessions"][0]["completed_at"] is None
    reloaded = public.get_public_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    assert reloaded["event"] == result["session"]["event"]
    with pytest.raises(PublicLiveConflictError):
        public.start_public_play_generator_playoff(db, **{**request, "idempotency_key": "playoff-stale-0001"})


def test_submitted_results_cannot_be_reopened_for_playoffs():
    db = FakeSupabase()
    token, session = public_session(db)
    db.db["live_sessions"][0]["state"]["generator_submission"] = {"status": "pending"}
    with pytest.raises(ValueError, match="Submitted results"):
        public.start_public_play_generator_playoff(db, club_id="club", session_key=session["session_key"],
            edit_token=token, requester_hash=requester(), expected_version=session["version"],
            idempotency_key="playoff-locked-0001", playoff_format="groups_of_four")


def test_public_playoff_cannot_complete_before_final_is_played():
    db = FakeSupabase()
    token, session = public_session(db)
    common = dict(club_id="club", session_key=session["session_key"], edit_token=token, requester_hash=requester())
    session = public.start_public_play_generator_playoff(db, **common,
        expected_version=session["version"], idempotency_key="playoff-pending-start", playoff_format="top_eight")["session"]
    with pytest.raises(ValueError, match="Save every playoff"):
        public.complete_public_play_generator_session(db, **common,
            expected_version=session["version"], idempotency_key="playoff-premature-complete")
    assert db.db["live_sessions"][0]["status"] == "active"


def test_admin_playoff_persists_preview_and_teams_with_optimistic_version():
    db = AdminDatabase()
    actor = dict(actor_email="admin@example.test", actor_role="admin", source="test")
    session = admin.create_play_generator_session(db, club_id="club", generator_kind="round_robin",
        play_format="doubles", title="Playoffs", participant_names=[f"Player {i}" for i in range(8)],
        player_ids=[], total_rounds=1, court_count=2, preview_fingerprint=None, **actor)["session"]
    session = admin.save_play_generator_round(db, club_id="club", session_key=session["session_key"],
        round_number=1, scores=[{"match_id": m["id"], "score_a": 11, "score_b": 5} for m in matches(session["event"]["rounds"][0])],
        expected_version=session["version"], **actor)["session"]
    session = admin.start_play_generator_playoff(db, club_id="club", session_key=session["session_key"],
        playoff_format="top_eight", expected_version=session["version"], **actor)["session"]
    assert session["event"]["rounds"][1]["label"] == "Playoff semifinals"
    assert admin.get_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]["event"] == session["event"]
    with pytest.raises(ValueError, match="Save every playoff"):
        admin.complete_play_generator_session(db, club_id="club", session_key=session["session_key"],
            expected_version=session["version"], **actor)


def test_public_api_validates_format_token_and_version_before_creating_bracket():
    db = FakeSupabase()
    token, session = public_session(db)
    app = FastAPI()
    def raise_error(exc):
        code = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, PublicLiveConflictError) else 400
        raise HTTPException(code, str(exc))
    install_public_play_generator_routes(app, get_club=lambda slug: {"id": "club"},
        get_supabase_client=lambda: db, public_club_payload=lambda club, slug: club,
        require_public_writes=lambda: None, require_service_role=lambda: None,
        requester_hash=lambda request: requester(), raise_public_error=raise_error,
        public_writes_enabled=lambda: True, service_role_configured=lambda: True)
    client = TestClient(app)
    url = f"/clubs/club/play-generators/sessions/{session['session_key']}/playoff"
    body = {"edit_token": token, "expected_version": session["version"], "idempotency_key": "playoff-api-0001", "playoff_format": "top_eight"}
    assert client.post(url, json={**body, "playoff_format": "bad"}).status_code == 422
    assert client.post(url, json={**body, "edit_token": "wrong", "idempotency_key": "playoff-api-wrong-0001"}).status_code == 403
    response = client.post(url, json=body)
    assert response.status_code == 200
    assert response.json()["session"]["event"]["playoff"]["format"] == "top_eight"
    assert client.post(url, json=body).json()["idempotent_replay"]
