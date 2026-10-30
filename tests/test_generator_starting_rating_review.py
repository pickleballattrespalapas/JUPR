from copy import deepcopy

import pytest

from jupr_app.services import generator_starting_rating_review as guard
from jupr_app.services import admin_play_generator_service as admin
from jupr_app.services import public_play_generator_service as public
from jupr_app.services.starting_rating_review_service import StartingRatingReviewRequired
from test_admin_play_generator_service import FakeSupabase as AdminDB
from test_public_play_generator_service import FakeSupabase as PublicDB, key, requester, token_secret


def event(mode="rated", fmt="doubles", ids=(1, 2, 3, 4)):
    return {"ratingMode": mode, "playFormat": fmt,
            "participants": [{"id": f"p-{i}", "name": f"Player {i}", "player_id": i} for i in ids]}


def review_needed(*args, **kwargs):
    raise StartingRatingReviewRequired({"match_format": kwargs["match_format"], "players": [
        {"player_id": kwargs["player_ids"][0], "review_required": True, "name": "New Player"}]})


def test_unrated_and_empty_hub_groups_do_not_need_rating_reads(monkeypatch):
    monkeypatch.setattr(guard, "require_starting_rating_review", review_needed)
    guard.require_generator_starting_ratings(None, club_id="club", event=event("unrated"))
    guard.require_generator_starting_ratings(None, club_id="club", event=event(ids=()))


def test_both_formats_reviewed_and_old_roster_remains_playable(monkeypatch):
    calls = []
    monkeypatch.setattr(guard, "require_starting_rating_review", lambda db, **kw: calls.append(kw))
    prior = event(fmt="doubles_singles")
    guard.require_generator_starting_ratings(None, club_id="club", event=prior)
    assert [(c["match_format"], c["player_ids"]) for c in calls] == [
        ("doubles", [1, 2, 3, 4]), ("singles", [1, 2, 3, 4])]
    calls.clear()
    guard.require_generator_starting_ratings(None, club_id="club", event=prior, before=prior)
    assert calls == []
    after = event(fmt="doubles_singles", ids=(1, 2, 3, 4, 5))
    guard.require_generator_starting_ratings(None, club_id="club", event=after, before=prior)
    assert all(c["player_ids"] == [5] for c in calls)


def test_profile_substitution_is_reviewed_even_when_roster_slot_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(guard, "require_starting_rating_review", lambda db, **kw: calls.append(kw))
    before = event(); after = deepcopy(before)
    after["participants"][0]["player_id"] = 8
    guard.require_generator_starting_ratings(None, club_id="club", event=after, before=before)
    assert calls[0]["player_ids"] == [8]


def test_rated_unlinked_player_requires_a_profile(monkeypatch):
    current = event(); current["participants"][1]["player_id"] = None
    with pytest.raises(guard.GeneratorProfileReviewRequired, match="Player 2"):
        guard.require_generator_starting_ratings(None, club_id="club", event=current)


def test_publication_reviews_only_scored_rated_formats(monkeypatch):
    calls = []
    monkeypatch.setattr(guard, "require_starting_rating_review", lambda db, **kw: calls.append(kw))
    guard.require_published_starting_ratings(None, club_id="club", matches=[
        {"match_format": "doubles", "rating_scope": "unrated", "t1_p1": 99},
        {"match_format": "singles", "rating_scope": "overall_only", "t1_p1": 1, "t2_p1": 2},
    ])
    assert calls == [{"club_id": "club", "player_ids": [1, 2], "match_format": "singles"}]


@pytest.mark.parametrize("surface", ["admin", "public"])
def test_review_rejection_prevents_start_without_creating_a_live_session(monkeypatch, surface):
    monkeypatch.setattr(guard, "require_starting_rating_review", review_needed)
    db = AdminDB() if surface == "admin" else PublicDB()
    rows = [{"id": i, "club_id": "club", "name": f"Player {i}"} for i in range(1, 5)]
    storage = db.storage if surface == "admin" else db.db
    storage["players"] = rows
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles", title="Rated RR",
                 participant_names=[p["name"] for p in rows], participant_player_ids={p["name"]: p["id"] for p in rows},
                 total_rounds=3, court_count=1, preview_fingerprint=None, rating_mode="rated")
    with pytest.raises(ValueError, match="Review starting estimates") as failure:
        if surface == "admin":
            admin.create_play_generator_session(db, **setup, player_ids=[], actor_email="admin@example.invalid",
                actor_role="administrator", source="test")
        else:
            public.create_public_play_generator_session(db, **setup, idempotency_key=key("review-first"),
                requester_hash=requester(), token_secret=token_secret())
    assert storage["live_sessions"] == []
    assert failure.value.detail["code"] == "STARTING_RATING_REVIEW_REQUIRED"
    if surface == "public":
        assert storage["public_live_operations"][0]["status"] == "rejected"


def test_confirmed_public_start_retry_does_not_repeat_review(monkeypatch):
    monkeypatch.setattr(guard, "require_starting_rating_review", lambda *_a, **_k: None)
    db = PublicDB()
    db.db["players"] = [{"id": i, "club_id": "club", "name": f"Player {i}"} for i in range(1, 5)]
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles", title="Rated RR",
        participant_names=[f"Player {i}" for i in range(1, 5)], participant_player_ids={f"Player {i}": i for i in range(1, 5)},
        total_rounds=3, court_count=1, preview_fingerprint=None, rating_mode="rated", idempotency_key=key("review-retry"),
        requester_hash=requester(), token_secret=token_secret())
    first = public.create_public_play_generator_session(db, **setup)
    monkeypatch.setattr(guard, "require_starting_rating_review", review_needed)
    retry = public.create_public_play_generator_session(db, **setup)
    assert retry["session"]["session_key"] == first["session"]["session_key"]
    assert len(db.db["live_sessions"]) == 1


def test_rated_checkin_waits_for_review_without_stopping_current_game(monkeypatch):
    from tests.test_hub_arrival_registration import empty_hub, arrive
    from tests.test_admin_round_robin_hub_service import command, raw_session
    db, hub = empty_hub()
    for number in range(1, 5):
        hub = arrive(db, hub, f"Guest {number}")
    hub = command(db, hub, "auto_fill_courts")
    playing = next(c for c in hub["workspace"]["courts"] if c["current"])
    current = deepcopy(playing["current"])
    # Model an in-progress session created before this review rollout.
    raw_session(db)["state"]["page_state"]["event"]["ratingMode"] = "rated"
    monkeypatch.setattr(guard, "require_starting_rating_review", review_needed)

    hub = arrive(db, hub, "New Guest")
    assert len(hub["workspace"]["groups"][0]["players"]) == 4
    assert hub["registration"]["pending_count"] == 1
    pending = next(c for c in hub["registration"]["checkins"] if c["name"] == "New Guest")
    assert "select a club profile" in pending["review_reason"]
    after = next(c["current"] for c in hub["workspace"]["courts"] if c["id"] == playing["id"])
    for field in ("match_id", "side_a", "side_b", "score_a", "score_b"):
        assert after[field] == current[field]

    saved = command(db, hub, "score_and_release", {"court_id": playing["id"],
        "session_key": current["session_key"], "match_id": current["match_id"], "score_a": 11, "score_b": 7})
    assert saved["hub"]["status"] == "active"
    assert saved["registration"]["pending_count"] == 1
    assert saved["workspace"]["groups"][0]["play_format"] == "doubles"


def test_active_mode_change_requires_review_but_completed_intent_waits_for_approval(monkeypatch):
    from test_generator_submissions import Database, completed_session, submit, review_args
    from jupr_app.services import generator_session_details_service as details
    from jupr_app.services import generator_submission_service as submissions

    db = Database()
    saved, token = completed_session(db)
    row = db.db["live_sessions"][0]
    row["status"] = "active"
    args = dict(club_id="club", session_key=saved["session_key"], expected_version=row["updated_at"],
                title="New title", rating_mode="rated", actor_email="admin@example.invalid", actor_role="administrator")
    with pytest.raises(guard.GeneratorProfileReviewRequired):
        details.update_admin_generator_details(db, **args)
    assert row["state"]["page_state"]["event"]["ratingMode"] == "unrated"

    row["status"] = "completed"
    details.update_admin_generator_details(db, **args)
    current = public.get_public_play_generator_session(db, club_id="club", session_key=saved["session_key"])["session"]
    current = submit(db, current, token)
    db.starting_reviews.clear()
    with pytest.raises(StartingRatingReviewRequired):
        submissions.review_generator_submission(db, **review_args(current))
    assert db.db["live_sessions"][0]["state"]["generator_submission"]["status"] == "pending"
    assert not db.db["matches"]


@pytest.mark.parametrize("committed", [False, True])
def test_direct_publish_recovers_committed_receipt_before_new_rating_review(monkeypatch, committed):
    db = AdminDB()
    saved = admin.create_play_generator_session(db, club_id="club", generator_kind="round_robin",
        play_format="doubles", title="Legacy RR", participant_names=[f"Player {i}" for i in range(1, 5)],
        player_ids=[], total_rounds=3, court_count=1, preview_fingerprint=None,
        actor_email="admin@example.invalid", actor_role="administrator", source="test")["session"]
    payloads = [{"live_match_id": "game-1", "context_id": "context-1", "match_format": "doubles",
                 "rating_scope": "overall_only", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4}]
    _, fingerprint = admin.stable_direct_match_request(club_id="club", match_format="doubles", matches=payloads)
    result = {"ok": True, "committed": True, "request_fingerprint": fingerprint}
    db.storage["admin_direct_match_entry_operations"] = ([{
        "club_id": "club", "idempotency_key": "legacy-publish", "match_format": "doubles",
        "request_fingerprint": fingerprint, "result_json": result,
    }] if committed else [])
    monkeypatch.setattr(admin, "_publish_payloads", lambda *a, **k: payloads)
    monkeypatch.setattr(admin, "load_data", lambda *a: (None,) * 11)
    monkeypatch.setattr(admin, "submit_atomic_direct_matches", lambda *a, **k: result)
    monkeypatch.setattr(guard, "require_starting_rating_review", review_needed)
    args = dict(club_id="club", session_key=saved["session_key"], match_date="2026-10-30",
                expected_version=saved["version"], idempotency_key="legacy-publish", operation_key="operation-1",
                actor_email="admin@example.invalid", actor_role="administrator", source="test")
    if committed:
        published = admin.publish_play_generator_matches(db, **args)
        assert published["published_count"] == 1
        assert published["result"] == result
    else:
        with pytest.raises(StartingRatingReviewRequired):
            admin.publish_play_generator_matches(db, **args)
        assert not db.storage["live_sessions"][0]["state"]["official_publish"]["published_match_ids"]
