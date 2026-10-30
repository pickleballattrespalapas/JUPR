from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.starting_rating_review_service import (
    StartingRatingReviewConflict, StartingRatingReviewRequired,
    confirm_starting_ratings, get_starting_rating_review, require_starting_rating_review,
)


def snapshot(*, history=0, confirmation=None, rating=1200):
    return {"player": {"id": 1, "name": "Shannon", "rating": rating, "starting_rating": rating},
            "rated_matches": history, "recorded_rated_matches": history,
            "confirmation": confirmation, "state_fingerprint": "a" * 64}


class DB:
    def __init__(self, raw=None, error=None):
        self.raw = raw or snapshot()
        self.error = error
        self.calls = []

    def rpc(self, name, params):
        self.calls.append((name, params))

        def execute():
            if self.error:
                raise self.error
            raw = deepcopy(self.raw)
            if name == "confirm_starting_rating_reviews_v1":
                review = params["p_reviews"][0]
                raw["player"]["rating"] = raw["player"]["starting_rating"] = review["rating_elo"]
                raw["confirmation"] = {"rating_elo": review["rating_elo"], "requested_rating": review["requested_rating"], "confirmed_at": "2026-10-30T18:00:00Z"}
                return SimpleNamespace(data={"ok": True, "confirmed_count": 1, "players": [raw]})
            return SimpleNamespace(data={"players": [raw]})

        return SimpleNamespace(execute=execute)


def preview(db, request=None, format_name="doubles"):
    return get_starting_rating_review(db, club_id="club", player_ids=[1], match_format=format_name,
                                     requested_ratings={"1": request} if request else {})["players"][0]


def confirm(db, **patch):
    review = {"player_id": 1, "rating_jupr": 3.25, "expected_state_fingerprint": "a" * 64, "requested_rating": "3.5", **patch}
    return confirm_starting_ratings(db, club_id="club", match_format="doubles", reviews=[review], actor_email="admin@test", actor_role="administrator")


def test_history_keeps_calculated_rating_without_mismatch_warning():
    row = preview(DB(snapshot(history=9, rating=1371.392)), "4.0")
    assert row["rating_jupr"] == 1371.392 / 400
    assert row["requested_jupr"] == 4
    assert row["has_rated_history"]
    assert not row["review_required"]
    assert not row["requested_rating_differs"]
    assert not row["can_edit_starting_rating"]


def test_unplayed_estimate_requires_confirmation_without_import():
    db = DB()
    row = preview(db)
    assert row["review_required"]
    assert not row["requested_rating_differs"]
    assert row["can_edit_starting_rating"]
    with pytest.raises(StartingRatingReviewRequired) as exc:
        require_starting_rating_review(db, club_id="club", player_ids=[1])
    assert exc.value.detail["code"] == "STARTING_RATING_REVIEW_REQUIRED"
    assert exc.value.detail["players"][0]["player_id"] == 1


def test_confirmation_is_remembered_until_rating_or_requested_level_changes():
    confirmation = {"rating_elo": 1300, "requested_rating": "3.5", "confirmed_at": "2026-10-30T18:00:00Z"}
    db = DB(snapshot(rating=1300, confirmation=confirmation))
    assert not preview(db)["review_required"]
    assert not preview(db, "3.50")["review_required"], "Equivalent numeric formatting is not a new request"
    assert preview(db, "3.75")["review_required"]
    db.raw["player"]["rating"] = 1400
    assert preview(db)["review_required"]


def test_range_remains_a_human_decision_and_can_be_acknowledged():
    raw = snapshot()
    row = preview(DB(raw), "3.75–4.0")
    assert row["requested_jupr"] is None
    assert row["requested_rating_differs"] and row["review_required"]
    raw["confirmation"] = {"rating_elo": 1200, "requested_rating": "3.75–4.0", "confirmed_at": "today"}
    assert not preview(DB(raw), "3.75–4.0")["review_required"]


def test_singles_history_and_confirmation_do_not_promise_seed_editing():
    row = preview(DB(), "3.5", "singles")
    assert row["review_required"] and not row["can_edit_starting_rating"]
    assert "cannot be changed here" in row["review_note"]


def test_atomic_confirmation_carries_expected_snapshot_and_requested_text():
    db = DB()
    result = confirm(db)
    assert result["ok"] and not result["review_required"]
    name, payload = db.calls[0]
    assert name == "confirm_starting_rating_reviews_v1"
    assert payload["p_reviews"] == [{"player_id": 1, "rating_elo": 1300, "expected_state_fingerprint": "a" * 64, "requested_rating": "3.5"}]


@pytest.mark.parametrize("rating", [True, None, float("nan"), float("inf"), 0.9, 7.1])
def test_invalid_seed_does_not_reach_database(rating):
    db = DB()
    with pytest.raises(ValueError):
        confirm(db, rating_jupr=rating)
    assert not db.calls


@pytest.mark.parametrize("error", ["STARTING_REVIEW_CONFLICT", "STARTING_REVIEW_HISTORY_EXISTS", "STARTING_REVIEW_REPLAY_ACTIVE", "canceling statement due to lock timeout", "deadlock detected"])
def test_concurrency_errors_require_refresh_without_retrying_writes(error):
    db = DB(error=RuntimeError(error))
    with pytest.raises(StartingRatingReviewConflict):
        confirm(db)
    assert len(db.calls) == 1


def test_unavailable_evidence_fails_closed():
    db = DB(error=RuntimeError("RPC unavailable"))
    with pytest.raises(RuntimeError, match="unavailable"):
        require_starting_rating_review(db, club_id="club", player_ids=[1])


def test_one_batched_read_per_selected_roster_and_empty_roster_needs_no_read():
    db = DB()
    require_starting_rating_review(db, club_id="club", player_ids=[])
    assert not db.calls
    preview(db)
    assert len(db.calls) == 1 and db.calls[0][1]["p_player_ids"] == [1]
