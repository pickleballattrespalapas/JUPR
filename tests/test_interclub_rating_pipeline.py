from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.interclub_rating_service import (
    InterclubRatingError, build_rating_projection, process_interclub_ratings,
)


def source():
    players = [{"id": i, "club_id": "a" if i < 5 else "b", "starting_rating": 1400, "rating": 1400}
               for i in range(1, 9)]
    entries = [{"id": f"e{i}", "season_id": "season", "club_id": p["club_id"], "player_id": i,
                "starting_rating": 4.0 if i < 5 else 3.5} for i, p in enumerate(players, 1)]
    game = {"id": "g1", "status": "completed", "a": 11, "b": 7, "played_at": "2026-01-01T10:00:00Z"}
    doc = {"encounters": [{"club_a": "a", "club_b": "b", "pairings": [
        {"kind": "women", "players_a": ["e1", "e2"], "players_b": ["e5", "e6"], "games": [game]}]}]}
    batch = {"id": "batch", "season_id": "season", "meet_id": "meet", "revision": 2,
             "state": "approved", "ratings_status": "pending", "approved_at": "2026-01-02T00:00:00Z", "document": doc}
    return {"players": players, "entries": entries, "matches": [], "sources": [], "approved": [batch]}


def effects(plan, game, stream):
    return [e for e in plan["effects"] if e["game_id"] == game and e["stream"] == stream]


def test_independent_rating_streams_and_owned_club_islands():
    plan = build_rating_projection(source())
    overall = effects(plan, "g1", "overall")[0]
    league = effects(plan, "g1", "league")[0]
    assert overall["after_elo"] - overall["before_elo"] != pytest.approx(league["after_elo"] - league["before_elo"])
    assert overall["before_elo"] == 1400
    assert league["before_elo"] == 1600
    assert plan["rated_games"] == 1
    assert {(p["club_id"], p["id"]) for p in plan["players"] if p["matches_played"]} == {("a", 1), ("a", 2), ("b", 5), ("b", 6)}


def test_correction_replays_intervening_local_games_and_later_interclub_game():
    data = source()
    later = deepcopy(data["approved"][0])
    later.update(id="later", meet_id="later-meet")
    later_game = later["document"]["encounters"][0]["pairings"][0]["games"][0]
    later_game.update(id="g2", played_at="2026-01-03T10:00:00Z")
    data["approved"].append(later)
    data["matches"] = [{"id": 44, "club_id": "a", "date": "2026-01-02T10:00:00Z", "t1_p1": 1, "t1_p2": 2,
                        "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8}]
    before = build_rating_projection(data)
    first = data["approved"][0]["document"]["encounters"][0]["pairings"][0]["games"][0]
    first.update(a=4, b=11)
    after = build_rating_projection(data)
    assert before["matches"][0]["t1_p1_r"] != after["matches"][0]["t1_p1_r"]
    assert effects(before, "g2", "overall")[0]["before_elo"] != effects(after, "g2", "overall")[0]["before_elo"]
    assert effects(before, "g2", "league")[0]["before_elo"] != effects(after, "g2", "league")[0]["before_elo"]
    assert before["players"][2]["rating"] != after["players"][2]["rating"]  # local opponent affected too
    assert build_rating_projection(data) == after


def test_unapproved_correction_keeps_last_approved_source_until_reapproval():
    data = source()
    batch = data["approved"].pop()
    data["sources"] = [{**batch, "batch_id": batch["id"]}]
    previous = build_rating_projection(data)
    revised = deepcopy(batch)
    revised["revision"] += 2
    revised["document"]["encounters"][0]["pairings"][0]["games"][0]["status"] = "unplayed"
    assert build_rating_projection(data) == previous
    data["approved"] = [revised]
    after = build_rating_projection(data)
    assert after["rated_games"] == 0
    assert all(p["rating"] == 1400 and p["matches_played"] == 0 for p in after["players"])


@pytest.mark.parametrize("status", ["retired", "forfeit", "unplayed", "pending"])
def test_noncompleted_games_and_rotating_singles_never_rate(status):
    data = source()
    encounter = data["approved"][0]["document"]["encounters"][0]
    encounter["pairings"][0]["games"][0]["status"] = status
    encounter["tiebreak"] = {"a": 21, "b": 18, "order_a": ["e1", "e2", "e3", "e4"], "order_b": ["e5", "e6", "e7", "e8"]}
    result = build_rating_projection(data)
    assert result["rated_games"] == 0 and result["effects"] == []


def test_actual_injury_substitute_gets_only_games_they_played():
    data = source()
    game = data["approved"][0]["document"]["encounters"][0]["pairings"][0]["games"][0]
    game["players_a"] = ["e3", "e2"]
    plan = build_rating_projection(data)
    assert {e["entry_id"] for e in plan["effects"]} == {"e3", "e2", "e5", "e6"}
    assert plan["players"][0]["matches_played"] == 0


def test_cross_club_player_injection_and_duplicate_game_rejected():
    data = source()
    pairing = data["approved"][0]["document"]["encounters"][0]["pairings"][0]
    pairing["players_a"] = ["e1", "e7"]
    with pytest.raises(InterclubRatingError, match="represent"):
        build_rating_projection(data)
    data = source()
    duplicate = deepcopy(data["approved"][0])
    duplicate["id"] = "another"
    data["approved"].append(duplicate)
    with pytest.raises(InterclubRatingError, match="more than one"):
        build_rating_projection(data)


def test_missing_immutable_seed_cannot_double_apply_current_rating():
    data = source()
    data["players"][0]["starting_rating"] = None
    data["players"][0]["rating"] = 1500
    with pytest.raises(InterclubRatingError, match="seed"):
        build_rating_projection(data)


@pytest.mark.parametrize("score", [(12, 0), (20, 11), (11, 10), (10, 8), (-1, 11)])
def test_impossible_completed_scores_rejected(score):
    data = source()
    game = data["approved"][0]["document"]["encounters"][0]["pairings"][0]["games"][0]
    game.update(a=score[0], b=score[1])
    with pytest.raises(InterclubRatingError):
        build_rating_projection(data)


class DB:
    def __init__(self, snapshot, *, conflict=False, fail=False):
        self.snapshot = snapshot
        self.calls = []
        self.conflict = conflict
        self.fail = fail
    def rpc(self, name, params):
        self.calls.append((name, params))
        if name == "pcs_interclub_rating_snapshot":
            value = {"snapshot": self.snapshot, "fingerprint": "fingerprint"}
        elif name == "pcs_apply_interclub_rating_projection":
            if self.fail:
                raise RuntimeError("private database error")
            if self.conflict:
                self.conflict = False
                value = {"status": "conflict"}
            else:
                value = {"status": "completed", "rated_games": 1}
        elif name == "pcs_fail_interclub_ratings":
            value = {"status": "failed"}
        else:
            raise AssertionError(name)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=value))


def test_exact_approval_revision_atomic_retry_and_safe_failure():
    snapshot = source()
    batch = snapshot["approved"][0]
    db = DB(snapshot, conflict=True)
    assert process_interclub_ratings(db, batch)["status"] == "completed"
    assert len([c for c in db.calls if c[0] == "pcs_apply_interclub_rating_projection"]) == 2
    assert all(c[1]["p_expected_revision"] == 2 for c in db.calls if c[0] == "pcs_apply_interclub_rating_projection")
    db = DB(snapshot, fail=True)
    result = process_interclub_ratings(db, batch)
    assert result["status"] == "failed" and "private" not in result["error"]
    assert db.calls[-1][0] == "pcs_fail_interclub_ratings"
    stale = {**batch, "revision": 1}
    db = DB(snapshot)
    assert process_interclub_ratings(db, stale)["status"] == "failed"
    assert not any(c[0] == "pcs_apply_interclub_rating_projection" for c in db.calls)


def test_completed_same_revision_is_idempotent_without_second_commit():
    snapshot = source()
    batch = snapshot["approved"][0]
    batch["ratings_status"] = "completed"
    snapshot["sources"] = [{**batch, "batch_id": batch["id"]}]
    db = DB(snapshot)
    result = process_interclub_ratings(db, batch)
    assert result["idempotent"] is True
    assert len(db.calls) == 1


def test_future_played_time_is_not_eligible_for_ratings():
    data = source()
    data["approved"][0]["document"]["encounters"][0]["pairings"][0]["games"][0]["played_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(InterclubRatingError, match="future"):
        build_rating_projection(data)


def test_equal_timestamps_use_play_order_not_random_game_identifiers():
    data = source()
    pairing = data["approved"][0]["document"]["encounters"][0]["pairings"][0]
    first = pairing["games"][0]
    first["id"] = "z-first"
    second = {**first, "id": "a-second", "a": 11, "b": 1}
    pairing["games"].append(second)
    plan = build_rating_projection(data)
    first_effect = effects(plan, "z-first", "league")[0]
    second_effect = effects(plan, "a-second", "league")[0]
    assert first_effect["before_elo"] == 1600
    assert second_effect["before_elo"] == first_effect["after_elo"]


def test_local_repair_replays_pending_graph_and_keeps_failure_durable():
    from jupr_app.services.interclub_rating_service import reconcile_interclub_for_club

    class LocalDB(DB):
        def __init__(self, *args, pending=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.pending = pending
        def table(self, name):
            assert name == "pcs_interclub_rating_repairs"
            outer = self
            class Query:
                def select(self, *args): return self
                def eq(self, *args): return self
                def limit(self, *args): return self
                def execute(self): return SimpleNamespace(data=[{"club_id": "a"}] if outer.pending else [])
            return Query()
        def rpc(self, name, params):
            if name == "pcs_fail_interclub_club_ratings":
                self.calls.append((name, params))
                return SimpleNamespace(execute=lambda: SimpleNamespace(data={"status": "failed"}))
            return super().rpc(name, params)

    db = LocalDB(source())
    assert reconcile_interclub_for_club(db, "a")["status"] == "completed"
    assert any(name == "pcs_apply_interclub_rating_projection" for name, _ in db.calls)
    failed = LocalDB(source(), fail=True)
    result = reconcile_interclub_for_club(failed, "a")
    assert result["status"] == "failed" and "score was saved" in result["error"]
    assert failed.calls[-1][0] == "pcs_fail_interclub_club_ratings"
    unrelated = LocalDB(source(), pending=False)
    assert reconcile_interclub_for_club(unrelated, "other")["status"] == "not_requested"
    assert unrelated.calls == []
