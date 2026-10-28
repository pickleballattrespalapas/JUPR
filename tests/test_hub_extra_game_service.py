from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event, generator_event_standings
from jupr_app.domain.round_robin_hub import apply_hub_command, build_hub_snapshot, _ensure_players, player_identity
from jupr_app.services.hub_extra_game_service import prepare_extra_game

NOW = "2026-10-28T03:00:00+00:00"


class Query:
    def __init__(self, db):
        self.db, self.filters, self.payload = db, {}, None

    def select(self, *_): return self
    def eq(self, key, value): self.filters[key] = value; return self
    def limit(self, *_): return self
    def insert(self, payload): self.payload = deepcopy(payload); return self

    def execute(self):
        if self.payload is not None:
            self.db.inserts += 1
            key = self.payload["session_key"]
            if key in self.db.rows:
                raise RuntimeError("unique session key")
            self.db.rows[key] = {**self.payload, "version": 1}
            if self.db.lose_response:
                self.db.lose_response = False
                raise TimeoutError("insert response lost after commit")
            return SimpleNamespace(data=[deepcopy(self.db.rows[key])])
        return SimpleNamespace(data=[deepcopy(row) for row in self.db.rows.values()
            if all(row.get(key) == value for key, value in self.filters.items())])


class DB:
    def __init__(self): self.rows, self.inserts, self.lose_response = {}, 0, False
    def table(self, name): assert name == "live_sessions"; return Query(self)


def fixture():
    events = {}
    for key, count in (("lower", 5), ("higher", 7)):
        events[key] = start_generator_event(create_generator_preview(generator_kind="round_robin",
            play_format="doubles", title=key, participant_names=[f"{key} {i}" for i in range(count)],
            total_rounds=1, court_count=0))
    hub = {"id": "hub", "groups": [{"session_key": key} for key in events],
        "courts": [{"id": "c1", "label": "Court 1", "open": True, "dedicated_session_key": None,
            "current": None, "next": None}], "players": {}}
    _ensure_players(hub, events, NOW)
    lower = {"session_key": "lower", "participant_id": events["lower"]["rounds"][0]["byeParticipantIds"][0]}
    higher = [{"session_key": "higher", "participant_id": pid} for pid in events["higher"]["rounds"][0]["byeParticipantIds"]]
    payload = {"side_a": [lower, higher[0]], "side_b": higher[1:], "standings_policy": "exhibition"}
    return hub, events, payload


def prepare(db, hub, events, payload, fingerprint="fingerprint"):
    return prepare_extra_game(db, club_id="club", hub_id="hub", hub=hub, events=events, payload=payload,
        operation_key="exact-extra-game-request", request_fingerprint=fingerprint, actor_email="operator@example.test", now=NOW)


def test_extra_game_keeps_source_memberships_results_and_exact_chosen_teams():
    hub, events, payload = fixture()
    original_hub, original_events = deepcopy(hub), deepcopy(events)
    draft, attached, updated = prepare(DB(), hub, events, payload)
    key = draft["session_key"]
    assert hub == original_hub and events == original_events
    assert all(updated[k] == event for k, event in events.items())
    assert draft["state"]["page_state"]["event"]["status"] == "preview"
    assert draft["state"]["page_state"]["event"]["hubSupplementalDraft"] is True
    extra = updated[key]
    assert extra["hubSupplemental"]["standingsPolicy"] == "exhibition"
    assert extra["hubFinishAfterRound"] == 1 and len(extra["rounds"]) == 1
    assert extra["ratingMode"] == "unrated"
    assert extra["rounds"][0]["matches"][0]["sideA"] == ["p-1", "p-2"]
    assert extra["rounds"][0]["matches"][0]["sideB"] == ["p-3", "p-4"]
    for ref in extra["hubSupplemental"]["sourceParticipants"]:
        assert attached["participant_identities"][ref["session_key"]][ref["participant_id"]] == attached["participant_identities"][key][ref["extra_participant_id"]]


def test_lost_insert_response_and_exact_retry_reuse_same_durable_draft():
    hub, events, payload = fixture()
    db = DB(); db.lose_response = True
    first = prepare(db, hub, events, payload)
    second = prepare(db, hub, events, payload)
    assert first == second
    assert len(db.rows) == db.inserts == 1
    with pytest.raises(ValueError, match="different or attached"):
        prepare(db, hub, events, payload, fingerprint="changed-request")
    assert db.inserts == 1


def test_attached_session_is_never_overwritten_or_reattached_by_preparation():
    hub, events, payload = fixture()
    db = DB(); draft, attached, updated = prepare(db, hub, events, payload)
    key = draft["session_key"]
    db.rows[key]["state"]["page_state"]["event"] = deepcopy(updated[key])
    db.rows[key]["state"]["round_robin_hub_id"] = "hub"
    before = deepcopy(db.rows)
    with pytest.raises(ValueError, match="different or attached"):
        prepare(db, hub, events, payload)
    assert db.rows == before and db.inserts == 1


def test_modified_draft_cannot_be_reused_under_an_old_operation_fingerprint():
    hub, events, payload = fixture()
    db = DB(); draft, _, _ = prepare(db, hub, events, payload)
    stored = db.rows[draft["session_key"]]["state"]["page_state"]["event"]
    stored["rounds"][0]["matches"][0]["scoreA"] = 11
    with pytest.raises(ValueError, match="different or attached"):
        prepare(db, hub, events, payload)
    assert db.inserts == 1


@pytest.mark.parametrize("failure", ["duplicate", "paused", "unfinished", "policy"])
def test_invalid_extra_selection_creates_no_draft(failure):
    hub, events, payload = fixture()
    if failure == "duplicate": payload["side_b"][1] = deepcopy(payload["side_a"][0])
    if failure == "paused":
        ref = payload["side_a"][0]
        hub["players"][hub["participant_identities"][ref["session_key"]][ref["participant_id"]]]["paused"] = True
    if failure == "unfinished":
        payload["side_a"][0]["participant_id"] = events["lower"]["rounds"][0]["matches"][0]["sideA"][0]
    if failure == "policy": payload["standings_policy"] = "home_groups"
    db = DB()
    with pytest.raises(ValueError): prepare(db, hub, events, payload)
    assert db.inserts == 0


def test_distinct_roster_entries_for_one_profile_cannot_fill_two_extra_slots():
    hub, events, payload = fixture()
    for ref in payload["side_a"]:
        next(p for p in events[ref["session_key"]]["participants"] if p["id"] == ref["participant_id"])["player_id"] = 42
    db = DB()
    with pytest.raises(ValueError, match="Each person"):
        prepare(db, hub, events, payload)
    assert db.inserts == 0


def test_supplemental_assignment_shares_claims_and_scores_only_extra_results():
    hub, events, payload = fixture()
    before_standings = {key: generator_event_standings(value) for key, value in events.items()}
    draft, hub, events = prepare(DB(), hub, events, payload)
    key = draft["session_key"]
    match = events[key]["rounds"][0]["matches"][0]
    hub, events = apply_hub_command(hub, events, action="assign_game", payload={"session_key": key,
        "match_id": match["id"], "round_number": 1, "court_id": "c1"}, now=NOW)
    snapshot = build_hub_snapshot(hub, events, now=NOW)
    source_ref = payload["side_a"][0]
    player = next(p for p in events[source_ref["session_key"]]["participants"] if p["id"] == source_ref["participant_id"])
    identity = player_identity(player, source_ref["session_key"], hub)
    assert any(p["identity"] == identity and p["on_court"] for p in snapshot["players"])
    assert all(not any(p["identity"] == identity for p in m["side_a"] + m["side_b"]) for m in snapshot["eligible_queue"])
    another_db = DB()
    with pytest.raises(ValueError, match="court and reservation"):
        prepare(another_db, hub, events, payload)
    assert another_db.inserts == 0
    hub, events = apply_hub_command(hub, events, action="score_and_release", payload={"session_key": key,
        "match_id": match["id"], "round_number": 1, "court_id": "c1", "score_a": 11, "score_b": 7}, now=NOW)
    assert events[key]["status"] == "completed" and len(events[key]["rounds"]) == 1
    assert sum(row["matches"] for row in generator_event_standings(events[key])) == 4
    assert all(generator_event_standings(events[k]) == standings for k, standings in before_standings.items())
