from types import SimpleNamespace
from copy import deepcopy

import pytest

from jupr_app.services.public_live_operation_service import PublicLiveRecoveryRequiredError
from jupr_app.services import public_play_generator_service
from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session,
    create_public_play_generator_session,
    get_public_play_generator_session,
    mark_public_play_generator_round_played,
    mutate_public_play_generator_roster,
    preview_public_play_generator,
    save_public_play_generator_round,
    skip_public_play_generator_round,
)


class Query:
    def __init__(self, db, name):
        self.db = db
        self.name = name
        self.filters = []
        self.limit_n = None
        self.order_key = None
        self.desc = False
        self.payload = None
        self.update_payload = None

    def select(self, *_args, **_kwargs): return self
    def eq(self, key, value): self.filters.append((key, value)); return self
    def in_(self, key, values): self.filters.append((key, set(values))); return self
    def gte(self, *_args): return self
    def limit(self, value): self.limit_n = int(value); return self
    def order(self, key, desc=False): self.order_key = key; self.desc = bool(desc); return self
    def insert(self, payload): self.payload = payload; return self
    def update(self, payload): self.update_payload = dict(payload); return self

    def matches(self, row):
        for key, value in self.filters:
            if isinstance(value, set):
                if row.get(key) not in value: return False
            elif str(row.get(key)) != str(value): return False
        return True

    def execute(self):
        rows = self.db.setdefault(self.name, [])
        if self.payload is not None:
            values = self.payload if isinstance(self.payload, list) else [self.payload]
            rows.extend(dict(row) for row in values)
            return SimpleNamespace(data=values)
        selected = [row for row in rows if self.matches(row)]
        if self.update_payload is not None:
            for row in selected: row.update(self.update_payload)
            return SimpleNamespace(data=selected)
        if self.order_key:
            selected = sorted(selected, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.desc)
        if self.limit_n is not None: selected = selected[: self.limit_n]
        return SimpleNamespace(data=selected)


class FakeSupabase:
    def __init__(self):
        self.db = {"live_sessions": [], "public_live_operations": [], "players": []}
    def table(self, name): return Query(self.db, name)


def requester(): return "a" * 64

def token_secret(): return "x" * 48

def key(label): return f"public-generator-{label}-00000001"

def matches(round_row):
    return list(round_row.get("matches") or []) or [match for court in round_row.get("courts") or [] for match in court.get("matches") or []]


@pytest.mark.parametrize("linked_indices", [(), (1, 3), (0, 1, 2, 3, 4)])
@pytest.mark.parametrize("rating_mode", ["rated", "unrated"])
def test_five_player_roster_preview_start_and_reload_preserve_profile_links(linked_indices, rating_mode):
    supabase = FakeSupabase()
    names = ["Manuel", "Jose", "Victor", "Chay", "Beto"]
    supabase.db["players"] = [
        {"id": 101 + index, "club_id": "club", "name": name}
        for index, name in enumerate(names)
    ]
    players_before = deepcopy(supabase.db["players"])
    setup = dict(
        club_id="club", generator_kind="round_robin", play_format="doubles",
        title="3.5+ RR", participant_names=names,
        participant_player_ids={names[index]: 101 + index for index in linked_indices},
        total_rounds=5, court_count=1, rating_mode=rating_mode,
    )
    preview = preview_public_play_generator(supabase, **setup)["preview"]
    expected_ids = [101 + index if index in linked_indices else None for index in range(5)]
    assert [player.get("player_id") for player in preview["participants"]] == expected_ids
    assert [player["name"] for player in preview["participants"]] == names
    assert len(preview["rounds"]) == 5
    assert all(len(matches(row)) == 1 and len(row["byeParticipantIds"]) == 1 for row in preview["rounds"])
    assert sorted(pid for row in preview["rounds"] for pid in row["byeParticipantIds"]) == [f"p-{i}" for i in range(1, 6)]
    assert not supabase.db["live_sessions"], "Preview must not create a session"

    created = create_public_play_generator_session(
        supabase, **setup, preview_fingerprint=preview["previewFingerprint"],
        idempotency_key=key("mixed-roster"), requester_hash=requester(), token_secret=token_secret(),
    )
    session = created["session"]
    saved = save_public_play_generator_round(
        supabase, club_id="club", session_key=session["session_key"], round_number=1,
        scores=[{"match_id": matches(session["event"]["rounds"][0])[0]["id"], "score_a": 11, "score_b": 7}],
        edit_token=created["edit_token"], expected_version=session["version"],
        idempotency_key=key("mixed-score"), requester_hash=requester(),
    )["session"]
    reloaded = get_public_play_generator_session(supabase, club_id="club", session_key=saved["session_key"])["session"]
    assert [player.get("player_id") for player in reloaded["event"]["participants"]] == expected_ids
    assert reloaded["event"]["rounds"][0]["status"] == "saved"
    assert supabase.db["players"] == players_before, "Manual names must not create or modify club profiles"


@pytest.mark.parametrize("invalid_player", [
    None,
    {"id": 101, "club_id": "other-club", "name": "Manuel"},
    {"id": 101, "club_id": "club", "name": "Someone Else"},
])
def test_mixed_roster_still_rejects_missing_cross_club_and_mismatched_profiles(invalid_player):
    supabase = FakeSupabase()
    supabase.db["players"] = [invalid_player] if invalid_player else []
    with pytest.raises(public_play_generator_service.PublicPlayGeneratorError, match="Choose Manuel from the player list again"):
        preview_public_play_generator(
            supabase, club_id="club", generator_kind="round_robin", play_format="doubles",
            title="Roster validation", participant_names=["Manuel", "Jose", "Victor", "Chay", "Beto"],
            participant_player_ids={"Manuel": 101}, total_rounds=5, court_count=1,
        )


def test_public_eight_plus_four_preserves_round_and_persists_optional_extra_game():
    supabase = FakeSupabase()
    created = create_public_play_generator_session(
        supabase, club_id="club", generator_kind="round_robin", play_format="doubles",
        title="Late arrivals", participant_names=[f"Player {i}" for i in range(8)],
        participant_player_ids={}, total_rounds=3, court_count=2, preview_fingerprint=None,
        idempotency_key=key("late-create"), requester_hash=requester(), token_secret=token_secret(),
    )
    session = created["session"]
    first = deepcopy(session["event"]["rounds"][0])
    common = dict(club_id="club", session_key=session["session_key"],
                  participant_id=None, player_id=None, substitute_scope="rest", roster_order=[],
                  edit_token=created["edit_token"], requester_hash=requester())
    for i in range(4):
        session = mutate_public_play_generator_roster(supabase, **common,
            action="add", name=f"Arrival {i}", expected_version=session["version"],
            idempotency_key=key(f"late-add-{i}"))["session"]
        assert session["event"]["rounds"][0] == first
    seat = dict(action="seat_arrivals", name=None, court_number=3,
                participant_ids=[f"p-new-{i}" for i in range(1, 5)],
                expected_version=session["version"], idempotency_key=key("late-seat"))
    added = mutate_public_play_generator_roster(supabase, **common, **seat)["session"]
    replay = mutate_public_play_generator_roster(supabase, **common, **seat)["session"]
    assert replay["version"] == added["version"]
    stored = get_public_play_generator_session(supabase, club_id="club", session_key=session["session_key"])["session"]
    assert stored["event"]["rounds"][0]["matches"][:2] == first["matches"]
    assert len(stored["event"]["rounds"][0]["matches"]) == 3
    assert stored["event"]["courtCount"] == 3
    saved = save_public_play_generator_round(supabase, club_id="club", session_key=session["session_key"],
        round_number=1, scores=[{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in matches(stored["event"]["rounds"][0])],
        edit_token=created["edit_token"], expected_version=stored["version"], idempotency_key=key("late-score"), requester_hash=requester())["session"]
    advanced = advance_public_play_generator_session(supabase, club_id="club", session_key=session["session_key"],
        edit_token=created["edit_token"], expected_version=saved["version"], idempotency_key=key("late-advance"), requester_hash=requester())["session"]
    assert advanced["current_round_number"] == 2
    assert len(advanced["event"]["rounds"][1]["matches"]) == 3
    assert not advanced["event"]["rounds"][1]["byeParticipantIds"]


@pytest.mark.parametrize("initial_count,arrivals", [(6, 2), (8, 4)])
@pytest.mark.parametrize("scoring_mode", ["scored", "unscored"])
def test_automatic_courts_expand_and_mix_late_arrivals_after_round_is_finished(initial_count, arrivals, scoring_mode):
    db = FakeSupabase()
    created = create_public_play_generator_session(
        db, club_id="club", generator_kind="round_robin", play_format="doubles",
        title="Automatic courts", participant_names=[f"Player {i}" for i in range(initial_count)],
        participant_player_ids={}, total_rounds=3, court_count=0, scoring_mode=scoring_mode,
        preview_fingerprint=None, idempotency_key=key("auto-create"),
        requester_hash=requester(), token_secret=token_secret(),
    )
    session = created["session"]
    first = deepcopy(session["event"]["rounds"][0])
    common = dict(club_id="club", session_key=session["session_key"],
                  edit_token=created["edit_token"], requester_hash=requester())
    for i in range(arrivals):
        session = mutate_public_play_generator_roster(
            db, **common, action="add", name=f"Arrival {i}", participant_id=None,
            player_id=None, substitute_scope="rest", roster_order=[],
            expected_version=session["version"], idempotency_key=key(f"auto-add-{i}"),
        )["session"]
        assert session["event"]["rounds"][0] == first, "Players already on court must stay put"

    if scoring_mode == "scored":
        session = save_public_play_generator_round(
            db, **common, round_number=1, expected_version=session["version"],
            scores=[{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in matches(first)],
            idempotency_key=key("auto-score"),
        )["session"]
        session = advance_public_play_generator_session(
            db, **common, expected_version=session["version"], idempotency_key=key("auto-advance"),
        )["session"]
    else:
        session = mark_public_play_generator_round_played(
            db, **common, round_number=1, expected_version=session["version"], idempotency_key=key("auto-played"),
        )["session"]
    reloaded = get_public_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    assert reloaded["current_round_number"] == 2
    second = reloaded["event"]["rounds"][1]
    assert len(matches(second)) == (initial_count + arrivals) // 4
    assert not second["byeParticipantIds"]
    all_ids = {p["id"] for p in reloaded["event"]["participants"]}
    scheduled = [pid for m in matches(second) for pid in m["sideA"] + m["sideB"]]
    assert len(scheduled) == len(set(scheduled)) == len(all_ids)
    assert set(scheduled) == all_ids
    for old_match in matches(first):
        old_group = set(old_match["sideA"] + old_match["sideB"])
        for new_match in matches(second):
            assert len(old_group.intersection(new_match["sideA"] + new_match["sideB"])) <= 2
    old_partners = {frozenset(m[side]) for m in matches(first) for side in ("sideA", "sideB")}
    new_partners = {frozenset(m[side]) for m in matches(second) for side in ("sideA", "sideB")}
    assert not old_partners.intersection(new_partners)
    assert all(len(matches(r)) == (initial_count + arrivals) // 4 for r in reloaded["event"]["rounds"][1:])


def test_public_round_robin_preview_create_score_skip_and_roster():
    supabase = FakeSupabase()
    preview = preview_public_play_generator(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Public Singles",
        participant_names=["A", "B", "C", "D", "E"],
        participant_player_ids={},
        total_rounds=4,
        court_count=2,
    )["preview"]
    assert len(preview["rounds"]) == 4
    assert any(row["byeParticipantIds"] for row in preview["rounds"])

    created = create_public_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Public Singles",
        participant_names=["A", "B", "C", "D", "E"],
        participant_player_ids={},
        total_rounds=4,
        court_count=2,
        preview_fingerprint=preview["previewFingerprint"],
        idempotency_key=key("create"),
        requester_hash=requester(),
        token_secret=token_secret(),
    )
    session = created["session"]
    edit = created["edit_token"]
    first = session["event"]["rounds"][0]
    scored = save_public_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=1,
        scores=[{"match_id": row["id"], "score_a": 11, "score_b": 7} for row in matches(first)],
        edit_token=edit,
        expected_version=session["version"],
        idempotency_key=key("scores"),
        requester_hash=requester(),
    )["session"]
    advanced = advance_public_play_generator_session(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        edit_token=edit,
        expected_version=scored["version"],
        idempotency_key=key("advance"),
        requester_hash=requester(),
    )["session"]
    roster = mutate_public_play_generator_roster(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        action="add",
        participant_id=None,
        name="F",
        player_id=None,
        substitute_scope="rest",
        roster_order=[],
        edit_token=edit,
        expected_version=advanced["version"],
        idempotency_key=key("roster"),
        requester_hash=requester(),
    )["session"]
    skipped = skip_public_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=2,
        reason="Weather",
        edit_token=edit,
        expected_version=roster["version"],
        idempotency_key=key("skip"),
        requester_hash=requester(),
    )["session"]
    assert skipped["event"]["rounds"][0]["status"] == "saved"
    assert skipped["event"]["rounds"][1]["status"] == "skipped"
    assert any(row["name"] == "F" for row in skipped["event"]["participants"])
    assert get_public_play_generator_session(supabase, club_id="club", session_key=session["session_key"])["session"]["unrated"] is True


def test_public_ladder_previews_only_round_one_and_requires_results_to_advance():
    supabase = FakeSupabase()
    preview = preview_public_play_generator(
        supabase,
        club_id="club",
        generator_kind="ladder",
        play_format="doubles",
        title="Public Ladder",
        participant_names=[f"P{idx}" for idx in range(1, 10)],
        participant_player_ids={},
        total_rounds=3,
        court_count=2,
    )["preview"]
    assert len(preview["rounds"]) == 1
    created = create_public_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="ladder",
        play_format="doubles",
        title="Public Ladder",
        participant_names=[f"P{idx}" for idx in range(1, 10)],
        participant_player_ids={},
        total_rounds=3,
        court_count=2,
        preview_fingerprint=preview["previewFingerprint"],
        idempotency_key=key("ladder-create"),
        requester_hash=requester(),
        token_secret=token_secret(),
    )
    with pytest.raises(Exception):
        advance_public_play_generator_session(
            supabase,
            club_id="club",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key=key("ladder-early"),
            requester_hash=requester(),
        )


def _create_singles_session(supabase, *, operation_key="public-generator-recovery-create"):
    return create_public_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Recovery test",
        participant_names=["A", "B", "C", "D"],
        participant_player_ids={},
        total_rounds=2,
        court_count=2,
        preview_fingerprint=None,
        idempotency_key=operation_key,
        requester_hash=requester(),
        token_secret=token_secret(),
    )


def test_completed_generator_create_without_session_stops_before_a_duplicate(monkeypatch):
    supabase = FakeSupabase()
    operation = {
        "operation_key": "f" * 64,
        "status": "completed",
        "result_json": {},
        "request_fingerprint": "a" * 64,
    }
    monkeypatch.setattr(
        public_play_generator_service,
        "begin_public_live_operation",
        lambda *_args, **_kwargs: (operation, True),
    )
    monkeypatch.setattr(
        public_play_generator_service,
        "_find_creation_row",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PublicLiveRecoveryRequiredError) as caught:
        _create_singles_session(supabase)

    message = str(caught.value)
    assert "contact club staff" in message
    assert "Don’t start another one" in message
    assert "Retry the same request" not in message


def test_uncertain_generator_create_retries_the_same_request(monkeypatch):
    supabase = FakeSupabase()
    original_execute = Query.execute

    def fail_live_session_insert(query):
        if query.name == "live_sessions" and query.payload is not None:
            raise RuntimeError("simulated response loss")
        return original_execute(query)

    monkeypatch.setattr(Query, "execute", fail_live_session_insert)

    with pytest.raises(PublicLiveRecoveryRequiredError) as caught:
        _create_singles_session(supabase, operation_key="public-generator-uncertain-create")

    message = str(caught.value)
    assert "Retry the same request" in message
    assert "contact club staff" not in message


def test_pending_generator_change_retries_that_same_action():
    supabase = FakeSupabase()
    created = _create_singles_session(supabase)
    supabase.db["live_sessions"][0]["pending_operation_key"] = "pending-change"
    first_round = created["session"]["event"]["rounds"][0]

    with pytest.raises(PublicLiveRecoveryRequiredError) as caught:
        save_public_play_generator_round(
            supabase,
            club_id="club",
            session_key=created["session"]["session_key"],
            round_number=1,
            scores=[
                {"match_id": row["id"], "score_a": 11, "score_b": 7}
                for row in matches(first_round)
            ],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key="public-generator-pending-change",
            requester_hash=requester(),
        )

    message = str(caught.value)
    assert "Retry that same action" in message
    assert "contact club staff" not in message
