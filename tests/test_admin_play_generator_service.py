from __future__ import annotations

from types import SimpleNamespace
from copy import deepcopy
import pytest

from jupr_app.services.admin_play_generator_service import (
    advance_play_generator_session,
    create_play_generator_session,
    get_play_generator_session,
    list_play_generator_sessions,
    mutate_play_generator_roster,
    preview_play_generator,
    save_play_generator_round,
    skip_play_generator_round,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.limit_count = None
        self.order_key = None
        self.order_desc = False
        self.insert_payload = None
        self.update_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def in_(self, key, values):
        self.filters.append((key, set(values)))
        return self

    def limit(self, value):
        self.limit_count = int(value)
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def _matches(self, row):
        for key, value in self.filters:
            if isinstance(value, set):
                if row.get(key) not in value:
                    return False
            elif str(row.get(key)) != str(value):
                return False
        return True

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            inserted = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            rows.extend(dict(row) for row in inserted)
            return SimpleNamespace(data=inserted)
        selected = [row for row in rows if self._matches(row)]
        if self.update_payload is not None:
            for row in selected:
                row.update(self.update_payload)
            return SimpleNamespace(data=selected)
        if self.order_key:
            selected = sorted(
                selected,
                key=lambda row: str(row.get(self.order_key) or ""),
                reverse=self.order_desc,
            )
        if self.limit_count is not None:
            selected = selected[: self.limit_count]
        return SimpleNamespace(data=selected)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "live_sessions": [],
            "players": [],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


@pytest.mark.parametrize("linked_indices", [(), (1, 3), (0, 1, 2, 3, 4)])
@pytest.mark.parametrize("rating_mode", ["rated", "unrated"])
def test_sparse_profiles_preserve_guest_positions_through_preview_start_and_reload(linked_indices, rating_mode, monkeypatch):
    from jupr_app.services import generator_starting_rating_review as rating_review
    monkeypatch.setattr(rating_review, "require_starting_rating_review", lambda *_a, **_k: None)
    db = FakeSupabase()
    names = ["Manuel", "Jose", "Victor", "Chay", "Beto"]
    db.storage["players"] = [{"id": 101 + index, "club_id": "club", "name": name} for index, name in enumerate(names)]
    links = {names[index]: 101 + index for index in linked_indices}
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles", title="Mixed roster",
                 participant_names=names, player_ids=[], participant_player_ids=links,
                 total_rounds=3, court_count=1, rating_mode=rating_mode)
    preview = preview_play_generator(db, **setup)["preview"]
    expected = [links.get(name) for name in names]
    assert [p.get("player_id") for p in preview["participants"]] == expected
    if rating_mode == "rated" and len(linked_indices) < len(names):
        with pytest.raises(ValueError, match="select a club profile"):
            create_play_generator_session(db, **setup, preview_fingerprint=preview["previewFingerprint"],
                actor_email="admin@example.invalid", actor_role="administrator", source="test")
        assert not db.storage["live_sessions"]
        return
    session = create_play_generator_session(db, **setup, preview_fingerprint=preview["previewFingerprint"],
        actor_email="admin@example.invalid", actor_role="administrator", source="test")["session"]
    restored = get_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    assert [p.get("player_id") for p in restored["event"]["participants"]] == expected
    assert restored["event"]["ratingMode"] == rating_mode


def test_sparse_links_preserve_explicit_ids_with_identical_names_in_different_groups():
    db = FakeSupabase()
    db.storage["players"] = [{"id": number, "club_id": "club", "name": "Alex Same"} for number in (101, 102)]
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles", title="Group",
        participant_names=["Alex Same", "Guest B", "Guest C", "Guest D"], player_ids=[], total_rounds=3, court_count=1)
    first = preview_play_generator(db, **setup, participant_player_ids={"Alex Same": 101})["preview"]
    second = preview_play_generator(db, **setup, participant_player_ids={"Alex Same": 102})["preview"]
    assert first["participants"][0]["player_id"] == 101
    assert second["participants"][0]["player_id"] == 102
    assert first["previewFingerprint"] != second["previewFingerprint"]
    with pytest.raises(ValueError, match="changed after preview"):
        create_play_generator_session(db, **setup, participant_player_ids={"Alex Same": 102},
            preview_fingerprint=first["previewFingerprint"], actor_email="admin@example.invalid", actor_role="administrator", source="test")
    assert db.storage["live_sessions"] == []


@pytest.mark.parametrize("links,message", [
    ({"A": 999}, "not found in this club"),
    ({"Unknown name": 1}, "not in this roster"),
    ({"A": 1, "B": 1}, "two places"),
    ({"A": 1, " a ": 2}, "one club profile"),
])
def test_sparse_links_reject_foreign_or_ambiguous_profiles(links, message):
    db = FakeSupabase()
    db.storage["players"] = [{"id": 1, "club_id": "club", "name": "A"},
        {"id": 2, "club_id": "club", "name": "Other"}, {"id": 999, "club_id": "other-club", "name": "A"}]
    with pytest.raises(ValueError, match=message):
        preview_play_generator(db, club_id="club", generator_kind="round_robin", play_format="doubles", title="Group",
            participant_names=["A", "B", "C", "D"], player_ids=[], participant_player_ids=links, total_rounds=3, court_count=1)


def test_legacy_complete_id_list_remains_compatible_with_sparse_links():
    db = FakeSupabase()
    names = ["A", "B", "C", "D"]
    db.storage["players"] = [{"id": index, "club_id": "club", "name": name} for index, name in enumerate(names, 1)]
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles", title="Group",
                 participant_names=names, total_rounds=3, court_count=1)
    legacy = preview_play_generator(db, **setup, player_ids=[1, 2, 3, 4])["preview"]
    sparse = preview_play_generator(db, **setup, player_ids=[], participant_player_ids=dict(zip(names, [1, 2, 3, 4])))["preview"]
    assert legacy["previewFingerprint"] == sparse["previewFingerprint"]
    with pytest.raises(ValueError, match="not both"):
        preview_play_generator(db, **setup, player_ids=[1, 2, 3, 4], participant_player_ids={"A": 1})


def test_moneyball_still_requires_all_profiles_with_sparse_input():
    db = FakeSupabase()
    db.storage["players"] = [{"id": 1, "club_id": "club", "name": "A"}]
    with pytest.raises(ValueError, match="club|player|profile"):
        preview_play_generator(db, club_id="club", generator_kind="round_robin", play_format="doubles", title="Moneyball",
            participant_names=["A", "B", "C", "D"], player_ids=[], participant_player_ids={"A": 1},
            total_rounds=3, court_count=1, event_format="moneyball_expectations")


def test_admin_eight_plus_four_preserves_active_games_and_persists_spare_court():
    supabase = FakeSupabase()
    actor = dict(actor_email="admin@example.test", actor_role="admin", source="test")
    session = create_play_generator_session(supabase, club_id="club", generator_kind="round_robin",
        play_format="doubles", title="Late arrivals", participant_names=[f"Player {i}" for i in range(8)],
        player_ids=[], total_rounds=3, court_count=2, preview_fingerprint=None, **actor)["session"]
    first = deepcopy(session["event"]["rounds"][0])
    common = dict(club_id="club", session_key=session["session_key"], participant_id=None,
                  player_id=None, substitute_scope="rest", roster_order=[], **actor)
    for i in range(4):
        session = mutate_play_generator_roster(supabase, **common, action="add", name=f"Arrival {i}",
            expected_version=session["version"])["session"]
        assert session["event"]["rounds"][0] == first
    session = mutate_play_generator_roster(supabase, **common, action="seat_arrivals", name=None,
        participant_ids=[f"p-new-{i}" for i in range(1, 5)], court_number=3,
        expected_version=session["version"])["session"]
    stored = get_play_generator_session(supabase, club_id="club", session_key=session["session_key"])["session"]
    assert stored["event"]["rounds"][0]["matches"][:2] == first["matches"]
    assert len(stored["event"]["rounds"][0]["matches"]) == 3
    assert stored["event"]["courtCount"] == 3


def _matches(round_row):
    if round_row.get("matches"):
        return list(round_row.get("matches") or [])
    return [
        match
        for court in round_row.get("courts") or []
        for match in court.get("matches") or []
    ]


def test_preview_and_create_round_robin_session():
    supabase = FakeSupabase()
    preview = preview_play_generator(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Singles RR",
        participant_names=["A", "B", "C", "D", "E"],
        player_ids=[],
        total_rounds=4,
        court_count=2,
    )

    assert len(preview["preview"]["rounds"]) == 4
    assert preview["schedule_rows"]

    created = create_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Singles RR",
        participant_names=["A", "B", "C", "D", "E"],
        player_ids=[],
        total_rounds=4,
        court_count=2,
        preview_fingerprint=preview["preview"]["previewFingerprint"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )

    session = created["session"]
    assert session["status"] == "active"
    assert session["generator_kind"] == "round_robin"
    assert session["play_format"] == "singles"
    assert session["event"]["rounds"][0]["status"] == "active"


def test_round_scores_skip_advance_and_roster_changes_are_durable():
    supabase = FakeSupabase()
    created = create_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="doubles",
        title="Adaptive RR",
        participant_names=["A", "B", "C", "D", "E"],
        player_ids=[],
        total_rounds=3,
        court_count=1,
        preview_fingerprint=None,
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )
    session = created["session"]
    first_match = _matches(session["event"]["rounds"][0])[0]

    saved = save_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=1,
        scores=[{"match_id": first_match["id"], "score_a": 11, "score_b": 6}],
        expected_version=session["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]
    assert saved["event"]["rounds"][0]["status"] == "saved"

    advanced = advance_play_generator_session(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        expected_version=saved["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]
    assert advanced["current_round_number"] == 2

    roster = mutate_play_generator_roster(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        action="add",
        participant_id=None,
        name="New Player",
        player_id=None,
        substitute_scope="rest",
        roster_order=[],
        expected_version=advanced["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]
    assert any(row["name"] == "New Player" for row in roster["event"]["participants"])

    skipped = skip_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=2,
        reason="Weather",
        expected_version=roster["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]
    assert skipped["event"]["rounds"][1]["status"] == "skipped"


def test_ladder_session_lists_only_round_one_until_advance():
    supabase = FakeSupabase()
    created = create_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="ladder",
        play_format="doubles",
        title="Ladder",
        participant_names=[f"P{idx}" for idx in range(1, 10)],
        player_ids=[],
        total_rounds=3,
        court_count=2,
        preview_fingerprint=None,
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )
    session = created["session"]
    assert len(session["event"]["rounds"]) == 1

    scores = [
        {"match_id": match["id"], "score_a": 11, "score_b": 7}
        for match in _matches(session["event"]["rounds"][0])
    ]
    saved = save_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=1,
        scores=scores,
        expected_version=session["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]
    advanced = advance_play_generator_session(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        expected_version=saved["version"],
        actor_email="admin@example.com",
        actor_role="admin",
        source="test",
    )["session"]

    assert len(advanced["event"]["rounds"]) == 2
    assert advanced["current_round_number"] == 2

    listing = list_play_generator_sessions(
        supabase,
        club_id="club",
        generator_kind="ladder",
    )
    assert listing["count"] == 1
    assert get_play_generator_session(
        supabase,
        club_id="club",
        session_key=session["session_key"],
    )["session"]["generator_kind"] == "ladder"


@pytest.mark.parametrize("mode,linked,expected", [
    ("admin_play_generator", False, True),
    ("public_play_generator", False, False),
    ("public_play_generator", True, True),
])
def test_admin_session_capabilities_match_existing_mutation_authority(mode, linked, expected):
    from jupr_app.services import admin_play_generator_service as service
    row = {"club_id": "club", "session_key": "session", "state": {"mode": mode,
        "page_state": {"event": {"generatorKind": "round_robin"}}}}
    if linked:
        row["state"]["round_robin_hub_id"] = "hub"
    payload = service._session_payload(row)
    assert payload["capabilities"]["can_manage"] is expected
    assert payload["capabilities"]["can_score"] is expected
    assert bool(payload["capabilities"]["unavailable_reason"]) is not expected
    row["state"]["generator_submission"] = {"status": "pending"}
    assert service._session_payload(row)["capabilities"]["can_score"] is False


def test_saved_match_publication_excludes_not_played_even_with_legacy_scores():
    from jupr_app.services import admin_play_generator_service as service
    matches = [{"id": status or "legacy", "status": status, "scoreA": 11, "scoreB": 5}
               for status in (None, "scored", "not_played", "skipped", "cancelled", "canceled")]
    event = {"rounds": [{"number": 1, "status": "saved", "matches": matches}]}
    assert [match["id"] for _, match in service._saved_matches(event)] == ["legacy", "scored"]


def test_skip_published_results_rejects_before_domain_mutation():
    db = FakeSupabase()
    actor = dict(actor_email="admin@example.test", actor_role="administrator", source="test")
    created = create_play_generator_session(db, club_id="club", generator_kind="round_robin", play_format="doubles",
        title="Published", participant_names=["A", "B", "C", "D"], player_ids=[], total_rounds=1,
        court_count=1, preview_fingerprint=None, **actor)["session"]
    row = db.storage["live_sessions"][0]
    row["state"]["official_publish"]["published_match_ids"] = ["official-match"]
    snapshot = deepcopy(row)
    with pytest.raises(ValueError, match="Published results"):
        skip_play_generator_round(db, club_id="club", session_key=created["session_key"], round_number=1,
            reason="Not actually played", expected_version=created["version"], **actor)
    assert row == snapshot
