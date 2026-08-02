from __future__ import annotations

from types import SimpleNamespace

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
