from types import SimpleNamespace

import pytest

from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session,
    create_public_play_generator_session,
    get_public_play_generator_session,
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
