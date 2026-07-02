from __future__ import annotations

from jupr_app.services.public_live_write_service import (
    create_public_round_robin_session,
    update_public_round_robin_scores,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, db: dict[str, dict], table_name: str):
        self.db = db
        self.table_name = table_name
        self.payload = None
        self.filters: dict[str, object] = {}
        self.select_expr = "*"

    def upsert(self, payload, **_kwargs):
        self.payload = dict(payload)
        return self

    def select(self, expr="*", *_args, **_kwargs):
        self.select_expr = str(expr or "*")
        return self

    def eq(self, key, value):
        self.filters[str(key)] = value
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table_name != "live_sessions":
            return FakeResponse([])
        if self.payload is not None:
            key = f"{self.payload['club_id']}::{self.payload['session_key']}"
            existing = self.db.get(key, {})
            row = {**existing, **self.payload}
            self.db[key] = row
            return FakeResponse([row])
        rows = list(self.db.values())
        for key, value in self.filters.items():
            rows = [row for row in rows if row.get(key) == value]
        return FakeResponse(rows[:1])


class FakeSupabase:
    def __init__(self):
        self.db: dict[str, dict] = {}

    def table(self, table_name):
        return FakeQuery(self.db, str(table_name))


def test_public_round_robin_create_and_score_update():
    supabase = FakeSupabase()

    created = create_public_round_robin_session(
        supabase,
        club_id="tres_palapas",
        event_name="Public Test RR",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
    )

    edit_token = created["edit_token"]
    session = created["session"]
    assert session["session_key"]
    assert session["title"] == "Public Test RR"
    assert session["rounds"]

    match_id = session["rounds"][0]["matches"][0]["id"]
    updated = update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=session["session_key"],
        edit_token=edit_token,
        scores=[{"match_id": match_id, "score_a": 11, "score_b": 8}],
    )

    updated_session = updated["session"]
    assert updated_session["rounds"][0]["matches"][0]["score_a"] == 11
    assert updated_session["rounds"][0]["matches"][0]["score_b"] == 8
    assert updated_session["standings"]
