from types import SimpleNamespace

from jupr_app.services.admin_jupr_live_service import (
    advance_admin_jupr_live_league_round,
    create_admin_jupr_live_session,
    publish_admin_jupr_live_matches,
    update_admin_jupr_live_scores,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.lt_filters = []
        self.insert_payload = None
        self.update_payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def lt(self, key, value):
        self.lt_filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            row = {"id": f"row-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[dict(row)])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        for key, expected in self.lt_filters:
            scoped = [row for row in scoped if row.get(key) is not None and str(row.get(key)) < str(expected)]
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in scoped])


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [
                {"club_id": "club", "id": 1, "name": "Alex"},
                {"club_id": "club", "id": 2, "name": "Blair"},
                {"club_id": "club", "id": 3, "name": "Casey"},
                {"club_id": "club", "id": 4, "name": "Devon"},
            ],
            "live_sessions": [],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def _league_matches(event):
    matches = []
    for round_row in event.get("rounds", []):
        for court in round_row.get("courts", []):
            for mini in court.get("miniRounds", []):
                matches.extend(mini.get("matches", []))
    return matches


def test_jupr_live_creates_league_ladder_event(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "1")
    created = create_admin_jupr_live_session(
        FakeSupabase(),
        club_id="club",
        title="League Night",
        event_type="league_ladder",
        participant_names=["Alex", "Blair", "Casey", "Devon"],
        player_ids=[1, 2, 3, 4],
        total_rounds=2,
        court_sizes=[4],
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="CREATE LIVE SESSION",
    )
    event = created["session"]["state"]["page_state"]["event"]
    assert event["type"] == "league"
    assert event["totalRounds"] == 2
    assert _league_matches(event)


def test_jupr_live_league_scores_publish_and_advance(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "1")
    supabase = FakeSupabase()
    created = create_admin_jupr_live_session(
        supabase,
        club_id="club",
        title="League Night",
        event_type="league_ladder",
        participant_names=["Alex", "Blair", "Casey", "Devon"],
        player_ids=[1, 2, 3, 4],
        total_rounds=2,
        court_sizes=[4],
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="CREATE LIVE SESSION",
    )
    session_key = created["session"]["session_key"]
    event = created["session"]["state"]["page_state"]["event"]
    score_rows = [{"match_id": match["id"], "score_a": 11, "score_b": 7} for match in _league_matches(event)]
    scored = update_admin_jupr_live_scores(
        supabase,
        club_id="club",
        session_key=session_key,
        scores=score_rows,
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="SAVE LIVE SCORES",
    )
    assert scored["changed_scores"] == len(score_rows)
    monkeypatch.setattr("jupr_app.services.admin_jupr_live_service.load_data", lambda *_args, **_kwargs: ([], [], [], [], [], [], [], {}, {}, False, None))
    monkeypatch.setattr("jupr_app.services.admin_jupr_live_service.submit_match_batch", lambda *_args, **_kwargs: SimpleNamespace(ok=True, data={"inserted": len(score_rows)}, errors=[]))
    published = publish_admin_jupr_live_matches(
        supabase,
        club_id="club",
        session_key=session_key,
        match_date="2026-01-01T00:00:00Z",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="PUBLISH LIVE MATCHES",
    )
    assert published["published_count"] == len(score_rows)
    advanced = advance_admin_jupr_live_league_round(
        supabase,
        club_id="club",
        session_key=session_key,
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="ADVANCE LIVE ROUND",
    )
    assert advanced["session"]["current_round_number"] == 2
