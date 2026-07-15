from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_match_log_service import (
    apply_admin_match_log_duplicate_cleanup,
    apply_admin_match_log_edits,
    build_admin_match_log,
    resolve_admin_match_log_duplicate_false_positive,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None
        self.update_payload = None
        self.insert_payload = None
        self.delete_mode = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append(("eq", key, value))
        return self

    def in_(self, key, values):
        self.filters.append(("in", key, {str(value) for value in values or []}))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def update(self, payload):
        self.update_payload = dict(payload or {})
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def delete(self):
        self.delete_mode = True
        return self

    def _apply_filters(self, rows):
        result = list(rows)
        for op, key, expected in self.filters:
            if op == "eq":
                result = [row for row in result if str(row.get(key)) == str(expected)]
            elif op == "in":
                result = [row for row in result if str(row.get(key)) in expected]
        return result

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            for row in rows:
                table.append(dict(row))
            return SimpleNamespace(data=rows)
        matched = self._apply_filters(table)
        if self.delete_mode:
            self.storage[self.table_name] = [row for row in table if row not in matched]
            return SimpleNamespace(data=matched)
        if self.update_payload is not None:
            for row in matched:
                row.update(self.update_payload)
            return SimpleNamespace(data=matched)
        if self.order_key:
            matched = sorted(matched, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            matched = matched[: self.limit_value]
        return SimpleNamespace(data=matched)


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeQuery(self.tables, name)


def fake_tables():
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
            {"club_id": "club", "id": 3, "name": "Casey"},
            {"club_id": "club", "id": 4, "name": "Devon"},
        ],
        "matches": [
            {
                "club_id": "club",
                "id": 1,
                "date": "2026-03-01T10:00:00Z",
                "league": "Open",
                "week_tag": "Week 1",
                "match_type": "Live Match",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8,
                "notes": "private note should not appear",
            },
            {
                "club_id": "club",
                "id": 2,
                "date": "2026-03-01T10:02:00Z",
                "league": "Open",
                "week_tag": "Week 1",
                "match_type": "Live Match",
                "t1_p1": 4,
                "t1_p2": 3,
                "t2_p1": 2,
                "t2_p2": 1,
                "score_t1": 8,
                "score_t2": 11,
            },
            {
                "club_id": "club",
                "id": 3,
                "date": "2026-03-02T10:00:00Z",
                "league": "POPUP",
                "week_tag": "Event",
                "match_type": "PopUp",
                "t1_p1": 1,
                "t1_p2": 3,
                "t2_p1": 2,
                "t2_p2": 4,
                "score_t1": 9,
                "score_t2": 11,
            },
        ],
        "club_people": [
            {"id": "cp1", "display_name": "Social Alex", "normalized_name": "social alex", "linked_player_id": None},
            {"id": "cp2", "display_name": "Social Blair", "normalized_name": "social blair", "linked_player_id": None},
            {"id": "cp3", "display_name": "Social Casey", "normalized_name": "social casey", "linked_player_id": None},
            {"id": "cp4", "display_name": "Social Devon", "normalized_name": "social devon", "linked_player_id": None},
        ],
        "live_events": [
            {
                "id": "event-1",
                "club_id": "club",
                "name": "Friday Social",
                "event_date": "2026-03-05",
                "result_mode": "social_unrated",
                "status": "saved",
                "submission_mode": "admin",
            }
        ],
        "live_event_participants": [
            {"id": "part-1", "event_id": "event-1", "club_person_id": "cp1", "participant_key": "a"},
            {"id": "part-2", "event_id": "event-1", "club_person_id": "cp2", "participant_key": "b"},
            {"id": "part-3", "event_id": "event-1", "club_person_id": "cp3", "participant_key": "c"},
            {"id": "part-4", "event_id": "event-1", "club_person_id": "cp4", "participant_key": "d"},
        ],
        "live_event_matches": [
            {
                "id": "social-1",
                "event_id": "event-1",
                "match_key": "r1-c1",
                "played_on": "2026-03-05",
                "round_number": 1,
                "court_number": 1,
                "mini_round_number": 1,
                "status": "saved",
                "submission_mode": "admin",
                "t1_p1_participant_id": "part-1",
                "t1_p2_participant_id": "part-2",
                "t2_p1_participant_id": "part-3",
                "t2_p2_participant_id": "part-4",
                "score_t1": 7,
                "score_t2": 11,
            }
        ],
        "admin_match_log_duplicate_resolutions": [],
        "admin_activity_log": [],
        "admin_audit_events": [],
        "player_badges": [],
        "badge_eval_queue": [],
    }


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(fake_tables())


def test_admin_match_log_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", raising=False)
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)

    payload = build_admin_match_log(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["status"] == "streamlit_fallback"
    assert payload["matches"] == []
    assert payload["correction_plan"]["mode"] == "planning_only"
    assert payload["resolved_duplicate_groups"] == []


def test_admin_match_log_duplicate_scan(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)

    payload = build_admin_match_log(fake_supabase(), club_id="club", filter_type="League", limit=20)

    assert payload["enabled"] is True
    assert payload["apply_enabled"] is False
    assert payload["summary"]["duplicate_groups"] == 1
    assert payload["summary"]["resolved_duplicate_groups"] == 0
    assert payload["duplicate_groups"][0]["keep_id"] == 1
    assert payload["duplicate_groups"][0]["delete_ids"] == [2]
    assert payload["duplicate_delete_preview"]["recommended_replay_scope"] == "ALL"
    assert payload["duplicate_delete_preview"]["recompute_scope"] == {"standings": True, "ratings": True}
    assert payload["matches"][0]["team1"][0]["name"] in {"Alex", "Devon"}
    assert "notes" not in payload["matches"][0]


def test_admin_match_log_filters_popup(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")

    payload = build_admin_match_log(fake_supabase(), club_id="club", filter_type="Pop-Up", limit=20)

    assert payload["summary"]["returned_matches"] == 1
    assert payload["matches"][0]["match_type"] == "PopUp"
    assert payload["duplicate_groups"] == []


def test_admin_match_log_apply_edits(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    tables = fake_tables()
    supabase = FakeSupabase(tables)

    result = apply_admin_match_log_edits(
        supabase,
        club_id="club",
        patches=[{"id": 1, "week_tag": "Week 2"}],
        actor_email="admin@example.com",
        actor_role="club_owner",
        correction_note="Fix week",
        confirmation_text="APPLY",
    )

    assert result["ok"] is True
    assert result["updated_count"] == 1
    assert tables["matches"][0]["week_tag"] == "Week 2"
    assert tables["matches"][0]["updated_by"] == "admin@example.com"


def test_admin_match_log_duplicate_cleanup(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    tables = fake_tables()
    supabase = FakeSupabase(tables)

    result = apply_admin_match_log_duplicate_cleanup(
        supabase,
        club_id="club",
        delete_ids=[2],
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="DELETE",
    )

    assert result["ok"] is True
    assert result["deleted_count"] == 1
    assert [row["id"] for row in tables["matches"]] == [1, 3]
    assert result["recompute_scope"] == {"standings": True, "ratings": True}


def test_admin_match_log_duplicate_no_issue_resolution(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    tables = fake_tables()
    supabase = FakeSupabase(tables)

    result = resolve_admin_match_log_duplicate_false_positive(
        supabase,
        club_id="club",
        match_ids=[1, 2],
        actor_email="admin@example.com",
        actor_role="club_owner",
        reason="Legitimate repeated matchup with same score.",
        confirmation_text="NO ISSUE",
    )

    assert result["ok"] is True
    assert result["mode"] == "duplicate_no_issue"
    assert result["match_ids"] == [1, 2]
    assert [row["id"] for row in tables["matches"]] == [1, 2, 3]
    assert tables["admin_match_log_duplicate_resolutions"][0]["match_id_key"] == "1,2"
    assert tables["admin_match_log_duplicate_resolutions"][0]["resolution"] == "no_issue"
    assert tables["admin_activity_log"][0]["action_type"] == "match_duplicate_false_positive_resolved"

    payload = build_admin_match_log(supabase, club_id="club", filter_type="League", limit=20)

    assert payload["summary"]["duplicate_groups"] == 0
    assert payload["summary"]["duplicate_delete_count"] == 0
    assert payload["summary"]["resolved_duplicate_groups"] == 1
    assert payload["duplicate_groups"] == []
    assert payload["duplicate_delete_preview"] is None
    assert payload["resolved_duplicate_groups"][0]["ids"] == [1, 2]
    assert payload["resolved_duplicate_groups"][0]["resolution"]["reason"] == "Legitimate repeated matchup with same score."


def test_admin_match_log_cleanup_rejects_no_issue_resolution(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    tables = fake_tables()
    supabase = FakeSupabase(tables)

    resolve_admin_match_log_duplicate_false_positive(
        supabase,
        club_id="club",
        match_ids=[1, 2],
        actor_email="admin@example.com",
        actor_role="club_owner",
        reason="Legitimate repeated matchup with same score.",
        confirmation_text="NO ISSUE",
    )

    try:
        apply_admin_match_log_duplicate_cleanup(
            supabase,
            club_id="club",
            delete_ids=[2],
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="DELETE",
        )
    except ValueError as exc:
        assert "not currently active duplicate cleanup candidates" in str(exc)
    else:
        raise AssertionError("Expected resolved duplicate cleanup to be rejected")
