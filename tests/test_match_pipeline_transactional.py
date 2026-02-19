from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain import match_pipeline


class _SelectQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, _fields: str):
        return self

    def eq(self, _col: str, _value):
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _DummySupabase:
    def __init__(self, match_rows):
        self._match_rows = match_rows

    def table(self, name: str):
        if name != "matches":
            return _SelectQuery([])
        return _SelectQuery(self._match_rows)


def test_record_match_rolls_back_and_returns_structured_error(monkeypatch):
    writes = []

    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "_find_existing_match_by_idempotency_key", lambda **_: None)
    monkeypatch.setattr(match_pipeline, "_build_processing_context", lambda **_: {
        "name_to_id": {},
        "df_players_all": pd.DataFrame([]),
        "df_leagues": pd.DataFrame([]),
        "df_meta": pd.DataFrame([]),
    })

    def fake_process(_match_list, **kwargs):
        kwargs["match_writer"]({"club_id": "club-1", "score_t1": 11, "score_t2": 9}, None, "admin", "key-1")
        raise RuntimeError("boom")

    monkeypatch.setattr(match_pipeline, "process_matches", fake_process)
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"id": 42}]))
    monkeypatch.setattr(
        match_pipeline,
        "sb_delete",
        lambda *_args, **kwargs: writes.append(("delete", kwargs["filters"])) or SimpleNamespace(data=[{"id": 42}]),
    )

    result = match_pipeline.record_match(
        supabase=object(),
        club_id="club-1",
        match_payload={
            "idempotency_key": "key-1",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 9,
        },
    )

    assert result["success"] is False
    assert result["match_id"] == 42
    assert result["error"] is not None
    assert ("delete", {"club_id": "club-1", "id": 42}) in writes


def test_update_match_rolls_back_patch_and_returns_structured_error(monkeypatch):
    writes = []
    supabase = _DummySupabase(match_rows=[{"id": 8, "score_t1": 11, "score_t2": 9, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4}])

    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())

    def fake_sb_update(_supabase, table, payload, *, filters):
        writes.append((table, dict(payload), dict(filters)))
        return SimpleNamespace(data=[{"id": filters["id"]}])

    monkeypatch.setattr(match_pipeline, "sb_update", fake_sb_update)
    monkeypatch.setattr(match_pipeline, "_rebuild_state", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))

    result = match_pipeline.update_match(
        supabase=supabase,
        club_id="club-1",
        match_id=8,
        patch={"score_t1": 1, "score_t2": 11},
    )

    assert result["success"] is False
    assert result["match_id"] == 8
    assert result["error"] is not None
    assert "rolled_back_match_patch" in result["warnings"]
    assert writes[0][1] == {"score_t1": 1, "score_t2": 11}
    assert writes[1][2] == {"club_id": "club-1", "id": 8}


def test_record_match_logs_audit_event(monkeypatch):
    logged = []

    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "_find_existing_match_by_idempotency_key", lambda **_: None)
    monkeypatch.setattr(match_pipeline, "_build_processing_context", lambda **_: {
        "name_to_id": {},
        "df_players_all": pd.DataFrame([]),
        "df_leagues": pd.DataFrame([]),
        "df_meta": pd.DataFrame([]),
    })

    def fake_process(match_list, **kwargs):
        row = dict(match_list[0])
        row["club_id"] = "club-1"
        kwargs["match_writer"](row, None, "admin", "key-1")
        return {"inserted": 1}

    monkeypatch.setattr(match_pipeline, "process_matches", fake_process)
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"id": 12}]))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **payload: logged.append(payload))

    result = match_pipeline.record_match(
        supabase=object(),
        club_id="club-1",
        match_payload={
            "idempotency_key": "key-1",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 9,
        },
    )

    assert result["success"] is True
    assert logged
    assert logged[0]["action_type"] == "record_match"
    assert logged[0]["payload"]["match_id"] == 12


def test_record_match_requires_idempotency_key():
    try:
        match_pipeline.record_match(
            supabase=object(),
            club_id="club-1",
            match_payload={"score_t1": 11, "score_t2": 9},
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "idempotency_key is required" in str(exc)


def test_record_match_returns_existing_on_idempotency_hit(monkeypatch):
    existing = {"id": 55, "club_id": "club-1", "idempotency_key": "dup-key"}
    process_calls = []

    monkeypatch.setattr(match_pipeline, "_find_existing_match_by_idempotency_key", lambda **_: existing)
    monkeypatch.setattr(match_pipeline, "process_matches", lambda *_args, **_kwargs: process_calls.append(True))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=object(),
        club_id="club-1",
        match_payload={"idempotency_key": "dup-key", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result["match_id"] == 55
    assert result["idempotent_hit"] is True
    assert result["existing"]["id"] == 55
    assert process_calls == []


def test_require_club_scope_rejects_missing_club_id_in_payload():
    try:
        match_pipeline.require_club_scope("club-1", {"id": 42})
        assert False, "expected MatchPipelineError"
    except match_pipeline.MatchPipelineError as exc:
        assert "club_id scope is required" in str(exc)


def test_require_club_scope_rejects_mismatched_club_id():
    try:
        match_pipeline.require_club_scope("club-1", {"club_id": "club-2", "id": 42})
        assert False, "expected MatchPipelineError"
    except match_pipeline.MatchPipelineError as exc:
        assert "club_id scope mismatch" in str(exc)
