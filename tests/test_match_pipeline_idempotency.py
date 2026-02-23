from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain import match_pipeline


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def select(self, _fields: str):
        return self

    def eq(self, _col: str, _value):
        return self

    def in_(self, _col: str, _values):
        return self

    def order(self, _col: str, desc: bool = False):
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        class _Resp:
            data = self._rows

        return _Resp()


class _Supabase:
    def __init__(self, existing_rows=None):
        self._existing_rows = existing_rows or []

    def table(self, name: str):
        if name == "matches":
            return _Query(self._existing_rows)
        return _Query([])


def test_record_match_idempotency_hit_returns_existing_and_skips_processing(monkeypatch):
    supabase = _Supabase(existing_rows=[{"id": 9, "idempotency_key": "dup-1", "club_id": "club-1"}])
    process_calls = []

    monkeypatch.setattr(match_pipeline, "_enforce_write_preflight", lambda _supabase: None)
    monkeypatch.setattr(match_pipeline, "_process_persisted_matches", lambda **_kwargs: process_calls.append(True))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "dup-1", "date": "2026-01-01T12:00:00+00:00", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result["idempotent_hit"] is True
    assert result["match_id"] == 9
    assert process_calls == []


def test_record_match_new_idempotency_processes_matches(monkeypatch):
    supabase = _Supabase(existing_rows=[])

    process_calls = []
    monkeypatch.setattr(match_pipeline, "_enforce_write_preflight", lambda _supabase: None)
    monkeypatch.setattr(match_pipeline, "_process_persisted_matches", lambda **_kwargs: process_calls.append(True) or {"inserted": 1})
    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"id": 10}]))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "new-1", "date": "2026-01-01T12:00:00+00:00", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result.get("idempotent_hit") is not True
    assert result["match_id"] == 10
    assert process_calls


def test_record_match_blocks_writes_when_preflight_fails(monkeypatch):
    supabase = _Supabase(existing_rows=[])

    monkeypatch.setattr(match_pipeline, "_enforce_write_preflight", lambda _supabase: (_ for _ in ()).throw(RuntimeError("read-only mode")))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    with pytest.raises(RuntimeError, match="read-only mode"):
        match_pipeline.record_match(
            supabase=supabase,
            club_id="club-1",
            match_payload={"idempotency_key": "new-2", "date": "2026-01-01T12:00:00+00:00", "score_t1": 11, "score_t2": 9},
        )


def test_record_match_missing_date_raises(monkeypatch):
    supabase = _Supabase(existing_rows=[])
    monkeypatch.setattr(match_pipeline, "_enforce_write_preflight", lambda _supabase: None)

    with pytest.raises(match_pipeline.MissingMatchDatetime):
        match_pipeline.record_match(
            supabase=supabase,
            club_id="club-1",
            match_payload={"idempotency_key": "missing-date", "score_t1": 11, "score_t2": 9},
        )
