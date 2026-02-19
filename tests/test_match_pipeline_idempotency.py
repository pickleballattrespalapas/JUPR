from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain import match_pipeline


class _Query:
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


class _Supabase:
    def __init__(self, existing_rows=None):
        self._existing_rows = existing_rows or []

    def table(self, name: str):
        assert name == "matches"
        return _Query(self._existing_rows)


def test_record_match_idempotency_hit_returns_existing_and_skips_rebuild(monkeypatch):
    supabase = _Supabase(existing_rows=[{"id": 9, "idempotency_key": "dup-1", "club_id": "club-1"}])
    rebuild_calls = []

    monkeypatch.setattr(match_pipeline, "_rebuild_state", lambda **_: rebuild_calls.append(True))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "dup-1", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result["idempotent_hit"] is True
    assert result["match_id"] == 9
    assert rebuild_calls == []


def test_record_match_new_idempotency_inserts_and_rebuilds(monkeypatch):
    supabase = _Supabase(existing_rows=[])
    rebuild_calls = []

    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "_snapshot_ratings_state", lambda **_: {"players": [], "league_ratings": []})
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"id": 10}]))
    monkeypatch.setattr(match_pipeline, "_rebuild_state", lambda **_: rebuild_calls.append(True) or {"matches_processed": 1})
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "new-1", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result.get("idempotent_hit") is not True
    assert result["match_id"] == 10
    assert rebuild_calls == [True]
