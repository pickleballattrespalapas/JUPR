from __future__ import annotations

import pandas as pd

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

    monkeypatch.setattr(match_pipeline, "process_matches", lambda *_args, **_kwargs: process_calls.append(True))
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "dup-1", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result["idempotent_hit"] is True
    assert result["match_id"] == 9
    assert process_calls == []


def test_record_match_new_idempotency_processes_matches(monkeypatch):
    supabase = _Supabase(existing_rows=[])

    monkeypatch.setattr(match_pipeline, "_build_processing_context", lambda **_: {
        "name_to_id": {},
        "df_players_all": pd.DataFrame([]),
        "df_leagues": pd.DataFrame([]),
        "df_meta": pd.DataFrame([]),
    })

    def fake_process(match_list, **kwargs):
        row = dict(match_list[0])
        row["club_id"] = "club-1"
        kwargs["match_writer"](row, None, "admin", "new-1")
        return {"inserted": 1}

    monkeypatch.setattr(match_pipeline, "process_matches", fake_process)
    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: type("R", (), {"data": [{"id": 10}]})())
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    result = match_pipeline.record_match(
        supabase=supabase,
        club_id="club-1",
        match_payload={"idempotency_key": "new-1", "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is True
    assert result.get("idempotent_hit") is not True
    assert result["match_id"] == 10
