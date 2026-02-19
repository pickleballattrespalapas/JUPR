from __future__ import annotations

from types import SimpleNamespace

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
    monkeypatch.setattr(match_pipeline, "_snapshot_ratings_state", lambda **_: {"players": [], "league_ratings": []})
    monkeypatch.setattr(match_pipeline, "_restore_ratings_state", lambda **_: writes.append("restore_ratings"))
    monkeypatch.setattr(match_pipeline, "sb_insert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"id": 42}]))
    monkeypatch.setattr(
        match_pipeline,
        "sb_delete",
        lambda *_args, **kwargs: writes.append(("delete", kwargs["filters"])) or SimpleNamespace(data=[{"id": 42}]),
    )
    monkeypatch.setattr(match_pipeline, "_rebuild_state", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))

    result = match_pipeline.record_match(
        supabase=object(),
        club_id="club-1",
        match_payload={"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 9},
    )

    assert result["success"] is False
    assert result["match_id"] == 42
    assert result["error"] is not None
    assert "ratings_restored_from_snapshot" in result["warnings"]
    assert ("delete", {"club_id": "club-1", "id": 42}) in writes
    assert "restore_ratings" in writes


def test_update_match_rolls_back_patch_and_returns_structured_error(monkeypatch):
    writes = []
    supabase = _DummySupabase(match_rows=[{"id": 8, "score_t1": 11, "score_t2": 9, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4}])

    monkeypatch.setattr(match_pipeline, "_run_write", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "_snapshot_ratings_state", lambda **_: {"players": [], "league_ratings": []})
    monkeypatch.setattr(match_pipeline, "_restore_ratings_state", lambda **_: writes.append("restore_ratings"))

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
    assert "restore_ratings" in writes
    assert writes[0][1] == {"score_t1": 1, "score_t2": 11}
    assert writes[1][2] == {"club_id": "club-1", "id": 8}
