from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain import match_pipeline
from jupr_app.domain.match_processing import MissingMatchDatetime, NonCanonicalMatchWrite, UnknownPlayerId, process_matches


class _Query:
    def __init__(self, storage: dict[str, list[dict]], table_name: str):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self._order: list[tuple[str, bool]] = []

    def select(self, _fields: str):
        return self

    def eq(self, column: str, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values):
        self.filters.append(("in", column, set(values)))
        return self

    def order(self, column: str, desc: bool = False):
        self._order.append((column, desc))
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        rows = [dict(r) for r in self.storage.get(self.table_name, [])]
        for op, col, val in self.filters:
            if op == "eq":
                rows = [r for r in rows if str(r.get(col)) == str(val)]
            elif op == "in":
                rows = [r for r in rows if r.get(col) in val]
        for col, desc in self._order:
            rows = sorted(rows, key=lambda r, c=col: r.get(c), reverse=desc)
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self, storage: dict[str, list[dict]]):
        self.storage = storage
        self.postgrest = object()

    def table(self, name: str):
        return _Query(self.storage, name)


def test_process_matches_without_persisted_match_ids_raises():
    supabase = _Supabase({"players": []})
    try:
        process_matches(
            [{"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8, "league": "Open"}],
            supabase=supabase,
            club_id="club-1",
            name_to_id={},
            df_players_all=pd.DataFrame(
                [
                    {"id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
                    {"id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
                    {"id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
                    {"id": 4, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
                ]
            ),
            df_leagues=pd.DataFrame([]),
            df_meta=pd.DataFrame([]),
        )
    except NonCanonicalMatchWrite:
        return

    raise AssertionError("Expected NonCanonicalMatchWrite when process_matches is called with non-persisted rows")


def test_moneyball_ingest_uses_record_match_and_process_path_never_inserts_matches(monkeypatch):
    storage = {
        "matches": [],
        "players": [
            {"club_id": "club-1", "id": 1, "name": "A", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
            {"club_id": "club-1", "id": 2, "name": "B", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
            {"club_id": "club-1", "id": 3, "name": "C", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
            {"club_id": "club-1", "id": 4, "name": "D", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
        ],
        "league_ratings": [],
        "leagues_metadata": [],
        "badge_eval_queue": [],
    }
    supabase = _Supabase(storage)

    inserts_seen: list[str] = []

    def fake_insert(_supabase, table: str, payload):
        inserts_seen.append(table)
        if table != "matches":
            raise AssertionError(f"Unexpected direct insert on table {table}")
        row = dict(payload)
        row["id"] = len(storage["matches"]) + 1
        storage["matches"].append(row)
        return SimpleNamespace(data=[row])

    monkeypatch.setattr(match_pipeline, "_enforce_write_preflight", lambda _supabase: None)
    monkeypatch.setattr(match_pipeline, "sb_insert", fake_insert)
    monkeypatch.setattr("jupr_app.domain.match_processing.sb_update", lambda *_args, **_kwargs: SimpleNamespace(data=[{"ok": True}]))
    monkeypatch.setattr("jupr_app.domain.match_processing.sb_upsert", lambda *_args, **_kwargs: SimpleNamespace(data=[{"ok": True}]))
    monkeypatch.setattr(match_pipeline, "sb_update", lambda *_args, **_kwargs: SimpleNamespace(data=[{"ok": True}]))
    monkeypatch.setattr(match_pipeline, "sb_retry", lambda fn: fn())
    monkeypatch.setattr(match_pipeline, "log_event", lambda **_: None)

    calls = []
    real_record_match = match_pipeline.record_match

    def spy_record_match(*, supabase, club_id, match_payload):
        calls.append((club_id, dict(match_payload)))
        return real_record_match(supabase=supabase, club_id=club_id, match_payload=match_payload)

    monkeypatch.setattr(match_pipeline, "record_match", spy_record_match)

    result = match_pipeline.ingest_and_process_match(
        payload={
            "idempotency_key": "moneyball:evt-1:1",
            "context_type": "moneyball",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 8,
            "date": "2026-01-01T12:00:00+00:00",
            "league": "Moneyball",
        },
        ctx={"supabase": supabase, "club_id": "club-1"},
    )

    assert result["success"] is True
    assert calls and calls[0][0] == "club-1"
    assert inserts_seen == ["matches"]


def test_process_matches_unknown_player_id_fails_without_creating_player(monkeypatch):
    storage = {
        "players": [
            {"club_id": "club-1", "id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
            {"club_id": "club-1", "id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
            {"club_id": "club-1", "id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0, "starting_rating": 1200.0},
        ]
    }
    supabase = _Supabase(storage)

    player_rows_before = len(storage["players"])

    with pytest.raises(UnknownPlayerId):
        process_matches(
            [{"id": 77, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 999, "score_t1": 11, "score_t2": 6, "league": "Open"}],
            supabase=supabase,
            club_id="club-1",
            name_to_id={},
            df_players_all=pd.DataFrame(storage["players"]),
            df_leagues=pd.DataFrame([]),
            df_meta=pd.DataFrame([]),
        )

    assert len(storage["players"]) == player_rows_before


def test_ingestion_with_player_names_creates_players_via_safe_add_player_and_persists(monkeypatch):
    supabase = _Supabase({"matches": []})

    safe_add_calls: list[str] = []

    def fake_safe_add_player(*, supabase, club_id, name, rating_jupr):
        safe_add_calls.append(name)
        return True, 100 + len(safe_add_calls)

    persisted_payloads: list[dict] = []

    def fake_record_match(*, supabase, club_id, match_payload):
        persisted_payloads.append(dict(match_payload))
        return {"success": True, "match_id": 10}

    monkeypatch.setattr(match_pipeline, "safe_add_player", fake_safe_add_player)
    monkeypatch.setattr(match_pipeline, "record_match", fake_record_match)

    result = match_pipeline.ingest_match_with_identity_resolution(
        supabase=supabase,
        club_id="club-1",
        match_payload={
            "idempotency_key": "ingest:names:1",
            "t1_p1": "Alice",
            "t1_p2": "Bob",
            "t2_p1": 3,
            "t2_p2": "4",
            "score_t1": 11,
            "score_t2": 7,
        },
    )

    assert result["success"] is True
    assert safe_add_calls == ["Alice", "Bob"]
    assert persisted_payloads
    assert persisted_payloads[0]["t1_p1"] == 101
    assert persisted_payloads[0]["t1_p2"] == 102
    assert persisted_payloads[0]["t2_p1"] == 3
    assert persisted_payloads[0]["t2_p2"] == 4


def test_process_matches_missing_date_raises():
    supabase = _Supabase({"players": [{"id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0}, {"id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0}, {"id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0}, {"id": 4, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0}]})

    with pytest.raises(MissingMatchDatetime):
        process_matches(
            [{"id": 10, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8, "league": "Open"}],
            supabase=supabase,
            club_id="club-1",
            name_to_id={},
            df_players_all=pd.DataFrame([
                {"id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
                {"id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
                {"id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
                {"id": 4, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            ]),
            df_leagues=pd.DataFrame([]),
            df_meta=pd.DataFrame([]),
        )
