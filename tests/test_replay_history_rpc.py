from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain import replay_history


class _Query:
    def __init__(self, supabase: "_Supabase", table: str):
        self.supabase = supabase
        self.table = table
        self.filters: list[tuple[str, object]] = []
        self._mode = "select"
        self._payload: dict | None = None

    def select(self, _fields: str):
        return self

    def eq(self, col: str, value):
        self.filters.append((col, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def insert(self, payload: dict):
        self._mode = "insert"
        self._payload = dict(payload)
        return self

    def update(self, payload: dict):
        self._mode = "update"
        self._payload = dict(payload)
        return self

    def delete(self):
        self._mode = "delete"
        return self

    def execute(self):
        rows = self.supabase.tables.setdefault(self.table, [])

        if self._mode == "insert":
            rows.append(dict(self._payload or {}))
            return SimpleNamespace(data=[dict(self._payload or {})])

        if self._mode == "update":
            updated = []
            for row in rows:
                if all(row.get(col) == val for col, val in self.filters):
                    row.update(self._payload or {})
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)

        if self._mode == "delete":
            kept = [dict(r) for r in rows if not all(r.get(col) == val for col, val in self.filters)]
            self.supabase.tables[self.table] = kept
            return SimpleNamespace(data=[])

        rows = [dict(row) for row in rows]
        for col, val in self.filters:
            rows = [r for r in rows if r.get(col) == val]
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self):
        self.tables = {
            "players": [
                {"id": 2, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 1, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 3, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 4, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
            ],
            "matches": [
                {
                    "id": 10,
                    "club_id": "club-1",
                    "date": "2026-01-01T00:00:00+00:00",
                    "league": "B League",
                    "match_type": "League",
                    "t1_p1": 2,
                    "t1_p2": 1,
                    "t2_p1": 4,
                    "t2_p2": 3,
                    "score_t1": 11,
                    "score_t2": 8,
                }
            ],
            "replay_lock": [],
        }
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, name: str):
        return _Query(self, name)

    def rpc(self, name: str, payload: dict):
        self.rpc_calls.append((name, payload))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))


def test_chunk_rows_by_payload_limit_splits_large_payload():
    rows = [
        {"league_name": "A", "player_id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 1, "starting_rating": 1200.0},
        {"league_name": "A", "player_id": 2, "rating": 1201.0, "wins": 1, "losses": 0, "matches_played": 1, "starting_rating": 1200.0},
    ]

    chunks = replay_history._chunk_rows_by_payload_limit(rows, max_payload_bytes=150)

    assert len(chunks) == 2
    assert chunks[0][0]["player_id"] == 1
    assert chunks[1][0]["player_id"] == 2


def test_replay_history_uses_replace_league_ratings_rpc(monkeypatch):
    supabase = _Supabase()

    monkeypatch.setattr(replay_history, "sb_update", lambda *args, **kwargs: SimpleNamespace(data=[]))

    replay_history.replay_history(
        supabase=supabase,
        club_id="club-1",
        df_meta=pd.DataFrame([{"league_name": "B League", "k_factor": 24}]),
        target_reset="B League",
    )

    assert len(supabase.rpc_calls) == 1
    rpc_name, payload = supabase.rpc_calls[0]
    assert rpc_name == "replace_league_ratings"
    assert payload["p_club_id"] == "club-1"
    assert payload["p_reset"] is True
    assert [row["player_id"] for row in payload["p_rows"]] == [1, 2, 3, 4]
