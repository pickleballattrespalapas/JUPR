from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from postgrest.exceptions import APIError

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.replay_lock import ReplayAlreadyRunningError, acquire_replay_lock


class _TableQuery:
    def __init__(self, supabase: "_FakeSupabase", table: str):
        self.supabase = supabase
        self.table = table
        self.filters: list[tuple[str, object]] = []
        self._order: list[tuple[str, bool]] = []
        self._payload: dict | None = None
        self._mode = "select"

    def select(self, _fields: str):
        self._mode = "select"
        return self

    def eq(self, column: str, value):
        self.filters.append((column, value))
        return self

    def order(self, column: str, desc: bool = False):
        self._order.append((column, desc))
        return self

    def update(self, payload: dict):
        self._mode = "update"
        self._payload = dict(payload)
        return self

    def delete(self):
        self._mode = "delete"
        return self

    def insert(self, payload: dict):
        self._mode = "insert"
        self._payload = dict(payload)
        return self

    def execute(self):
        rows = self.supabase.storage.setdefault(self.table, [])
        if self._mode == "insert":
            if self.table == "replay_lock":
                club_id = str(self._payload.get("club_id"))
                if any(str(r.get("club_id")) == club_id for r in rows):
                    raise APIError({"code": "23505", "message": "duplicate key"})
            rows.append(dict(self._payload or {}))
            return SimpleNamespace(data=[dict(self._payload or {})])

        filtered = [dict(r) for r in rows if all(str(r.get(c)) == str(v) for c, v in self.filters)]

        if self._mode == "delete":
            self.supabase.storage[self.table] = [
                dict(r)
                for r in rows
                if not all(str(r.get(c)) == str(v) for c, v in self.filters)
            ]
            return SimpleNamespace(data=filtered)

        if self._mode == "update":
            updated = []
            for row in rows:
                if all(str(row.get(c)) == str(v) for c, v in self.filters):
                    row.update(self._payload or {})
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)

        for column, desc in reversed(self._order):
            filtered = sorted(filtered, key=lambda r: str(r.get(column) or ""), reverse=desc)
        return SimpleNamespace(data=filtered)


class _RpcCall:
    def __init__(self, supabase: "_FakeSupabase", name: str, payload: dict):
        self.supabase = supabase
        self.name = name
        self.payload = payload

    def execute(self):
        if self.name != "replace_league_ratings":
            return SimpleNamespace(data=[])

        club_id = str(self.payload["p_club_id"])
        if self.payload.get("p_reset"):
            self.supabase.storage["league_ratings"] = [
                row
                for row in self.supabase.storage.get("league_ratings", [])
                if str(row.get("club_id")) != club_id
            ]
        self.supabase.storage.setdefault("league_ratings", []).extend(
            [dict(row) for row in self.payload.get("p_rows", [])]
        )
        return SimpleNamespace(data=[])


class _FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [
                {"id": 1, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 2, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 3, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
                {"id": 4, "club_id": "club-1", "starting_rating": 1200.0, "rating": 1200.0},
            ],
            "matches": [
                {
                    "id": 1,
                    "club_id": "club-1",
                    "date": "2026-01-01T00:00:00+00:00",
                    "league": "A League",
                    "match_type": "League",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 8,
                },
                {
                    "id": 2,
                    "club_id": "club-1",
                    "date": "2026-01-02T00:00:00+00:00",
                    "league": "A League",
                    "match_type": "League",
                    "t1_p1": 1,
                    "t1_p2": 3,
                    "t2_p1": 2,
                    "t2_p2": 4,
                    "score_t1": 9,
                    "score_t2": 11,
                },
                {
                    "id": 3,
                    "club_id": "club-1",
                    "date": "2026-01-03T00:00:00+00:00",
                    "league": "A League",
                    "match_type": "League",
                    "t1_p1": 1,
                    "t1_p2": 4,
                    "t2_p1": 2,
                    "t2_p2": 3,
                    "score_t1": 11,
                    "score_t2": 6,
                },
            ],
            "league_ratings": [],
            "replay_lock": [],
        }

    def table(self, name: str):
        return _TableQuery(self, name)

    def rpc(self, name: str, payload: dict):
        return _RpcCall(self, name, payload)


def _hash_league_ratings(rows: list[dict]) -> str:
    normalized = [
        {
            "club_id": str(r["club_id"]),
            "league_name": str(r["league_name"]),
            "player_id": int(r["player_id"]),
            "rating": round(float(r["rating"]), 6),
            "wins": int(r["wins"]),
            "losses": int(r["losses"]),
            "matches_played": int(r["matches_played"]),
            "starting_rating": round(float(r["starting_rating"]), 6),
        }
        for r in rows
    ]
    payload = json.dumps(sorted(normalized, key=lambda r: (r["league_name"], r["player_id"])), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_replay_twice_same_result():
    supabase = _FakeSupabase()

    replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)
    first_hash = _hash_league_ratings(supabase.storage["league_ratings"])

    replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)
    second_hash = _hash_league_ratings(supabase.storage["league_ratings"])

    assert first_hash == second_hash


def test_snapshot_consistency():
    supabase = _FakeSupabase()

    replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)

    ratings = {1: 1200.0, 2: 1200.0, 3: 1200.0, 4: 1200.0}
    ordered_matches = sorted(supabase.storage["matches"], key=lambda m: (str(m["date"]), int(m["id"])))

    for match in ordered_matches:
        p1, p2, p3, p4 = int(match["t1_p1"]), int(match["t1_p2"]), int(match["t2_p1"]), int(match["t2_p2"])
        s1, s2 = int(match["score_t1"]), int(match["score_t2"])

        sr1, sr2, sr3, sr4 = ratings[p1], ratings[p2], ratings[p3], ratings[p4]
        d1, d2 = calculate_hybrid_elo((sr1 + sr2) / 2, (sr3 + sr4) / 2, s1, s2)

        assert round(float(match["t1_p1_r"]), 6) == round(sr1, 6)
        assert round(float(match["t1_p2_r"]), 6) == round(sr2, 6)
        assert round(float(match["t2_p1_r"]), 6) == round(sr3, 6)
        assert round(float(match["t2_p2_r"]), 6) == round(sr4, 6)

        ratings[p1] += float(d1)
        ratings[p2] += float(d1)
        ratings[p3] += float(d2)
        ratings[p4] += float(d2)

        assert round(float(match["t1_p1_r_end"]), 6) == round(ratings[p1], 6)
        assert round(float(match["t1_p2_r_end"]), 6) == round(ratings[p2], 6)
        assert round(float(match["t2_p1_r_end"]), 6) == round(ratings[p3], 6)
        assert round(float(match["t2_p2_r_end"]), 6) == round(ratings[p4], 6)


def test_lock_prevents_concurrent():
    supabase = _FakeSupabase()

    acquire_replay_lock(supabase, "club-1")

    try:
        acquire_replay_lock(supabase, "club-1")
        raised = False
    except ReplayAlreadyRunningError:
        raised = True

    assert raised


def test_replay_history_enforces_running_lock():
    supabase = _FakeSupabase()
    supabase.storage["replay_lock"] = [{"club_id": "club-1", "status": "running"}]

    try:
        replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)
        raised = False
    except RuntimeError:
        raised = True

    assert raised


def test_replay_history_updates_lock_and_summary(monkeypatch):
    supabase = _FakeSupabase()
    summary = replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)

    assert summary["log_summary"]["lock"]["final_status"] == "success"
    assert supabase.storage["replay_lock"][0]["status"] == "success"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr("jupr_app.domain.replay_history.sb_rpc", _boom)
    try:
        replay_history(supabase=supabase, club_id="club-1", df_meta=None, target_reset=FULL_RESET_LABEL)
    except RuntimeError:
        pass

    assert supabase.storage["replay_lock"][0]["status"] == "failed"
