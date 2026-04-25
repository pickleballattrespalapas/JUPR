from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.data.load import load_data
from jupr_app.domain.bulk_match_editor import apply_bulk_match_edits, compute_recompute_scope
from jupr_app.domain.match_delete import delete_rated_matches_with_replay
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self._op = "select"
        self._payload = None
        self._select_cols = None
        self._filters = []
        self._order = []
        self._range = None
        self._limit = None

    def select(self, _cols):
        self._op = "select"
        self._select_cols = _cols
        return self

    def insert(self, payload, returning=None):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def delete(self, returning=None):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def neq(self, col, val):
        self._filters.append(("neq", col, val))
        return self

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, set(vals)))
        return self

    def or_(self, expr):
        self._filters.append(("or", expr, None))
        return self

    def order(self, col, desc=False):
        self._order.append((col, desc))
        return self

    def range(self, start, end):
        self._range = (int(start), int(end))
        return self

    def limit(self, n):
        self._limit = int(n)
        return self

    def execute(self):
        return self.sb.execute(self)


class _Supabase:
    def __init__(self):
        self.tables = {
            "matches": [],
            "players": [],
            "league_ratings": [],
            "leagues_metadata": [],
            "badges": [],
            "player_badges": [],
            "admin_audit_events": [],
        }
        self.rpc_calls = []

    def table(self, name):
        return _Query(self, name)

    def rpc(self, name, payload):
        self.rpc_calls.append((name, payload))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=len(payload.get("rows", []))))

    def execute(self, q: _Query):
        rows = self.tables.setdefault(q.table, [])
        data = list(rows)

        for op, col, val in q._filters:
            if op == "eq":
                data = [r for r in data if str(r.get(col)) == str(val)]
            elif op == "neq":
                data = [r for r in data if str(r.get(col)) != str(val)]
            elif op == "is":
                data = [r for r in data if r.get(col) is val]
            elif op == "in":
                data = [r for r in data if r.get(col) in val]
            elif op == "or":
                parts = [p.strip() for p in col.split(",") if p.strip()]
                keep = []
                for row in data:
                    matched = False
                    for part in parts:
                        field, _, raw = part.partition(".eq.")
                        if str(row.get(field)) == raw:
                            matched = True
                            break
                    if matched:
                        keep.append(row)
                data = keep

        for col, desc in reversed(q._order):
            data = sorted(data, key=lambda r: r.get(col), reverse=desc)

        if q._range is not None:
            s, e = q._range
            data = data[s : e + 1]

        if q._limit is not None:
            data = data[: q._limit]

        if q._op == "select":
            return SimpleNamespace(data=data)
        if q._op == "insert":
            payload = q._payload if isinstance(q._payload, list) else [q._payload]
            for row in payload:
                rows.append(dict(row))
            return SimpleNamespace(data=payload)
        if q._op == "update":
            for row in data:
                row.update(dict(q._payload))
            return SimpleNamespace(data=data)
        if q._op == "delete":
            kept = [row for row in rows if row not in data]
            self.tables[q.table] = kept
            return SimpleNamespace(data=data)

        return SimpleNamespace(data=[])


class _StrictSchemaSupabase(_Supabase):
    def __init__(self, missing_match_columns: set[str]):
        super().__init__()
        self.missing_match_columns = set(missing_match_columns)

    def execute(self, q: _Query):
        if q._op == "select" and q.table == "matches" and q._select_cols:
            requested = {c.strip() for c in str(q._select_cols).split(",")}
            missing = requested.intersection(self.missing_match_columns)
            if missing:
                raise RuntimeError(f"column does not exist: {sorted(missing)}")
        return super().execute(q)


def _seed_players():
    return [
        {"id": 1, "club_id": "club", "name": "A", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 2, "club_id": "club", "name": "B", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 3, "club_id": "club", "name": "C", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 4, "club_id": "club", "name": "D", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
    ]


def test_replay_history_ignores_soft_deleted_matches():
    sb = _Supabase()
    sb.tables["players"] = _seed_players()
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "date": "2024-01-01T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 5, "deleted_at": None},
        {"id": 2, "club_id": "club", "date": "2024-01-02T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 0, "deleted_at": "2026-01-01T00:00:00Z"},
    ]

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    assert result["matches_rewritten"] == 1


def test_replay_history_missing_rating_scope_still_excludes_soft_deleted():
    sb = _StrictSchemaSupabase(missing_match_columns={"rating_scope"})
    sb.tables["players"] = _seed_players()
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "date": "2024-01-01T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 5, "deleted_at": None},
        {"id": 2, "club_id": "club", "date": "2024-01-02T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 0, "deleted_at": "2026-01-01T00:00:00Z"},
    ]

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    assert result["matches_rewritten"] == 1


def test_replay_history_includes_null_rating_scope_but_skips_unrated():
    sb = _Supabase()
    sb.tables["players"] = _seed_players()
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "date": "2024-01-01T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 5, "deleted_at": None, "rating_scope": None},
        {"id": 2, "club_id": "club", "date": "2024-01-02T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 7, "deleted_at": None, "rating_scope": "unrated"},
    ]

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    assert result["matches_rewritten"] == 1


def test_delete_rated_matches_soft_deletes_and_replays(monkeypatch):
    sb = _Supabase()
    sb.tables["matches"] = [
        {"id": 10, "club_id": "club", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "deleted_at": None}
    ]

    monkeypatch.setattr("jupr_app.domain.match_delete.recompute_last_game_at_for_players", lambda **kwargs: None)
    monkeypatch.setattr("jupr_app.domain.match_delete.replay_history", lambda **kwargs: {"ok": True})

    result = delete_rated_matches_with_replay(
        supabase=sb,
        club_id="club",
        match_ids=[10],
        df_meta=pd.DataFrame(),
        actor="admin",
        source="match_log",
    )

    assert result["deleted_count"] == 1
    assert result["replay_result"] == {"ok": True}
    assert sb.tables["matches"][0]["deleted_at"] is not None


def test_bulk_scope_marks_scores_and_players_as_rating_affecting():
    scope = compute_recompute_scope([
        {"id": 1, "score_t1": 11},
        {"id": 2, "t1_p1": 99},
    ])
    assert scope == {"standings": True, "ratings": True}


def test_bulk_editor_validates_duplicate_players_and_negative_scores(monkeypatch):
    sb = _Supabase()
    sb.tables["players"] = _seed_players()
    sb.tables["matches"] = [
        {
            "id": 1,
            "club_id": "club",
            "league": "Main",
            "date": "2024-01-01T00:00:00Z",
            "week_tag": "Week 1",
            "match_type": "League",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 7,
        }
    ]

    monkeypatch.setattr("jupr_app.domain.bulk_match_editor.enqueue_badge_eval", lambda *a, **k: {"queued": False})
    monkeypatch.setattr("jupr_app.domain.bulk_match_editor.run_live_badge_awards", lambda *a, **k: {"mode": "inline"})

    with pytest.raises(ValueError, match="duplicate player"):
        apply_bulk_match_edits(sb, "club", [{"id": 1, "t1_p1": 2}], actor="admin")

    with pytest.raises(ValueError, match="cannot be negative"):
        apply_bulk_match_edits(sb, "club", [{"id": 1, "score_t1": -1}], actor="admin")

    with pytest.raises(ValueError, match="not in this club"):
        apply_bulk_match_edits(sb, "club", [{"id": 1, "t2_p2": 999}], actor="admin")


def test_bulk_editor_recomputes_last_game_for_removed_and_added_players(monkeypatch):
    sb = _Supabase()
    sb.tables["players"] = _seed_players() + [
        {"id": 5, "club_id": "club", "name": "E", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
    ]
    sb.tables["matches"] = [
        {
            "id": 1,
            "club_id": "club",
            "league": "Main",
            "date": "2024-01-01T00:00:00Z",
            "week_tag": "Week 1",
            "match_type": "League",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 7,
        }
    ]
    seen = {}

    monkeypatch.setattr("jupr_app.domain.bulk_match_editor.enqueue_badge_eval", lambda *a, **k: {"queued": False})
    monkeypatch.setattr("jupr_app.domain.bulk_match_editor.run_live_badge_awards", lambda *a, **k: {"mode": "inline"})

    def _capture_recompute(**kwargs):
        seen["player_ids"] = set(kwargs["player_ids"])

    monkeypatch.setattr("jupr_app.domain.player_activity.recompute_last_game_at_for_players", _capture_recompute)

    result = apply_bulk_match_edits(sb, "club", [{"id": 1, "t1_p1": 5}], actor="admin")

    assert result["updated_count"] == 1
    assert seen["player_ids"] == {1, 2, 3, 4, 5}


class _LoadSpyQuery(_Query):
    def __init__(self, sb, table):
        super().__init__(sb, table)
        self.sb = sb

    def is_(self, col, val):
        if self.table == "matches" and col == "deleted_at" and val is None:
            self.sb.called_matches_soft_filter = True
        return super().is_(col, val)


class _LoadSpySupabase(_Supabase):
    def __init__(self):
        super().__init__()
        self.called_matches_soft_filter = False

    def table(self, name):
        return _LoadSpyQuery(self, name)


def test_load_data_excludes_deleted_matches(monkeypatch):
    monkeypatch.setenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", "1")
    sb = _LoadSpySupabase()
    load_data(sb, "club", match_limit=5)
    assert sb.called_matches_soft_filter is True
