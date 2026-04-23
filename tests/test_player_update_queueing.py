from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    queue_player_updates_for_affected_subscribers,
)


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self._op = "select"
        self._payload = None
        self._filters = []
        self._order = None
        self._limit = None
        self._range = None

    def select(self, *_args, **_kwargs):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, set(vals)))
        return self

    def order(self, col, desc=False):
        self._order = (col, bool(desc))
        return self

    def limit(self, n):
        self._limit = int(n)
        return self

    def range(self, start, end):
        self._range = (int(start), int(end))
        return self

    def execute(self):
        return self.sb.execute(self)


class _Supabase:
    def __init__(self):
        self.tables = {
            "matches": [],
            "players": [],
            "league_ratings": [],
            "player_profile_update_subscriptions": [],
            "player_profile_update_outbox": [],
            "badge_eval_queue": [],
            "player_badges": [],
            "badges": [],
        }

    def table(self, name):
        return _Query(self, name)

    def execute(self, q: _Query):
        rows = self.tables.setdefault(q.table, [])
        data = list(rows)

        for op, col, val in q._filters:
            if op == "eq":
                data = [r for r in data if str(r.get(col)) == str(val)]
            elif op == "in":
                data = [r for r in data if r.get(col) in val]

        if q._order:
            col, desc = q._order
            data = sorted(data, key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)

        if q._range:
            start, end = q._range
            data = data[start : end + 1]

        if q._limit is not None:
            data = data[: q._limit]

        if q._op == "select":
            return SimpleNamespace(data=data)

        if q._op == "insert":
            payload = q._payload if isinstance(q._payload, list) else [q._payload]
            inserted = []
            for row in payload:
                row = dict(row)
                if q.table == "player_profile_update_outbox":
                    duplicate = any(
                        str(cur.get("subscription_id")) == str(row.get("subscription_id"))
                        and str(cur.get("week_start")) == str(row.get("week_start"))
                        and str(cur.get("week_end")) == str(row.get("week_end"))
                        for cur in rows
                    )
                    if duplicate:
                        raise RuntimeError("duplicate key value violates unique constraint")
                row.setdefault("id", len(rows) + 1)
                rows.append(row)
                inserted.append(row)
            return SimpleNamespace(data=inserted)

        if q._op == "update":
            for row in data:
                row.update(q._payload)
            return SimpleNamespace(data=data)

        return SimpleNamespace(data=[])


def _players_df():
    return pd.DataFrame(
        [
            {"id": 1, "name": "A", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 2, "name": "B", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 3, "name": "C", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 4, "name": "D", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        ]
    )


def test_queue_helper_queues_per_subscribed_player_per_week_window():
    sb = _Supabase()
    sb.tables["player_profile_update_subscriptions"] = [
        {"id": "sub-1", "club_id": "club", "player_id": 1, "email": "a@example.com", "request_status": REQUEST_STATUS_ACTIVE},
        {"id": "sub-2", "club_id": "club", "player_id": 2, "email": "b@example.com", "request_status": REQUEST_STATUS_ACTIVE},
    ]

    summary = queue_player_updates_for_affected_subscribers(
        sb,
        club_id="club",
        affected_player_ids=[1, 2, 2, 3],
        match_dates=["2026-02-10", "2026-02-11", date(2026, 2, 17)],
    )

    assert summary["affected_players"] == 3
    assert summary["active_subscriptions"] == 2
    assert summary["week_windows"] == 2
    assert summary["queued"] == 4
    assert summary["already_queued"] == 0
    assert summary["no_active_subscription"] == 1
    assert summary["failed"] == 0
    assert len(sb.tables["player_profile_update_outbox"]) == 4


def test_queue_helper_counts_duplicate_outbox_rows_as_already_queued():
    sb = _Supabase()
    sb.tables["player_profile_update_subscriptions"] = [
        {"id": "sub-1", "club_id": "club", "player_id": 1, "email": "a@example.com", "request_status": REQUEST_STATUS_ACTIVE},
    ]
    sb.tables["player_profile_update_outbox"] = [
        {
            "id": 1,
            "subscription_id": "sub-1",
            "club_id": "club",
            "player_id": 1,
            "week_start": "2026-02-09",
            "week_end": "2026-02-15",
            "email": "a@example.com",
            "send_status": "pending",
        }
    ]

    summary = queue_player_updates_for_affected_subscribers(
        sb,
        club_id="club",
        affected_player_ids=[1],
        match_dates=["2026-02-10"],
    )

    assert summary["queued"] == 0
    assert summary["already_queued"] == 1
    assert summary["failed"] == 0


def test_process_matches_queueing_errors_do_not_break_match_processing(monkeypatch):
    sb = _Supabase()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("queue down")

    monkeypatch.setattr("jupr_app.domain.match_processing.queue_player_updates_for_affected_subscribers", _boom)

    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 8}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )

    assert result["inserted"] == 1
    assert result["player_update_queue"]["mode"] == "error"
    assert "queue down" in result["player_update_queue"]["error"]


def test_process_matches_includes_player_update_queue_summary():
    sb = _Supabase()
    sb.tables["player_profile_update_subscriptions"] = [
        {"id": "sub-1", "club_id": "club", "player_id": 1, "email": "a@example.com", "request_status": REQUEST_STATUS_ACTIVE},
    ]

    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 8, "date": "2026-02-10"}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )

    assert result["player_update_queue"]["mode"] == "queued"
    assert result["player_update_queue"]["queued"] == 1
    assert result["player_update_queue"]["no_active_subscription"] == 3


def test_process_matches_skips_queue_when_no_matches_inserted(monkeypatch):
    sb = _Supabase()
    calls = {"count": 0}

    def _track(*_args, **_kwargs):
        calls["count"] += 1
        return {}

    monkeypatch.setattr("jupr_app.domain.match_processing.queue_player_updates_for_affected_subscribers", _track)

    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 0, "s2": 0}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )

    assert result["inserted"] == 0
    assert calls["count"] == 0
    assert result["player_update_queue"]["mode"] == "skipped"
