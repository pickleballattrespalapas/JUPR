from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from jupr_app.domain.notifications import player_update_sender as sender


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self.filters: list[tuple[str, str, object]] = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column: str, value: object):
        self.filters.append(("eq", column, value))
        return self

    def gte(self, column: str, value: object):
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column: str, value: object):
        self.filters.append(("lte", column, value))
        return self

    def execute(self):
        return self.sb.execute(self)


class _Supabase:
    def __init__(self):
        self.tables = {"matches": []}

    def table(self, name: str):
        return _Query(self, name)

    def execute(self, query: _Query):
        rows = list(self.tables.get(query.table, []))
        for op, column, value in query.filters:
            if op == "eq":
                rows = [row for row in rows if str(row.get(column)) == str(value)]
            elif op == "gte":
                rows = [row for row in rows if str(row.get(column) or "") >= str(value)]
            elif op == "lte":
                rows = [row for row in rows if str(row.get(column) or "") <= str(value)]
        return SimpleNamespace(data=rows)


def test_generate_and_queue_filters_to_subscribers_with_matches(monkeypatch):
    ctx = SimpleNamespace(supabase=object(), club_id="club")
    active_rows = [
        {"id": "s1", "player_id": 1},
        {"id": "s2", "player_id": 2},
        {"id": "s3", "player_id": 3},
    ]

    saved_players: list[int] = []
    queued_players: list[int] = []

    monkeypatch.setattr(sender, "list_active_subscriptions", lambda *_args, **_kwargs: active_rows)
    monkeypatch.setattr(sender, "get_player_ids_with_matches_in_range", lambda *_args, **_kwargs: {1, 3})
    monkeypatch.setattr(
        sender,
        "_save_digest_for_subscription",
        lambda _ctx, *, subscription, start_date, end_date: saved_players.append(int(subscription["player_id"])),
    )
    monkeypatch.setattr(
        sender,
        "_queue_outbox_for_subscription",
        lambda _ctx, *, subscription, start_date, end_date: queued_players.append(int(subscription["player_id"])),
    )

    result = sender.generate_and_queue_digests_for_active_subscriptions(
        ctx,
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 28),
        only_players_with_matches=True,
    )

    assert saved_players == [1, 3]
    assert queued_players == [1, 3]
    assert result == {
        "active_subscriptions": 3,
        "players_with_matches": 2,
        "eligible_subscriptions": 2,
        "saved": 2,
        "queued": 2,
        "skipped_no_matches": 1,
        "failed": 0,
    }


def test_generate_and_queue_unfiltered_queues_all_subscribers(monkeypatch):
    ctx = SimpleNamespace(supabase=object(), club_id="club")
    active_rows = [{"id": "s1", "player_id": 1}, {"id": "s2", "player_id": 2}]
    queued_players: list[int] = []

    monkeypatch.setattr(sender, "list_active_subscriptions", lambda *_args, **_kwargs: active_rows)
    monkeypatch.setattr(
        sender,
        "_save_digest_for_subscription",
        lambda _ctx, *, subscription, start_date, end_date: None,
    )
    monkeypatch.setattr(
        sender,
        "_queue_outbox_for_subscription",
        lambda _ctx, *, subscription, start_date, end_date: queued_players.append(int(subscription["player_id"])),
    )

    result = sender.generate_and_queue_digests_for_active_subscriptions(
        ctx,
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 28),
        only_players_with_matches=False,
    )

    assert queued_players == [1, 2]
    assert result["active_subscriptions"] == 2
    assert result["eligible_subscriptions"] == 2
    assert result["players_with_matches"] == 0
    assert result["skipped_no_matches"] == 0


def test_get_player_ids_with_matches_in_range_includes_range_bounds_and_ignores_invalid_ids():
    sb = _Supabase()
    sb.tables["matches"] = [
        {"club_id": "club", "date": "2026-02-10", "t1_p1": 1, "t1_p2": None, "t2_p1": "", "t2_p2": "abc"},
        {"club_id": "club", "date": "2026-02-11T20:15:00Z", "t1_p1": "3", "t1_p2": 4, "t2_p1": None, "t2_p2": " "},
        {"club_id": "club", "date": "2026-02-12T09:00:00Z", "t1_p1": 9, "t1_p2": 9, "t2_p1": 9, "t2_p2": 9},
    ]
    ctx = SimpleNamespace(supabase=sb, club_id="club")

    player_ids = sender.get_player_ids_with_matches_in_range(
        ctx,
        start_date=date(2026, 2, 10),
        end_date=date(2026, 2, 11),
    )

    assert player_ids == {1, 3, 4}
