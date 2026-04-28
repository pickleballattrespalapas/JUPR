from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_UNSUBSCRIBED,
    mark_unsubscribed,
)


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self._op = "select"
        self._payload = None
        self._filters = []

    def select(self, *_args, **_kwargs):
        self._op = "select"
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

    def execute(self):
        return self.sb.execute(self)


class _Supabase:
    def __init__(self):
        self.tables = {"player_profile_update_subscriptions": []}

    def table(self, name):
        return _Query(self, name)

    def execute(self, q: _Query):
        rows = self.tables.setdefault(q.table, [])
        matched = list(rows)
        for op, col, val in q._filters:
            if op == "eq":
                matched = [r for r in matched if str(r.get(col)) == str(val)]
            elif op == "in":
                matched = [r for r in matched if r.get(col) in val]

        if q._op == "select":
            return SimpleNamespace(data=matched)
        if q._op == "update":
            for row in matched:
                row.update(q._payload)
            return SimpleNamespace(data=matched)
        return SimpleNamespace(data=[])


def test_mark_unsubscribed_sets_status_and_timestamp():
    sb = _Supabase()
    sb.tables["player_profile_update_subscriptions"] = [
        {"id": "sub-1", "request_status": REQUEST_STATUS_ACTIVE},
    ]

    row = mark_unsubscribed(sb, "sub-1")

    assert row["request_status"] == REQUEST_STATUS_UNSUBSCRIBED
    assert row.get("unsubscribed_at")


def test_mark_unsubscribed_rejects_already_unsubscribed_rows():
    sb = _Supabase()
    sb.tables["player_profile_update_subscriptions"] = [
        {"id": "sub-1", "request_status": REQUEST_STATUS_UNSUBSCRIBED},
    ]

    try:
        mark_unsubscribed(sb, "sub-1")
        assert False, "Expected already-unsubscribed rows to be ignored and raise"
    except RuntimeError as exc:
        assert "could not be marked unsubscribed" in str(exc).lower()
