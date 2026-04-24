from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.ui.live.shared import (
    _append_roster_names,
    _create_and_resolve_admin_players,
    _rows_from_admin_editor_df,
)
from jupr_app.ui.pages.jupr_live_admin import ADMIN_CONFIG


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self._op = "select"
        self._payload = None
        self._filters = []

    def select(self, _cols):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def limit(self, _n):
        return self

    def execute(self):
        rows = self.sb.tables.setdefault(self.table, [])
        if self._op == "insert":
            payload = dict(self._payload)
            payload.setdefault("id", self.sb.next_player_id)
            self.sb.next_player_id += 1
            rows.append(payload)
            return SimpleNamespace(data=[payload])
        data = list(rows)
        for col, val in self._filters:
            data = [r for r in data if str(r.get(col)) == str(val)]
        return SimpleNamespace(data=data)


class _Supabase:
    def __init__(self):
        self.tables = {"players": []}
        self.next_player_id = 100

    def table(self, name):
        return _Query(self, name)


def _ctx(players: list[dict]):
    sb = _Supabase()
    sb.tables["players"] = [dict(p) for p in players]
    return SimpleNamespace(
        supabase=sb,
        club_id="club-1",
        name_to_id={str(p["name"]): int(p["id"]) for p in players},
    )


def test_admin_config_uses_ordered_roster_builder():
    assert ADMIN_CONFIG.requires_roster_resolution is True
    assert ADMIN_CONFIG.use_admin_roster_builder is True


def test_append_roster_names_preserves_custom_order_and_appends_existing_players():
    player_name_to_id = {"Amy": 1, "Brooke": 2, "Chris": 3}
    rows = _append_roster_names(
        [],
        ["Brooke", "Zoe"],
        player_name_to_id=player_name_to_id,
        default_new_player_rating=3.5,
    )
    rows = _append_roster_names(
        rows,
        ["Amy"],
        player_name_to_id=player_name_to_id,
        default_new_player_rating=3.5,
    )
    assert [r["display_name"] for r in rows] == ["Brooke", "Zoe", "Amy"]
    assert [r["order"] for r in rows] == [1, 2, 3]


def test_rows_from_editor_preserve_admin_order():
    player_name_to_id = {"Amy": 1, "Brooke": 2}
    df = pd.DataFrame(
        [
            {"Order": 2, "Name": "Amy", "Resolution": "existing_player", "Matched Player": "Amy", "Starting JUPR": 3.5},
            {"Order": 1, "Name": "Zoe", "Resolution": "create_new_player", "Matched Player": "", "Starting JUPR": 3.8},
        ]
    )
    rows = _rows_from_admin_editor_df(df, player_name_to_id=player_name_to_id)
    assert [r["display_name"] for r in rows] == ["Zoe", "Amy"]


def test_create_and_resolve_mixed_existing_and_new_players(monkeypatch):
    ctx = _ctx([
        {"id": 1, "club_id": "club-1", "name": "Amy"},
        {"id": 2, "club_id": "club-1", "name": "Brooke"},
    ])
    created_calls = []

    def _safe_add_player(**kwargs):
        created_calls.append(dict(kwargs))
        ctx.supabase.table("players").insert(
            {
                "club_id": kwargs["club_id"],
                "name": kwargs["name"],
                "rating": float(kwargs["rating_jupr"]) * 400.0,
                "starting_rating": float(kwargs["rating_jupr"]) * 400.0,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "active": True,
            }
        ).execute()
        return True, ""

    monkeypatch.setattr("jupr_app.ui.live.shared.safe_add_player", _safe_add_player)
    player_name_to_id = {"Amy": 1, "Brooke": 2}
    participant_names, resolved_ids, review_messages, created_names = _create_and_resolve_admin_players(
        ctx,
        roster_rows=[
            {"order": 2, "display_name": "Amy", "resolution_status": "existing_player", "player_id": 1},
            {"order": 1, "display_name": "Zoe", "resolution_status": "create_new_player", "starting_jupr_rating": 3.7},
        ],
        default_new_player_rating=3.5,
        player_name_to_id=player_name_to_id,
    )

    assert review_messages == []
    assert participant_names == ["Zoe", "Amy"]
    assert resolved_ids["Amy"] == 1
    assert resolved_ids["Zoe"] >= 100
    assert created_names == ["Zoe"]
    assert created_calls[0]["rating_jupr"] == 3.7


def test_needs_review_requires_explicit_selection(monkeypatch):
    ctx = _ctx([{"id": 1, "club_id": "club-1", "name": "Lance Zonneveld Jr."}])
    monkeypatch.setattr("jupr_app.ui.live.shared.safe_add_player", lambda **kwargs: (True, ""))
    participant_names, resolved_ids, review_messages, _ = _create_and_resolve_admin_players(
        ctx,
        roster_rows=[
            {
                "order": 1,
                "display_name": "Lance Zonneveld",
                "resolution_status": "needs_review",
                "selected_existing_name": "Lance Zonneveld Jr.",
            }
        ],
        default_new_player_rating=3.5,
        player_name_to_id={"Lance Zonneveld Jr.": 1},
    )
    assert participant_names == ["Lance Zonneveld"]
    assert resolved_ids == {}
    assert len(review_messages) == 1
    assert "Review roster" in review_messages[0]


def test_existing_player_only_roster_still_resolves_without_creation(monkeypatch):
    ctx = _ctx([
        {"id": 1, "club_id": "club-1", "name": "Amy"},
        {"id": 2, "club_id": "club-1", "name": "Brooke"},
    ])
    monkeypatch.setattr("jupr_app.ui.live.shared.safe_add_player", lambda **kwargs: (False, "should not be called"))
    participant_names, resolved_ids, review_messages, created_names = _create_and_resolve_admin_players(
        ctx,
        roster_rows=[
            {"order": 1, "display_name": "Brooke", "resolution_status": "existing_player", "player_id": 2},
            {"order": 2, "display_name": "Amy", "resolution_status": "existing_player", "player_id": 1},
        ],
        default_new_player_rating=3.5,
        player_name_to_id={"Amy": 1, "Brooke": 2},
    )
    assert participant_names == ["Brooke", "Amy"]
    assert resolved_ids == {"Brooke": 2, "Amy": 1}
    assert review_messages == []
    assert created_names == []
