from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.ui.live.shared import (
    _append_roster_names,
    _default_admin_roster_row,
    _default_state,
    _create_and_resolve_admin_players,
    _existing_player_rating_jupr,
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
        df_players_all=pd.DataFrame(players),
    )


def test_admin_config_uses_ordered_roster_builder():
    assert ADMIN_CONFIG.requires_roster_resolution is True
    assert ADMIN_CONFIG.use_admin_roster_builder is True


def test_admin_roster_state_has_no_silent_new_player_rating():
    assert _default_state(ADMIN_CONFIG)["default_new_player_rating"] is None

    row = _default_admin_roster_row(
        _ctx([]),
        "Zoe",
        order=1,
        player_name_to_id={},
        default_new_player_rating=None,
    )
    assert row["resolution_status"] == "create_new_player"
    assert row["starting_jupr_rating"] is None


def test_append_roster_names_preserves_custom_order_and_appends_existing_players():
    player_name_to_id = {"Amy": 1, "Brooke": 2, "Chris": 3}
    rows = _append_roster_names(
        _ctx([]),
        [],
        ["Brooke", "Zoe"],
        player_name_to_id=player_name_to_id,
        default_new_player_rating=3.5,
    )
    rows = _append_roster_names(
        _ctx([]),
        rows,
        ["Amy"],
        player_name_to_id=player_name_to_id,
        default_new_player_rating=3.5,
    )
    assert [r["display_name"] for r in rows] == ["Brooke", "Zoe", "Amy"]
    assert [r["order"] for r in rows] == [1, 2, 3]


def test_rows_from_editor_preserve_admin_order():
    player_name_to_id = {"Amy": 1, "Brooke": 2}
    ctx = _ctx([{"id": 1, "club_id": "club-1", "name": "Amy", "rating": 1720.0}])
    df = pd.DataFrame(
        [
            {"Order": 2, "Name": "Amy", "Resolution": "existing_player", "Matched Player": "Amy", "Current / Starting JUPR": 1.5},
            {"Order": 1, "Name": "Zoe", "Resolution": "create_new_player", "Matched Player": "", "Current / Starting JUPR": 3.8},
        ]
    )
    rows = _rows_from_admin_editor_df(
        df,
        player_name_to_id=player_name_to_id,
        ctx=ctx,
        default_new_player_rating=3.5,
    )
    assert [r["display_name"] for r in rows] == ["Zoe", "Amy"]
    assert rows[1]["starting_jupr_rating"] == 4.3


def test_existing_player_uses_current_rating_from_players_table():
    ctx = _ctx([{"id": 1, "club_id": "club-1", "name": "Richard Bartolowits", "rating": 1720.0}])
    assert _existing_player_rating_jupr(ctx, 1) == 4.3
    row = _default_admin_roster_row(
        ctx,
        "Richard Bartolowits",
        order=1,
        player_name_to_id={"Richard Bartolowits": 1},
        default_new_player_rating=3.5,
    )
    assert row["resolution_status"] == "existing_player"
    assert row["starting_jupr_rating"] == 4.3


def test_new_player_uses_default_rating_in_default_row():
    ctx = _ctx([])
    row = _default_admin_roster_row(
        ctx,
        "Lance Zonneveld",
        order=1,
        player_name_to_id={},
        default_new_player_rating=3.9,
    )
    assert row["resolution_status"] == "create_new_player"
    assert row["starting_jupr_rating"] == 3.9


def test_mixed_roster_keeps_distinct_existing_and_new_ratings():
    ctx = _ctx([
        {"id": 1, "club_id": "club-1", "name": "Kirsten Giacomini", "rating": 1720.0},
    ])
    rows = _append_roster_names(
        ctx,
        [],
        ["Kirsten Giacomini", "Lance Zonneveld"],
        player_name_to_id={"Kirsten Giacomini": 1},
        default_new_player_rating=3.5,
    )
    by_name = {row["display_name"]: row for row in rows}
    assert by_name["Kirsten Giacomini"]["starting_jupr_rating"] == 4.3
    assert by_name["Lance Zonneveld"]["starting_jupr_rating"] == 3.5


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


def test_create_new_player_requires_explicit_starting_rating(monkeypatch):
    ctx = _ctx([])
    created_calls = []
    monkeypatch.setattr(
        "jupr_app.ui.live.shared.safe_add_player",
        lambda **kwargs: created_calls.append(kwargs) or (True, ""),
    )

    participant_names, resolved_ids, review_messages, created_names = (
        _create_and_resolve_admin_players(
            ctx,
            roster_rows=[
                {
                    "order": 1,
                    "display_name": "Zoe",
                    "resolution_status": "create_new_player",
                    "starting_jupr_rating": None,
                }
            ],
            default_new_player_rating=None,
            player_name_to_id={},
        )
    )

    assert participant_names == ["Zoe"]
    assert resolved_ids == {}
    assert created_names == []
    assert created_calls == []
    assert review_messages == [
        "Review roster: Zoe needs an explicit Starting JUPR before a new player can be created."
    ]


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


def test_existing_player_creation_path_ignores_starting_jupr_and_does_not_overwrite(monkeypatch):
    ctx = _ctx([
        {"id": 1, "club_id": "club-1", "name": "Amy", "rating": 1720.0},
    ])
    safe_add_calls = []

    def _safe_add_player(**kwargs):
        safe_add_calls.append(kwargs)
        return True, ""

    monkeypatch.setattr("jupr_app.ui.live.shared.safe_add_player", _safe_add_player)
    _create_and_resolve_admin_players(
        ctx,
        roster_rows=[
            {
                "order": 1,
                "display_name": "Amy",
                "resolution_status": "existing_player",
                "player_id": 1,
                "starting_jupr_rating": 1.0,
            }
        ],
        default_new_player_rating=3.5,
        player_name_to_id={"Amy": 1},
    )
    assert safe_add_calls == []
