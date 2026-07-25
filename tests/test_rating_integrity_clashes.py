from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.data.load import load_data
from jupr_app.domain.bulk_match_editor import apply_bulk_match_edits, compute_recompute_scope
from jupr_app.domain.match_delete import delete_rated_matches_with_replay
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.singles_match_processing import process_singles_matches


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

    def select(self, _cols, *, count=None):
        self._op = "select"
        self._select_cols = _cols
        self._count = count
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
        self.select_cap: int | None = None
        self.select_ranges: list[tuple[str, tuple[int, int] | None]] = []

    def table(self, name):
        return _Query(self, name)

    def rpc(self, name, payload):
        self.rpc_calls.append((name, payload))

        def _execute():
            rows = payload.get("rows", [])
            if name == "bulk_update_match_snapshots":
                table = self.tables["matches"]
                for patch in rows:
                    for row in table:
                        if (
                            str(row.get("club_id")) == str(patch.get("club_id"))
                            and int(row.get("id")) == int(patch.get("id"))
                        ):
                            row.update(dict(patch))
            elif name in {
                "bulk_update_players_stats",
                "bulk_update_player_singles_stats",
            }:
                table = self.tables["players"]
                for patch in rows:
                    for row in table:
                        if (
                            str(row.get("club_id")) == str(patch.get("club_id"))
                            and int(row.get("id")) == int(patch.get("id"))
                        ):
                            row.update(
                                {
                                    key: value
                                    for key, value in patch.items()
                                    if key not in {"id", "club_id"}
                                }
                            )
            return SimpleNamespace(data=len(rows))

        return SimpleNamespace(execute=_execute)

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

        exact_count = len(data)
        if q._range is not None:
            s, e = q._range
            data = data[s : e + 1]

        if q._limit is not None:
            data = data[: q._limit]
        if q._op == "select" and self.select_cap is not None:
            data = data[: self.select_cap]

        if q._op == "select":
            self.select_ranges.append((q.table, q._range))
            return SimpleNamespace(
                data=data,
                count=exact_count if getattr(q, "_count", None) == "exact" else None,
            )
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
    def __init__(
        self,
        missing_match_columns: set[str] | None = None,
        missing_player_columns: set[str] | None = None,
    ):
        super().__init__()
        self.missing_match_columns = set(missing_match_columns or set())
        self.missing_player_columns = set(missing_player_columns or set())

    def execute(self, q: _Query):
        if q._op == "select" and q.table == "matches" and q._select_cols:
            requested = {c.strip() for c in str(q._select_cols).split(",")}
            missing = requested.intersection(self.missing_match_columns)
            if missing:
                raise RuntimeError(f"column does not exist: {sorted(missing)}")
        if q._op == "select" and q.table == "players" and q._select_cols:
            requested = {c.strip() for c in str(q._select_cols).split(",")}
            missing = requested.intersection(self.missing_player_columns)
            if missing:
                raise RuntimeError(f"column does not exist: {sorted(missing)}")
        return super().execute(q)


def _seed_players():
    return [
        {
            "id": player_id,
            "club_id": "club",
            "name": name,
            "rating": 1200,
            "starting_rating": 1200,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "singles_replay_baseline": {
                "rating": 1200,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "last_game_at": None,
            },
        }
        for player_id, name in enumerate(("A", "B", "C", "D"), start=1)
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


def test_replay_history_missing_rating_scope_fails_closed_before_writes():
    sb = _StrictSchemaSupabase(missing_match_columns={"rating_scope"})
    sb.tables["players"] = _seed_players()
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "date": "2024-01-01T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 5, "deleted_at": None},
        {"id": 2, "club_id": "club", "date": "2024-01-02T00:00:00Z", "league": "Main", "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 0, "deleted_at": "2026-01-01T00:00:00Z"},
    ]

    with pytest.raises(
        RuntimeError,
        match="column does not exist.*rating_scope",
    ):
        replay_history(
            supabase=sb,
            club_id="club",
            df_meta=pd.DataFrame(),
            target_reset=FULL_RESET_LABEL,
        )

    assert sb.rpc_calls == []


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


def _singles_replay_players():
    return [
        {
            "id": 1,
            "club_id": "club",
            "name": "A",
            "rating": 1200,
            "starting_rating": 1200,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "singles_rating": 9999,
            "singles_wins": 99,
            "singles_losses": 99,
            "singles_matches_played": 198,
            "singles_replay_baseline": {
                "rating": 1240,
                "wins": 2,
                "losses": 0,
                "matches_played": 2,
                "last_game_at": "2026-04-01T12:00:00Z",
            },
        },
        {
            "id": 2,
            "club_id": "club",
            "name": "B",
            "rating": 1200,
            "starting_rating": 1200,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "singles_rating": 9999,
            "singles_wins": 99,
            "singles_losses": 99,
            "singles_matches_played": 198,
            "singles_replay_baseline": {
                "rating": 1210,
                "wins": 1,
                "losses": 1,
                "matches_played": 2,
                "last_game_at": "2026-04-01T12:00:00Z",
            },
        },
    ]


def _managed_singles_match(
    match_id: int,
    *,
    date: str,
    score_t1: int = 11,
    score_t2: int = 7,
    deleted_at=None,
    rating_scope: str | None = None,
):
    return {
        "id": match_id,
        "club_id": "club",
        "date": date,
        "league": "Singles",
        "match_type": "Singles",
        "match_format": "singles",
        "rating_scope": rating_scope,
        "singles_replay_managed": True,
        "t1_p1": 1,
        "t1_p2": None,
        "t2_p1": 2,
        "t2_p2": None,
        "score_t1": score_t1,
        "score_t2": score_t2,
        "deleted_at": deleted_at,
        "rating_bonus_elo": None,
    }


def test_full_replay_rebuilds_only_managed_active_singles_from_baseline():
    sb = _Supabase()
    sb.tables["players"] = _singles_replay_players()
    sb.tables["matches"] = [
        _managed_singles_match(10, date="2026-05-01T12:00:00Z"),
        _managed_singles_match(
            11,
            date="2026-05-02T12:00:00Z",
            deleted_at="2026-05-03T12:00:00Z",
        ),
        _managed_singles_match(
            12,
            date="2026-05-03T12:00:00Z",
            rating_scope="unrated",
        ),
    ]

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    assert result["singles_replay_supported"] is True
    assert result["singles_players_updated"] == 2
    assert result["singles_matches_rewritten"] == 2
    assert result["singles_active_rated"] == 1
    assert result["singles_active_unrated"] == 1
    assert result["singles_deleted_skipped"] == 1
    player_1 = next(row for row in sb.tables["players"] if row["id"] == 1)
    player_2 = next(row for row in sb.tables["players"] if row["id"] == 2)
    assert player_1["singles_wins"] == 3
    assert player_1["singles_losses"] == 0
    assert player_1["singles_matches_played"] == 3
    assert player_2["singles_wins"] == 1
    assert player_2["singles_losses"] == 2
    assert player_2["singles_matches_played"] == 3
    assert player_1["singles_last_game_at"] == "2026-05-01T12:00:00+00:00"
    assert player_2["singles_last_game_at"] == "2026-05-01T12:00:00+00:00"
    assert sb.tables["matches"][0]["t1_p1_r"] == 1240
    assert sb.tables["matches"][0]["t2_p1_r"] == 1210
    assert sb.tables["matches"][2]["elo_delta"] == 0


def test_full_replay_fails_closed_on_invalid_active_managed_singles():
    sb = _Supabase()
    sb.tables["players"] = _singles_replay_players()
    sb.tables["matches"] = [
        _managed_singles_match(
            10,
            date="2026-05-01T12:00:00Z",
            score_t1=7,
            score_t2=7,
        )
    ]

    with pytest.raises(RuntimeError, match="invalid final score"):
        replay_history(
            supabase=sb,
            club_id="club",
            df_meta=pd.DataFrame(),
            target_reset=FULL_RESET_LABEL,
        )

    assert sb.rpc_calls == []
    assert next(row for row in sb.tables["players"] if row["id"] == 1)[
        "singles_rating"
    ] == 9999


def test_full_replay_resets_players_no_longer_referenced_by_managed_rows():
    sb = _Supabase()
    players = _singles_replay_players()
    players.append(
        {
            "id": 3,
            "club_id": "club",
            "name": "C",
            "rating": 1200,
            "starting_rating": 1200,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "singles_rating": 1800,
            "singles_wins": 9,
            "singles_losses": 1,
            "singles_matches_played": 10,
            "singles_replay_baseline": {
                "rating": 1320,
                "wins": 1,
                "losses": 1,
                "matches_played": 2,
                "last_game_at": "2026-03-01T12:00:00Z",
            },
        }
    )
    sb.tables["players"] = players
    sb.tables["matches"] = [
        _managed_singles_match(10, date="2026-05-01T12:00:00Z")
    ]

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    player_3 = next(row for row in sb.tables["players"] if row["id"] == 3)
    assert result["singles_players_updated"] == 3
    assert (
        player_3["singles_rating"],
        player_3["singles_wins"],
        player_3["singles_losses"],
        player_3["singles_matches_played"],
        player_3["singles_last_game_at"],
    ) == (1320.0, 1, 1, 2, "2026-03-01T12:00:00+00:00")


def test_full_replay_reads_all_exact_pages_before_writing():
    sb = _Supabase()
    players = _singles_replay_players()
    for player_id in range(3, 7):
        players.append(
            {
                "id": player_id,
                "club_id": "club",
                "name": f"P{player_id}",
                "rating": 1200,
                "starting_rating": 1200,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "singles_rating": 9999,
                "singles_wins": 99,
                "singles_losses": 99,
                "singles_matches_played": 198,
                "singles_replay_baseline": {
                    "rating": 1200 + player_id,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "last_game_at": None,
                },
            }
        )
    sb.tables["players"] = players
    sb.tables["matches"] = [
        _managed_singles_match(
            match_id,
            date=f"2026-05-{match_id:02d}T12:00:00Z",
        )
        for match_id in range(10, 15)
    ]
    sb.select_cap = 2

    result = replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    assert result["singles_players_updated"] == 6
    assert result["singles_matches_scanned_total"] == 5
    player_ranges = [
        row_range
        for table, row_range in sb.select_ranges
        if table == "players" and row_range is not None
    ]
    managed_match_ranges = [
        row_range
        for table, row_range in sb.select_ranges
        if table == "matches" and row_range is not None
    ]
    assert player_ranges[:3] == [(0, 999), (2, 1001), (4, 1003)]
    assert managed_match_ranges[-3:] == [(0, 999), (2, 1001), (4, 1003)]


def test_full_replay_fails_closed_when_required_baseline_column_is_unavailable():
    sb = _StrictSchemaSupabase(
        missing_player_columns={"singles_replay_baseline"}
    )
    sb.tables["players"] = _seed_players()

    with pytest.raises(
        RuntimeError,
        match="Managed singles player baselines are required",
    ):
        replay_history(
            supabase=sb,
            club_id="club",
            df_meta=pd.DataFrame(),
            target_reset=FULL_RESET_LABEL,
        )

    assert sb.rpc_calls == []


def test_singles_write_plan_uses_chronological_order_within_one_batch():
    sb = _Supabase()
    sb.tables["players"] = _singles_replay_players()

    result = process_singles_matches(
        [
            {
                "date": "2026-05-02T12:00:00Z",
                "league": "Singles",
                "match_type": "Singles",
                "match_format": "singles",
                "t1_p1": 1,
                "t2_p1": 2,
                "score_t1": 11,
                "score_t2": 8,
            },
            {
                "date": "2026-05-01T12:00:00Z",
                "league": "Singles",
                "match_type": "Singles",
                "match_format": "singles",
                "t1_p1": 2,
                "t2_p1": 1,
                "score_t1": 11,
                "score_t2": 9,
            },
        ],
        supabase=sb,
        club_id="club",
        name_to_id={},
        build_write_plan_only=True,
    )

    assert [
        row["date"] for row in result["write_plan"]["match_rows"]
    ] == [
        "2026-05-01T12:00:00+00:00",
        "2026-05-02T12:00:00+00:00",
    ]


def test_singles_writer_fails_before_insert_when_player_snapshot_is_missing():
    sb = _Supabase()
    sb.tables["players"] = [_singles_replay_players()[0]]

    with pytest.raises(
        RuntimeError,
        match="Authoritative singles player rows are incomplete",
    ):
        process_singles_matches(
            [
                {
                    "date": "2026-05-01T12:00:00Z",
                    "league": "Singles",
                    "match_type": "Singles",
                    "match_format": "singles",
                    "t1_p1": 1,
                    "t2_p1": 2,
                    "score_t1": 11,
                    "score_t2": 8,
                }
            ],
            supabase=sb,
            club_id="club",
            name_to_id={},
        )

    assert sb.tables["matches"] == []


def test_excluding_latest_managed_singles_restores_exact_baseline(monkeypatch):
    sb = _Supabase()
    sb.tables["players"] = _singles_replay_players()
    sb.tables["matches"] = [
        _managed_singles_match(10, date="2026-05-01T12:00:00Z")
    ]
    sb.tables["admin_activity_log"] = []
    replay_history(
        supabase=sb,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )
    assert next(row for row in sb.tables["players"] if row["id"] == 1)[
        "singles_matches_played"
    ] == 3
    monkeypatch.setattr(
        "jupr_app.domain.match_delete.recompute_last_game_at_for_players",
        lambda **_kwargs: None,
    )

    result = delete_rated_matches_with_replay(
        supabase=sb,
        club_id="club",
        match_ids=[10],
        df_meta=pd.DataFrame(),
        actor="admin",
        source="match_log",
    )

    assert result["deleted_count"] == 1
    assert result["replay_error"] is None
    assert result["recovery_required"] is False
    assert result["replay_result"]["singles_replay_supported"] is True
    player_1 = next(row for row in sb.tables["players"] if row["id"] == 1)
    player_2 = next(row for row in sb.tables["players"] if row["id"] == 2)
    assert (
        player_1["singles_rating"],
        player_1["singles_wins"],
        player_1["singles_losses"],
        player_1["singles_matches_played"],
        player_1["singles_last_game_at"],
    ) == (1240.0, 2, 0, 2, "2026-04-01T12:00:00+00:00")
    assert (
        player_2["singles_rating"],
        player_2["singles_wins"],
        player_2["singles_losses"],
        player_2["singles_matches_played"],
        player_2["singles_last_game_at"],
    ) == (1210.0, 1, 1, 2, "2026-04-01T12:00:00+00:00")
    assert sb.tables["matches"][0]["deleted_at"] is not None


def test_legacy_singles_exclusion_fails_before_mutation():
    sb = _Supabase()
    sb.tables["matches"] = [
        {
            **_managed_singles_match(10, date="2026-05-01T12:00:00Z"),
            "singles_replay_managed": False,
        }
    ]

    with pytest.raises(ValueError, match="Legacy singles rows"):
        delete_rated_matches_with_replay(
            supabase=sb,
            club_id="club",
            match_ids=[10],
            df_meta=pd.DataFrame(),
        )

    assert sb.tables["matches"][0]["deleted_at"] is None


def test_match_exclusion_fails_before_mutation_when_recovery_metadata_is_unavailable():
    sb = _StrictSchemaSupabase(missing_match_columns={"singles_replay_managed"})
    sb.tables["matches"] = [
        {
            **_managed_singles_match(10, date="2026-05-01T12:00:00Z"),
            "singles_replay_managed": True,
        }
    ]

    with pytest.raises(ValueError, match="recovery metadata is unavailable"):
        delete_rated_matches_with_replay(
            supabase=sb,
            club_id="club",
            match_ids=[10],
            df_meta=pd.DataFrame(),
        )

    assert sb.tables["matches"][0]["deleted_at"] is None


def test_managed_singles_exclusion_rejects_invalid_baseline_before_mutation():
    sb = _Supabase()
    sb.tables["players"] = _singles_replay_players()
    sb.tables["players"][1]["singles_replay_baseline"]["last_game_at"] = (
        "not-a-timestamp"
    )
    sb.tables["matches"] = [
        _managed_singles_match(10, date="2026-05-01T12:00:00Z")
    ]

    with pytest.raises(ValueError, match="baseline is incomplete"):
        delete_rated_matches_with_replay(
            supabase=sb,
            club_id="club",
            match_ids=[10],
            df_meta=pd.DataFrame(),
        )

    assert sb.tables["matches"][0]["deleted_at"] is None


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
    assert result["recovery_required"] is False
    assert sb.tables["matches"][0]["deleted_at"] is not None


def test_match_exclusion_reports_committed_mutation_when_activity_recovery_fails(
    monkeypatch,
):
    sb = _Supabase()
    sb.tables["matches"] = [
        {
            "id": 10,
            "club_id": "club",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "deleted_at": None,
        }
    ]
    replay_calls = []

    def fail_activity_recovery(**_kwargs):
        raise RuntimeError("activity unavailable")

    def successful_replay(**kwargs):
        replay_calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "jupr_app.domain.match_delete.recompute_last_game_at_for_players",
        fail_activity_recovery,
    )
    monkeypatch.setattr(
        "jupr_app.domain.match_delete.replay_history",
        successful_replay,
    )

    result = delete_rated_matches_with_replay(
        supabase=sb,
        club_id="club",
        match_ids=[10],
        df_meta=pd.DataFrame(),
        actor="admin",
        source="match_log",
    )

    assert result["deleted_count"] == 1
    assert result["recovery_required"] is True
    assert "Player activity recovery failed" in result["replay_error"]
    assert result["replay_result"] == {"ok": True}
    assert len(replay_calls) == 1
    assert sb.tables["matches"][0]["deleted_at"] is not None


def test_match_exclusion_zero_update_is_recovery_required_not_success(
    monkeypatch,
):
    class _ZeroMatchUpdateSupabase(_Supabase):
        def execute(self, query):
            if query.table == "matches" and query._op == "update":
                return SimpleNamespace(data=[])
            return super().execute(query)

    sb = _ZeroMatchUpdateSupabase()
    sb.tables["matches"] = [
        {
            "id": 10,
            "club_id": "club",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "deleted_at": None,
        }
    ]
    monkeypatch.setattr(
        "jupr_app.domain.match_delete.recompute_last_game_at_for_players",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "jupr_app.domain.match_delete.replay_history",
        lambda **_kwargs: {"ok": True},
    )

    result = delete_rated_matches_with_replay(
        supabase=sb,
        club_id="club",
        match_ids=[10],
        df_meta=pd.DataFrame(),
    )

    assert result["deleted_count"] == 0
    assert result["deleted_ids"] == []
    assert result["recovery_required"] is True
    assert "did not attest every selected mutation" in result["replay_error"]
    assert sb.tables["matches"][0]["deleted_at"] is None


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
