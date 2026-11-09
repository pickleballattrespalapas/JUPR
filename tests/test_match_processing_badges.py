from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from postgrest.exceptions import APIError

from jupr_app.domain.bulk_match_editor import apply_bulk_match_edits
from jupr_app.domain.gamification.live_awards import run_live_badge_awards
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.gamification.badge_lease import BadgeLeaseLost


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
        self._select = "*"

    def select(self, cols):
        self._op = "select"
        self._select = cols
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None, ignore_duplicates=False):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
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

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
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
    def __init__(self, *, queue_missing=False, missing_optional_pb=False):
        self.queue_missing = queue_missing
        self.missing_optional_pb = missing_optional_pb
        self.tables = {
            "matches": [],
            "players": [],
            "league_ratings": [],
            "leagues_metadata": [],
            "badges": [
                {"badge_id": "participant", "state": "live", "eval_triggers": ["match_recorded", "match_updated"]},
                {"badge_id": "first_win", "state": "live", "eval_triggers": ["match_recorded", "match_updated"]},
            ],
            "player_badges": [],
        }

    def table(self, name):
        return _Query(self, name)

    def execute(self, q: _Query):
        if q.table == "badge_eval_queue" and self.queue_missing:
            raise APIError({"code": "PGRST205", "message": "missing"})

        rows = self.tables.setdefault(q.table, [])
        data = list(rows)
        for op, col, val in q._filters:
            if op == "eq":
                data = [r for r in data if str(r.get(col)) == str(val)]
            elif op == "in":
                data = [r for r in data if r.get(col) in val]
            elif op == "is" and val is None:
                data = [r for r in data if r.get(col) is None]

        if q._order:
            col, desc = q._order
            data = sorted(data, key=lambda r: str(r.get(col) or ""), reverse=desc)
        if q._range is not None:
            start, end = q._range
            data = data[start : end + 1]
        if q._limit is not None:
            data = data[: q._limit]

        if q._op == "select":
            if (
                q.table == "player_badges"
                and self.missing_optional_pb
                and "awarded_by" in q._select
            ):
                raise APIError({"code": "42703", "message": "column player_badges.awarded_by does not exist"})
            return SimpleNamespace(data=data)

        if q._op == "insert":
            payload = q._payload if isinstance(q._payload, list) else [q._payload]
            inserted = []
            for row in payload:
                row = dict(row)
                row.setdefault("id", len(rows) + 1)
                rows.append(row)
                inserted.append(dict(row))
            return SimpleNamespace(data=inserted)

        if q._op == "update":
            for row in data:
                row.update(q._payload)
            return SimpleNamespace(data=data)

        if q._op == "upsert":
            payload = q._payload if isinstance(q._payload, list) else [q._payload]
            keys = [k.strip() for k in (q._on_conflict or "").split(",") if k.strip()]
            for row in payload:
                row = dict(row)
                existing = None
                if keys:
                    for cur in rows:
                        if all(str(cur.get(k)) == str(row.get(k)) for k in keys):
                            existing = cur
                            break
                if existing:
                    existing.update(row)
                else:
                    row.setdefault("id", len(rows) + 1)
                    rows.append(row)
            return SimpleNamespace(data=payload)

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


def test_match_insert_awards_inline_when_queue_table_missing():
    sb = _Supabase(queue_missing=True)
    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 5}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    badge_ids = {row["badge_id"] for row in sb.tables["player_badges"]}
    assert result["badge_summary"]["mode"] == "inline"
    assert "first_win" in badge_ids


def test_live_awards_work_when_recompute_columns_missing():
    sb = _Supabase(queue_missing=True, missing_optional_pb=True)
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "league": "Main", "date": "2024-01-01T00:00:00Z", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 4}
    ]
    sb.tables["players"] = _players_df().to_dict("records")

    summary = run_live_badge_awards(sb, club_id="club", player_ids=[1], event_type="match_recorded")
    assert summary["mode"] == "inline"
    assert summary["awarded_count"] >= 1


def test_live_awards_are_idempotent_via_upsert_key():
    sb = _Supabase(missing_optional_pb=True)
    sb.tables["matches"] = [
        {"id": 1, "club_id": "club", "league": "Main", "date": "2024-01-01T00:00:00Z", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 4}
    ]
    sb.tables["players"] = _players_df().to_dict("records")

    run_live_badge_awards(sb, club_id="club", player_ids=[1], event_type="match_recorded")
    run_live_badge_awards(sb, club_id="club", player_ids=[1], event_type="match_recorded")
    keys = {(r["club_id"], r["player_id"], r["badge_id"], r["context_id"]) for r in sb.tables["player_badges"]}
    assert len(keys) == len(sb.tables["player_badges"])


def test_bulk_edit_uses_inline_fallback_without_breaking():
    sb = _Supabase(queue_missing=True, missing_optional_pb=True)
    sb.tables["players"] = _players_df().to_dict("records")
    sb.tables["matches"] = [
        {"id": 10, "club_id": "club", "league": "Main", "date": "2024-01-01T00:00:00Z", "week_tag": None, "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 5}
    ]
    result = apply_bulk_match_edits(sb, "club", [{"id": 10, "notes": "edited"}], actor="admin")
    assert result["updated_count"] == 1
    assert result["badge_summary"]["mode"] == "inline"


def test_popup_matches_do_not_trigger_live_awards():
    sb = _Supabase(queue_missing=True)
    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 5, "match_type": "PopUp", "is_popup": True}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    assert result["badge_summary"]["mode"] == "skipped"
    assert sb.tables["player_badges"] == []


def _isolated_mode(monkeypatch, mode):
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external" if mode == "external" else "embedded")
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1" if mode == "leased" else "0")


def _forbid_inline(monkeypatch, module):
    def lost(*args, **kwargs):
        raise BadgeLeaseLost("Worker no longer owns the job")
    monkeypatch.setattr(f"{module}.process_badge_eval_queue", lost)
    monkeypatch.setattr(f"{module}.run_live_badge_awards", lambda *a, **k: pytest.fail("Unfenced inline fallback executed"))


@pytest.mark.parametrize("mode", ["external", "leased"])
@pytest.mark.parametrize("queue_missing", [False, True])
def test_match_save_keeps_scores_and_never_falls_back_when_isolated(monkeypatch, mode, queue_missing):
    _isolated_mode(monkeypatch, mode)
    _forbid_inline(monkeypatch, "jupr_app.domain.match_processing")
    sb = _Supabase(queue_missing=queue_missing)
    result = process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 5}],
        supabase=sb, club_id="club", name_to_id={}, df_players_all=_players_df(),
        df_leagues=pd.DataFrame(), df_meta=pd.DataFrame(),
    )
    assert len(sb.tables["matches"]) == 1
    assert sb.tables["matches"][0]["score_t1"] == 11
    assert sb.tables["player_badges"] == []
    summary = result["badge_summary"]
    assert summary["mode"] == ("queue_error" if queue_missing else "queue")
    if queue_missing:
        assert summary["recovery_required"]
        assert summary["recovery_match_refs"] == ["1"]
        assert result["player_update_queue"]["mode"] == "error"
    else:
        assert summary["deferred"] and summary["notification_badge_refs"] == ["1"]


@pytest.mark.parametrize("mode", ["external", "leased"])
@pytest.mark.parametrize("queued", [False, True])
def test_shared_side_effects_never_fallback_on_enqueue_or_lease_failure(monkeypatch, mode, queued):
    from jupr_app.domain.matches import side_effects
    _isolated_mode(monkeypatch, mode)
    _forbid_inline(monkeypatch, "jupr_app.domain.matches.side_effects")
    monkeypatch.setattr(side_effects, "enqueue_badge_eval", lambda *a, **k: {"queued": queued})
    result = side_effects.run_badge_side_effects(supabase=object(), club_id="club",
        has_badge_eligible_match=True, affected_players={1, 2, 3, 4}, db_matches=[{}],
        match_payloads=[{}], dedupe_match_id="saved-operation")
    assert result["mode"] == ("queue" if queued else "queue_error")
    assert result.get("recovery_required", False) is (not queued)


@pytest.mark.parametrize("mode", ["external", "leased"])
def test_bulk_edit_keeps_earlier_enqueue_failure_visible(monkeypatch, mode):
    _isolated_mode(monkeypatch, mode)
    _forbid_inline(monkeypatch, "jupr_app.domain.bulk_match_editor")
    monkeypatch.setattr("jupr_app.domain.bulk_match_editor.enqueue_badge_eval",
        lambda *a, **k: {"queued": k["match_id"] != "10"})
    sb = _Supabase()
    sb.tables["players"] = _players_df().to_dict("records")
    sb.tables["matches"] = [
        {"id": mid, "club_id": "club", "league": "Main", "date": "2024-01-01T00:00:00Z",
         "week_tag": None, "match_type": "League", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3,
         "t2_p2": 4, "score_t1": 11, "score_t2": 5} for mid in [10, 11]]
    result = apply_bulk_match_edits(sb, "club", [{"id": mid, "notes": "edited"} for mid in [10, 11]], actor="admin")
    assert result["updated_count"] == 2
    assert result["badge_summary"]["mode"] == "queue_error"
    assert result["badge_summary"]["recovery_match_refs"] == ["10"]
    assert result["badge_summary"]["recovery_required"]
    assert any("award evaluations could not be queued" in warning for warning in result["warnings"])
    assert sb.tables["player_badges"] == []
