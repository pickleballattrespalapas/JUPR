from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.match_processing import process_matches
from jupr_app.ui.live.shared import (
    build_league_round_official_payloads,
    build_rr_official_payloads,
    build_tournament_official_payloads,
)
from jupr_app.ui.pages.jupr_live_admin import ADMIN_CONFIG


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self._op = "select"
        self._payload = None
        self._filters = []
        self._limit = None

    def select(self, _cols):
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
        }
        self.fail_on_rating_scope_insert = False

    def table(self, name):
        return _Query(self, name)

    def execute(self, q: _Query):
        rows = self.tables.setdefault(q.table, [])
        data = list(rows)
        for op, col, val in q._filters:
            if op == "eq":
                data = [r for r in data if str(r.get(col)) == str(val)]
        if q._limit is not None:
            data = data[: q._limit]
        if q._op == "select":
            return SimpleNamespace(data=data)
        if q._op == "insert":
            payload = q._payload if isinstance(q._payload, list) else [q._payload]
            if q.table == "matches" and self.fail_on_rating_scope_insert:
                if any("rating_scope" in row for row in payload):
                    raise RuntimeError("column \"rating_scope\" of relation \"matches\" does not exist")
            for row in payload:
                rows.append(dict(row))
            return SimpleNamespace(data=payload)
        if q._op == "update":
            for row in data:
                row.update(dict(q._payload))
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


def _patch_side_effects(monkeypatch):
    monkeypatch.setattr(
        "jupr_app.domain.match_processing.enqueue_badge_eval",
        lambda *args, **kwargs: {"queued": False},
    )
    monkeypatch.setattr(
        "jupr_app.domain.match_processing.run_live_badge_awards",
        lambda *args, **kwargs: {"mode": "inline", "awarded_count": 0},
    )
    monkeypatch.setattr(
        "jupr_app.domain.match_processing.queue_player_updates_for_affected_subscribers",
        lambda *args, **kwargs: {"week_windows": 0, "queued": 0, "already_queued": 0, "no_active_subscription": 0, "failed": 0},
    )


def test_admin_config_uses_rating_mode_not_official_context():
    assert ADMIN_CONFIG.show_rating_mode is True
    assert ADMIN_CONFIG.show_official_context is False


def test_official_payload_builders_preserve_rating_scope_context():
    state = {"official_league": "ignored", "official_week_tag": "ignored"}
    rr_event = {
        "official_context": {
            "league": "JUPR Live",
            "week_tag": "",
            "match_type": "JUPR Live Rated",
            "rating_scope": "overall_only",
            "is_popup": False,
        },
        "participants": [
            {"id": "p1", "player_id": 1},
            {"id": "p2", "player_id": 2},
            {"id": "p3", "player_id": 3},
            {"id": "p4", "player_id": 4},
        ],
        "rounds": [{"number": 1, "matches": [{"id": "m1", "teamA": ["p1", "p2"], "teamB": ["p3", "p4"], "scoreA": 11, "scoreB": 9}]}],
    }
    rr_payload = build_rr_official_payloads(state, rr_event)[0]
    assert rr_payload["rating_scope"] == "overall_only"
    assert rr_payload["match_type"] == "JUPR Live Rated"
    assert rr_payload["league"] == "JUPR Live"
    assert rr_payload["week_tag"] == ""

    rr_event["official_context"]["match_type"] = "JUPR Live Unrated"
    rr_event["official_context"]["rating_scope"] = "unrated"
    rr_payload_unrated = build_rr_official_payloads(state, rr_event)[0]
    assert rr_payload_unrated["rating_scope"] == "unrated"
    assert rr_payload_unrated["match_type"] == "JUPR Live Unrated"

    league_event = {
        "official_context": dict(rr_event["official_context"]),
        "participants": rr_event["participants"],
        "currentRoundNumber": 1,
        "rounds": [
            {
                "number": 1,
                "courts": [
                    {
                        "courtNumber": 1,
                        "miniRounds": [
                            {
                                "number": 1,
                                "matches": [{"id": "lm1", "teamA": ["p1", "p2"], "teamB": ["p3", "p4"], "scoreA": 11, "scoreB": 3}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    league_payload = build_league_round_official_payloads(state, league_event)[0]
    assert league_payload["rating_scope"] == "unrated"
    assert league_payload["match_type"] == "JUPR Live Unrated"


def test_tournament_payload_builder_preserves_context():
    state = {"official_league": "", "official_week_tag": ""}
    event = {
        "name": "Live Admin Cup",
        "official_context": {
            "league": "JUPR Live",
            "week_tag": "",
            "match_type": "JUPR Live Rated",
            "rating_scope": "overall_only",
            "is_popup": False,
        },
        "teams": [
            {"id": "a", "player1_id": 1, "player2_id": 2},
            {"id": "b", "player1_id": 3, "player2_id": 4},
        ],
        "rounds": [{"number": 1, "matches": [{"slot": 1, "participantAId": "a", "participantBId": "b", "scoreA": 11, "scoreB": 6, "winnerId": "a"}]}],
    }
    payload = build_tournament_official_payloads(state, event)[0]
    assert payload["rating_scope"] == "overall_only"
    assert payload["league"] == "JUPR Live"
    assert payload["match_type"] == "JUPR Live Rated"


def test_process_matches_overall_only_updates_players_not_league_ratings(monkeypatch):
    _patch_side_effects(monkeypatch)
    sb = _Supabase()
    sb.tables["players"] = _players_df().to_dict("records")
    result = process_matches(
        [{
            "league": "JUPR Live",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "s1": 11,
            "s2": 7,
            "rating_scope": "overall_only",
        }],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    assert result["inserted"] == 1
    assert sb.tables["league_ratings"] == []
    updated_players = {int(p["id"]): p for p in sb.tables["players"]}
    assert int(updated_players[1]["matches_played"]) == 1
    assert float(updated_players[1]["rating"]) != 1200.0


def test_process_matches_unrated_records_match_without_rating_changes(monkeypatch):
    _patch_side_effects(monkeypatch)
    sb = _Supabase()
    initial_players = _players_df().to_dict("records")
    sb.tables["players"] = [dict(row) for row in initial_players]
    result = process_matches(
        [{
            "league": "JUPR Live",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "s1": 11,
            "s2": 8,
            "rating_scope": "unrated",
        }],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    assert result["inserted"] == 1
    assert result["skipped_unrated"] == 1
    assert sb.tables["league_ratings"] == []
    before = {int(row["id"]): row for row in initial_players}
    after = {int(row["id"]): row for row in sb.tables["players"]}
    assert int(after[1]["matches_played"]) == int(before[1]["matches_played"])
    assert float(after[1]["rating"]) == float(before[1]["rating"])
    assert len(sb.tables["matches"]) == 1
    assert sb.tables["matches"][0]["rating_scope"] == "unrated"
    assert float(sb.tables["matches"][0]["elo_delta"]) == 0.0


def test_process_matches_unrated_plan_has_match_but_no_aggregate_writes(
    monkeypatch,
):
    _patch_side_effects(monkeypatch)
    sb = _Supabase()
    sb.tables["players"] = _players_df().to_dict("records")

    result = process_matches(
        [{
            "date": "2026-07-26",
            "league": "Open",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "s1": 11,
            "s2": 8,
            "rating_scope": "unrated",
        }],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        build_write_plan_only=True,
    )

    assert result["inserted"] == 1
    assert result["skipped_unrated"] == 1
    assert len(result["write_plan"]["match_rows"]) == 1
    assert result["write_plan"]["match_rows"][0]["rating_scope"] == "unrated"
    assert result["write_plan"]["player_updates"] == []
    assert result["write_plan"]["league_rating_updates"] == []
    assert sb.tables["matches"] == []


def test_process_matches_without_metadata_does_not_create_unmanaged_league_ratings(monkeypatch):
    _patch_side_effects(monkeypatch)
    sb = _Supabase()
    sb.tables["players"] = _players_df().to_dict("records")
    process_matches(
        [{"league": "Main", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 4}],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    assert sb.tables["league_ratings"] == []
    assert len(sb.tables["matches"]) == 1


def test_process_matches_insert_falls_back_when_rating_scope_column_missing(monkeypatch):
    _patch_side_effects(monkeypatch)
    sb = _Supabase()
    sb.fail_on_rating_scope_insert = True
    sb.tables["players"] = _players_df().to_dict("records")
    result = process_matches(
        [{
            "league": "JUPR Live",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "s1": 11,
            "s2": 9,
            "rating_scope": "overall_only",
        }],
        supabase=sb,
        club_id="club",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    assert result["inserted"] == 1
    assert len(sb.tables["matches"]) == 1
    assert "rating_scope" not in sb.tables["matches"][0]
