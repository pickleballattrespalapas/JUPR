from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.session_ladder_engine import generateRoundGames
from services import session_ladder_api as api


class _Query:
    def __init__(self, db: dict[str, list[dict]], table: str):
        self.db = db
        self.table = table
        self._filters: list[tuple[str, object]] = []
        self._op = "select"
        self._payload: dict | None = None
        self._conflict: str | None = None
        self._limit: int | None = None

    def select(self, _fields: str):
        self._op = "select"
        return self

    def eq(self, col: str, val):
        self._filters.append((col, val))
        return self

    def limit(self, n: int):
        self._limit = int(n)
        return self

    def insert(self, payload: dict, returning: str | None = None):
        _ = returning
        self._op = "insert"
        self._payload = dict(payload)
        return self

    def upsert(self, payload: dict, on_conflict: str, returning: str | None = None):
        _ = returning
        self._op = "upsert"
        self._payload = dict(payload)
        self._conflict = str(on_conflict)
        return self

    def update(self, payload: dict):
        self._op = "update"
        self._payload = dict(payload)
        return self

    def execute(self):
        rows = self.db.setdefault(self.table, [])

        def match(row):
            return all(row.get(k) == v for k, v in self._filters)

        if self._op == "select":
            out = [dict(r) for r in rows if match(r)]
            if self._limit is not None:
                out = out[: self._limit]
            return SimpleNamespace(data=out)

        if self._op == "insert":
            payload = dict(self._payload or {})
            if self.table == "players":
                payload.setdefault("id", len(rows) + 1)
            else:
                payload.setdefault("id", f"{self.table}_{len(rows)+1}")
            rows.append(payload)
            return SimpleNamespace(data=[dict(payload)])

        if self._op == "upsert":
            payload = dict(self._payload or {})
            keys = [k.strip() for k in str(self._conflict or "").split(",") if k.strip()]
            for i, row in enumerate(rows):
                if all(row.get(k) == payload.get(k) for k in keys):
                    rows[i] = {**row, **payload, "id": row.get("id")}
                    return SimpleNamespace(data=[dict(rows[i])])
            if self.table == "players":
                payload.setdefault("id", len(rows) + 1)
            else:
                payload.setdefault("id", f"{self.table}_{len(rows)+1}")
            rows.append(payload)
            return SimpleNamespace(data=[dict(payload)])

        if self._op == "update":
            upd = []
            for i, row in enumerate(rows):
                if match(row):
                    rows[i] = {**row, **(self._payload or {})}
                    upd.append(dict(rows[i]))
            return SimpleNamespace(data=upd)

        raise RuntimeError("unsupported op")


class _Supabase:
    def __init__(self):
        self.storage: dict[str, list[dict]] = {
            "leagues_metadata": [
                {"id": 1, "club_id": "club-1", "league_name": "League One", "is_active": True},
                {"id": 2, "club_id": "club-1", "league_name": "League Two", "is_active": True},
            ],
            "players": [
                {"id": 1, "club_id": "club-1", "name": "A One", "normalized_name": "a one", "rating": 1500, "active": True},
                {"id": 2, "club_id": "club-1", "name": "B Two", "normalized_name": "b two", "rating": 1400, "active": True},
                {"id": 3, "club_id": "club-1", "name": "C Three", "normalized_name": "c three", "rating": 1300, "active": True},
                {"id": 4, "club_id": "club-1", "name": "D Four", "normalized_name": "d four", "rating": 1200, "active": True},
                {"id": 5, "club_id": "club-1", "name": "E Five", "normalized_name": "e five", "rating": 1100, "active": True},
                {"id": 6, "club_id": "club-1", "name": "F Six", "normalized_name": "f six", "rating": 1000, "active": True},
                {"id": 7, "club_id": "club-1", "name": "G Seven", "normalized_name": "g seven", "rating": 900, "active": True},
                {"id": 8, "club_id": "club-1", "name": "H Eight", "normalized_name": "h eight", "rating": 800, "active": True},
            ],
            "session_ladder_sessions": [],
            "session_ladder_roster_entries": [],
            "session_ladder_court_pods": [],
            "session_ladder_court_pod_players": [],
            "session_ladder_games": [],
            "league_ratings": [],
            "session_ladder_rating_history": [],
            "session_ladder_attendance": [],
            "session_ladder_awards_attendance": [],
        }

    def table(self, name: str):
        return _Query(self.storage, name)


def _auth(role: str = "manager") -> dict:
    return {"club_id": "club-1", "role": role, "user_id": "u-1", "admin_logged_in": False}


def _seed_and_start_round_1(sb: _Supabase, monkeypatch) -> str:
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", True)
    sid = api.post_create_session(
        supabase=sb,
        auth=_auth("manager"),
        payload={
            "league_key": "league one",
            "season_id": None,
            "session_starts_at": "2026-09-03T18:00:00Z",
            "courts_available": 2,
            "players_per_court": 4,
        },
    )["session"]["id"]

    # service expects roster open first
    sb.storage["session_ladder_sessions"][0]["state"] = "roster_open"

    for pid, rating in zip(range(1, 9), [1500, 1400, 1300, 1200, 1100, 1000, 900, 800]):
        api.post_add_roster_entries(
            supabase=sb,
            auth=_auth("manager"),
            payload={
                "session_id": sid,
                "mode": "manual_existing",
                "player_id": pid,
                "status": "CHECKED_IN",
                "rating_snapshot": rating,
            },
        )

    api.post_seed_courts_by_rating(supabase=sb, auth=_auth("manager"), session_id=sid)
    api.post_lock_seeding(supabase=sb, auth=_auth("manager"), session_id=sid)
    api.post_start_round(supabase=sb, auth=_auth("manager"), session_id=sid, round_number=1)
    return sid


def test_feature_flag_blocks_non_admin_endpoints(monkeypatch):
    sb = _Supabase()
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", False)
    with pytest.raises(RuntimeError, match="feature.sessionLadder disabled"):
        api.post_create_session(
            supabase=sb,
            auth=_auth("manager"),
            payload={
                "league_key": "League One",
                "season_id": None,
                "session_starts_at": "2026-09-03T18:00:00Z",
                "courts_available": 1,
                "players_per_court": 4,
            },
        )


def test_feature_flag_allows_elevated_admin_preview_access(monkeypatch):
    sb = _Supabase()
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", False)
    created = api.post_create_session(
        supabase=sb,
        auth={**_auth("manager"), "admin_logged_in": True},
        payload={
            "league_key": "League One",
            "season_id": None,
            "session_starts_at": "2026-09-03T18:00:00Z",
            "courts_available": 1,
            "players_per_court": 4,
        },
    )
    assert created["session"]["id"]


def test_auth_blocks_mutation_for_non_manager_non_admin(monkeypatch):
    sb = _Supabase()
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", True)
    with pytest.raises(PermissionError, match="manager/admin"):
        api.post_create_session(
            supabase=sb,
            auth=_auth("player"),
            payload={
                "league_key": "League One",
                "season_id": None,
                "session_starts_at": "2026-09-03T18:00:00Z",
                "courts_available": 1,
                "players_per_court": 4,
            },
        )


def test_create_session_validates_and_canonicalizes_league_key(monkeypatch):
    sb = _Supabase()
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", True)

    created = api.post_create_session(
        supabase=sb,
        auth=_auth("manager"),
        payload={
            "league_key": "league one",
            "season_id": None,
            "session_starts_at": "2026-09-03T18:00:00Z",
            "courts_available": 2,
            "players_per_court": 4,
        },
    )
    assert created["session"]["league_id"] == "League One"

    with pytest.raises(ValueError, match="Unknown league key"):
        api.post_create_session(
            supabase=sb,
            auth=_auth("manager"),
            payload={
                "league_key": "Missing League",
                "season_id": None,
                "session_starts_at": "2026-09-03T18:00:00Z",
                "courts_available": 2,
                "players_per_court": 4,
            },
        )


def test_roster_add_modes_create_manual_bulk_and_csv(monkeypatch):
    sb = _Supabase()
    sid = _seed_and_start_round_1(sb, monkeypatch)

    # create new player
    out_new = api.post_add_roster_entries(
        supabase=sb,
        auth=_auth("manager"),
        payload={
            "session_id": sid,
            "mode": "create_new_player",
            "name": "Walk In",
            "rating": 1234,
            "status": "WALK_IN",
        },
    )
    assert out_new["added"]

    # bulk text
    out_bulk = api.post_add_roster_entries(
        supabase=sb,
        auth=_auth("manager"),
        payload={
            "session_id": sid,
            "mode": "bulk_text",
            "bulk_text": "Neo,1199,CHECKED_IN\nTrin,1188,EXPECTED",
        },
    )
    assert len(out_bulk["added"]) == 2

    # csv upload
    out_csv = api.post_add_roster_entries(
        supabase=sb,
        auth=_auth("manager"),
        payload={
            "session_id": sid,
            "mode": "csv_upload",
            "csv_text": "name,rating,status\nMorpheus,1177,CHECKED_IN\n",
        },
    )
    assert len(out_csv["added"]) == 1


def test_round_and_details_endpoints_include_stable_deep_links(monkeypatch):
    sb = _Supabase()
    sid = _seed_and_start_round_1(sb, monkeypatch)

    round1_pods = sorted([p for p in sb.storage["session_ladder_court_pods"] if p["round_number"] == 1], key=lambda x: x["court_number"])
    for pod in round1_pods:
        player_rows = sorted([r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]], key=lambda x: x["player_order"])
        p = [r["player_id"] for r in player_rows]
        template = generateRoundGames(p, "4p")
        for g in template:
            api.post_submit_game_result(
                supabase=sb,
                auth=_auth("manager"),
                payload={
                    "session_id": sid,
                    "court_pod_id": pod["id"],
                    "game_number": g["game_number"],
                    "teamA_player_ids": g["teamA"],
                    "teamB_player_ids": g["teamB"],
                    "scoreA": 21,
                    "scoreB": 17,
                },
            )

    close = api.post_close_round(supabase=sb, auth=_auth("manager"), session_id=sid, round_number=1, allow_override=True, override_reason="test override")
    assert close["courts"]
    assert all("/sessions/" in c["court_sheet_route"] for c in close["courts"])

    details = api.get_session_details(supabase=sb, auth=_auth("manager"), session_id=sid)
    assert details["courts"]
    assert all("/sessions/" in c["court_sheet_route"] for c in details["courts"])


def test_complete_and_publish_endpoints(monkeypatch):
    sb = _Supabase()
    sid = _seed_and_start_round_1(sb, monkeypatch)

    for rnd in (1, 2):
        pods = sorted([p for p in sb.storage["session_ladder_court_pods"] if p["round_number"] == rnd], key=lambda x: x["court_number"])
        for pod in pods:
            player_rows = sorted([r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]], key=lambda x: x["player_order"])
            p = [r["player_id"] for r in player_rows]
            for g in generateRoundGames(p, "4p"):
                api.post_submit_game_result(
                    supabase=sb,
                    auth=_auth("manager"),
                    payload={
                        "session_id": sid,
                        "court_pod_id": pod["id"],
                        "game_number": g["game_number"],
                        "teamA_player_ids": g["teamA"],
                        "teamB_player_ids": g["teamB"],
                        "scoreA": 21,
                        "scoreB": 18,
                    },
                )
        api.post_close_round(supabase=sb, auth=_auth("manager"), session_id=sid, round_number=rnd, allow_override=True, override_reason="test override")
        if rnd == 1:
            api.post_start_round(supabase=sb, auth=_auth("manager"), session_id=sid, round_number=2)

    done = api.post_complete_session(supabase=sb, auth=_auth("manager"), session_id=sid)
    assert done["session"]["state"] == "completed"
    assert done["session"]["rating_update"]["applied"] is False

    done_again = api.post_complete_session(supabase=sb, auth=_auth("manager"), session_id=sid)
    assert done_again["session"]["rating_update"]["applied"] is False

    pub = api.post_publish_session(supabase=sb, auth=_auth("manager"), session_id=sid)
    assert pub["session"]["state"] == "published"
    assert pub["session"].get("recap")
    assert pub["session"].get("leaderboard")

    locked_pod = sorted([p for p in sb.storage["session_ladder_court_pods"] if p["round_number"] == 1], key=lambda x: x["court_number"])[0]
    player_rows = sorted([r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == locked_pod["id"]], key=lambda x: x["player_order"])
    t = generateRoundGames([r["player_id"] for r in player_rows], "4p")[0]
    with pytest.raises(RuntimeError, match="Session is locked"):
        api.post_submit_game_result(
            supabase=sb,
            auth=_auth("manager"),
            payload={
                "session_id": sid,
                "court_pod_id": locked_pod["id"],
                "game_number": t["game_number"],
                "teamA_player_ids": t["teamA"],
                "teamB_player_ids": t["teamB"],
                "scoreA": 11,
                "scoreB": 9,
            },
        )
