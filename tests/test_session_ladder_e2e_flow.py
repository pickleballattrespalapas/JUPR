from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.session_ladder_engine import generateRoundGames
from services import session_ladder_api as api


class _Query:
    def __init__(self, db: dict[str, list[dict]], table: str):
        self.db = db
        self.table = table
        self._filters = []
        self._op = "select"
        self._payload = None
        self._conflict = None

    def select(self, _fields: str):
        self._op = "select"
        return self

    def eq(self, col: str, value):
        self._filters.append((col, value))
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

    def delete(self):
        self._op = "delete"
        return self

    def execute(self):
        rows = self.db.setdefault(self.table, [])

        def match(row):
            return all(row.get(k) == v for k, v in self._filters)

        if self._op == "select":
            return SimpleNamespace(data=[dict(r) for r in rows if match(r)])

        if self._op == "insert":
            payload = dict(self._payload or {})
            payload.setdefault("id", len(rows) + 1 if self.table == "players" else f"{self.table}_{len(rows)+1}")
            rows.append(payload)
            return SimpleNamespace(data=[dict(payload)])

        if self._op == "upsert":
            payload = dict(self._payload or {})
            keys = [k.strip() for k in str(self._conflict or "").split(",") if k.strip()]
            for i, row in enumerate(rows):
                if all(row.get(k) == payload.get(k) for k in keys):
                    rows[i] = {**row, **payload, "id": row.get("id")}
                    return SimpleNamespace(data=[dict(rows[i])])
            payload.setdefault("id", len(rows) + 1 if self.table == "players" else f"{self.table}_{len(rows)+1}")
            rows.append(payload)
            return SimpleNamespace(data=[dict(payload)])

        if self._op == "update":
            out = []
            for i, row in enumerate(rows):
                if match(row):
                    rows[i] = {**row, **(self._payload or {})}
                    out.append(dict(rows[i]))
            return SimpleNamespace(data=out)

        raise RuntimeError("unsupported")


class _Supabase:
    def __init__(self):
        self.storage = {
            "leagues_metadata": [
                {"id": 1, "club_id": "club-1", "league_name": "League One", "is_active": True},
            ],
            "players": [
                {"id": 1, "club_id": "club-1", "name": "A", "normalized_name": "a", "rating": 1500, "active": True},
                {"id": 2, "club_id": "club-1", "name": "B", "normalized_name": "b", "rating": 1400, "active": True},
                {"id": 3, "club_id": "club-1", "name": "C", "normalized_name": "c", "rating": 1300, "active": True},
                {"id": 4, "club_id": "club-1", "name": "D", "normalized_name": "d", "rating": 1200, "active": True},
                {"id": 5, "club_id": "club-1", "name": "E", "normalized_name": "e", "rating": 1100, "active": True},
                {"id": 6, "club_id": "club-1", "name": "F", "normalized_name": "f", "rating": 1000, "active": True},
                {"id": 7, "club_id": "club-1", "name": "G", "normalized_name": "g", "rating": 900, "active": True},
                {"id": 8, "club_id": "club-1", "name": "H", "normalized_name": "h", "rating": 800, "active": True},
            ],
            "session_ladder_sessions": [],
            "session_ladder_roster_entries": [],
            "session_ladder_court_pods": [],
            "session_ladder_court_pod_players": [],
            "session_ladder_games": [],
        }

    def table(self, name: str):
        return _Query(self.storage, name)


def _auth():
    return {"club_id": "club-1", "role": "manager", "user_id": "u-1", "admin_logged_in": False}


def test_e2e_roster_seed_score_close_creates_round2(monkeypatch):
    sb = _Supabase()
    monkeypatch.setattr(api, "FEATURE_SESSION_LADDER", True)

    sid = api.post_create_session(
        supabase=sb,
        auth=_auth(),
        payload={
            "league_key": "league one",
            "season_id": None,
            "session_starts_at": "2026-09-03T18:00:00Z",
            "courts_available": 2,
            "players_per_court": 4,
        },
    )["session"]["id"]

    sb.storage["session_ladder_sessions"][0]["state"] = "roster_open"

    for pid, rating in zip(range(1, 9), [1500, 1400, 1300, 1200, 1100, 1000, 900, 800]):
        api.post_add_roster_entries(
            supabase=sb,
            auth=_auth(),
            payload={
                "session_id": sid,
                "mode": "manual_existing",
                "player_id": pid,
                "status": "CHECKED_IN",
                "rating_snapshot": rating,
            },
        )

    api.post_seed_courts_by_rating(supabase=sb, auth=_auth(), session_id=sid)
    api.post_lock_seeding(supabase=sb, auth=_auth(), session_id=sid)
    api.post_start_round(supabase=sb, auth=_auth(), session_id=sid, round_number=1)

    round1 = [p for p in sb.storage["session_ladder_court_pods"] if p["round_number"] == 1]
    assert len(round1) == 2

    for pod in round1:
        players = sorted([r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]], key=lambda x: x["player_order"])
        pids = [r["player_id"] for r in players]
        for g in generateRoundGames(pids, "4p"):
            api.post_submit_game_result(
                supabase=sb,
                auth=_auth(),
                payload={
                    "session_id": sid,
                    "court_pod_id": pod["id"],
                    "game_number": g["game_number"],
                    "teamA_player_ids": g["teamA"],
                    "teamB_player_ids": g["teamB"],
                    "scoreA": 11,
                    "scoreB": 7,
                },
            )

    close = api.post_close_round(supabase=sb, auth=_auth(), session_id=sid, round_number=1, allow_override=True, override_reason="test override")
    assert close["generated_next_round"] is True

    round2 = [p for p in sb.storage["session_ladder_court_pods"] if p["round_number"] == 2]
    assert len(round2) == 2
