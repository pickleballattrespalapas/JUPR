from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain import session_ladder_service as svc


class _Query:
    def __init__(self, db: dict[str, list[dict]], table: str):
        self.db = db
        self.table = table
        self._filters: list[tuple[str, object]] = []
        self._op = "select"
        self._payload = None
        self._conflict = None

    def select(self, _fields: str):
        self._op = "select"
        return self

    def eq(self, col: str, val):
        self._filters.append((col, val))
        return self

    def insert(self, payload: dict):
        self._op = "insert"
        self._payload = dict(payload)
        return self

    def upsert(self, payload: dict, on_conflict: str):
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
            row = dict(self._payload)
            row.setdefault("id", f"{self.table}_{len(rows)+1}")
            rows.append(row)
            return SimpleNamespace(data=[dict(row)])

        if self._op == "upsert":
            keys = [k.strip() for k in self._conflict.split(",")]
            payload = dict(self._payload)
            for idx, row in enumerate(rows):
                if all(row.get(k) == payload.get(k) for k in keys):
                    rows[idx] = {**row, **payload, "id": row.get("id")}
                    return SimpleNamespace(data=[dict(rows[idx])])
            payload.setdefault("id", f"{self.table}_{len(rows)+1}")
            rows.append(payload)
            return SimpleNamespace(data=[dict(payload)])

        if self._op == "update":
            updated = []
            for idx, row in enumerate(rows):
                if match(row):
                    rows[idx] = {**row, **self._payload}
                    updated.append(dict(rows[idx]))
            return SimpleNamespace(data=updated)

        if self._op == "delete":
            removed = [dict(r) for r in rows if match(r)]
            self.db[self.table] = [dict(r) for r in rows if not match(r)]
            return SimpleNamespace(data=removed)

        raise RuntimeError("Unsupported operation")


class _Supabase:
    def __init__(self):
        self.storage: dict[str, list[dict]] = {
            "session_ladder_sessions": [],
            "session_ladder_roster_entries": [],
            "session_ladder_court_pods": [],
            "session_ladder_court_pod_players": [],
            "session_ladder_games": [],
            "players": [
                {"id": i, "club_id": "club-1", "name": f"P{i}", "rating": 1200.0}
                for i in range(1, 17)
            ],
            "league_ratings": [],
            "session_ladder_rating_history": [],
            "session_ladder_attendance": [],
            "session_ladder_awards_attendance": [],
        }

    def table(self, name: str):
        return _Query(self.storage, name)


def _seed_basic_session(sb: _Supabase) -> str:
    created = svc.createSession(
        sb,
        club_id="club-1",
        league_id="league-a",
        season_id=None,
        session_starts_at="2026-09-01T18:00:00Z",
        courts_available=2,
        players_per_court=4,
        rounds_planned=2,
        created_by="admin",
    )
    sid = created["id"]
    svc._transition_state(sb, sessionId=sid, to_state="roster_open", updated_by="admin")
    for idx, rating in enumerate([1600, 1500, 1400, 1300, 1200, 1100, 1000, 900], start=1):
        svc.updateRosterStatus(
            sb,
            sessionId=sid,
            player_id=idx,
            status="CHECKED_IN",
            rating_snapshot=rating,
            updated_by="admin",
        )
    svc.seedCourtsByRating(sb, sessionId=sid, seeded_by="admin")
    svc.lockSeeding(sb, sessionId=sid, updated_by="admin")
    return sid


def test_can_transition_state_machine_paths():
    assert svc.canTransition("draft", "roster_open")
    assert svc.canTransition("round_2_closed", "completed")
    assert not svc.canTransition("draft", "round_1_active")
    assert not svc.canTransition("published", "completed")


def test_session_state_values_match_db_allowed_list():
    db_allowed_states = {
        "draft",
        "roster_open",
        "seeded_locked",
        "round_1_active",
        "round_1_closed",
        "round_2_active",
        "round_2_closed",
        "round_3_active",
        "round_3_closed",
        "completed",
        "published",
    }
    assert set(svc.SESSION_STATES) == db_allowed_states


def test_close_round_is_idempotent_for_next_round_generation():
    sb = _Supabase()
    sid = _seed_basic_session(sb)
    svc.startRound(sb, sessionId=sid, roundNumber=1, updated_by="admin")

    pods = sorted(sb.storage["session_ladder_court_pods"], key=lambda r: r["court_number"])
    p1 = pods[0]["id"]
    p2 = pods[1]["id"]

    for pod in (p1, p2):
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        templates = svc.generateRoundGames(p, "4p")
        for game in templates:
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=pod,
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=21,
                scoreB=15,
                edited_by="admin",
            )

    first = svc.closeRound(sb, sessionId=sid, roundNumber=1, updated_by="admin", allow_override=True, override_reason="test override")
    second = svc.closeRound(sb, sessionId=sid, roundNumber=1, updated_by="admin", allow_override=True, override_reason="test override")

    assert first["generated_next_round"] is True
    assert second["generated_next_round"] is False

    round2_pods = [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 2]
    assert len(round2_pods) == 2

    round2_games = [
        g
        for g in sb.storage["session_ladder_games"]
        if any(p["id"] == g["court_pod_id"] for p in round2_pods)
    ]
    assert len(round2_games) == 6  # 2 courts x 3 games


def test_validate_completeness_complete_and_incomplete_cases():
    sb = _Supabase()
    sid = _seed_basic_session(sb)
    svc.startRound(sb, sessionId=sid, roundNumber=1, updated_by="admin")

    pod = [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1][0]
    pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
    p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
    for game in svc.generateRoundGames(p, "4p")[:2]:
        svc.submitGameResult(
            sb,
            sessionId=sid,
            court_pod_id=pod["id"],
            game_number=game["game_number"],
            teamA_player_ids=game["teamA"],
            teamB_player_ids=game["teamB"],
            scoreA=21,
            scoreB=19,
            edited_by="admin",
        )

    incomplete = svc.validateSessionCompleteness(sb, sid)
    assert incomplete.ok is False
    assert incomplete.missing

    for other_pod in [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1]:
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == other_pod["id"]]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        for game in svc.generateRoundGames(p, "4p"):
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=other_pod["id"],
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=21,
                scoreB=18,
                edited_by="admin",
            )

    complete = svc.validateSessionCompleteness(sb, sid)
    assert complete.ok is True


def test_complete_and_publish_session_flow():
    sb = _Supabase()
    sid = _seed_basic_session(sb)

    svc.startRound(sb, sessionId=sid, roundNumber=1, updated_by="admin")
    for pod in [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1]:
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        for game in svc.generateRoundGames(p, "4p"):
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=pod["id"],
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=21,
                scoreB=18,
                edited_by="admin",
            )
    svc.closeRound(sb, sessionId=sid, roundNumber=1, updated_by="admin", allow_override=True, override_reason="test override")

    svc.startRound(sb, sessionId=sid, roundNumber=2, updated_by="admin")
    for pod in [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 2]:
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        for game in svc.generateRoundGames(p, "4p"):
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=pod["id"],
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=21,
                scoreB=16,
                edited_by="admin",
            )
    svc.closeRound(sb, sessionId=sid, roundNumber=2, updated_by="admin", allow_override=True, override_reason="test override")

    completed = svc.completeSession(sb, sessionId=sid, updated_by="admin")
    assert completed["state"] == "completed"
    assert completed["rating_update_hook_triggered"] is True
    assert completed["rating_update"]["applied"] is False

    completed_again = svc.completeSession(sb, sessionId=sid, updated_by="admin")
    assert completed_again["rating_update"]["applied"] is False

    published = svc.publishSession(sb, sessionId=sid, updated_by="admin")
    assert published["state"] == "published"
    assert published.get("recap")
    assert published.get("leaderboard")
    assert published.get("attendance")
    assert published.get("published_at")
    assert sb.storage["session_ladder_rating_history"]
    assert {row["league_name"] for row in sb.storage["league_ratings"]} == {"league-a"}

    first_published_at = published["published_at"]
    first_recap = published["recap"]
    first_leaderboard = published["leaderboard"]
    first_attendance = published["attendance"]
    first_attendance_count = len(sb.storage["session_ladder_attendance"])

    published_again = svc.publishSession(sb, sessionId=sid, updated_by="admin")
    assert published_again["published_at"] == first_published_at
    assert published_again["recap"] == first_recap
    assert published_again["leaderboard"] == first_leaderboard
    assert published_again["attendance"] == first_attendance
    assert len(sb.storage["session_ladder_attendance"]) == first_attendance_count


def test_submit_game_result_upsert_idempotent():
    sb = _Supabase()
    sid = _seed_basic_session(sb)
    pod = [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1][0]
    pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
    p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
    game = svc.generateRoundGames(p, "4p")[0]

    svc.submitGameResult(
        sb,
        sessionId=sid,
        court_pod_id=pod["id"],
        game_number=1,
        teamA_player_ids=game["teamA"],
        teamB_player_ids=game["teamB"],
        scoreA=21,
        scoreB=19,
        edited_by="admin",
    )
    svc.submitGameResult(
        sb,
        sessionId=sid,
        court_pod_id=pod["id"],
        game_number=1,
        teamA_player_ids=game["teamA"],
        teamB_player_ids=game["teamB"],
        scoreA=21,
        scoreB=17,
        edited_by="admin",
    )

    rows = [r for r in sb.storage["session_ladder_games"] if r["court_pod_id"] == pod["id"] and r["game_number"] == 1]
    assert len(rows) == 1
    assert rows[0]["score_b"] == 17


def test_submit_game_result_blocked_when_session_locked():
    sb = _Supabase()
    sid = _seed_basic_session(sb)
    sb.storage["session_ladder_sessions"][0]["state"] = "published"
    pod = [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1][0]
    pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
    p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
    game = svc.generateRoundGames(p, "4p")[0]

    with pytest.raises(RuntimeError, match="Session is locked"):
        svc.submitGameResult(
            sb,
            sessionId=sid,
            court_pod_id=pod["id"],
            game_number=1,
            teamA_player_ids=game["teamA"],
            teamB_player_ids=game["teamB"],
            scoreA=11,
            scoreB=8,
            edited_by="admin",
        )


def test_round_close_auto_completes_final_round():
    sb = _Supabase()
    sid = _seed_basic_session(sb)

    svc.startRound(sb, sessionId=sid, roundNumber=1, updated_by="admin")
    for pod in [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 1]:
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        for game in svc.generateRoundGames(p, "4p"):
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=pod["id"],
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=11,
                scoreB=8,
                edited_by="admin",
            )
    out1 = svc.closeRound(sb, sessionId=sid, roundNumber=1, updated_by="admin", allow_override=True, override_reason="ok")
    assert out1["auto_completed"] is False

    svc.startRound(sb, sessionId=sid, roundNumber=2, updated_by="admin")
    for pod in [r for r in sb.storage["session_ladder_court_pods"] if r["round_number"] == 2]:
        pod_players = [r for r in sb.storage["session_ladder_court_pod_players"] if r["court_pod_id"] == pod["id"]]
        p = [x["player_id"] for x in sorted(pod_players, key=lambda r: r["player_order"])]
        for game in svc.generateRoundGames(p, "4p"):
            svc.submitGameResult(
                sb,
                sessionId=sid,
                court_pod_id=pod["id"],
                game_number=game["game_number"],
                teamA_player_ids=game["teamA"],
                teamB_player_ids=game["teamB"],
                scoreA=11,
                scoreB=7,
                edited_by="admin",
            )
    out2 = svc.closeRound(sb, sessionId=sid, roundNumber=2, updated_by="admin", allow_override=True, override_reason="ok")
    assert out2["auto_completed"] is True
    session = sb.storage["session_ladder_sessions"][0]
    assert session["state"] == "completed"
