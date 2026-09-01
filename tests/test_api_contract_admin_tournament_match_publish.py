from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.services.admin_tournament_podium_review_service import (
    PODIUM_REVIEW_ACTION,
    PODIUM_REVIEW_CONTRACT,
    build_admin_tournament_podium_review_fingerprint,
)


def match_publish_tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "start_date": "2026-04-10",
                "created_at": "2026-03-01T00:00:00Z",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_family_label": "Men's Doubles",
                "division_name": "4.0",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "name": "Men's Doubles 4.0",
                "status": "draft",
                "updated_at": "2026-04-10T16:00:00Z",
            }
        ],
        "tournament_teams": [
            {
                "id": "team_1",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "team_number": 1,
                "player1_id": 1,
                "player2_id": 2,
                "updated_at": "2026-04-10T16:00:00Z",
            },
            {
                "id": "team_2",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "team_number": 2,
                "player1_id": 3,
                "player2_id": 4,
                "updated_at": "2026-04-10T16:00:00Z",
            },
        ],
        "tournament_games": [
            {
                "id": "game_1",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "stage": "ROUND_ROBIN",
                "rr_round_number": 1,
                "rr_slot_number": 1,
                "team_a_id": "team_1",
                "team_b_id": "team_2",
                "score_a": 11,
                "score_b": 8,
                "winner_team_id": "team_1",
                "loser_team_id": "team_2",
                "finalized_at": "2026-04-10T17:00:00Z",
                "updated_at": "2026-04-10T17:00:00Z",
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        ],
        "league_ratings": [],
        "leagues_metadata": [],
        "matches": [],
        "admin_activity_log": [],
    }


def install_official_publish_prerequisites(tables: dict[str, list[dict]]) -> None:
    """Make a publish fixture satisfy tournament-wide closeout prerequisites."""

    draw = tables["tournament_event_draws"][0]
    draw_id = str(draw["id"])
    updated_at = str(draw.get("updated_at") or "2026-04-10T17:00:00Z")
    draw["updated_at"] = updated_at
    teams = [row for row in tables["tournament_teams"] if str(row.get("draw_id")) == draw_id]
    for team in teams:
        team.setdefault("updated_at", updated_at)
    if len(teams) < 3:
        used_player_ids = {
            int(player_id)
            for team in teams
            for player_id in (team.get("player1_id"), team.get("player2_id"))
            if player_id is not None
        }
        next_player_id = max(used_player_ids or {0}) + 1
        singles = all(team.get("player2_id") is None for team in teams)
        team = {
            "id": "team_3",
            "tournament_id": "tour_1",
            "draw_id": draw_id,
            "team_number": 3,
            "player1_id": next_player_id,
            "player2_id": None if singles else next_player_id + 1,
            "updated_at": updated_at,
        }
        tables["tournament_teams"].append(team)
        teams.append(team)
        tables.setdefault("players", []).append(
            {"club_id": "club", "id": next_player_id, "name": f"Player {next_player_id}", "rating": 1200}
        )
        if team["player2_id"] is not None:
            tables["players"].append(
                {"club_id": "club", "id": next_player_id + 1, "name": f"Player {next_player_id + 1}", "rating": 1200}
            )
    games = [row for row in tables["tournament_games"] if str(row.get("draw_id")) == draw_id]
    for game in games:
        game.setdefault("updated_at", updated_at)
    podium = [
        {
            "id": f"podium_{placement}",
            "tournament_id": "tour_1",
            "draw_id": draw_id,
            "placement": placement,
            "team_id": str(teams[placement - 1]["id"]),
            "source": "ROUND_ROBIN",
            "updated_at": updated_at,
        }
        for placement in (1, 2, 3)
    ]
    tables["tournament_podium"] = podium
    badges: list[dict] = []
    for podium_row in podium:
        placement = int(podium_row["placement"])
        team = next(row for row in teams if row["id"] == podium_row["team_id"])
        for player_id in (team.get("player1_id"), team.get("player2_id")):
            if player_id is None:
                continue
            badges.append(
                {
                    "club_id": "club",
                    "player_id": player_id,
                    "badge_id": PODIUM_BADGE_MAP[placement],
                    "context_type": "tournament",
                    "context_id": f"tour_1:draw:{draw_id}:podium:{placement}",
                    "revoked_at": None,
                }
            )
    tables["player_badges"] = badges
    fingerprint = build_admin_tournament_podium_review_fingerprint(
        draw=draw,
        teams=teams,
        games=games,
        podium=podium,
    )
    reviews = [
        row
        for row in tables.setdefault("admin_activity_log", [])
        if row.get("action_type") != PODIUM_REVIEW_ACTION
    ]
    reviews.append(
        {
            "club_id": "club",
            "actor_email": "reviewer@example.com",
            "action_type": PODIUM_REVIEW_ACTION,
            "entity_type": "tournament_event_draw",
            "entity_id": draw_id,
            "after_json": {
                "podium_review_evidence": {
                    "contract": PODIUM_REVIEW_CONTRACT,
                    "tournament_id": "tour_1",
                    "draw_id": draw_id,
                    "review_fingerprint": fingerprint,
                }
            },
        }
    )
    tables["admin_activity_log"] = reviews
    tables.setdefault("tournament_admin_operations", [])
    tournament = tables["tournaments"][0]
    event_option = next(
        (
            row
            for row in tables.get("tournament_event_options", [])
            if str(row.get("id") or "") == str(draw.get("event_option_id") or "")
        ),
        {},
    )
    family = str(event_option.get("event_family_label") or event_option.get("label") or "").strip()
    division = str(event_option.get("division_name") or event_option.get("label") or "").strip()
    division_label = (
        f"{family} / {division}"
        if family and division and family != division
        else division or family or str(draw.get("name") or "Tournament Draw")
    )
    match_format = (
        "singles"
        if teams and all(team.get("player2_id") is None for team in teams)
        else "doubles"
    )
    tables["leagues_metadata"] = [
        {
            "id": 1,
            "club_id": "club",
            "league_name": f"Tournament · {tournament['name']} · {division_label}",
            "k_factor": 32,
            "status": "active",
            "is_active": True,
            "ended_at": None,
            "match_format": match_format,
        }
    ]


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)


def test_admin_tournament_publish_matches_contract(monkeypatch):
    tables = match_publish_tables()
    install_official_publish_prerequisites(tables)
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        captured["club_id"] = kwargs.get("club_id")
        return {"inserted": len(match_list), "badge_summary": {"mode": "test"}, "player_update_queue": {"mode": "test"}}

    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", fake_process_matches)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_official_matches_publish"
    assert payload["match_count"] == 1
    assert payload["game_count"] == 1
    match_payload = captured["match_list"][0]
    assert match_payload["match_type"] == "Tournament"
    assert match_payload["tournament_id"] == "tour_1"
    assert match_payload["tournament_game_id"] == "game_1"
    assert match_payload["t1_p1"] == 1
    assert match_payload["t1_p2"] == 2
    assert match_payload["t2_p1"] == 3
    assert match_payload["t2_p2"] == 4
    assert match_payload["score_t1"] == 11
    assert match_payload["score_t2"] == 8
    assert "Spring Classic" in match_payload["league"]
    audit = next(
        row
        for row in tables["admin_activity_log"]
        if row["action_type"] == "publish_tournament_games_to_matches_admin"
    )
    assert audit["flagged_for_review"] is True


def test_admin_tournament_publish_matches_applies_semifinal_bonus_to_winner_only_payload(monkeypatch):
    tables = match_publish_tables()
    tables["tournament_games"][0].update({"stage": "PLAYOFF", "playoff_round": "SF", "playoff_game_code": "P1"})
    install_official_publish_prerequisites(tables)
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        return {"inserted": len(match_list), "winner_bonus_summary": {"match_count": 1, "player_elo_total": 12.0}}

    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", fake_process_matches)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES", "playoff_winner_bonus_elo": 6},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["playoff_winner_bonus_elo"] == 6.0
    assert payload["bonus_match_count"] == 1
    match_payload = captured["match_list"][0]
    assert match_payload["winner_bonus_elo"] == 6.0
    assert match_payload["rating_bonus_elo"] == 6.0
    assert match_payload["winner_bonus_reason"] == "tournament_semifinal_winner_bonus"
    audit = next(
        row
        for row in tables["admin_activity_log"]
        if row["action_type"] == "publish_tournament_games_to_matches_admin"
    )
    assert audit["after_json"]["bonus_tournament_game_ids"] == ["game_1"]


def test_admin_tournament_publish_matches_blocks_duplicate(monkeypatch):
    tables = match_publish_tables()
    install_official_publish_prerequisites(tables)
    tables["matches"] = [
        {
            "id": "match-game-1",
            "club_id": "club",
            "tournament_id": "tour_1",
            "tournament_game_id": "game_1",
            "context_type": "tournament_game",
            "context_id": "game_1",
            "match_format": "doubles",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 8,
            "row_version": 1,
            "deleted_at": None,
        }
    ]
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", lambda *_args, **_kwargs: {"inserted": 1})

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "already" in response.json()["detail"].lower()
    assert "official Match Log link" in response.json()["detail"]


def test_incomplete_publish_is_blocked_before_durable_intent(monkeypatch):
    tables = match_publish_tables()
    tables["tournament_admin_operations"] = []
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-official-publish")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "1",
    )
    monkeypatch.setenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
        "1",
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "official publishing is blocked" in response.json()["detail"]
    assert tables["matches"] == []
    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []


def test_admin_tournament_publish_matches_requires_confirmation(monkeypatch):
    tables = match_publish_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH"},
    )

    assert response.status_code == 400
    assert "PUBLISH MATCHES" in response.json()["detail"]


def test_admin_tournament_publish_matches_requires_doubles(monkeypatch):
    tables = match_publish_tables()
    tables["tournament_teams"][0]["player2_id"] = None
    install_official_publish_prerequisites(tables)
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", lambda *_args, **_kwargs: {"inserted": 1})

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "doubles teams" in response.json()["detail"]
