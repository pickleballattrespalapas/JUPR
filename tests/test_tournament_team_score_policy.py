from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services import admin_tournament_team_competition_service as service
from services.api import admin_tournament_team_competition_routes as routes
from services.api.admin_tournament_team_competition_routes import ScoreRequest


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters = []
        self.limit_count = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_count = int(value)
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        if self.limit_count is not None:
            rows = rows[: self.limit_count]
        return SimpleNamespace(data=rows)


class _Rpc:
    def __init__(self, owner, name, params):
        self.owner = owner
        self.name = name
        self.params = params

    def execute(self):
        self.owner.rpc_calls.append((self.name, self.params))
        return SimpleNamespace(data={"ok": True, "score_review": self.params["p_score_review"]})


class _Supabase:
    def __init__(self):
        self.tables = {
            "tournament_team_match_games": [
                {
                    "id": "game-1",
                    "tournament_id": "tournament-1",
                    "matchup_id": "matchup-1",
                }
            ],
            "tournament_team_matchups": [
                {
                    "id": "matchup-1",
                    "tournament_id": "tournament-1",
                    "event_option_id": "event-1",
                }
            ],
            "tournament_event_options": [
                {
                    "id": "event-1",
                    "tournament_id": "tournament-1",
                    "scoring_default": "GAME_TO_11",
                }
            ],
        }
        self.rpc_calls = []

    def table(self, name):
        return _Query(self.tables.get(name, []))

    def rpc(self, name, params):
        return _Rpc(self, name, params)


def _score(supabase, **overrides):
    kwargs = {
        "club_id": "club-1",
        "tournament_id": "tournament-1",
        "match_game_id": "game-1",
        "score_a": 11,
        "score_b": 7,
        "expected_game_version": 1,
        "expected_matchup_version": 1,
        "actor_email": "owner@example.com",
        "idempotency_key": "team-score-request-1",
    }
    kwargs.update(overrides)
    return service.score_team_match_game(supabase, **kwargs)


def test_team_score_uses_configured_policy_and_reviewed_atomic_rpc(monkeypatch):
    monkeypatch.setattr(service, "require_admin_team_tournament_runtime", lambda: None)
    supabase = _Supabase()

    result = _score(supabase)

    assert result["score_review"]["status"] == "ordinary"
    name, params = supabase.rpc_calls[0]
    assert name == "admin_score_tournament_team_match_game_reviewed_cas"
    assert params["p_score_review"]["scoring_format"] == "GAME_TO_11"
    assert params["p_score_review"]["score_a"] == 11
    assert params["p_score_review"]["accepted"] is True


def test_team_unusual_score_requires_explicit_acknowledgement(monkeypatch):
    monkeypatch.setattr(service, "require_admin_team_tournament_runtime", lambda: None)
    supabase = _Supabase()

    with pytest.raises(ValueError, match="explicit acknowledgement"):
        _score(supabase, score_a=76, score_b=11)
    assert supabase.rpc_calls == []

    accepted = _score(
        supabase,
        score_a=76,
        score_b=11,
        unusual_score_acknowledged=True,
    )
    assert accepted["score_review"]["status"] == "unusual"
    assert accepted["score_review"]["acknowledged"] is True


def test_team_impossible_score_is_rejected_before_atomic_operation(monkeypatch):
    monkeypatch.setattr(service, "require_admin_team_tournament_runtime", lambda: None)
    supabase = _Supabase()

    with pytest.raises(ValueError, match="Impossible tournament score"):
        _score(supabase, score_a=10, score_b=7)
    assert supabase.rpc_calls == []


def test_team_score_api_contract_carries_acknowledgement() -> None:
    defaulted = ScoreRequest.model_validate(
        {
            "idempotency_key": "team-score-request-1",
            "confirmation_text": "SAVE TEAM SCORE",
            "score_a": 11,
            "score_b": 7,
            "expected_game_version": 1,
            "expected_matchup_version": 1,
        }
    )
    acknowledged = ScoreRequest.model_validate(
        {
            **defaulted.model_dump(),
            "unusual_score_acknowledged": True,
        }
    )

    assert defaulted.unusual_score_acknowledged is False
    assert acknowledged.unusual_score_acknowledged is True


def test_team_score_route_forwards_acknowledgement(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(routes, "is_admin_team_tournament_enabled", lambda: True)
    monkeypatch.setattr(
        routes, "require_tournament_admin_mutation_runtime", lambda _surface: None
    )
    monkeypatch.setattr(
        routes,
        "_resolve_manage_role_or_403",
        lambda **_kwargs: ("owner@example.com", "club_owner"),
    )

    def score(_supabase, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(routes, "score_team_match_game", score)
    app = FastAPI()
    routes.install_admin_tournament_team_competition_routes(
        app,
        get_supabase_client=lambda: object(),
    )

    response = TestClient(app).post(
        "/admin/clubs/club-1/tournaments/admin/tournaments/tournament-1/"
        "team-competition/games/game-1/score",
        headers={"Authorization": "Bearer local"},
        json={
            "idempotency_key": "team-score-request-1",
            "confirmation_text": "SAVE TEAM SCORE",
            "score_a": 76,
            "score_b": 11,
            "unusual_score_acknowledged": True,
            "expected_game_version": 1,
            "expected_matchup_version": 1,
        },
    )

    assert response.status_code == 200, response.text
    assert captured["unusual_score_acknowledged"] is True
