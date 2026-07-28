from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.domain.gamification.top_performer_awards import TOP_PERFORMER_BADGE_IDS
from jupr_app.services.admin_league_awards_service import _verify_badge_rows


def _storage() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Open",
                "is_active": True,
                "status": "active",
                "min_games": 2,
                "awards_config": {"default_min_games": 2, "default_depth": 1},
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
            {"club_id": "club", "id": 3, "name": "Casey"},
        ],
        "league_ratings": [
            {"club_id": "club", "player_id": 1, "league_name": "Open", "rating": 1600, "starting_rating": 1400, "wins": 5, "losses": 1, "matches_played": 6, "is_active": True},
            {"club_id": "club", "player_id": 2, "league_name": "Open", "rating": 1500, "starting_rating": 1500, "wins": 4, "losses": 2, "matches_played": 6, "is_active": True},
            {"club_id": "club", "player_id": 3, "league_name": "Open", "rating": 1300, "starting_rating": 1320, "wins": 1, "losses": 5, "matches_played": 6, "is_active": True},
        ],
        "admin_activity_log": [],
        "badges": [{"badge_id": badge_id} for badge_id in TOP_PERFORMER_BADGE_IDS.values()],
        "player_badges": [],
    }


def _install(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_league_awards_preview_contract(monkeypatch):
    supabase = FakeSupabase(_storage())
    _install(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/league-manager/leagues/Open/awards/preview",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_awards_preview"
    assert payload["award_count"] >= 1
    assert any(row["player_name"] == "Alex" for row in payload["awards"])


def test_admin_league_awards_close_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(_storage())
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Open/awards/close",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "CLOSE", "award_badges": False},
    )

    assert response.status_code == 400
    assert "CLOSE LEAGUE" in response.json()["detail"]


def test_admin_league_awards_close_updates_metadata_and_audits(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Open/awards/close",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "CLOSE LEAGUE", "award_badges": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["league"]["status"] == "ended"
    assert tables["leagues_metadata"][0]["is_active"] is False
    assert tables["leagues_metadata"][0]["end_awards"]["top_performers"]
    assert tables["leagues_metadata"][0]["end_awards"]["workflow"]["mint"]["status"] == "skipped_by_legacy_request"
    assert {row["action_type"] for row in tables["admin_activity_log"]} >= {
        "league_awards_freeze_admin",
        "league_awards_preview_admin",
        "league_awards_overrides_admin",
    }


def _post(client: TestClient, suffix: str, payload: dict) -> object:
    return client.post(
        f"/admin/clubs/club/league-manager/leagues/Open/awards/{suffix}",
        headers={"Authorization": "Bearer local"},
        json=payload,
    )


def _advance_to_overrides(client: TestClient) -> dict:
    frozen = _post(
        client,
        "freeze",
        {
            "confirmation_text": "FREEZE LEAGUE AWARDS",
            "idempotency_key": "freeze:operation-1",
            "source": "test_league_awards",
        },
    )
    assert frozen.status_code == 200
    preview = _post(
        client,
        "preview",
        {"idempotency_key": "preview:operation-1", "source": "test_league_awards"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    confirmed = _post(
        client,
        "overrides",
        {
            "idempotency_key": "overrides:operation-1",
            "preview_fingerprint": preview_payload["wizard"]["preview"]["fingerprint"],
            "overrides": [],
            "source": "test_league_awards",
        },
    )
    assert confirmed.status_code == 200
    return confirmed.json()


def test_admin_league_awards_badge_verification_includes_context_type():
    expected = [
        {
            "player_id": 1,
            "badge_id": "league_mvp_gold",
            "context_type": "league",
            "context_id": "Open:top_performer:mvp:1",
        }
    ]
    tables = {
        "player_badges": [
            {
                "club_id": "club",
                "player_id": 1,
                "badge_id": "league_mvp_gold",
                "context_type": "tournament",
                "context_id": "Open:top_performer:mvp:1",
            }
        ]
    }
    supabase = FakeSupabase(tables)

    with pytest.raises(RuntimeError, match="found 0 of 1"):
        _verify_badge_rows(supabase, club_id="club", expected=expected)

    tables["player_badges"][0]["context_type"] = "league"
    assert _verify_badge_rows(supabase, club_id="club", expected=expected) == expected


def test_admin_league_awards_persists_freeze_preview_and_override_reason(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    client = TestClient(app)

    frozen = _post(
        client,
        "freeze",
        {
            "confirmation_text": "FREEZE LEAGUE AWARDS",
            "idempotency_key": "freeze:override-test",
            "source": "test_league_awards",
        },
    )
    assert frozen.status_code == 200
    assert frozen.json()["wizard"]["status"] == "frozen"
    assert tables["leagues_metadata"][0]["status"] == "ended"

    preview = _post(
        client,
        "preview",
        {"idempotency_key": "preview:override-test", "source": "test_league_awards"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    award = preview_payload["wizard"]["preview"]["awards"][0]
    replacement_id = next(row["player_id"] for row in preview_payload["eligible_players"] if row["player_id"] != award["player_id"])

    missing_reason = _post(
        client,
        "overrides",
        {
            "idempotency_key": "overrides:missing-reason",
            "preview_fingerprint": preview_payload["wizard"]["preview"]["fingerprint"],
            "overrides": [{"category_key": award["category_key"], "rank": award["rank"], "player_id": replacement_id}],
        },
    )
    assert missing_reason.status_code == 400
    assert "reason" in missing_reason.json()["detail"].lower()

    saved = _post(
        client,
        "overrides",
        {
            "idempotency_key": "overrides:documented",
            "preview_fingerprint": preview_payload["wizard"]["preview"]["fingerprint"],
            "overrides": [
                {
                    "category_key": award["category_key"],
                    "rank": award["rank"],
                    "player_id": replacement_id,
                    "reason": "Committee correction after score review",
                }
            ],
        },
    )
    assert saved.status_code == 200
    payload = saved.json()
    override_key = f"{award['category_key']}:{award['rank']}"
    assert payload["wizard"]["status"] == "overrides_confirmed"
    assert payload["wizard"]["override_notes"][override_key] == "Committee correction after score review"
    assert tables["leagues_metadata"][0]["end_awards"]["workflow"]["override_notes"][override_key]


def test_admin_league_awards_mint_never_false_succeeds_and_same_key_retries(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_league_awards_service.mint_top_performer_badges",
        lambda *_args, **_kwargs: [object()],
    )
    client = TestClient(app)
    confirmed = _advance_to_overrides(client)

    mint_request = {
        "confirmation_text": "MINT AWARDS",
        "idempotency_key": "mint:retry-same-operation",
        "source": "test_league_awards",
    }
    failed = _post(client, "mint", mint_request)
    assert failed.status_code == 500
    assert "does not report success" in failed.json()["detail"]
    workflow = tables["leagues_metadata"][0]["end_awards"]["workflow"]
    assert workflow["status"] == "mint_failed"
    assert workflow["mint"]["status"] == "failed"

    for award in confirmed["wizard"]["final_awards"]:
        tables["player_badges"].append(
            {
                "club_id": "club",
                "player_id": award["player_id"],
                "badge_id": TOP_PERFORMER_BADGE_IDS[award["category_key"]],
                "context_type": "league",
                "context_id": f"Open:top_performer:{award['category_key']}:{award['rank']}",
            }
        )

    retried = _post(client, "mint", mint_request)
    assert retried.status_code == 200
    payload = retried.json()
    assert payload["wizard"]["status"] == "minted"
    assert payload["wizard"]["mint"]["attempt_count"] == 2
    assert payload["badge_verified_count"] == payload["badge_expected_count"] == len(tables["player_badges"])

    replayed = _post(client, "mint", mint_request)
    assert replayed.status_code == 200
    assert replayed.json()["idempotent_replay"] is True


def test_admin_league_awards_mint_fails_closed_when_badge_definitions_are_missing(monkeypatch):
    tables = _storage()
    missing_badge_id = tables["badges"].pop()["badge_id"]
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    client = TestClient(app)
    confirmed = _advance_to_overrides(client)
    workflow_before = tables["leagues_metadata"][0]["end_awards"]["workflow"]
    revision_before = workflow_before["revision"]

    response = _post(
        client,
        "mint",
        {
            "confirmation_text": "MINT AWARDS",
            "idempotency_key": "mint:missing-badge-definition",
            "source": "test_league_awards",
        },
    )

    assert confirmed["badge_definitions_ready"] is False
    assert missing_badge_id in confirmed["missing_badge_ids"]
    assert response.status_code == 500
    assert "mint was not attempted" in response.json()["detail"]
    assert "supabase/migrations/20260720014744_seed_top_performer_badges.sql" in response.json()["detail"]
    workflow_after = tables["leagues_metadata"][0]["end_awards"]["workflow"]
    assert workflow_after["status"] == "overrides_confirmed"
    assert workflow_after["revision"] == revision_before
    assert workflow_after["mint"]["attempt_count"] == 0


def test_admin_league_awards_archive_requires_verified_mint_and_is_idempotent(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    client = TestClient(app)

    blocked = _post(
        client,
        "archive",
        {
            "confirmation_text": "ARCHIVE LEAGUE",
            "idempotency_key": "archive:blocked-before-mint",
        },
    )
    assert blocked.status_code == 400
    assert "mint" in blocked.json()["detail"].lower()

    _advance_to_overrides(client)
    workflow = tables["leagues_metadata"][0]["end_awards"]["workflow"]
    workflow["status"] = "minted"
    workflow["mint"] = {"status": "verified", "expected_count": 0, "verified_count": 0, "attempt_count": 1, "attempts": []}

    archive_request = {
        "confirmation_text": "ARCHIVE LEAGUE",
        "idempotency_key": "archive:verified-operation",
        "source": "test_league_awards",
    }
    archived = _post(client, "archive", archive_request)
    assert archived.status_code == 200
    assert archived.json()["wizard"]["status"] == "archived"
    assert tables["leagues_metadata"][0]["status"] == "archived"

    replayed = _post(client, "archive", archive_request)
    assert replayed.status_code == 200
    assert replayed.json()["idempotent_replay"] is True


def test_admin_league_awards_writes_require_separate_flag(monkeypatch):
    supabase = FakeSupabase(_storage())
    _install(monkeypatch, supabase)
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", raising=False)

    response = _post(
        TestClient(app),
        "freeze",
        {"confirmation_text": "FREEZE LEAGUE AWARDS", "idempotency_key": "freeze:closed-gate"},
    )

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()


def test_admin_league_awards_production_refusal_precedes_data_access(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "enabled")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("data access must not run")
        ),
    )

    response = _post(
        TestClient(app),
        "freeze",
        {
            "confirmation_text": "FREEZE LEAGUE AWARDS",
            "idempotency_key": "freeze:production-refusal",
        },
    )

    assert response.status_code == 403
    assert "staging-only" in response.json()["detail"].lower()


def test_admin_league_awards_strict_audit_fails_before_freeze_mutation(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_league_awards_service.write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="audit unavailable"),
    )

    response = _post(
        TestClient(app),
        "freeze",
        {"confirmation_text": "FREEZE LEAGUE AWARDS", "idempotency_key": "freeze:strict-audit"},
    )

    assert response.status_code == 500
    assert "audit log write required" in response.json()["detail"]
    assert tables["leagues_metadata"][0]["status"] == "active"
    assert tables["leagues_metadata"][0].get("end_awards") is None
