from dataclasses import replace
from pathlib import Path

import pytest

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.services.production_feature_policy import production_feature_enabled
from scripts.activate_operations_badges import select_additions
from scripts.deployment_verifier import PRODUCTION_ENABLED_FEATURE_FLAGS, PRODUCTION_PRE_OPERATIONS_ENABLED_FEATURE_FLAGS, PRODUCTION_PRE_TEAM_LEAGUES_ENABLED_FEATURE_FLAGS


@pytest.fixture
def production(monkeypatch):
    values = {"JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "server-fixture",
        "JUPR_PRODUCTION_WRITE_POLICY": "enabled", "JUPR_STAGING_WRITE_WAVE": "none",
        "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS": "1", "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE": "1"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("feature", ["badges", "generators"])
def test_reviewed_production_club_is_required(production, feature):
    assert production_feature_enabled(feature, "tres_palapas")
    assert not production_feature_enabled(feature, "another_club")
    assert not production_feature_enabled("interclub", "tres_palapas")


@pytest.mark.parametrize("key,value", [
    ("JUPR_ENV", "staging"), ("FLY_APP_NAME", "juprleagues-api-staging"),
    ("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co"),
    ("SUPABASE_SERVICE_ROLE_KEY", ""), ("JUPR_PRODUCTION_WRITE_POLICY", "disabled"),
    ("JUPR_STAGING_WRITE_WAVE", "open"),
])
def test_production_gate_rejects_environment_drift(production, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    assert not production_feature_enabled("badges")
    assert not production_feature_enabled("generators")


def test_activation_excludes_existing_and_revoked_badge_identities():
    c = BadgeCandidate(badge_id="clutch_performer", player_id=1, club_id="tres_palapas", context_type="overall", context_id="new", match_id=None)
    existing = [{"player_id": 1, "badge_id": "clutch_performer", "context_type": "overall", "context_id": "legacy", "revoked_at": "2026-01-01"}]
    other = replace(c, player_id=2)
    assert select_additions([c, other, other, replace(c, badge_id="good_sport")], existing) == [other]


def test_release_preserves_every_preexisting_production_gate():
    assert PRODUCTION_PRE_OPERATIONS_ENABLED_FEATURE_FLAGS <= PRODUCTION_ENABLED_FEATURE_FLAGS
    assert PRODUCTION_PRE_TEAM_LEAGUES_ENABLED_FEATURE_FLAGS - PRODUCTION_PRE_OPERATIONS_ENABLED_FEATURE_FLAGS == {
        "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS", "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
        "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", "JUPR_ENABLE_NEXT_ADMIN_SHELL"}


def test_deferred_routes_are_absent_from_release():
    from services.api.main import app
    routes = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/admin/clubs/{club_id}/leaderboard-settings" in routes
    assert "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/review" in routes
    assert not any("interclub" in path or path.startswith("/admin/platform") or path == "/clubs/create" for path in routes)
    root = Path(__file__).resolve().parents[1]
    for path in ["apps/web/app/admin/interclub", "apps/web/app/admin/platform", "apps/web/app/clubs/create"]:
        assert not (root / path).exists()


def test_generator_route_gate_opens_only_the_reviewed_club(production):
    from fastapi import HTTPException
    from services.api.admin_play_generator_routes import _require_write_gate
    _require_write_gate("tres_palapas")
    with pytest.raises(HTTPException) as error:
        _require_write_gate("another_club")
    assert error.value.status_code == 403
