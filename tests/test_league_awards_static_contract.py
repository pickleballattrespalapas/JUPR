from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOP_PERFORMER_BADGE_SEED = "supabase/migrations/20260720014744_seed_top_performer_badges.sql"


def test_league_awards_browser_is_a_persisted_fastapi_wizard() -> None:
    panel = (ROOT / "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx").read_text()
    assert "Recover saved state" in panel
    assert "FREEZE LEAGUE AWARDS" in panel
    assert "preview_fingerprint" in panel
    assert "Override reason" in panel
    assert "MINT AWARDS" in panel
    assert "ARCHIVE LEAGUE" in panel
    assert "SUPABASE_SERVICE_ROLE_KEY" not in panel
    assert "NEXT_PUBLIC_SUPABASE_SERVICE" not in panel
    assert ".from(" not in panel
    assert "supabase.table" not in panel


def test_league_awards_fastapi_exposes_separate_recoverable_actions() -> None:
    routes = (ROOT / "services/api/admin_league_manager_routes.py").read_text()
    for suffix in ["awards\")", "awards/freeze", "awards/preview", "awards/overrides", "awards/mint", "awards/archive"]:
        assert suffix in routes
    assert "authenticate_bearer" in routes
    assert "resolve_admin_role" in routes


def test_league_awards_mint_requires_read_after_write_verification() -> None:
    service = (ROOT / "jupr_app/services/admin_league_awards_service.py").read_text()
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE" in service
    assert "SUPABASE_SERVICE_ROLE_KEY" in service
    assert "_verify_badge_rows" in service
    assert "Badge mint could not be verified" in service
    assert 'workflow["status"] = "mint_failed"' in service
    assert 'workflow["status"] = "minted"' in service
    assert "idempotency_key" in service
    assert "end_awards" in service


def test_league_awards_mint_fails_closed_without_seeded_badge_definitions() -> None:
    service = (ROOT / "jupr_app/services/admin_league_awards_service.py").read_text()
    panel = (ROOT / "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx").read_text()
    assert "_badge_definition_readiness" in service
    assert "badge_definitions_ready" in service
    assert "mint was not attempted" in service
    assert TOP_PERFORMER_BADGE_SEED in service
    assert "badge_definitions_ready" in panel
    assert "Badge minting is blocked" in panel


def test_league_awards_staging_gate_is_explicitly_disabled_in_production_config() -> None:
    staging = (ROOT / "fly.staging.toml").read_text()
    production = tomllib.loads((ROOT / "fly.toml").read_text())
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text()
    flag = "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE"
    assert flag in staging
    assert flag in workflow
    assert production["env"][flag] == "0"


def test_league_awards_manual_evidence_keeps_parity_partial() -> None:
    evidence = (ROOT / "docs/league_awards_parity_evidence.md").read_text()
    assert "Parity remains Partial" in evidence
    assert "manual staging gate" in evidence.lower()
    assert "false-success" in evidence
