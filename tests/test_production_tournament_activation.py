from __future__ import annotations

import pytest
from jupr_app.services.production_tournament_guard import production_tournament_writes_enabled
from jupr_app.services.admin_tournament_guarded_operation import require_tournament_admin_mutation_runtime, tournament_admin_guarded_runtime_enabled
from jupr_app.services.admin_tournament_ops_service import require_admin_tournament_official_publish_runtime
from jupr_app.services.admin_tournament_team_competition_service import require_admin_team_tournament_runtime
from jupr_app.services.public_tournament_team_service import require_public_team_tournament_mutation_runtime
from jupr_app.services.public_tournament_commerce_service import require_tournament_commerce_mutation_runtime, tournament_commerce_runtime_status

def enable(monkeypatch):
    vals={
      "JUPR_ENV":"production","JUPR_PRODUCTION_WRITE_POLICY":"enabled","JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION":"1",
      "SUPABASE_SERVICE_ROLE_KEY":"server-only","JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS":"1",
      "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS":"1","JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH":"1",
      "JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION":"1","JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES":"1",
      "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES":"1","JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS":"0","JUPR_EMAIL_MODE":"dry_run"}
    for k,v in vals.items(): monkeypatch.setenv(k,v)

def test_gate_fails_closed(monkeypatch):
    enable(monkeypatch); monkeypatch.delenv("JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION")
    assert not production_tournament_writes_enabled()
    with pytest.raises(PermissionError): require_tournament_admin_mutation_runtime("operations")

def test_reviewed_surfaces_open(monkeypatch):
    enable(monkeypatch)
    assert production_tournament_writes_enabled()
    assert tournament_admin_guarded_runtime_enabled("operations")
    require_tournament_admin_mutation_runtime("operations")
    require_admin_tournament_official_publish_runtime()
    require_admin_team_tournament_runtime()
    require_public_team_tournament_mutation_runtime()
    require_tournament_commerce_mutation_runtime(actor_type="PUBLIC_REGISTRANT")
    assert tournament_commerce_runtime_status()["offline_payment_only"] is True
