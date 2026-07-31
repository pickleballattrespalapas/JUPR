from pathlib import Path


def test_team_league_operation_history_uses_real_timestamp_column():
    source = Path("jupr_app/services/team_league_service.py").read_text()
    block = source[source.index('"team_league_operations"'):source.index('players = [', source.index('"team_league_operations"'))]
    assert 'order="started_at"' in block
    assert 'order="created_at"' not in block


def test_league_creation_has_structural_mode():
    service = Path("jupr_app/services/admin_league_manager_create_service.py").read_text()
    routes = Path("services/api/admin_league_manager_routes.py").read_text()
    migration = Path("supabase/migrations/20260731210000_league_competition_mode.sql").read_text()
    assert 'league_type: str = "Individual"' in service
    assert 'pattern=r"^(Individual|Team)$"' in routes
    assert "prevent_league_competition_mode_change" in migration
    assert "require_team_league_mode" in migration
