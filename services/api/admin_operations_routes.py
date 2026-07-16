from __future__ import annotations

from typing import Any

from jupr_app.services.admin_operations_service import build_admin_operations_status
from services.api.admin_badge_diagnostics_routes import install_admin_badge_diagnostics_routes
from services.api.admin_challenge_ladder_routes import install_admin_challenge_ladder_routes
from services.api.admin_jupr_live_routes import install_admin_jupr_live_routes
from services.api.admin_league_manager_routes import install_admin_league_manager_routes
from services.api.admin_match_canonical_audit_routes import install_admin_match_canonical_audit_routes
from services.api.admin_match_log_routes import install_admin_match_log_routes
from services.api.admin_match_uploader_routes import install_admin_match_uploader_routes
from services.api.admin_moneyball_routes import install_admin_moneyball_routes
from services.api.admin_player_editor_routes import install_admin_player_editor_routes
from services.api.admin_player_updates_routes import install_admin_player_updates_routes
from services.api.admin_replay_routes import install_admin_replay_routes
from services.api.admin_support_requests_routes import install_admin_support_requests_routes
from services.api.admin_tools_routes import install_admin_tools_routes
from services.api.admin_tournament_routes import install_admin_tournament_routes
from services.api.admin_tournament_setup_routes import install_admin_tournament_setup_routes
from services.api.admin_verified_updates_routes import install_admin_verified_updates_routes
from services.api.admin_weekly_recap_routes import install_admin_weekly_recap_routes


def install_admin_operations_routes(app, *, get_supabase_client=None) -> None:
    """Register admin operations status and guarded planning/write routes."""

    @app.get("/admin/operations/status")
    def get_admin_operations_status() -> dict[str, Any]:
        return build_admin_operations_status()

    if get_supabase_client is not None:
        install_admin_match_log_routes(app, get_supabase_client=get_supabase_client)
        install_admin_replay_routes(app, get_supabase_client=get_supabase_client)
        install_admin_match_uploader_routes(app, get_supabase_client=get_supabase_client)
        install_admin_player_editor_routes(app, get_supabase_client=get_supabase_client)
        install_admin_player_updates_routes(app, get_supabase_client=get_supabase_client)
        install_admin_verified_updates_routes(app, get_supabase_client=get_supabase_client)
        install_admin_support_requests_routes(app, get_supabase_client=get_supabase_client)
        install_admin_tools_routes(app, get_supabase_client=get_supabase_client)
        install_admin_moneyball_routes(app, get_supabase_client=get_supabase_client)
        install_admin_jupr_live_routes(app, get_supabase_client=get_supabase_client)
        install_admin_challenge_ladder_routes(app, get_supabase_client=get_supabase_client)
        install_admin_match_canonical_audit_routes(app, get_supabase_client=get_supabase_client)
        install_admin_league_manager_routes(app, get_supabase_client=get_supabase_client)
        install_admin_weekly_recap_routes(app, get_supabase_client=get_supabase_client)
        install_admin_badge_diagnostics_routes(app, get_supabase_client=get_supabase_client)
        install_admin_tournament_setup_routes(app, get_supabase_client=get_supabase_client)
        install_admin_tournament_routes(app, get_supabase_client=get_supabase_client)
