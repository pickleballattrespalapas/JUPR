from __future__ import annotations

from typing import Any

from jupr_app.services.admin_operations_service import build_admin_operations_status
from services.api.admin_match_log_routes import install_admin_match_log_routes


def install_admin_operations_routes(app, *, get_supabase_client=None) -> None:
    """Register admin operations status and read-only planning routes."""

    @app.get("/admin/operations/status")
    def get_admin_operations_status() -> dict[str, Any]:
        return build_admin_operations_status()

    if get_supabase_client is not None:
        install_admin_match_log_routes(app, get_supabase_client=get_supabase_client)
