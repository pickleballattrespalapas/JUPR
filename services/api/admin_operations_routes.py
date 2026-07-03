from __future__ import annotations

from typing import Any

from jupr_app.services.admin_operations_service import build_admin_operations_status


def install_admin_operations_routes(app) -> None:
    """Register status-only routes for the Next admin migration cockpit."""

    @app.get("/admin/operations/status")
    def get_admin_operations_status() -> dict[str, Any]:
        return build_admin_operations_status()
