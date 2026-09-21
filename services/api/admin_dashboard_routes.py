"""Authenticated read boundary for the Admin Home action center."""
from __future__ import annotations

from typing import Any

from fastapi import Path, Response

from jupr_app.services.admin_dashboard_service import build_admin_dashboard
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header


def install_admin_dashboard_routes(app, *, get_supabase_client) -> None:
    @app.get("/admin/clubs/{club_id}/dashboard")
    def get_admin_dashboard(
        response: Response,
        club_id: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,100}$"),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        db = get_supabase_client()
        _, assignments = require_admin_assignments(
            get_supabase_client=lambda: db,
            authorization=authorization,
            requested_club_id=club_id,
        )
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Authorization"
        return build_admin_dashboard(db, club_id=club_id, assignments=assignments)
