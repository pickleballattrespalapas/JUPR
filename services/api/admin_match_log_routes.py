from __future__ import annotations

from typing import Any

from fastapi import Query
from supabase import Client

from jupr_app.services.admin_match_log_service import build_admin_match_log


def install_admin_match_log_routes(app, *, get_supabase_client) -> None:
    """Register non-mutating Match Log planning routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/match-log")
    def get_admin_match_log(
        club_id: str,
        filter_type: str = Query(default="All", alias="filter"),
        match_id: int | None = Query(default=None),
        league: str | None = Query(default=None),
        week_tag: str | None = Query(default=None),
        start_date: str | None = Query(default=None),
        end_date: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=1000),
    ) -> dict[str, Any]:
        supabase: Client = get_supabase_client()
        return build_admin_match_log(
            supabase,
            club_id=str(club_id),
            filter_type=filter_type,
            match_id=match_id,
            league=league,
            week_tag=week_tag,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
