from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Response
from supabase import Client

from jupr_app.services.public_weekly_recap_service import (
    build_public_weekly_recaps,
    build_weekly_recap_pdf_bytes,
)
from services.api.admin_operations_routes import install_admin_operations_routes
from services.api.public_email_preferences_routes import install_public_email_preferences_routes
from services.api.public_support_intake_routes import install_public_support_intake_routes
from services.api.public_tournament_pairing_routes import install_public_tournament_pairing_routes
from services.api.public_tournament_registration_routes import install_public_tournament_registration_routes
from services.api.public_verified_updates_routes import install_public_verified_updates_routes


def install_public_weekly_recap_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public Weekly Recap plus late public/admin status routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/weekly-recaps")
    def get_club_weekly_recaps(
        club_slug: str,
        week_start: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        recaps = build_public_weekly_recaps(supabase, club_id=club_id, week_start=week_start)
        return {"club": public_club_payload(club, club_slug), **recaps}

    @app.get("/clubs/{club_slug}/weekly-recaps/{week_start}")
    def get_club_weekly_recap_detail(club_slug: str, week_start: str) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        recaps = build_public_weekly_recaps(supabase, club_id=club_id, week_start=week_start)
        if not recaps.get("selected_recap"):
            raise HTTPException(status_code=404, detail="weekly recap not found")
        return {"club": public_club_payload(club, club_slug), **recaps}

    @app.get("/clubs/{club_slug}/weekly-recaps/{week_start}/pdf")
    def get_club_weekly_recap_pdf(club_slug: str, week_start: str) -> Response:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        recaps = build_public_weekly_recaps(supabase, club_id=club_id, week_start=week_start)
        selected = recaps.get("selected_recap")
        if not selected:
            raise HTTPException(status_code=404, detail="weekly recap not found")
        pdf = build_weekly_recap_pdf_bytes(
            selected.get("recap") or {},
            week_start=str(selected.get("week_start") or week_start),
            week_end=str(selected.get("week_end") or ""),
        )
        filename = f"weekly_recap_{str(selected.get('week_start') or week_start)}.pdf"
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    install_public_tournament_registration_routes(
        app,
        get_club=get_club,
        get_supabase_client=get_supabase_client,
        public_club_payload=public_club_payload,
    )
    install_public_tournament_pairing_routes(
        app,
        get_club=get_club,
        get_supabase_client=get_supabase_client,
        public_club_payload=public_club_payload,
    )
    install_public_support_intake_routes(
        app,
        get_club=get_club,
        get_supabase_client=get_supabase_client,
        public_club_payload=public_club_payload,
    )
    install_public_verified_updates_routes(
        app,
        get_club=get_club,
        get_supabase_client=get_supabase_client,
        public_club_payload=public_club_payload,
    )
    install_public_email_preferences_routes(app, get_supabase_client=get_supabase_client)
    install_admin_operations_routes(app, get_supabase_client=get_supabase_client)
