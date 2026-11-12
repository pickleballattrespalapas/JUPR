from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import HTTPException, Query, Response

from jupr_app.services.round_robin_history_service import get_round_robin_history, list_round_robin_history
from services.api.club_site_routes import published_shell


def install_public_round_robin_history_routes(app, *, get_supabase_client):
    def context(slug, response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client()
        site = published_shell(db, slug)
        if site["document"]["visibility"] == "unlisted":
            response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return db, site["club_id"]

    @app.get("/clubs/{club_slug}/round-robins")
    def history(club_slug: str, response: Response,
                kind: Literal["all", "round_robin", "moneyball"] = "all",
                from_date: date | None = None, to_date: date | None = None,
                offset: int = Query(default=0, ge=0, le=100000),
                limit: int = Query(default=24, ge=1, le=50)):
        db, club_id = context(club_slug, response)
        try:
            return list_round_robin_history(db, club_id=club_id, kind=kind,
                from_date=from_date, to_date=to_date, offset=offset, limit=limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "Round robin results are temporarily unavailable.") from exc

    @app.get("/clubs/{club_slug}/round-robins/{session_key}")
    def results(club_slug: str, session_key: str, response: Response):
        db, club_id = context(club_slug, response)
        try:
            return get_round_robin_history(db, club_id=club_id, session_key=session_key)
        except LookupError as exc:
            raise HTTPException(404, "This round robin is not available.") from exc
        except Exception as exc:
            raise HTTPException(503, "Round robin results are temporarily unavailable.") from exc
