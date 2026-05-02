from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from supabase import Client, create_client

from jupr_app.services.leaderboard_service import get_public_leaderboard


app = FastAPI(title="JUPR API", version="0.1.0")


def _get_supabase_credentials() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    key = service_role_key or anon_key

    if not url or not key:
        raise RuntimeError(
            "Supabase config missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or SUPABASE_ANON_KEY for read-only local development)."
        )
    return url, key


def get_supabase_client() -> Client:
    url, key = _get_supabase_credentials()
    return create_client(url, key)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "jupr-api"}


@app.get("/clubs/{club_slug}")
def get_club(club_slug: str) -> dict[str, Any]:
    slug = str(club_slug).strip()
    if not slug:
        raise HTTPException(status_code=400, detail="club_slug is required")

    supabase = get_supabase_client()

    row = (
        supabase.table("clubs_config")
        .select("club_id,club_slug,club_name,display_name,is_public")
        .eq("club_slug", slug)
        .limit(1)
        .execute()
        .data
        or []
    )

    if not row:
        fallback = (
            supabase.table("players")
            .select("club_id")
            .eq("club_id", slug)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not fallback:
            raise HTTPException(status_code=404, detail="club not found")
        return {
            "club_id": slug,
            "club_slug": slug,
            "club_name": slug,
            "display_name": slug,
            "is_public": True,
        }

    return row[0]


@app.get("/clubs/{club_slug}/leaderboards")
def get_club_leaderboard(club_slug: str, league_name: str | None = Query(default=None)) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    rows = get_public_leaderboard(supabase=supabase, club_id=club_id, league_name=league_name)
    return {
        "club": {
            "club_id": club.get("club_id", club_id),
            "club_slug": club.get("club_slug", club_slug),
            "club_name": club.get("club_name") or club.get("display_name") or club_slug,
        },
        "leaderboard": rows,
    }
