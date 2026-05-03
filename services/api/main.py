from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client, create_client

from jupr_app.data.load import load_data
from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES
from jupr_app.services.context import ServiceContext
from jupr_app.services.leaderboard_service import get_public_leaderboard
from jupr_app.services.match_service import submit_match_batch


app = FastAPI(title="JUPR API", version="0.1.0")


class MatchBatchRequest(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_admin_score_entry"


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


def _authorize_score_entry(*, token: str | None, requested_permission: str) -> str:
    """Temporary guard for v1 admin workflow.

    This is intentionally a placeholder and NOT production auth. It verifies an
    environment token and leaves room for future Supabase JWT + role validation.
    """

    if requested_permission != PERMISSION_ENTER_SCORES:
        raise HTTPException(status_code=403, detail="insufficient permission")

    expected = os.getenv("JUPR_ADMIN_API_TOKEN", "").strip()
    provided = str(token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Admin score-entry token is not configured. Set JUPR_ADMIN_API_TOKEN. "
                "This endpoint currently uses a placeholder token guard."
            ),
        )
    if provided != expected:
        raise HTTPException(status_code=401, detail="invalid admin token")

    return "token_guard_placeholder"


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


@app.post("/admin/clubs/{club_id}/matches/batch")
def submit_admin_match_batch(
    club_id: str,
    payload: MatchBatchRequest,
    x_admin_token: str | None = Header(default=None),
    x_admin_permission: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_mode = _authorize_score_entry(token=x_admin_token, requested_permission=str(x_admin_permission or ""))

    supabase = get_supabase_client()
    df_players_all, _, df_leagues, _, df_meta, _, _, _, _, name_to_id = load_data(supabase, club_id)

    service_ctx = ServiceContext(
        supabase=supabase,
        club_id=str(club_id),
        source=payload.source,
        actor_role="scorekeeper",
    )
    result = submit_match_batch(
        service_ctx,
        payload.matches,
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    if not result.ok:
        raise HTTPException(status_code=400, detail="; ".join(result.errors) or "Unable to submit match batch")

    return {
        "ok": True,
        "auth_mode": auth_mode,
        "required_permission": PERMISSION_ENTER_SCORES,
        "result": result.data,
    }
