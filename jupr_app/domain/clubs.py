from __future__ import annotations

import os
from typing import Any

from jupr_app.ui import branding


def get_default_club_id() -> str:
    """Resolve default club id from env, preserving Tres Palapas fallback."""
    return str(os.getenv("JUPR_DEFAULT_CLUB_ID") or branding.CLUB_ID).strip() or branding.CLUB_ID


def _fallback_config(club_id: str) -> dict[str, Any]:
    return {
        "id": club_id or branding.CLUB_ID,
        "slug": "tres-palapas",
        "name": branding.CLUB_NAME,
        "tagline": branding.TAGLINE,
        "support_email": branding.SUPPORT_EMAIL,
        "public_base_url": branding.PUBLIC_BASE_URL_FALLBACK,
        "logo_url": None,
        "primary_color": None,
        "is_active": True,
    }


def get_club_config(supabase: Any, club_id: str) -> dict[str, Any]:
    """Load club config from DB, falling back for legacy deployments/missing table."""
    requested_club_id = str(club_id or "").strip() or get_default_club_id()
    fallback = _fallback_config(requested_club_id)

    if supabase is None:
        return fallback

    try:
        response = (
            supabase.table("clubs")
            .select("id,slug,name,tagline,support_email,public_base_url,logo_url,primary_color,is_active")
            .eq("id", requested_club_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return fallback

    rows = getattr(response, "data", None) or []
    if not rows:
        return fallback

    row = rows[0] or {}
    merged = {**fallback, **row}
    return merged


__all__ = ["get_default_club_id", "get_club_config"]
