from __future__ import annotations

import os
import re
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


_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


def validate_club_slug(slug: str) -> str:
    normalized = str(slug or "").strip().lower()
    if not normalized:
        raise ValueError("Club slug is required.")
    if not _SLUG_RE.fullmatch(normalized):
        raise ValueError("Club slug must contain only lowercase letters, numbers, or hyphens.")
    return normalized


def get_club_by_slug(supabase: Any, slug: str, include_inactive: bool = False) -> dict[str, Any] | None:
    normalized = validate_club_slug(slug)
    query = supabase.table("clubs").select("*").eq("slug", normalized)
    if not include_inactive:
        query = query.eq("is_active", True)
    response = query.limit(1).execute()
    rows = getattr(response, "data", None) or []
    return rows[0] if rows else None


def create_club_config(
    supabase: Any,
    *,
    club_id: str,
    slug: str,
    name: str,
    tagline: str | None = None,
    support_email: str | None = None,
    public_base_url: str | None = None,
    created_by_email: str | None = None,
    features_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_slug = validate_club_slug(slug)
    if get_club_by_slug(supabase, normalized_slug, include_inactive=True):
        raise ValueError(f"Club slug '{normalized_slug}' already exists.")

    payload: dict[str, Any] = {
        "id": str(club_id or "").strip(),
        "slug": normalized_slug,
        "name": str(name or "").strip(),
        "tagline": tagline,
        "support_email": support_email,
        "public_base_url": public_base_url,
        "created_by_email": created_by_email,
        "features_json": features_json or {},
    }
    if not payload["id"]:
        raise ValueError("club_id is required.")
    if not payload["name"]:
        raise ValueError("name is required.")

    response = supabase.table("clubs").insert(payload).execute()
    rows = getattr(response, "data", None) or []
    return rows[0] if rows else payload


def list_clubs(supabase: Any, include_inactive: bool = False) -> list[dict[str, Any]]:
    query = supabase.table("clubs").select("*")
    if not include_inactive:
        query = query.eq("is_active", True)
    response = query.execute()
    return list(getattr(response, "data", None) or [])


__all__ = ["get_default_club_id", "get_club_config", "validate_club_slug", "get_club_by_slug", "create_club_config", "list_clubs"]
