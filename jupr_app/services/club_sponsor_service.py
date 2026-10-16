"""Reusable club sponsor profiles; tournaments retain independent snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any
from uuid import UUID

from jupr_app.services.tournament_sponsor_service import (
    BUCKET, asset_prefix, logo_urls, valid_asset_path, website_url,
)


class SponsorConflictError(ValueError):
    pass


def _profile(supabase: Any, club_id: str, sponsor_id: str) -> dict[str, Any] | None:
    rows = (supabase.table("club_sponsors").select("*").eq("club_id", club_id)
            .eq("id", str(UUID(sponsor_id))).limit(1).execute().data or [])
    return rows[0] if rows else None


def _with_logo(supabase: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Only sign assets whose full path is scoped to this record's club/source.
    safe = [dict(row, logo_path=row.get("logo_path") if valid_asset_path(
        row.get("logo_path") or "", club_id=row["club_id"],
        tournament_id=row.get("logo_tournament_id") or "") else "") for row in rows]
    urls = logo_urls(supabase, safe)
    return [dict(row, logo_url=urls.get(row["logo_path"], "")) for row in safe]


def list_club_sponsors(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = supabase.table("club_sponsors").select("*").eq("club_id", club_id).order("name").execute().data or []
    return _with_logo(supabase, rows)


def save_club_sponsor(supabase: Any, *, club_id: str, sponsor_id: str,
                      tournament_id: str, value: dict[str, Any],
                      expected_updated_at: str | None) -> dict[str, Any]:
    sponsor_id = str(UUID(sponsor_id))
    name = " ".join(str(value.get("name") or "").split())
    if not name or len(name) > 120:
        raise ValueError("Enter a sponsor name up to 120 characters.")
    clean = {"name": name, "website": website_url(value.get("website"))}
    for key, maximum in (("public_description", 500), ("contact_name", 120),
                         ("contact_email", 254), ("contact_phone", 80)):
        text = str(value.get(key) or "").strip()
        if len(text) > maximum:
            raise ValueError(f"Keep {key.replace('_', ' ')} to {maximum} characters or fewer.")
        clean[key] = text
    if clean["contact_email"] and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", clean["contact_email"]):
        raise ValueError("Enter a valid sponsor contact email.")
    path = str(value.get("logo_path") or "")
    if path and not valid_asset_path(path, club_id=club_id, tournament_id=tournament_id):
        raise ValueError("This logo does not belong to this tournament.")
    clean.update(logo_path=path, logo_tournament_id=tournament_id if path else None)
    existing = _profile(supabase, club_id, sponsor_id)
    if existing and all(existing.get(k) == v for k, v in clean.items()):
        return _with_logo(supabase, [existing])[0]  # Safe retry after a lost response.
    if existing and existing.get("updated_at") != expected_updated_at:
        raise SponsorConflictError("This saved sponsor changed. Reload club sponsors before updating it.")
    duplicates = (supabase.table("club_sponsors").select("id").eq("club_id", club_id)
                  .eq("name_key", name.lower()).limit(1).execute().data or [])
    if duplicates and str(duplicates[0]["id"]) != sponsor_id:
        raise SponsorConflictError("A club sponsor with this name is already saved. Choose it from Saved club sponsor.")
    clean["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        if existing:
            result = (supabase.table("club_sponsors").update(clean).eq("club_id", club_id)
                      .eq("id", sponsor_id).eq("updated_at", expected_updated_at).execute().data or [])
        else:
            result = supabase.table("club_sponsors").insert(dict(clean, id=sponsor_id, club_id=club_id)).execute().data or []
    except Exception as exc:
        if getattr(exc, "code", None) == "23505":
            raise SponsorConflictError("This sponsor was just saved by another staff member. Reload club sponsors.") from exc
        raise
    if not result:
        raise SponsorConflictError("This saved sponsor changed. Reload club sponsors before updating it.")
    return _with_logo(supabase, result)[0]


def use_club_sponsor(supabase: Any, *, club_id: str, tournament_id: str,
                     sponsor_id: str) -> dict[str, Any]:
    row = _profile(supabase, club_id, sponsor_id)
    if not row:
        raise LookupError("Saved sponsor not found in this club.")
    path = row.get("logo_path") or ""
    if path:
        if not valid_asset_path(path, club_id=club_id, tournament_id=row.get("logo_tournament_id") or ""):
            raise ValueError("The saved logo is unavailable. Edit the sponsor and upload it again.")
        # Copy into the destination tournament's existing private asset namespace.
        # Public pages/emails keep using their established validation and snapshots.
        destination = asset_prefix(club_id, tournament_id) + hashlib.sha256(path.encode()).hexdigest()[:32] + ".webp"
        if path != destination:
            data = supabase.storage.from_(BUCKET).download(path)
            supabase.storage.from_(BUCKET).upload(destination, data, {
                "content-type": "image/webp", "upsert": "true", "cache-control": "3600"})
        path = destination
    result = {"id": str(row["id"]), "club_sponsor_id": str(row["id"]), "name": row["name"],
              "website": row["website"], "public_description": row["public_description"],
              "logo_path": path, "tier": "supporting", "level": "", "notes": "", "is_visible": True}
    return dict(result, logo_url=logo_urls(supabase, [result]).get(path, ""))
