"""Tournament-scoped sponsor records and private, server-validated logo assets."""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

BUCKET = "tournament-sponsor-logos"
MAX_BYTES = 5 * 1024 * 1024
TIERS = ("presenting", "premier", "supporting")
TIER_LABELS = {"presenting": "Premier / Presenting", "premier": "Supporting sponsors", "supporting": "Community sponsors"}


def website_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 2048 or re.search(r"[\s\\\x00-\x1f]", text):
        raise ValueError("Enter a valid sponsor website.")
    if not re.match(r"^[a-z][a-z0-9+.-]*:", text, re.I):
        text = "https://" + text
    try:
        parts = urlsplit(text)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname or "." not in parts.hostname or parts.username or parts.password:
            raise ValueError()
        _ = parts.port
    except ValueError as exc:
        raise ValueError("Enter an HTTP or HTTPS sponsor website.") from exc
    return urlunsplit(parts)


def asset_prefix(club_id: str, tournament_id: str) -> str:
    # Club IDs are text (Tres uses tres_palapas), while tournament IDs are UUIDs.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(club_id)):
        raise ValueError("Invalid club for this logo.")
    return f"{club_id}/{UUID(str(tournament_id))}/"


def valid_asset_path(path: str, *, club_id: str, tournament_id: str) -> bool:
    if not path:
        return False
    try:
        prefix = asset_prefix(club_id, tournament_id)
    except ValueError:
        return False
    return path.startswith(prefix) and bool(re.fullmatch(r"[0-9a-f]{32}\.webp", path[len(prefix):]))


def normalize_sponsors(value: Any, *, club_id: str, tournament_id: str, strict: bool = True) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 50:
        if strict:
            raise ValueError("Use a sponsor list with no more than 50 sponsors.")
        return []
    result = []
    ids = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            if strict:
                raise ValueError("Invalid sponsor record.")
            continue
        name = str(row.get("name") or "").strip()[:120]
        if not name:
            if strict:
                raise ValueError("Every sponsor needs a name.")
            continue
        tier = row.get("tier") or "supporting"  # Legacy labels never imply title sponsorship.
        if tier not in TIERS:
            if strict:
                raise ValueError("Choose a valid sponsor tier.")
            tier = "supporting"
        try:
            website = website_url(row.get("website"))
        except ValueError:
            if strict:
                raise
            website = ""
        path = str(row.get("logo_path") or "")
        if path and not valid_asset_path(path, club_id=club_id, tournament_id=tournament_id):
            if strict:
                raise ValueError("This logo does not belong to this tournament. Upload it again.")
            path = ""
        sponsor_id = str(row.get("id") or f"legacy-{index}")[:100]
        if sponsor_id in ids:
            if strict:
                raise ValueError("Sponsor IDs must be unique.")
            continue
        ids.add(sponsor_id)
        result.append({"id": sponsor_id, "name": name, "tier": tier,
                       "level": str(row.get("level") or "").strip()[:80],
                       "website": website, "notes": str(row.get("notes") or "")[:2000],
                       "logo_path": path, "is_visible": row.get("is_visible") is not False,
                       "sort_order": index})
    return result


def validate_sponsor_payload(payload: dict[str, Any] | None, *, club_id: str, tournament_id: str) -> dict[str, Any]:
    clean = dict(payload or {})
    if "sponsors_json" in clean:
        clean["sponsors_json"] = normalize_sponsors(clean["sponsors_json"], club_id=club_id, tournament_id=tournament_id)
    return clean


def encode_logo(encoded: str) -> bytes:
    if len(encoded) > ((MAX_BYTES + 2) // 3) * 4:
        raise ValueError("Choose a logo under 5 MB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("The logo could not be read. Choose another image.") from exc
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose a logo under 5 MB.")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or getattr(image, "is_animated", False):
                raise ValueError("Use a static PNG, JPG, or WebP logo.")
            if max(image.size) > 4096:
                raise ValueError("Logos must be no larger than 4096 pixels on either side.")
            image.load()
            clean = ImageOps.exif_transpose(image).convert("RGBA")
            clean.thumbnail((1024, 1024))
            clean.info.clear()
            output = BytesIO()
            clean.save(output, format="WEBP", quality=90, method=4)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The logo could not be read. Choose another image.") from exc


def signed_logo_url(supabase: Any, path: str) -> str:
    result = supabase.storage.from_(BUCKET).create_signed_url(path, 3600)
    return str(result.get("signedURL") or result.get("signedUrl") or "")


def upload_logo(supabase: Any, *, club_id: str, tournament_id: str, encoded: str) -> dict[str, Any]:
    data = encode_logo(encoded)
    path = asset_prefix(club_id, tournament_id) + uuid4().hex + ".webp"
    supabase.storage.from_(BUCKET).upload(path, data, {"content-type": "image/webp", "upsert": "false", "cache-control": "3600"})
    return {"logo_path": path, "logo_url": signed_logo_url(supabase, path)}


def logo_urls(supabase: Any, sponsors: list[dict[str, Any]]) -> dict[str, str]:
    urls = {}
    for path in {s["logo_path"] for s in sponsors if s.get("logo_path")}:
        try:
            urls[path] = signed_logo_url(supabase, path)
        except Exception:
            # A storage outage must not hide the sponsor name or break tournament pages.
            continue
    return urls


def public_sponsors(supabase: Any, value: Any, *, club_id: str, tournament_id: str) -> list[dict[str, Any]]:
    sponsors = [s for s in normalize_sponsors(value or [], club_id=club_id, tournament_id=tournament_id, strict=False) if s["is_visible"]]
    urls = logo_urls(supabase, sponsors)
    return [{"id": s["id"], "name": s["name"], "tier": s["tier"], "level": s["level"],
             "website": s["website"], "sort_order": s["sort_order"], "logo_url": urls.get(s["logo_path"], "")} for s in sponsors]
