"""Read published sponsor details and embed bounded copies of private logos."""
from __future__ import annotations

import base64
from io import BytesIO
import logging
from typing import Any

from PIL import Image

from jupr_app.domain.notifications.tournament_email_sponsors import MAX_LOGO_BYTES, MAX_TOTAL_LOGO_BYTES, TIERS
from jupr_app.services.tournament_sponsor_service import BUCKET, MAX_BYTES, normalize_sponsors

LOGGER = logging.getLogger(__name__)


def tournament_email_sponsor_snapshot(supabase: Any, *, club_id: str, tournament_id: str) -> list[dict]:
    # Published settings are updated by Review/Publish. Setup draft rows and
    # private notes must never become email content.
    tournament = supabase.table("tournaments").select("id").eq("id", str(tournament_id)).eq("club_id", str(club_id)).limit(1).execute().data
    if not tournament:
        return []
    rows = supabase.table("tournament_registration_settings").select("sponsors_json").eq("tournament_id", str(tournament_id)).limit(1).execute().data or []
    value = rows[0].get("sponsors_json") if rows else []
    sponsors = normalize_sponsors(value or [], club_id=str(club_id), tournament_id=str(tournament_id), strict=False)
    fields = ("name", "tier", "level", "public_description", "website", "logo_path")
    return sorted([{key: row[key] for key in fields} for row in sponsors if row["is_visible"]],
                  key=lambda row: TIERS.index(row["tier"]))


def prepare_tournament_email_sponsors(supabase: Any, snapshot: list[dict]) -> list[dict]:
    result = []
    total = 0
    logos: dict[str, bytes] = {}
    for sponsor in snapshot:
        row = {key: sponsor[key] for key in ("name", "tier", "level", "public_description", "website")}
        path = sponsor.get("logo_path")
        if path and total < MAX_TOTAL_LOGO_BYTES:
            try:
                if path not in logos:
                    raw = supabase.storage.from_(BUCKET).download(path)
                    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
                        raise ValueError("Invalid logo size")
                    with Image.open(BytesIO(raw)) as image:
                        if image.format != "WEBP" or max(image.size) > 4096:
                            raise ValueError("Invalid stored logo")
                        image.thumbnail((400, 200))
                        output = BytesIO()
                        image.convert("RGBA").save(output, format="PNG", optimize=True)
                        logos[path] = output.getvalue()
                data = logos[path]
                if len(data) <= MAX_LOGO_BYTES and total + len(data) <= MAX_TOTAL_LOGO_BYTES:
                    row["logo_png_base64"] = base64.b64encode(data).decode("ascii")
                    total += len(data)
            except Exception:
                # Storage failure must not prevent a registration/edit link or
                # suppress the sponsor's name, description and website.
                LOGGER.warning("Tournament email logo unavailable; using sponsor text.")
        result.append(row)
    return result


def load_tournament_email_sponsors(supabase: Any, *, club_id: str, tournament_id: str) -> list[dict]:
    try:
        snapshot = tournament_email_sponsor_snapshot(supabase, club_id=club_id, tournament_id=tournament_id)
        return prepare_tournament_email_sponsors(supabase, snapshot)
    except Exception:
        LOGGER.warning("Tournament email sponsor details unavailable.")
        return []
