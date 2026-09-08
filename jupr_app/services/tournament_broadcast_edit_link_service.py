"""Personal registration links for reviewed tournament emails.

Previews and saved communications contain only registration identities and the
reviewed destination. Bearer tokens are minted for the claimed SMTP attempt.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.services.public_tournament_registration_edit_service import (
    _edit_url,
    _public_web_base_url,
    _stable_edit_secret,
)


def tournament_broadcast_edit_link_context(supabase: Any, *, club_id: str) -> dict[str, str]:
    _stable_edit_secret()  # Fail before confirmation if secure links are unavailable.
    clubs = supabase.table("clubs").select("id,slug").eq("id", club_id).limit(2).execute().data or []
    slug = str(clubs[0].get("slug") or "") if len(clubs) == 1 else ""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", slug):
        raise ValueError("The club's registration page could not be found. Try again without the Edit Registration button.")
    base_url = _public_web_base_url()
    parsed = urlsplit(base_url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
    if (not (parsed.scheme == "https" or local_http) or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("The registration website is not configured correctly. Try again without the Edit Registration button.")
    return {"club_slug": slug, "public_base_url": base_url}


def build_tournament_broadcast_edit_links(*, tournament_id: str, recipient: dict,
        context: dict[str, str]) -> list[dict[str, str]]:
    secret = _stable_edit_secret()
    return [{**registration, "edit_url": _edit_url(
        club_slug=context["club_slug"], tournament_id=tournament_id,
        registration_slug=None, public_base_url=context["public_base_url"],
        edit_token=build_registration_edit_token(tournament_id=tournament_id,
            registration_id=registration["registration_id"], email=recipient["email"], secret=secret),
    )} for registration in recipient["registration_edit_links"]]
