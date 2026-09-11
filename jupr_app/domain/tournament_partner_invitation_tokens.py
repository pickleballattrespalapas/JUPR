"""Expiring capabilities scoped to one invitation and one participant."""
from __future__ import annotations

import hmac
import json
import time
from typing import Any

from jupr_app.domain.tournament_registration_edit_tokens import (
    _b64url_decode, _b64url_encode, _sign, registration_edit_email_hash,
)

PURPOSE = "partner-invitation-v1"


def build_partner_invitation_token(*, invitation_id: str, club_id: str,
                                   tournament_id: str, role: str, email: str,
                                   expires_at: int, secret: str | None = None) -> str:
    if role not in {"requester", "target"}:
        raise ValueError("Invalid invitation participant.")
    payload = _b64url_encode(json.dumps({
        "purpose": PURPOSE, "id": invitation_id, "club": club_id,
        "tournament": tournament_id, "role": role,
        "email_hash": registration_edit_email_hash(email), "exp": expires_at,
    }, sort_keys=True, separators=(",", ":")).encode())
    return f"{payload}.{_sign(PURPOSE + ':' + payload, secret)}"


def verify_partner_invitation_token(token: str, *, club_id: str,
                                    now: int | None = None,
                                    secret: str | None = None) -> dict[str, Any]:
    try:
        if not token or len(token) > 4000:
            raise ValueError()
        payload, signature = token.split(".", 1)
        if not hmac.compare_digest(signature, _sign(PURPOSE + ':' + payload, secret)):
            raise ValueError()
        claims = json.loads(_b64url_decode(payload))
        if claims.get("purpose") != PURPOSE or claims.get("role") not in {"requester", "target"}:
            raise ValueError()
        if claims.get("club") != str(club_id) or not claims.get("id") or not claims.get("tournament"):
            raise ValueError()
        if int(claims["exp"]) <= int(time.time() if now is None else now):
            raise ValueError("expired")
        return claims
    except Exception as exc:
        raise ValueError("This partner request link is invalid or has expired. Open the Partner Board to send a new request.") from exc
