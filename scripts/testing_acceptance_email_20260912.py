"""Joe-authorized, single-recipient replacement email for an accepted test request.

Run on the existing production API runtime. No registration or partnership
mutation. A separate delivery record preserves the original email audit and
prevents repeated sends, including after an uncertain provider response.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import smtplib
from unittest.mock import patch
from uuid import uuid4

from jupr_app.config import get_email_mode, get_smtp_config
from jupr_app.data.client import make_supabase
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart
from jupr_app.domain.notifications.tournament_partner_invitation_email import invitation_email
from jupr_app.domain.notifications.tournament_email_sponsors import sponsor_inline_images
from jupr_app.services import public_tournament_partner_invitation_service as svc

INVITATION_ID = "pinv_c659519207064c7d967ac7006b54194b"
TOURNAMENT_ID = "563b7922-ae92-41d4-8286-75fe9846e944"
EMAIL_HASH = "5b8f8552f594869988693a1db3f1c7a0f3789639161e4aca62436556ec03c0e7"
KIND = "reserved_requester_nonreceipt_retry_20260912"
DELIVERIES = "tournament_partner_invitation_deliveries"
EXPECTED_API_SHA = "65f08d1b0eb5d39e741d8f05af96f1f3b222658c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(db, *, send: bool = False) -> dict:
    require(os.getenv("JUPR_ENV") == "production" and os.getenv("FLY_APP_NAME") == "juprleagues-api", "Wrong runtime")
    require(os.getenv("SUPABASE_URL", "").rstrip("/") == "https://dnoockbwfenunhcibwfn.supabase.co", "Wrong database")
    require(os.getenv("JUPR_IMAGE_BUILD_GIT_SHA") == EXPECTED_API_SHA, "Production version changed")
    require(get_email_mode() == "live", "Live email is required")
    row = svc._one(db, svc.TABLE, INVITATION_ID)
    require(row.get("club_id") == "tres_palapas" and str(row.get("tournament_id")) == TOURNAMENT_ID, "Request scope changed")
    require(row.get("requester_name") == "Testing" and row.get("status") == "RESERVED" and not svc._expired(row), "Request is no longer awaiting registration")
    require(hashlib.sha256(row["requester_email"].encode()).hexdigest() == EMAIL_HASH, "Requester address changed")
    require(row.get("target_selection_id") == "sel_25e88625d7" and row.get("event_option_id") == "div_05c05159c7", "Partner or division changed")
    require(svc._email_is_contact_allowed(row["requester_email"]), "Recipient cannot be contacted")
    ctx = svc._context(db, row)
    require(svc._display_name(ctx["target_registration"]) == "Joe Baumann" and svc._available(ctx), "Accepted partner unavailable")
    token = svc._token(row, ctx, "requester")
    review = svc.review_invitation(db, club_id="tres_palapas", club_slug="tres-palapas", token=token)
    require(review.get("role") == "requester" and review.get("status") == "RESERVED", "Requester link invalid")
    require(review["registration_prefill"]["event_option_id"] == row["event_option_id"], "Registration division mismatch")
    url = svc._public_web_base_url().rstrip("/") + review["registration_url"]
    require(url.startswith("https://pickleballclubsandwich.com/clubs/tres-palapas/tournament-registration") and "#partner_invitation=" in url, "Registration URL invalid")
    message = dict(
        to_email=row["requester_email"], title="Testing: Joe accepted - complete your Baja Classic registration",
        description="Joe Baumann accepted your partner request. You’re now listed together on the roster as pending registration. Your division and partner are already selected—complete your registration to confirm your team. Your reservation lasts until " + str(row["expires_at"])[:10] + ".",
        tournament_name=ctx["tournament"]["name"], division_name=review["division_name"],
        requester_name=row["requester_name"], target_name="Joe Baumann", message="",
        action_url=url, action_label="Complete registration",
        sponsors=svc.load_tournament_email_sponsors(db, club_id=row["club_id"], tournament_id=TOURNAMENT_ID),
    )
    prior = svc._rows(db.table(DELIVERIES).select("status").eq("invitation_id", INVITATION_ID).eq("kind", KIND).execute())
    if prior:
        return {"ok": prior[0]["status"] == "sent", "status": "already_" + prior[0]["status"], "messages_sent": 0}
    if not send:
        cfg = get_smtp_config()
        return {"ok": True, "status": "ready", "requester": "Testing", "partner": "Joe Baumann", "division": review["division_name"], "smtp_host": cfg.host, "from_email": cfg.from_email, "messages_sent": 0}

    html, plain = invitation_email(**{key: value for key, value in message.items() if key != "to_email"})
    images = sponsor_inline_images(message["sponsors"])
    attempt = uuid4().hex
    # Plain insert: the existing primary key also prevents concurrent replays.
    db.table(DELIVERIES).insert({"invitation_id": INVITATION_ID, "kind": KIND, "status": "sending", "attempt_id": attempt}).execute()
    status = "delivery_unknown"
    smtp_codes = []
    original_data = smtplib.SMTP.data
    def capture_ack(server, content):
        code, response = original_data(server, content)
        smtp_codes.append(code)
        return code, response
    try:
        # Capture the server's DATA acknowledgment without logging message text,
        # credentials or private registration links. This is a separate process.
        with patch.object(smtplib.SMTP, "data", capture_ack):
            message_id = send_email_with_inline_chart(to_email=message["to_email"], subject=message["title"],
                html_body=html, text_body=plain, chart_png_bytes=None, inline_png_images=images,
                message_id=attempt)
        status = "sent"
    finally:
        db.table(DELIVERIES).update({"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("invitation_id", INVITATION_ID).eq("kind", KIND).eq("attempt_id", attempt).execute()
    require(status == "sent", "Email was not sent live")
    return {"ok": True, "status": status, "requester": "Testing", "partner": "Joe Baumann", "messages_sent": 1,
            "message_id": message_id, "smtp_data_response_codes": smtp_codes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    try:
        db = make_supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        result = run(db, send=args.send)
    except Exception as exc:
        # Never log email capabilities, addresses, credentials or provider output.
        result = {"ok": False, "error_type": type(exc).__name__}
        if isinstance(exc, smtplib.SMTPResponseException):
            result["smtp_code"] = int(exc.smtp_code)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
