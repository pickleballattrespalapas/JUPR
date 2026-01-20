# jupr_app/domain/build_challenge_notice_message.py
from __future__ import annotations


def build_challenge_notice_message(
    *,
    challenge_id: int | None,
    tier_id: str,
    challenger_name: str,
    defender_name: str,
    challenger_contact: str,
    admin_name: str,
    admin_contact: str,
    ledger_ref: str | None = None,
) -> dict:
    """
    Returns:
      - email_full: "Subject: ..." + blank line + body
      - sms: short text version

    IMPORTANT:
      - No timestamps included.
      - 48h is based on the sent/received timestamp of the message itself.
    """
    cid = f"#{int(challenge_id)}" if challenge_id else "(pending id)"

    # Allow blanks / placeholders
    chal_contact = (challenger_contact or "").strip() or "[ADD CHALLENGER CONTACT]"
    adm_name = (admin_name or "").strip() or "Ladder Admin"
    adm_contact = (admin_contact or "").strip() or "[ADD ADMIN CONTACT]"

    subject = f"Challenge Ladder Notice — Action Required (48 hours) — {challenger_name} vs {defender_name}"

    ledger_line = ""
    if ledger_ref is not None and str(ledger_ref).strip():
        ledger_line = "\nLedger ref: " + str(ledger_ref).strip()

    body = f"""Hi {defender_name},

This message is an official Challenge Ladder notice.

Challenger: {challenger_name}
Challenger contact: {chal_contact}

Response required:
You have 48 hours from the timestamp on THIS message (email/text receipt time) to respond to BOTH:
1) {adm_name} (Ladder Admin): {adm_contact}
2) {challenger_name} (Challenger): {chal_contact}

Reply with one of:
- ACCEPT (and propose times to play)
- PASS (use Monthly Pass, if available)

Challenge ID: {cid}
Tier: {tier_id}{ledger_line}

Thank you,
{adm_name}
""".strip()

    sms = (
        f"Ladder Challenge Notice: {challenger_name} challenged you. "
        f"Reply within 48h (based on this message timestamp) to BOTH "
        f"Admin ({adm_contact}) and {challenger_name} ({chal_contact}). "
        f"Challenge {cid}."
    )

    return {"email_full": f"Subject: {subject}\n\n{body}", "sms": sms}
