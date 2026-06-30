from __future__ import annotations

from html import escape
from typing import Any

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, SMTPConfig, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _contact_line(*, requester_email: str, requester_phone: str) -> str:
    parts: list[str] = []
    if _safe_text(requester_email):
        parts.append(f"Email: {_safe_text(requester_email)}")
    if _safe_text(requester_phone):
        parts.append(f"Phone: {_safe_text(requester_phone)}")
    return " • ".join(parts)


def build_tournament_partner_request_subject(*, tournament_name: str, requester_name: str) -> str:
    requester = _safe_text(requester_name) or "A player"
    tournament = _safe_text(tournament_name) or "Tournament"
    return f"{requester} wants to partner with you for {tournament}"


def build_tournament_partner_request_email_html(
    *,
    tournament_name: str,
    target_name: str,
    requester_name: str,
    requester_email: str,
    requester_phone: str,
    event_label: str,
    division_label: str,
    day_label: str,
    message: str,
) -> str:
    contact = _contact_line(requester_email=requester_email, requester_phone=requester_phone)
    message_html = ""
    if _safe_text(message):
        message_html = f"<p><strong>Message:</strong><br>{escape(_safe_text(message))}</p>"
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>Partner request</h1>
<p>Hi {escape(_safe_text(target_name) or 'there')},</p>
<p>{escape(_safe_text(requester_name) or 'A player')} would like to request you as a partner.</p>
<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}<br>
<strong>Division:</strong> {escape(_safe_text(day_label))} / {escape(_safe_text(event_label))} / {escape(_safe_text(division_label))}</p>
<p><strong>Requester contact:</strong><br>{escape(contact)}</p>
{message_html}
<p>Your email address was not shared with the requester by this form. Contact them directly if you want to discuss playing together.</p>
</body></html>"""


def build_tournament_partner_request_email_text(
    *,
    tournament_name: str,
    target_name: str,
    requester_name: str,
    requester_email: str,
    requester_phone: str,
    event_label: str,
    division_label: str,
    day_label: str,
    message: str,
) -> str:
    lines = [
        f"Hi {_safe_text(target_name) or 'there'},",
        f"{_safe_text(requester_name) or 'A player'} would like to request you as a partner.",
        f"Tournament: {_safe_text(tournament_name)}",
        f"Division: {_safe_text(day_label)} / {_safe_text(event_label)} / {_safe_text(division_label)}",
        f"Requester contact: {_contact_line(requester_email=requester_email, requester_phone=requester_phone)}",
    ]
    if _safe_text(message):
        lines.extend(["", "Message:", _safe_text(message)])
    lines.extend([
        "",
        "Your email address was not shared with the requester by this form. Contact them directly if you want to discuss playing together.",
    ])
    return "\n".join(lines)


def send_tournament_partner_request_email(
    *,
    tournament_name: str,
    target_name: str,
    target_email: str,
    requester_name: str,
    requester_email: str,
    requester_phone: str,
    event_label: str,
    division_label: str,
    day_label: str,
    message: str = "",
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to_email = _safe_text(target_email)
    if not original_to_email:
        raise ValueError("Requested player does not have an email address on file.")
    if not _safe_text(requester_email) and not _safe_text(requester_phone):
        raise ValueError("Enter your email or phone number so the requested player can contact you.")

    subject = build_tournament_partner_request_subject(
        tournament_name=tournament_name,
        requester_name=requester_name,
    )
    mode = get_email_mode()
    effective_to = original_to_email
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not redirect_to:
            raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
        effective_to = redirect_to
        subject = f"[STAGING→{original_to_email}] {subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": original_to_email}

    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=subject,
        html_body=build_tournament_partner_request_email_html(
            tournament_name=tournament_name,
            target_name=target_name,
            requester_name=requester_name,
            requester_email=requester_email,
            requester_phone=requester_phone,
            event_label=event_label,
            division_label=division_label,
            day_label=day_label,
            message=message,
        ),
        text_body=build_tournament_partner_request_email_text(
            tournament_name=tournament_name,
            target_name=target_name,
            requester_name=requester_name,
            requester_email=requester_email,
            requester_phone=requester_phone,
            event_label=event_label,
            division_label=division_label,
            day_label=day_label,
            message=message,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}
