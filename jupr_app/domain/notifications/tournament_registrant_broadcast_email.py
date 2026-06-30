from __future__ import annotations

from html import escape
from typing import Any

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, SMTPConfig, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _message_html(message: str) -> str:
    escaped = escape(_safe_text(message))
    return escaped.replace("\n", "<br>")


def build_tournament_registrant_broadcast_subject(*, tournament_name: str, subject: str) -> str:
    clean_subject = _safe_text(subject) or "Tournament update"
    tournament = _safe_text(tournament_name)
    return f"{tournament}: {clean_subject}" if tournament and tournament.lower() not in clean_subject.lower() else clean_subject


def build_tournament_registrant_broadcast_email_html(
    *,
    tournament_name: str,
    recipient_name: str,
    subject: str,
    message: str,
) -> str:
    greeting_name = _safe_text(recipient_name) or "there"
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>{escape(_safe_text(subject) or 'Tournament update')}</h1>
<p>Hi {escape(greeting_name)},</p>
<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}</p>
<p>{_message_html(message)}</p>
<p style=\"color:#6b7280;font-size:12px\">You are receiving this because you registered for this tournament.</p>
</body></html>"""


def build_tournament_registrant_broadcast_email_text(
    *,
    tournament_name: str,
    recipient_name: str,
    subject: str,
    message: str,
) -> str:
    greeting_name = _safe_text(recipient_name) or "there"
    return "\n".join(
        [
            _safe_text(subject) or "Tournament update",
            "",
            f"Hi {greeting_name},",
            "",
            f"Tournament: {_safe_text(tournament_name)}",
            "",
            _safe_text(message),
            "",
            "You are receiving this because you registered for this tournament.",
        ]
    )


def send_tournament_registrant_broadcast_email(
    *,
    tournament_name: str,
    recipient_email: str,
    recipient_name: str,
    subject: str,
    message: str,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to_email = _safe_text(recipient_email)
    if not original_to_email:
        raise ValueError("Recipient email is required.")
    if not _safe_text(message):
        raise ValueError("Message is required.")

    final_subject = build_tournament_registrant_broadcast_subject(
        tournament_name=tournament_name,
        subject=subject,
    )
    mode = get_email_mode()
    effective_to = original_to_email
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not redirect_to:
            raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
        effective_to = redirect_to
        final_subject = f"[STAGING→{original_to_email}] {final_subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": original_to_email}

    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=final_subject,
        html_body=build_tournament_registrant_broadcast_email_html(
            tournament_name=tournament_name,
            recipient_name=recipient_name,
            subject=final_subject,
            message=message,
        ),
        text_body=build_tournament_registrant_broadcast_email_text(
            tournament_name=tournament_name,
            recipient_name=recipient_name,
            subject=final_subject,
            message=message,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}
