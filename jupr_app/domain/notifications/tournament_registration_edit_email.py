from __future__ import annotations

from html import escape
from typing import Any

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, SMTPConfig, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def build_tournament_registration_edit_email_subject(*, tournament_name: str) -> str:
    return f"Edit your {_safe_text(tournament_name) or 'Tournament'} registration"


def build_tournament_registration_edit_email_html(*, tournament_name: str, registered_email: str, edit_url: str) -> str:
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>Edit your tournament registration</h1>
<p>We received a request to edit your tournament registration.</p>
<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}<br><strong>Registered email:</strong> {escape(_safe_text(registered_email))}</p>
<p><a href=\"{escape(_safe_text(edit_url))}\" style=\"background:#2563eb;color:white;padding:10px 14px;text-decoration:none;border-radius:6px\">Edit my registration</a></p>
<p>Or copy and paste this link into your browser:<br>{escape(_safe_text(edit_url))}</p>
<p>This link expires in 48 hours. If you did not request this, you can ignore this email.</p>
</body></html>"""


def build_tournament_registration_edit_email_text(*, tournament_name: str, registered_email: str, edit_url: str) -> str:
    return "\n".join([
        "We received a request to edit your tournament registration.",
        f"Tournament: {_safe_text(tournament_name)}",
        f"Registered email: {_safe_text(registered_email)}",
        f"Edit my registration: {_safe_text(edit_url)}",
        "This link expires in 48 hours.",
        "If you did not request this, you can ignore this email.",
    ])


def send_tournament_registration_edit_email(*, tournament_name: str, registered_email: str, edit_url: str, smtp_config: SMTPConfig | None = None) -> dict[str, str]:
    original_to_email = _safe_text(registered_email)
    subject = build_tournament_registration_edit_email_subject(tournament_name=tournament_name)
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
        html_body=build_tournament_registration_edit_email_html(tournament_name=tournament_name, registered_email=original_to_email, edit_url=edit_url),
        text_body=build_tournament_registration_edit_email_text(tournament_name=tournament_name, registered_email=original_to_email, edit_url=edit_url),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}
