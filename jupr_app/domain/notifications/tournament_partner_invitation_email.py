from __future__ import annotations

from html import escape

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_STAGING_REDIRECT, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart
from jupr_app.domain.notifications.tournament_email_sponsors import with_sponsors_html, with_sponsors_text, sponsor_inline_images


def invitation_email(*, title: str, description: str, tournament_name: str,
                     division_name: str, requester_name: str, target_name: str,
                     message: str, action_url: str, action_label: str,
                     requester_email: str = "", sponsors: list[dict] | None = None) -> tuple[str, str]:
    lines = [description, f"Tournament: {tournament_name}", f"Division: {division_name}",
             f"Partners: {requester_name} and {target_name}"]
    if message:
        lines += ["", f"Message from {requester_name}:", message]
    if requester_email:
        lines += ["", f"Reply to {requester_name}: {requester_email}"]
    lines += ["", f"{action_label}: {action_url}"]
    body = f'<html><body style="font-family:Arial,sans-serif;font-size:18px;line-height:1.6;color:#0f172a"><h1>{escape(title)}</h1>'
    body += f'<p>{escape(description)}</p><p><strong>{escape(tournament_name)}</strong><br>{escape(division_name)}<br>{escape(requester_name)} and {escape(target_name)}</p>'
    if message:
        body += f'<p><strong>Message from {escape(requester_name)}</strong></p><blockquote style="white-space:pre-wrap">{escape(message)}</blockquote>'
    if requester_email:
        body += f'<p><a href="mailto:{escape(requester_email, quote=True)}">Reply to {escape(requester_name)}</a></p>'
    body += f'<p style="margin:28px 0"><a href="{escape(action_url, quote=True)}" style="display:inline-block;background:#174f43;color:white;padding:16px 24px;border-radius:10px;font-size:20px;font-weight:bold;text-decoration:none">{escape(action_label)}</a></p></body></html>'
    return with_sponsors_html(body, sponsors), with_sponsors_text("\n".join(lines), sponsors)


def send_partner_invitation_email(*, to_email: str, title: str, sponsors: list[dict] | None = None, **copy) -> str:
    mode = get_email_mode()
    html, plain = invitation_email(title=title, sponsors=sponsors, **copy)
    if mode == EMAIL_MODE_DRY_RUN:
        return "dry_run"
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        to_email = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not to_email:
            raise ValueError("Staging email redirect is not configured.")
        title = "[STAGING] " + title
    send_email_with_inline_chart(to_email=to_email, subject=title,
        html_body=html, text_body=plain, chart_png_bytes=None,
        reply_to=copy.get("requester_email") or None,
        inline_png_images=sponsor_inline_images(sponsors))
    return "staging_redirect" if mode == EMAIL_MODE_STAGING_REDIRECT else "sent"
