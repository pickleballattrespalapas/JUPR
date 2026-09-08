from __future__ import annotations

from datetime import date
from html import escape
from typing import Any
from urllib.parse import urlsplit

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, SMTPConfig, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.tournament_email_sponsors import with_sponsors_html, with_sponsors_text, sponsor_inline_images
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _message_html(message: str) -> str:
    escaped = escape(_safe_text(message))
    return escaped.replace("\n", "<br>")


def _event_lines(event: dict) -> list[str]:
    day = _safe_text(event.get("day"))
    event_date = _safe_text(event.get("event_date"))
    try:
        event_date = date.fromisoformat(event_date).strftime("%a, %b %d, %Y").replace(" 0", " ")
    except ValueError:
        pass
    when = " · ".join(dict.fromkeys(value for value in (day, event_date) if value))
    lines = [_safe_text(event.get("division")) or "Event"]
    if when:
        lines.append(when)
    if partner := _safe_text(event.get("partner_name")):
        lines.append(f"Partner: {partner}")
    elif event.get("partner_mode") == "NEEDS_PARTNER":
        lines.append("Looking for a partner")
    return lines


def _registration_events_sections(registrations: list[dict] | None) -> tuple[str, str]:
    if registrations is None:
        return "", ""
    html = '<h2 style="font-size:20px;margin-top:24px">Your registration events</h2>'
    text = ["Your registration events"]
    for registration in registrations:
        name = _safe_text(registration.get("name")) or "Registrant"
        cancelled = registration.get("registration_status") == "cancelled"
        label = name + (" — Registration cancelled" if cancelled else "")
        html += f'<h3 style="font-size:16px">{escape(label)}</h3>'
        text.extend(["", label])
        events = registration.get("events") or []
        if not events:
            html += "<p>No registration events are currently listed.</p>"
            text.append("No registration events are currently listed.")
        else:
            html += "<ul>"
            for event in events:
                lines = _event_lines(event)
                html += '<li style="margin-bottom:12px"><strong>' + escape(lines[0]) + '</strong>'
                html += "".join("<br>" + escape(line) for line in lines[1:]) + "</li>"
                text.append("- " + " · ".join(lines))
            html += "</ul>"
    return html, "\n".join(text) + "\n\n"


def build_tournament_registrant_broadcast_subject(*, tournament_name: str, subject: str) -> str:
    clean_subject = _safe_text(subject) or "Tournament update"
    tournament = _safe_text(tournament_name)
    return f"{tournament}: {clean_subject}" if tournament and tournament.lower() not in clean_subject.lower() else clean_subject


def _registration_edit_sections(registrations: list[dict] | None) -> tuple[str, str]:
    if not registrations:
        return "", ""
    html = '<h2 style="font-size:20px;margin-top:24px">Edit your registration</h2>'
    text = ["Edit your registration"]
    for registration in registrations:
        name = _safe_text(registration.get("name")) or "Registrant"
        url = _safe_text(registration.get("edit_url"))
        style = "display:inline-block;background:#2563eb;color:#ffffff;padding:12px 18px;text-decoration:none;border-radius:6px;font-weight:bold"
        html += f"<p><strong>{escape(name)}</strong></p>"
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Invalid registration edit URL.")
            html += f'<p><a href="{escape(url, quote=True)}" style="{style}">Edit Registration</a></p>'
            text.extend([name, f"Edit Registration: {url}", ""])
        else:
            # The admin preview shows the button without exposing a bearer link.
            html += f'<p><span role="link" aria-disabled="true" style="{style}">Edit Registration</span></p>'
            text.extend([name, "Edit Registration: [Personal link added when this email is sent]", ""])
    expiry = "Each private edit link expires 48 hours after this email is sent."
    return html + f'<p style="font-size:12px;color:#6b7280">{expiry}</p>', "\n".join([*text, expiry, "", ""])


def build_tournament_registrant_broadcast_email_html(
    *,
    tournament_name: str,
    recipient_name: str,
    subject: str,
    message: str,
    personalize_greeting: bool = True,
    email_sponsors: list[dict] | None = None,
    registration_events: list[dict] | None = None,
    registration_edit_links: list[dict] | None = None,
) -> str:
    greeting_name = _safe_text(recipient_name) or "there"
    greeting = f"<p>Hi {escape(greeting_name)},</p>" if personalize_greeting else ""
    tournament_line = f"<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}</p>" if personalize_greeting else ""
    events_html, _ = _registration_events_sections(registration_events)
    edit_html, _ = _registration_edit_sections(registration_edit_links)
    return with_sponsors_html(f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>{escape(_safe_text(subject) or 'Tournament update')}</h1>
{greeting}
{tournament_line}
<p style="font-size:18px;line-height:1.65;margin:24px 0">{_message_html(message)}</p>
{events_html}
{edit_html}
<p style=\"color:#6b7280;font-size:12px\">You are receiving this because you registered for this tournament.</p>
</body></html>""", email_sponsors)


def build_tournament_registrant_broadcast_email_text(
    *,
    tournament_name: str,
    recipient_name: str,
    subject: str,
    message: str,
    personalize_greeting: bool = True,
    email_sponsors: list[dict] | None = None,
    registration_events: list[dict] | None = None,
    registration_edit_links: list[dict] | None = None,
) -> str:
    greeting_name = _safe_text(recipient_name) or "there"
    _, events_text = _registration_events_sections(registration_events)
    _, edit_text = _registration_edit_sections(registration_edit_links)
    if not personalize_greeting:
        return with_sponsors_text(f"{_safe_text(subject)}\n\n{_safe_text(message)}\n\n{events_text}{edit_text}You are receiving this because you registered for this tournament.", email_sponsors)
    return with_sponsors_text("\n".join(
        [
            _safe_text(subject) or "Tournament update",
            "",
            f"Hi {greeting_name},",
            "",
            f"Tournament: {_safe_text(tournament_name)}",
            "",
            _safe_text(message),
            "",
            events_text + edit_text + "You are receiving this because you registered for this tournament.",
        ]
    ), email_sponsors)


def send_tournament_registrant_broadcast_email(
    *,
    tournament_name: str,
    recipient_email: str,
    recipient_name: str,
    subject: str,
    message: str,
    smtp_config: SMTPConfig | None = None,
    personalize_greeting: bool = True,
    message_id: str | None = None,
    email_sponsors: list[dict] | None = None,
    registration_events: list[dict] | None = None,
    registration_edit_links: list[dict] | None = None,
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
            personalize_greeting=personalize_greeting,
            email_sponsors=email_sponsors,
            registration_events=registration_events,
            registration_edit_links=registration_edit_links,
        ),
        text_body=build_tournament_registrant_broadcast_email_text(
            tournament_name=tournament_name,
            recipient_name=recipient_name,
            subject=final_subject,
            message=message,
            personalize_greeting=personalize_greeting,
            email_sponsors=email_sponsors,
            registration_events=registration_events,
            registration_edit_links=registration_edit_links,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
        **({"message_id": message_id} if message_id else {}),
        inline_png_images=sponsor_inline_images(email_sponsors),
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}
