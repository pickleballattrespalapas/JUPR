from __future__ import annotations

from decimal import Decimal, InvalidOperation
from html import escape
from typing import Any

from jupr_app.config import (
    EMAIL_MODE_DRY_RUN,
    EMAIL_MODE_LIVE,
    EMAIL_MODE_STAGING_REDIRECT,
    SMTPConfig,
    get_email_mode,
    get_env_or_default,
)
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart

PAYMENT_NOTE = "Payment will be taken on site at Tres Palapas, or an invoice will be sent to the registered email address."


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_money(value: Any) -> Decimal:
    text = _safe_text(value).replace("$", "").replace(",", "")
    if not text:
        return Decimal("0")
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return Decimal("0")


def format_money(value: Any) -> str:
    amount = _coerce_money(value).quantize(Decimal("0.01"))
    if amount == amount.to_integral_value():
        return f"${amount:.0f}"
    return f"${amount:.2f}"


def _event_view(selection: dict[str, Any], day: dict[str, Any] | None = None, event: dict[str, Any] | None = None) -> dict[str, Any]:
    day = day or {}
    event = event or {}
    return {
        "day_label": _safe_text(day.get("label") or selection.get("day_label") or selection.get("event_day_label")),
        "family_label": _safe_text(event.get("event_family_label") or selection.get("family_label") or selection.get("event_family")),
        "division_name": _safe_text(event.get("division_name") or event.get("label") or selection.get("division_name") or selection.get("division")),
        "skill_label": _safe_text(event.get("skill_label") or selection.get("skill_label")),
        "age_label": _safe_text(event.get("age_label") or selection.get("age_label")),
        "partner_mode": _safe_text(selection.get("partner_mode") or "NONE").upper() or "NONE",
        "partner_name": _safe_text(selection.get("partner_name")),
        "price_usd": _coerce_money(event.get("price_usd", selection.get("price_usd"))),
    }


def build_registration_confirmation_view_model(*, tournament: dict[str, Any] | None = None, registration: dict[str, Any] | None = None, selections: list[dict[str, Any]] | None = None, days: list[dict[str, Any]] | None = None, event_options: list[dict[str, Any]] | None = None, tournament_name: str | None = None, registration_id: str | None = None, display_name: str | None = None, email: str | None = None, confirmation_url: str | None = None, sender_from_name: str | None = None, sender_from_email: str | None = None) -> dict[str, Any]:
    tournament = tournament or {}
    registration = registration or {}
    day_lookup = {str(row.get("id")): row for row in (days or [])}
    event_lookup = {str(row.get("id")): row for row in (event_options or [])}
    selected_events = []
    total = Decimal("0")
    for selection in selections or []:
        event = event_lookup.get(str(selection.get("event_option_id") or "")) or selection.get("event") or {}
        day = day_lookup.get(str(selection.get("registration_day_id") or event.get("registration_day_id") or "")) or selection.get("day") or {}
        row = _event_view(selection, day, event)
        total += _coerce_money(row.get("price_usd"))
        selected_events.append(row)
    return {
        "tournament_name": _safe_text(tournament_name or tournament.get("name") or "Tournament"),
        "registration_id": _safe_text(registration_id or registration.get("id")),
        "display_name": _safe_text(display_name or registration.get("display_name") or "Player"),
        "email": _safe_text(email or registration.get("email")),
        "selected_events": selected_events,
        "total_price_usd": total,
        "payment_note": PAYMENT_NOTE,
        "confirmation_url": _safe_text(confirmation_url),
        "sender_from_name": _safe_text(sender_from_name),
        "sender_from_email": _safe_text(sender_from_email),
    }


def build_tournament_registration_confirmation_subject(view_model: dict) -> str:
    return f"Registration confirmed: {_safe_text(view_model.get('tournament_name')) or 'Tournament'}"


def _partner_display(event: dict[str, Any]) -> str:
    mode = _safe_text(event.get("partner_mode")).upper()
    name = _safe_text(event.get("partner_name"))
    if mode == "HAS_PARTNER":
        return f"Partner: {name}" if name else "Partner entered"
    if mode == "NEEDS_PARTNER":
        return "Needs partner"
    return "—"


def _division_display(event: dict[str, Any]) -> str:
    parts = [_safe_text(event.get("division_name"))]
    skill = _safe_text(event.get("skill_label"))
    age = _safe_text(event.get("age_label"))
    if skill and skill.lower() != "open":
        parts.append(skill)
    if age and age.lower() not in {"all ages", "all"}:
        parts.append(age)
    return " • ".join(p for p in parts if p) or "Division"


def build_tournament_registration_confirmation_html(view_model: dict) -> str:
    rows = []
    for event in view_model.get("selected_events") or []:
        rows.append("<tr>" + "".join([
            f"<td>{escape(_safe_text(event.get('day_label')))}</td>",
            f"<td>{escape(_safe_text(event.get('family_label')))}</td>",
            f"<td>{escape(_division_display(event))}</td>",
            f"<td>{escape(_partner_display(event))}</td>",
            f"<td>{escape(format_money(event.get('price_usd')))}</td>",
        ]) + "</tr>")
    url = _safe_text(view_model.get("confirmation_url"))
    link = f'<p><a href="{escape(url)}">View your registration confirmation</a></p>' if url else ""
    sender = ""
    if _safe_text(view_model.get("sender_from_email")):
        sender = f"<p>This email was sent from {escape(_safe_text(view_model.get('sender_from_name')))} &lt;{escape(_safe_text(view_model.get('sender_from_email')))}&gt;.</p>"
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>Registration confirmed</h1>
<p>Your registration for <strong>{escape(_safe_text(view_model.get('tournament_name')))}</strong> is confirmed.</p>
<p><strong>Registrant:</strong> {escape(_safe_text(view_model.get('display_name')))}<br><strong>Email:</strong> {escape(_safe_text(view_model.get('email')))}</p>
<table cellpadding=\"8\" cellspacing=\"0\" border=\"1\" style=\"border-collapse:collapse;width:100%\"><thead><tr><th>Day</th><th>Event</th><th>Division</th><th>Partner</th><th>Price</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><strong>Total due: {escape(format_money(view_model.get('total_price_usd')))}</strong></p>
<p>{escape(_safe_text(view_model.get('payment_note')))}</p>{link}{sender}
</body></html>"""


def build_tournament_registration_confirmation_text(view_model: dict) -> str:
    lines = [
        "Registration confirmed",
        f"Tournament: {_safe_text(view_model.get('tournament_name'))}",
        f"Registrant: {_safe_text(view_model.get('display_name'))}",
        f"Email: {_safe_text(view_model.get('email'))}",
        "Events:",
    ]
    for event in view_model.get("selected_events") or []:
        lines.append(f"- {_safe_text(event.get('day_label'))} | {_safe_text(event.get('family_label'))} | {_division_display(event)} | {_partner_display(event)} | {format_money(event.get('price_usd'))}")
    lines.extend([f"Total due: {format_money(view_model.get('total_price_usd'))}", _safe_text(view_model.get("payment_note"))])
    if _safe_text(view_model.get("confirmation_url")):
        lines.append(f"Confirmation page: {_safe_text(view_model.get('confirmation_url'))}")
    return "\n".join(lines)


def send_tournament_registration_confirmation_email(*, view_model: dict, smtp_config: SMTPConfig | None = None) -> dict[str, str]:
    original_to_email = _safe_text(view_model.get("email"))
    subject = build_tournament_registration_confirmation_subject(view_model)
    mode = get_email_mode()
    effective_to = original_to_email
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO").strip()
        if not redirect_to:
            raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
        effective_to = redirect_to
        subject = f"[STAGING→{original_to_email}] {subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": original_to_email}
    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=subject,
        html_body=build_tournament_registration_confirmation_html(view_model),
        text_body=build_tournament_registration_confirmation_text(view_model),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}
