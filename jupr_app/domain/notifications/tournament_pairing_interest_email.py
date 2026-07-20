from __future__ import annotations

from html import escape
from typing import Any

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, SMTPConfig, get_email_mode, get_env_or_default
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def organizer_email() -> str:
    return (
        get_env_or_default("JUPR_TOURNAMENT_ORGANIZER_EMAIL")
        or get_env_or_default("JUPR_SUPPORT_EMAIL")
        or "joe@juprleagues.com"
    ).strip()


def build_pairing_interest_subject(*, tournament_name: str, division_name: str) -> str:
    division = _safe_text(division_name)
    suffix = f" — {division}" if division else ""
    return f"New tournament pairing interest: {_safe_text(tournament_name) or 'Tournament'}{suffix}"


def build_pairing_interest_html(
    *,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    recipient_kind: str,
    accept_url: str | None = None,
) -> str:
    intro = "A player sent pairing interest from the public tournament board."
    action_url = _safe_text(accept_url) or _safe_text(board_url)
    action_label = "Open tournament board"
    next_step = "Organizers should review pending pairing interest before final tournament operations."
    if recipient_kind == "player":
        intro = "Another player sent interest in pairing with you from the public tournament board."
        action_label = "Review and accept request"
        next_step = "If you accept this request, JUPR will automatically pair both registrations for that division."
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>Tournament pairing interest</h1>
<p>{escape(intro)}</p>
<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}<br>
<strong>Division:</strong> {escape(_safe_text(division_name))}<br>
<strong>Interested player:</strong> {escape(_safe_text(requester_name))}<br>
<strong>Board entry:</strong> {escape(_safe_text(target_name))}</p>
<p><a href=\"{escape(action_url)}\" style=\"background:#2563eb;color:white;padding:10px 14px;text-decoration:none;border-radius:6px\">{escape(action_label)}</a></p>
<p>{escape(next_step)}</p>
<p><a href=\"{escape(_safe_text(board_url))}\">Open full partner board</a></p>
</body></html>"""


def build_pairing_interest_text(
    *,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    recipient_kind: str,
    accept_url: str | None = None,
) -> str:
    intro = "A player sent pairing interest from the public tournament board."
    action_url = _safe_text(accept_url) or _safe_text(board_url)
    next_step = "Organizers should review pending pairing interest before final tournament operations."
    if recipient_kind == "player":
        intro = "Another player sent interest in pairing with you from the public tournament board."
        next_step = "If you accept this request, JUPR will automatically pair both registrations for that division."
    return "\n".join([
        intro,
        f"Tournament: {_safe_text(tournament_name)}",
        f"Division: {_safe_text(division_name)}",
        f"Interested player: {_safe_text(requester_name)}",
        f"Board entry: {_safe_text(target_name)}",
        f"Review request: {action_url}",
        f"Open tournament board: {_safe_text(board_url)}",
        next_step,
    ])


def _send_pairing_email(
    *,
    to_email: str,
    subject: str,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    recipient_kind: str,
    accept_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to_email = _safe_text(to_email)
    if not original_to_email:
        return {"status": "skipped", "provider_message_id": "missing_recipient", "to_email": ""}
    mode = get_email_mode()
    effective_to = original_to_email
    effective_subject = subject
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not redirect_to:
            raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
        effective_to = redirect_to
        effective_subject = f"[STAGING→{original_to_email}] {subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": original_to_email}
    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=effective_subject,
        html_body=build_pairing_interest_html(
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            recipient_kind=recipient_kind,
            accept_url=accept_url,
        ),
        text_body=build_pairing_interest_text(
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            recipient_kind=recipient_kind,
            accept_url=accept_url,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider_message_id, "to_email": effective_to}


def send_pairing_interest_emails(
    *,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    target_email: str | None,
    organizer_to_email: str | None = None,
    board_url: str = "",
    accept_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, dict[str, str]]:
    subject = build_pairing_interest_subject(tournament_name=tournament_name, division_name=division_name)
    organizer_to = _safe_text(organizer_to_email) or organizer_email()
    return {
        "player": _send_pairing_email(
            to_email=_safe_text(target_email),
            subject=subject,
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            accept_url=accept_url,
            recipient_kind="player",
            smtp_config=smtp_config,
        ),
        "organizer": _send_pairing_email(
            to_email=organizer_to,
            subject=subject,
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            accept_url=None,
            recipient_kind="organizer",
            smtp_config=smtp_config,
        ),
    }


def build_pairing_status_subject(*, tournament_name: str, action: str) -> str:
    label = {
        "accepted": "accepted",
        "declined": "declined",
        "cancelled": "cancelled",
    }.get(_safe_text(action).lower(), "updated")
    return f"Tournament partner request {label}: {_safe_text(tournament_name) or 'Tournament'}"


def _pairing_status_copy(*, action: str, recipient_kind: str) -> tuple[str, str]:
    clean_action = _safe_text(action).lower()
    if clean_action == "accepted":
        return (
            "The partner request was accepted and both registrations are now paired.",
            "Review confirmed pairing",
        )
    if clean_action == "declined":
        return (
            "The requested player declined the partner request. No team was created.",
            "Review partner requests",
        )
    if clean_action == "cancelled":
        if recipient_kind == "target":
            return (
                "The interested player cancelled the pending partner request. No team was created.",
                "Review partner requests",
            )
        return (
            "The pending partner request was cancelled. No team was created.",
            "Review partner requests",
        )
    return ("The partner request status changed.", "Review partner requests")


def build_pairing_status_html(
    *,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    action: str,
    recipient_kind: str,
) -> str:
    intro, action_label = _pairing_status_copy(action=action, recipient_kind=recipient_kind)
    return f"""<!doctype html><html><body style=\"font-family:Arial,sans-serif;color:#1f2937\">
<h1>Tournament partner request update</h1>
<p>{escape(intro)}</p>
<p><strong>Tournament:</strong> {escape(_safe_text(tournament_name))}<br>
<strong>Division:</strong> {escape(_safe_text(division_name))}<br>
<strong>Requester:</strong> {escape(_safe_text(requester_name))}<br>
<strong>Requested player:</strong> {escape(_safe_text(target_name))}</p>
<p><a href=\"{escape(_safe_text(board_url))}\" style=\"background:#2563eb;color:white;padding:10px 14px;text-decoration:none;border-radius:6px\">{escape(action_label)}</a></p>
<p>Contact details remain private; use the secure registration link above for any further action.</p>
</body></html>"""


def build_pairing_status_text(
    *,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    action: str,
    recipient_kind: str,
) -> str:
    intro, _action_label = _pairing_status_copy(action=action, recipient_kind=recipient_kind)
    return "\n".join(
        [
            intro,
            f"Tournament: {_safe_text(tournament_name)}",
            f"Division: {_safe_text(division_name)}",
            f"Requester: {_safe_text(requester_name)}",
            f"Requested player: {_safe_text(target_name)}",
            f"Review partner requests: {_safe_text(board_url)}",
            "Contact details remain private; use the secure registration link for any further action.",
        ]
    )


def _send_pairing_status_email(
    *,
    to_email: str,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    board_url: str,
    action: str,
    recipient_kind: str,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to_email = _safe_text(to_email)
    if not original_to_email:
        return {"status": "skipped", "provider_message_id": "missing_recipient", "to_email": ""}
    mode = get_email_mode()
    subject = build_pairing_status_subject(tournament_name=tournament_name, action=action)
    effective_to = original_to_email
    effective_subject = subject
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not redirect_to:
            raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
        effective_to = redirect_to
        effective_subject = f"[STAGING→{original_to_email}] {subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": original_to_email}
    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=effective_subject,
        html_body=build_pairing_status_html(
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            action=action,
            recipient_kind=recipient_kind,
        ),
        text_body=build_pairing_status_text(
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=board_url,
            action=action,
            recipient_kind=recipient_kind,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {
        "status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect",
        "provider_message_id": provider_message_id,
        "to_email": effective_to,
    }


def send_pairing_status_emails(
    *,
    action: str,
    tournament_name: str,
    division_name: str,
    requester_name: str,
    target_name: str,
    requester_email: str | None,
    target_email: str | None,
    requester_url: str,
    target_url: str,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, dict[str, str]]:
    """Send only the lifecycle messages relevant to the transition actor.

    Accepted pairings notify both registrations. Declines notify the requester;
    requester cancellations notify the target. Callers pass an empty address for
    privacy-denied recipients, which produces an auditable ``skipped`` status.
    """

    clean_action = _safe_text(action).lower()
    recipients: dict[str, tuple[str, str]] = {}
    if clean_action == "accepted":
        recipients = {
            "requester": (_safe_text(requester_email), requester_url),
            "target": (_safe_text(target_email), target_url),
        }
    elif clean_action == "declined":
        recipients = {"requester": (_safe_text(requester_email), requester_url)}
    elif clean_action == "cancelled":
        recipients = {"target": (_safe_text(target_email), target_url)}

    return {
        recipient_kind: _send_pairing_status_email(
            to_email=email,
            tournament_name=tournament_name,
            division_name=division_name,
            requester_name=requester_name,
            target_name=target_name,
            board_url=url,
            action=clean_action,
            recipient_kind=recipient_kind,
            smtp_config=smtp_config,
        )
        for recipient_kind, (email, url) in recipients.items()
    }
