from __future__ import annotations

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


def _text(value: Any) -> str:
    return str(value or "").strip()


def build_team_invitation_subject(
    *, tournament_name: str, team_name: str, captain_name: str
) -> str:
    return (
        f"{_text(captain_name) or 'A captain'} invited you to "
        f"{_text(team_name) or 'a team'} for {_text(tournament_name) or 'a tournament'}"
    )


def build_team_invitation_email_html(
    *,
    tournament_name: str,
    team_name: str,
    captain_name: str,
    invited_name: str,
    invitation_url: str,
) -> str:
    return f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#1f2937">
<h1>Four-player team invitation</h1>
<p>Hi {escape(_text(invited_name) or 'there')},</p>
<p>{escape(_text(captain_name) or 'A captain')} invited you to join
<strong>{escape(_text(team_name))}</strong> for
<strong>{escape(_text(tournament_name))}</strong>.</p>
<p><a href="{escape(_text(invitation_url), quote=True)}">Accept or decline this invitation</a></p>
<p>This is a private, single-use link. If you were not expecting it, you may decline.</p>
</body></html>"""


def build_team_invitation_email_text(
    *,
    tournament_name: str,
    team_name: str,
    captain_name: str,
    invited_name: str,
    invitation_url: str,
) -> str:
    return "\n".join(
        [
            f"Hi {_text(invited_name) or 'there'},",
            (
                f"{_text(captain_name) or 'A captain'} invited you to join "
                f"{_text(team_name)} for {_text(tournament_name)}."
            ),
            "",
            f"Accept or decline: {_text(invitation_url)}",
            "",
            "This is a private, single-use invitation link.",
        ]
    )


def send_team_invitation_email(
    *,
    target_email: str,
    tournament_name: str,
    team_name: str,
    captain_name: str,
    invited_name: str,
    invitation_url: str,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to = _text(target_email).lower()
    if not original_to or "@" not in original_to:
        raise ValueError("The invited teammate needs a valid email address.")
    url = _text(invitation_url)
    if not url.startswith("https://") and not url.startswith("http://localhost"):
        raise ValueError("A safe tournament invitation URL is required.")
    subject = build_team_invitation_subject(
        tournament_name=tournament_name,
        team_name=team_name,
        captain_name=captain_name,
    )
    mode = get_email_mode()
    effective_to = original_to
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not redirect_to:
            raise ValueError(
                "JUPR_STAGING_EMAIL_REDIRECT_TO is required when "
                "JUPR_EMAIL_MODE=staging_redirect."
            )
        effective_to = redirect_to
        subject = f"[STAGING→{original_to}] {subject}"
    if mode == EMAIL_MODE_DRY_RUN:
        return {
            "status": "dry_run",
            "provider_message_id": "dry_run",
            "to_email": original_to,
        }
    provider_message_id = send_email_with_inline_chart(
        to_email=effective_to,
        subject=subject,
        html_body=build_team_invitation_email_html(
            tournament_name=tournament_name,
            team_name=team_name,
            captain_name=captain_name,
            invited_name=invited_name,
            invitation_url=url,
        ),
        text_body=build_team_invitation_email_text(
            tournament_name=tournament_name,
            team_name=team_name,
            captain_name=captain_name,
            invited_name=invited_name,
            invitation_url=url,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {
        "status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect",
        "provider_message_id": provider_message_id,
        "to_email": effective_to,
    }


# Full names retained for discoverability.
build_tournament_team_invitation_subject = build_team_invitation_subject
build_tournament_team_invitation_email_html = build_team_invitation_email_html
build_tournament_team_invitation_email_text = build_team_invitation_email_text
send_tournament_team_invitation_email = send_team_invitation_email
