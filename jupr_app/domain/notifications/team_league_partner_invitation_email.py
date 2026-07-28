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


def build_team_league_partner_invitation_subject(
    *, league_name: str, captain_name: str
) -> str:
    return (
        f"{_text(captain_name) or 'A player'} invited you to join "
        f"{_text(league_name) or 'a team league'}"
    )


def build_team_league_partner_invitation_email_html(
    *,
    club_name: str,
    league_name: str,
    team_name: str,
    captain_name: str,
    partner_name: str,
    confirmation_url: str,
    expires_label: str,
) -> str:
    return f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#1f2937">
<h1>Confirm your team-league partner</h1>
<p>Hi {escape(_text(partner_name) or 'there')},</p>
<p>{escape(_text(captain_name) or 'A player')} invited you to join
<strong>{escape(_text(team_name))}</strong> in
<strong>{escape(_text(league_name))}</strong> at {escape(_text(club_name))}.</p>
<p><a href="{escape(_text(confirmation_url), quote=True)}">Review and confirm the invitation</a></p>
<p>This private link expires {escape(_text(expires_label))}. If you were not expecting
this invitation, you can decline it from the same page.</p>
</body></html>"""


def build_team_league_partner_invitation_email_text(
    *,
    club_name: str,
    league_name: str,
    team_name: str,
    captain_name: str,
    partner_name: str,
    confirmation_url: str,
    expires_label: str,
) -> str:
    return "\n".join(
        [
            f"Hi {_text(partner_name) or 'there'},",
            (
                f"{_text(captain_name) or 'A player'} invited you to join "
                f"{_text(team_name)} in {_text(league_name)} at {_text(club_name)}."
            ),
            "",
            f"Review and confirm: {_text(confirmation_url)}",
            "",
            f"This private link expires {_text(expires_label)}.",
        ]
    )


def send_team_league_partner_invitation_email(
    *,
    target_email: str,
    club_name: str,
    league_name: str,
    team_name: str,
    captain_name: str,
    partner_name: str,
    confirmation_url: str,
    expires_label: str,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    original_to = _text(target_email).lower()
    if not original_to or "@" not in original_to:
        raise ValueError("The selected partner does not have a valid email address.")
    if not _text(confirmation_url).startswith("https://") and not _text(
        confirmation_url
    ).startswith("http://localhost"):
        raise ValueError("A safe partner-confirmation URL is required.")

    subject = build_team_league_partner_invitation_subject(
        league_name=league_name,
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
        html_body=build_team_league_partner_invitation_email_html(
            club_name=club_name,
            league_name=league_name,
            team_name=team_name,
            captain_name=captain_name,
            partner_name=partner_name,
            confirmation_url=confirmation_url,
            expires_label=expires_label,
        ),
        text_body=build_team_league_partner_invitation_email_text(
            club_name=club_name,
            league_name=league_name,
            team_name=team_name,
            captain_name=captain_name,
            partner_name=partner_name,
            confirmation_url=confirmation_url,
            expires_label=expires_label,
        ),
        chart_png_bytes=None,
        smtp_config=smtp_config,
    )
    return {
        "status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect",
        "provider_message_id": provider_message_id,
        "to_email": effective_to,
    }
