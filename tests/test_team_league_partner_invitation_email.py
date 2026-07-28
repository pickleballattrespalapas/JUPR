from __future__ import annotations

from jupr_app.domain.notifications import (
    team_league_partner_invitation_email as email,
)


def _kwargs() -> dict[str, str]:
    return {
        "club_name": "Club <North>",
        "league_name": "Fall & Friends",
        "team_name": "Dink <Duo>",
        "captain_name": "Alex <Captain>",
        "partner_name": "Blair <Partner>",
        "confirmation_url": (
            "https://staging.example.test/clubs/north/"
            "team-league-partner-confirmation?team=abc#token=secret"
        ),
        "expires_label": "August 10 <UTC>",
    }


def test_team_league_invitation_html_escapes_user_content() -> None:
    html = email.build_team_league_partner_invitation_email_html(**_kwargs())

    assert "Club &lt;North&gt;" in html
    assert "Fall &amp; Friends" in html
    assert "Dink &lt;Duo&gt;" in html
    assert "Alex &lt;Captain&gt;" in html
    assert "Blair &lt;Partner&gt;" in html
    assert "August 10 &lt;UTC&gt;" in html


def test_team_league_invitation_dry_run_never_calls_smtp(monkeypatch) -> None:
    monkeypatch.setattr(email, "get_email_mode", lambda: email.EMAIL_MODE_DRY_RUN)
    monkeypatch.setattr(
        email,
        "send_email_with_inline_chart",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SMTP was called")
        ),
    )

    result = email.send_team_league_partner_invitation_email(
        target_email="partner@example.com",
        **_kwargs(),
    )

    assert result == {
        "status": "dry_run",
        "provider_message_id": "dry_run",
        "to_email": "partner@example.com",
    }


def test_team_league_invitation_staging_redirect_preserves_target_in_subject(
    monkeypatch,
) -> None:
    sent: dict[str, str] = {}
    monkeypatch.setattr(
        email, "get_email_mode", lambda: email.EMAIL_MODE_STAGING_REDIRECT
    )
    monkeypatch.setattr(
        email,
        "get_env_or_default",
        lambda name: (
            "staging-inbox@example.test"
            if name == "JUPR_STAGING_EMAIL_REDIRECT_TO"
            else ""
        ),
    )

    def fake_send(**kwargs):
        sent.update(
            {
                "to_email": kwargs["to_email"],
                "subject": kwargs["subject"],
                "html_body": kwargs["html_body"],
            }
        )
        return "provider-1"

    monkeypatch.setattr(email, "send_email_with_inline_chart", fake_send)

    result = email.send_team_league_partner_invitation_email(
        target_email="partner@example.com",
        **_kwargs(),
    )

    assert result == {
        "status": "staging_redirect",
        "provider_message_id": "provider-1",
        "to_email": "staging-inbox@example.test",
    }
    assert sent["to_email"] == "staging-inbox@example.test"
    assert sent["subject"].startswith("[STAGING→partner@example.com]")
    assert "https://staging.example.test/" in sent["html_body"]
