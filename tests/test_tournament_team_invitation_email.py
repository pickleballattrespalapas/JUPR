from jupr_app.domain.notifications import tournament_team_invitation_email as email


def _kwargs():
    return {
        "tournament_name": "Summer <Open>",
        "team_name": "Dink & Drink",
        "captain_name": "Captain <Chris>",
        "invited_name": "Pat <Player>",
        "invitation_url": "https://example.test/invite#token=secret",
    }


def test_email_html_escapes_user_content():
    html = email.build_team_invitation_email_html(**_kwargs())

    assert "Summer &lt;Open&gt;" in html
    assert "Captain &lt;Chris&gt;" in html
    assert "Pat &lt;Player&gt;" in html
    assert "Dink &amp; Drink" in html


def test_dry_run_never_calls_smtp(monkeypatch):
    monkeypatch.setattr(email, "get_email_mode", lambda: email.EMAIL_MODE_DRY_RUN)
    monkeypatch.setattr(
        email,
        "send_email_with_inline_chart",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("SMTP was called")),
    )

    result = email.send_team_invitation_email(
        target_email="player@example.com",
        **_kwargs(),
    )

    assert result == {
        "status": "dry_run",
        "provider_message_id": "dry_run",
        "to_email": "player@example.com",
    }
