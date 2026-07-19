from pathlib import Path


WEB_APP = Path("apps/web/app")


def test_verified_updates_next_page_is_club_scoped_and_uses_status_api():
    root_page = (WEB_APP / "verified-updates/page.tsx").read_text(encoding="utf-8")
    club_page = (WEB_APP / "clubs/[clubSlug]/verified-updates/page.tsx").read_text(encoding="utf-8")
    form = (WEB_APP / "verified-updates/VerifiedUpdatesRequestForm.tsx").read_text(encoding="utf-8")

    assert 'const clubSlug = "tres-palapas"' not in root_page
    assert "params.clubSlug" in club_page
    assert "/verified-updates/status?player_id=" in form
    assert 'type="email" required maxLength={320}' in form
    assert 'type="submit"' in form
    assert "pending_admin_review" in form
    assert "Verified player updates are active" in form


def test_player_pages_link_to_club_scoped_verified_updates():
    players = (WEB_APP / "clubs/[clubSlug]/players/page.tsx").read_text(encoding="utf-8")
    profile = (WEB_APP / "clubs/[clubSlug]/players/[playerId]/page.tsx").read_text(encoding="utf-8")

    assert "/clubs/${clubSlug}/verified-updates?player_id=" in players
    assert "/clubs/${clubSlug}/verified-updates?player_id=" in profile


def test_email_preferences_client_posts_token_not_subscription_id():
    panel = (WEB_APP / "email-preferences/EmailPreferencesPanel.tsx").read_text(encoding="utf-8")
    sender = Path("jupr_app/domain/notifications/player_update_sender.py").read_text(encoding="utf-8")

    assert "unsubscribeEmailPreferences({ token, ut, scope })" in panel
    assert "subscription_id: subscriptionId" not in panel
    assert "/email-preferences?" in sender
    assert 'unsubscribe_params["sid"]' not in sender
