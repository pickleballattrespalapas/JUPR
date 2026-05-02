from jupr_app.ui.page_registry import (
    ADMIN_ONLY_PAGE_KEYS,
    HIDDEN_PAGE_KEYS,
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)
from jupr_app.ui.admin_page_permissions import ADMIN_PAGE_PERMISSION_MATRIX


def test_jupr_live_is_registered_as_public_page():
    public_labels = labels_for_keys(PUBLIC_NAV_KEYS)

    assert "jupr_live" in PAGE_KEY_TO_LABEL
    assert "jupr_live_admin" in PAGE_KEY_TO_LABEL
    assert PAGE_KEY_TO_LABEL["jupr_live"] == "🔴 JUPR Live"
    assert PAGE_KEY_TO_LABEL["jupr_live_admin"] == "🔴 JUPR Live Admin"
    assert LABEL_TO_PAGE_KEY["🔴 JUPR Live"] == "jupr_live"
    assert LABEL_TO_PAGE_KEY["🔴 JUPR Live Admin"] == "jupr_live_admin"
    assert "🔴 JUPR Live" in public_labels
    assert "🔴 JUPR Live Admin" not in public_labels
    assert "jupr_live_social" not in PAGE_KEY_TO_LABEL
    assert "🟢 JUPR Live Social" not in LABEL_TO_PAGE_KEY
    assert "🟢 JUPR Live Social" not in public_labels


def test_existing_public_pages_remain_in_shared_public_nav():
    public_labels = labels_for_keys(PUBLIC_NAV_KEYS)

    assert public_labels == [
        "Home",
        "🏆 Leaderboards",
        "📊 League Results",
        "🎯 Match Explorer",
        "🔍 Player Search",
        "📼 Badge Codex",
        "🪜 Challenge Ladder",
        "🔴 JUPR Live",
        "📝 Tournament Registration",
        "🤝 Partner Board",
        "Rating Rules",
        "🗞️ Weekly Recap",
        "❓ FAQs",
    ]


def test_restored_public_route_keys_are_visible_in_public_nav():
    restored_keys = {
        "league_results",
        "match_explorer",
        "badge_codex",
        "challenge_ladder",
        "jupr_live",
        "tournament_partner_board",
    }

    assert restored_keys.issubset(set(PUBLIC_NAV_KEYS))


def test_admin_only_pages_remain_excluded_from_public_nav():
    assert not set(PUBLIC_NAV_KEYS).intersection(ADMIN_ONLY_PAGE_KEYS)


def test_hidden_deep_link_pages_remain_hidden_from_public_nav():
    assert "verified_updates_request" in HIDDEN_PAGE_KEYS
    assert "tournament_roster" in HIDDEN_PAGE_KEYS
    assert "privacy_policy" in HIDDEN_PAGE_KEYS
    assert "terms_of_use" in HIDDEN_PAGE_KEYS
    assert "contact_support" in HIDDEN_PAGE_KEYS
    assert "data_corrections" in HIDDEN_PAGE_KEYS
    assert "email_preferences" in HIDDEN_PAGE_KEYS
    assert "profile_privacy" in HIDDEN_PAGE_KEYS
    assert "verified_updates_request" not in PUBLIC_NAV_KEYS
    assert "tournament_roster" not in PUBLIC_NAV_KEYS


def test_player_editor_remains_admin_only():
    public_labels = labels_for_keys(PUBLIC_NAV_KEYS)
    assert "player_editor" not in PUBLIC_NAV_KEYS
    assert PAGE_KEY_TO_LABEL["player_editor"] == "👥 Player Editor"
    assert "👥 Player Editor" not in public_labels


def test_every_admin_only_page_is_mapped_or_intentionally_blocked():
    intentionally_blocked = {"admin_login", "reset_password"}
    missing = sorted(
        key
        for key in ADMIN_ONLY_PAGE_KEYS
        if key not in ADMIN_PAGE_PERMISSION_MATRIX and key not in intentionally_blocked
    )
    assert missing == []
