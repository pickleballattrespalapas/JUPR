from jupr_app.ui.page_registry import (
    ADMIN_ONLY_PAGE_KEYS,
    HIDDEN_PAGE_KEYS,
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)
from jupr_app.ui.admin_page_permissions import ADMIN_PAGE_PERMISSION_MATRIX
from pathlib import Path


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
        "🤝 Players Needing Partners",
        "Rating Rules",
        "🗞️ Weekly Recap",
        "❓ FAQs",
    ]


def test_streamlit_dispatch_uses_registered_players_needing_partners_label():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    label = PAGE_KEY_TO_LABEL["tournament_partner_board"]

    assert f'"{label}": tournament_partner_board' in app
    assert '"🤝 Partner Board": tournament_partner_board' not in app


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


def test_admin_sidebar_excludes_public_pages_and_uses_safe_public_link_control():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "if x in ADMIN_ONLY_LABELS" in app
    assert "st.sidebar.page_link(" not in app
    assert 'st.sidebar.button("🌐 View Public Site", key="admin_view_public_site_btn")' in app




def test_view_public_site_clears_admin_query_params_and_enters_public_mode():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"admin_view_public_site_btn"' in app
    assert 'st.session_state["jupr_public_mode"] = True' in app
    assert 'st.session_state["jupr_admin_entry_active"] = False' in app
    assert '"admin",' in app
    assert '"next",' in app
    assert '"jupr_admin_access_token",' in app
    assert '"jupr_admin_refresh_token",' in app
    assert '"jupr_admin_restore_from_storage",' in app
    assert '"logout",' in app
    assert '"page",' in app

def test_mixed_admin_public_route_is_not_kept_canonical():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "requested_public_page_while_admin" in app
