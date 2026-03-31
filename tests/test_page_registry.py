from jupr_app.ui.page_registry import (
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)


def test_jupr_live_is_registered_as_public_page():
    public_labels = labels_for_keys(PUBLIC_NAV_KEYS)

    assert "jupr_live" in PUBLIC_NAV_KEYS
    assert "jupr_live_admin" not in PUBLIC_NAV_KEYS
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
        "🏆 Leaderboards",
        "📊 League Results",
        "🗞️ Weekly Recap",
        "📝 Tournament Registration",
        "🤝 Partner Board",
        "🎯 Match Explorer",
        "🔍 Player Search",
        "📼 Badge Codex",
        "🔴 JUPR Live",
        "🪜 Challenge Ladder",
        "❓ FAQs",
    ]


def test_player_editor_remains_admin_only():
    public_labels = labels_for_keys(PUBLIC_NAV_KEYS)
    assert "player_editor" not in PUBLIC_NAV_KEYS
    assert PAGE_KEY_TO_LABEL["player_editor"] == "👥 Player Editor"
    assert "👥 Player Editor" not in public_labels
