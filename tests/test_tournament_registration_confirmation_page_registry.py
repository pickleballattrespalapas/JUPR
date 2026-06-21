from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS, PAGE_DEFINITIONS


def test_tournament_registration_confirmation_registered_public_hidden():
    key = "tournament_registration_confirmation"
    definition = next(page for page in PAGE_DEFINITIONS if page.key == key)
    assert PAGE_KEY_TO_LABEL[key] == "✅ Registration Confirmation"
    assert definition.public is True
    assert key in HIDDEN_PAGE_KEYS
    assert key not in PUBLIC_NAV_KEYS
