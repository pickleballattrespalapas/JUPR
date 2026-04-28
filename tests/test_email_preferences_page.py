from pathlib import Path

from jupr_app.ui.pages.email_preferences import parse_unsubscribe_identifiers


def test_parse_unsubscribe_identifiers_prefers_token_and_supports_sid_fallback():
    token, sid = parse_unsubscribe_identifiers(token_q="abc", ut_q="", sid_q="s1", subscription_id_q="")
    assert token == "abc"
    assert sid == "s1"

    token, sid = parse_unsubscribe_identifiers(token_q="", ut_q="legacy", sid_q="", subscription_id_q="sub-2")
    assert token == "legacy"
    assert sid == "sub-2"


def test_email_preferences_page_registered_and_routed():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '"Email Preferences": email_preferences' in contents
    assert '"Privacy Policy": privacy_policy' in contents
    assert '"Terms of Use": terms_of_use' in contents
