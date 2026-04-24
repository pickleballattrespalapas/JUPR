from pathlib import Path


def test_public_nav_uses_html_anchors_instead_of_streamlit_radio():
    contents = Path("jupr_app/ui/public_nav.py").read_text(encoding="utf-8")

    assert "st.radio" not in contents
    assert "jupr-public-nav" in contents
    assert '"./"' in contents
    assert '"?page=leaderboards"' in contents
    assert '"?admin=1&page=admin_login"' in contents
    assert 'default_page: str = "home"' in contents
