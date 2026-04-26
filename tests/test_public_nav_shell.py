from pathlib import Path


def test_public_nav_uses_html_anchors_instead_of_streamlit_radio():
    contents = Path("jupr_app/ui/public_nav.py").read_text(encoding="utf-8")

    assert "st.radio" not in contents
    assert "jupr-public-nav" in contents
    assert '"./"' in contents
    assert 'params["page"] = page_key' in contents
    assert 'source="public_header:admin_login"' in contents
    assert 'source="public_header:admin_dashboard"' in contents
    assert 'source="public_header:logout"' in contents
    assert 'default_page: str = "home"' in contents


def test_public_nav_does_not_use_header_tag():
    contents = Path("jupr_app/ui/public_nav.py").read_text(encoding="utf-8")

    assert '<header class="jupr-public-nav">' not in contents
    assert '<div class="jupr-public-nav" role="navigation" aria-label="Public site navigation">' in contents


def test_public_mode_hides_only_streamlit_header():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "header{visibility:hidden;}" not in contents
    assert 'header[data-testid="stHeader"]{visibility:hidden;}' in contents
