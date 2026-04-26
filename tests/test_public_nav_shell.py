from pathlib import Path


def test_public_nav_uses_same_tab_routing_buttons():
    contents = Path("jupr_app/ui/public_nav.py").read_text(encoding="utf-8")

    assert "st.radio" not in contents
    assert 'st.button("Home"' not in contents  # built from map-driven labels
    assert "navigate_same_tab(" in contents
    assert 'source="public_header:home"' in contents
    assert '"leaderboards": "public_header:leaderboards"' in contents
    assert '"league_results": "public_header:league_results"' in contents
    assert '"match_explorer": "public_header:match_explorer"' in contents
    assert '"badge_codex": "public_header:badge_codex"' in contents
    assert '"challenge_ladder": "public_header:challenge_ladder"' in contents
    assert '"jupr_live": "public_header:jupr_live"' in contents
    assert '"players": "public_header:players"' in contents
    assert '"tournament_registration": "public_header:tournament_registration"' in contents
    assert '"tournament_partner_board": "public_header:tournament_partner_board"' in contents
    assert '"rating_rules": "public_header:rating_rules"' in contents
    assert '"weekly_recap": "public_header:weekly_recap"' in contents
    assert '"faqs": "public_header:faqs"' in contents
    assert '"public_header:admin_dashboard" if admin_authenticated else "public_header:admin_login"' in contents
    assert 'source="public_header:logout"' in contents
    assert "default_page: str = \"home\"" in contents


def test_public_nav_does_not_use_internal_anchor_links():
    contents = Path("jupr_app/ui/public_nav.py").read_text(encoding="utf-8")

    assert 'href="?' not in contents
    assert '<a class="jupr-public-nav-link' not in contents
    assert '<div class="jupr-public-nav" role="navigation" aria-label="Public site navigation">' in contents


def test_public_mode_hides_only_streamlit_header():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "header{visibility:hidden;}" not in contents
    assert 'header[data-testid="stHeader"]{visibility:hidden;}' in contents
