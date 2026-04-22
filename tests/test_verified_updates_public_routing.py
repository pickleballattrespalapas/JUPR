from pathlib import Path

from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS


def test_verified_updates_request_page_registered_as_hidden_public_route():
    assert "verified_updates_request" in PAGE_KEY_TO_LABEL
    assert PAGE_KEY_TO_LABEL["verified_updates_request"] == "📬 Verified Updates Request"
    assert "verified_updates_request" in HIDDEN_PAGE_KEYS
    assert "verified_updates_request" not in PUBLIC_NAV_KEYS


def test_players_page_uses_durable_verified_updates_route_params():
    contents = Path("jupr_app/ui/pages/players.py").read_text(encoding="utf-8")

    assert '"page": "verified_updates_request"' in contents
    assert '"player_id": int(pid)' in contents
    assert '"club_id": str(club_id)' in contents
    assert 'st.query_params["pid"] = str(pid)' in contents
    assert 'st.query_params["player_id"] = str(pid)' in contents
    assert 'st.query_params.pop("pid", None)' not in contents


def test_players_page_guards_invalid_selectbox_state_for_verified_route():
    contents = Path("jupr_app/ui/pages/players.py").read_text(encoding="utf-8")

    assert 'current_pick = st.session_state.get("player_search_id", "")' in contents
    assert 'if current_pick not in options:' in contents
    assert 'st.session_state["player_search_id"] = ""' in contents
    assert 'Missing required player_id for verified updates request' in contents
    assert 'Player not found for verified updates request' in contents


def test_streamlit_main_surfaces_page_render_exceptions_without_route_fallback():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'Page render failed.' in contents
    assert 'st.error("This page failed to render.")' in contents
    assert 'st.exception(exc)' in contents
    assert 'Append ?debug=1 to the URL to view exception details in development.' in contents
