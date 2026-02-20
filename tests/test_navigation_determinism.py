import streamlit_app
from jupr_app.ui.pages import admin_dashboard


def test_pending_nav_updates_target_next_run(monkeypatch):
    state = {}
    monkeypatch.setattr(streamlit_app.st, "session_state", state)

    streamlit_app._init_session()
    assert streamlit_app.st.session_state["_nav_target"] == "home"

    admin_dashboard._nav_to("🏟️ League Manager")
    assert streamlit_app.st.session_state["_nav_pending"] == "🏟️ League Manager"
    assert streamlit_app.st.session_state["_nav_target"] == "home"

    streamlit_app._process_pending_nav()

    assert streamlit_app.st.session_state["_nav_target"] == "🏟️ League Manager"
    assert streamlit_app.st.session_state["_nav_pending"] is None
