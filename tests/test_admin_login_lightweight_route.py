from pathlib import Path


def test_streamlit_app_has_minimal_auth_context_helper():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "def _build_minimal_context(" in app
    assert "schema_degraded=False" in app
    assert "schema_degraded_reason=None" in app


def test_streamlit_app_handles_auth_routes_before_get_data():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")

    early_branch = app.index("auth_route_requested =")
    get_data_call = app.index(") = get_data(selected_club_id)")

    assert early_branch < get_data_call
    assert "if auth_route_requested:" in app
    assert "from jupr_app.ui.pages import admin_login, reset_password" in app


def test_streamlit_app_early_auth_route_renders_without_full_data_load():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "auth_ctx = _build_minimal_context(" in app
    assert "admin_login.render(auth_ctx)" in app
    assert "reset_password.render(auth_ctx)" in app
    assert "return" in app[app.index("if auth_route_requested:"): app.index("# ---- Load data + ctx ----")]
