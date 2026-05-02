from jupr_app.domain.admin.roles import ROLE_ORGANIZER, ROLE_SUPER_ADMIN
from jupr_app.ui.admin_view_as import (
    can_use_view_as,
    resolve_effective_admin_role,
    sanitize_view_as_role,
)
from pathlib import Path


def test_super_admin_can_select_effective_role():
    role, source, view_as_role = resolve_effective_admin_role(
        "super_admin", "admin_role_assignments", "organizer"
    )
    assert role == "organizer"
    assert source == "super_admin_view_as"
    assert view_as_role == "organizer"


def test_non_super_admin_cannot_activate_view_as_mode():
    role, source, view_as_role = resolve_effective_admin_role(
        "organizer", "admin_role_assignments", "scorekeeper"
    )
    assert role == "organizer"
    assert source == "admin_role_assignments"
    assert view_as_role is None


def test_real_role_remains_super_admin_while_effective_role_changes():
    real_role = ROLE_SUPER_ADMIN
    effective_role, source, _ = resolve_effective_admin_role(
        real_role, "admin_role_assignments", ROLE_ORGANIZER
    )
    assert real_role == ROLE_SUPER_ADMIN
    assert effective_role == ROLE_ORGANIZER
    assert source == "super_admin_view_as"


def test_view_as_helpers_enforce_super_admin_only():
    assert can_use_view_as("super_admin") is True
    assert can_use_view_as("organizer") is False
    assert sanitize_view_as_role("organizer", "scorekeeper") is None


def test_view_as_selector_uses_explicit_widget_key():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'VIEW_AS_SELECTOR_KEY = "admin_view_as_selector_label"' in app
    assert 'key=VIEW_AS_SELECTOR_KEY' in app


def test_return_to_super_admin_clears_role_and_resets_selector_label():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'if st.sidebar.button("Return to Super Admin"):' in app
    assert 'st.session_state["admin_view_as_role"] = None' in app
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL" in app


def test_stale_selector_value_is_guarded_after_return_to_actual_super_admin():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "current_view_as == \"\" and selector_label != VIEW_AS_ACTUAL_LABEL" in app
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = selected_label" in app


def test_selectbox_change_paths_still_switch_between_actual_and_organizer():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"View as Organizer": "organizer"' in app
    assert "picked_role = view_as_options[picked_label]" in app
    assert "if picked_role != current_role:" in app
    assert 'st.session_state["admin_view_as_role"] = picked_role or None' in app
