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


def test_return_to_super_admin_sets_pending_reset_and_reruns():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'if st.sidebar.button("Return to Super Admin"):' in app
    assert 'st.session_state["admin_view_as_role"] = None' in app
    assert "st.session_state[VIEW_AS_RESET_PENDING_KEY] = True" in app
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL" not in app.split('if st.sidebar.button("Return to Super Admin"):', 1)[1].split('st.rerun()', 1)[0]


def test_stale_selector_value_is_guarded_after_return_to_actual_super_admin():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "last_selector_label = st.session_state.get(VIEW_AS_LAST_SELECTOR_KEY)" in app
    assert "selector_changed = picked_label != last_selector_label" in app
    assert "if selector_changed and picked_role != current_role:" in app


def test_pending_reset_applies_before_selectbox_render():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    pre_selectbox = app.split("picked_label = st.sidebar.selectbox", 1)[0]
    assert "if st.session_state.pop(VIEW_AS_RESET_PENDING_KEY, False):" in pre_selectbox
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL" in pre_selectbox


def test_selectbox_change_paths_still_switch_between_actual_and_organizer():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"View as Organizer": "organizer"' in app
    assert "picked_role = view_as_options[picked_label]" in app
    assert "if selector_changed and picked_role != current_role:" in app
    assert 'st.session_state["admin_view_as_role"] = picked_role or None' in app
