from jupr_app.domain.admin.roles import ROLE_ORGANIZER, ROLE_SUPER_ADMIN, ROLE_UNASSIGNED
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


def test_unassigned_role_stays_unassigned_through_effective_role_resolution():
    role, source, view_as_role = resolve_effective_admin_role(
        ROLE_UNASSIGNED, "default", "read_only"
    )
    assert role == ROLE_UNASSIGNED
    assert source == "default"
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


def test_return_to_super_admin_restores_real_role_without_explicit_rerun():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'if st.sidebar.button("Return to Super Admin"):' in app
    assert 'st.session_state["admin_view_as_role"] = None' in app
    assert 'st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL' in app
    assert 'st.session_state["admin_role"] = st.session_state.get("admin_real_role", "read_only")' in app
    assert 'st.session_state["admin_role_source"] = st.session_state.get("admin_real_role_source", "not_authenticated")' in app
    return_block = app.split('if st.sidebar.button("Return to Super Admin"):', 1)[1].split("visible_labels =", 1)[0]
    assert "st.rerun()" not in return_block


def test_selected_label_reconciles_view_as_role_without_explicit_rerun():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "mapped_role = picked_role or None" in app
    assert "if mapped_role != current_role:" in app
    assert 'st.session_state["admin_view_as_role"] = mapped_role' in app
    selectbox_block = app.split("picked_label = st.sidebar.selectbox", 1)[1].split("# Optional: allow pages to request a refresh", 1)[0]
    assert "st.rerun()" not in selectbox_block


def test_pending_reset_applies_before_selectbox_render():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    pre_selectbox = app.split("picked_label = st.sidebar.selectbox", 1)[0]
    assert "if st.session_state.pop(VIEW_AS_RESET_PENDING_KEY, False):" in pre_selectbox
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL" in pre_selectbox


def test_selectbox_change_paths_still_switch_between_actual_and_organizer():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"View as Organizer": "organizer"' in app
    assert "picked_role = view_as_options[picked_label]" in app
    assert "mapped_role = picked_role or None" in app
    assert "if mapped_role != current_role:" in app
    assert 'st.session_state["admin_view_as_role"] = mapped_role' in app
    assert "_apply_effective_admin_role_from_view_as(mapped_role)" in app


def test_view_as_helper_restores_super_admin_when_actual_selected():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "def _apply_effective_admin_role_from_view_as(mapped_role: str | None) -> None:" in app
    assert 'st.session_state["admin_role"] = st.session_state.get("admin_real_role", "read_only")' in app
    assert 'st.session_state["admin_role_source"] = st.session_state.get("admin_real_role_source", "not_authenticated")' in app


def test_view_as_helper_sets_read_only_source_for_super_admin_view_as():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.session_state["admin_role"] = mapped_role' in app
    assert 'st.session_state["admin_role_source"] = "super_admin_view_as"' in app


def test_pending_reset_handled_before_effective_role_resolution():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    reset_idx = app.index("if st.session_state.pop(VIEW_AS_RESET_PENDING_KEY, False):")
    resolve_idx = app.index("resolve_effective_admin_role(")
    assert reset_idx < resolve_idx


def test_pending_reset_clears_view_as_before_role_resolution():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    pre_resolve = app.split("resolve_effective_admin_role(", 1)[0]
    assert 'st.session_state["admin_view_as_role"] = None' in pre_resolve


def test_pending_reset_sets_selector_before_selectbox():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    pre_selectbox = app.split("picked_label = st.sidebar.selectbox", 1)[0]
    assert "st.session_state[VIEW_AS_SELECTOR_KEY] = VIEW_AS_ACTUAL_LABEL" in pre_selectbox


def test_impossible_state_guard_present_after_effective_role_resolution():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.session_state.get("admin_view_as_role") is None' in app
    assert 'st.session_state.get("admin_role_source") == "super_admin_view_as"' in app
    assert 'st.session_state["admin_role"] = st.session_state.get("admin_real_role", "read_only")' in app
    assert 'st.session_state["admin_role_source"] = st.session_state.get("admin_real_role_source", "not_authenticated")' in app


def test_read_only_super_admin_view_as_caption_not_possible_with_actual_super_admin_label():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "Actual Super Admin" in app
    assert "Current role: **read_only** (source: `super_admin_view_as`)." not in Path("jupr_app/ui/pages/admin_tools.py").read_text(encoding="utf-8")
