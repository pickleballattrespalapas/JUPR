from jupr_app.domain.admin.roles import ROLE_ORGANIZER, ROLE_SUPER_ADMIN
from jupr_app.ui.admin_view_as import (
    can_use_view_as,
    resolve_effective_admin_role,
    sanitize_view_as_role,
)


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
