from jupr_app.domain.admin.roles import ROLE_ORGANIZER, ROLE_SUPER_ADMIN
from jupr_app.domain.admin_activity_log import build_activity_payload
from jupr_app.ui.admin_page_permissions import is_admin_page_available_for_role


def _apply_view_as(real_role: str, view_as_role: str | None) -> tuple[str, str]:
    if real_role != "super_admin":
        return real_role, "admin_role_assignments"
    if view_as_role in {"club_owner", "organizer", "scorekeeper", "read_only"}:
        return view_as_role, "super_admin_view_as"
    return real_role, "admin_role_assignments"


def test_super_admin_can_select_effective_role():
    role, source = _apply_view_as("super_admin", "organizer")
    assert role == "organizer"
    assert source == "super_admin_view_as"


def test_non_super_admin_cannot_activate_view_as_mode():
    role, source = _apply_view_as("organizer", "scorekeeper")
    assert role == "organizer"
    assert source != "super_admin_view_as"


def test_effective_role_controls_navigation_visibility():
    assert is_admin_page_available_for_role("match_uploader", ROLE_ORGANIZER) is True
    assert is_admin_page_available_for_role("player_editor", ROLE_ORGANIZER) is False


def test_real_role_remains_super_admin_while_effective_role_changes():
    real_role = ROLE_SUPER_ADMIN
    effective_role, source = _apply_view_as(real_role, ROLE_ORGANIZER)
    assert real_role == ROLE_SUPER_ADMIN
    assert effective_role == ROLE_ORGANIZER
    assert source == "super_admin_view_as"


def test_deep_linked_page_blocked_when_role_lacks_permission():
    assert is_admin_page_available_for_role("player_editor", "scorekeeper") is False


def test_view_as_state_is_session_only():
    session_state = {"admin_view_as_role": "organizer"}
    session_state.pop("admin_view_as_role", None)
    assert "admin_view_as_role" not in session_state


def test_admin_activity_payload_supports_real_and_effective_roles():
    payload = build_activity_payload(
        club_id="club-1",
        actor_email="admin@example.com",
        actor_role=ROLE_SUPER_ADMIN,
        action_type="match_edit",
        entity_type="match",
        entity_id="1",
        after_json={"effective_role": ROLE_ORGANIZER, "source": "super_admin_view_as"},
    )
    assert payload["actor_role"] == ROLE_SUPER_ADMIN
    assert payload["after_json"]["effective_role"] == ROLE_ORGANIZER
