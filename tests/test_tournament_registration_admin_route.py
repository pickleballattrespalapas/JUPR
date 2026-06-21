from __future__ import annotations

from pathlib import Path

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS
from jupr_app.ui.admin_page_permissions import ADMIN_PAGE_PERMISSION_MATRIX
from jupr_app.ui.page_registry import PAGE_DEFINITIONS, PAGE_KEY_TO_LABEL


def _source(path: str) -> str:
    return Path(path).read_text()


def test_registration_admin_alias_registered_as_admin_only_not_public():
    assert PAGE_KEY_TO_LABEL["tournament_registration_admin"] == "🧾 Registration Management"

    page = next(page for page in PAGE_DEFINITIONS if page.key == "tournament_registration_admin")
    assert page.admin_only is True
    assert page.public is False


def test_registration_admin_alias_permission_requires_manage_tournaments():
    assert "tournament_registration_admin" in ADMIN_PAGE_PERMISSION_MATRIX
    assert ADMIN_PAGE_PERMISSION_MATRIX["tournament_registration_admin"] == (
        PERMISSION_MANAGE_TOURNAMENTS,
    )


def test_streamlit_pages_maps_registration_management_to_existing_module():
    source = _source("streamlit_app.py")
    assert '"🧾 Registration Management": tournament_registration' in source
    assert '"📝 Tournament Registration": tournament_registration' in source


def test_tournament_ops_manage_registrations_routes_to_admin_alias():
    source = _source("jupr_app/ui/pages/tournament_ops.py")
    button_index = source.index('st.button("Manage Registrations"')
    route_block = source[button_index : source.index("st.rerun()", button_index)]

    assert 'st.query_params["admin"] = "1"' in route_block
    assert 'st.query_params["page"] = "tournament_registration_admin"' in route_block
    assert 'st.query_params["page"] = "tournament_registration"' not in route_block
    assert 'st.query_params.pop("public", None)' in route_block


def test_tournament_registration_admin_render_preserves_alias_page_key():
    source = _source("jupr_app/ui/pages/tournament_registration.py")
    assert 'current_page_key = _safe_text(st.query_params.get("page"))' in source
    assert 'if admin_mode and current_page_key == "tournament_registration_admin"' in source
    assert '_select_admin_tournament(ctx, supabase, page_key=admin_page_key)' in source


def test_public_registration_route_remains_public_page_key():
    source = _source("jupr_app/ui/pages/tournament_registration.py")
    assert '_select_public_tournament(ctx, supabase, page_key="tournament_registration")' in source
    assert 'navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)' in source
