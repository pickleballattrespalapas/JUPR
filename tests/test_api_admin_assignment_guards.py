from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import HTTPException

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, PERMISSION_VIEW_AUDIT_LOG, ROLE_UNASSIGNED


@pytest.mark.parametrize(
    ("module_name", "helper_name", "helper_kwargs"),
    [
        (
            "services.api.admin_tools_routes",
            "_resolve_role_or_403",
            {"source": "test_admin_tools", "permission": PERMISSION_VIEW_AUDIT_LOG},
        ),
        (
            "services.api.admin_support_requests_routes",
            "_resolve_support_role_or_403",
            {"source": "test_admin_support", "write": False},
        ),
        (
            "services.api.admin_badge_diagnostics_routes",
            "_resolve_badge_diagnostics_role_or_403",
            {"source": "test_badges", "permission": PERMISSION_VIEW_AUDIT_LOG},
        ),
        (
            "services.api.admin_match_canonical_audit_routes",
            "_resolve_role_or_403",
            {"source": "test_canonical_audit", "permission": PERMISSION_VIEW_AUDIT_LOG},
        ),
        (
            "services.api.admin_match_log_routes",
            "_resolve_role_or_403",
            {"source": "test_match_log", "permission": PERMISSION_MANAGE_MATCHES},
        ),
    ],
)
def test_admin_helpers_reject_authenticated_user_without_explicit_club_assignment(
    monkeypatch,
    module_name: str,
    helper_name: str,
    helper_kwargs: dict[str, object],
) -> None:
    module = import_module(module_name)
    logged_payloads: list[dict] = []
    monkeypatch.setattr(
        module,
        "authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="reader@example.com", user_id="reader-1"),
    )
    monkeypatch.setattr(
        module,
        "resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role=ROLE_UNASSIGNED, assigned=False, source="default"),
    )
    monkeypatch.setattr(module, "write_admin_activity_log", lambda _supabase, payload: logged_payloads.append(payload))

    with pytest.raises(HTTPException) as exc_info:
        getattr(module, helper_name)(
            supabase=object(),
            club_id="club",
            authorization="Bearer local",
            **helper_kwargs,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "insufficient permission"
    assert logged_payloads[0]["after_json"]["reason"] == "missing_club_assignment"
