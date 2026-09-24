"""Club staff policy. Club identity always comes from the authenticated route."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

operator_request_authorized: ContextVar[bool] = ContextVar("operator_request_authorized", default=False)

ADMIN_ROLES = frozenset({"administrator", "club_owner", "super_admin"})
PROGRAM_TYPES = frozenset({"leagues", "tournaments", "round_robin", "ladder", "challenge_ladder", "live_play", "moneyball"})


def assignment_active(row: dict[str, Any], now: datetime | None = None) -> bool:
    if row.get("revoked_at"):
        return False
    value = row.get("expires_at")
    if not value:
        return True
    try:
        end = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if end.tzinfo is None:
            return False
        return end > (now or datetime.now(timezone.utc))
    except (ValueError, TypeError):
        return False


def validate_scopes(scopes: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not scopes or len(scopes) > 100:
        raise ValueError("Choose at least one scope (maximum 100).")
    result = []
    for scope in scopes:
        kind = str(scope.get("kind", ""))
        program = str(scope.get("program_type", ""))
        resource = str(scope.get("resource_id", "")).strip()
        if kind not in {"club", "program_type", "resource"}:
            raise ValueError("Unknown staff scope.")
        if kind != "club" and program not in PROGRAM_TYPES:
            raise ValueError("Choose a supported program type.")
        if kind == "resource" and (not resource or len(resource) > 200):
            raise ValueError("Choose the assigned program or session.")
        result.append({"kind": kind, "program_type": program if kind != "club" else "", "resource_id": resource if kind == "resource" else ""})
    return result


def permits(scopes: list[dict[str, Any]], program: str, resources: set[str]) -> bool:
    return any(
        scope.get("kind") == "club"
        or (scope.get("program_type") == program and (
            scope.get("kind") == "program_type"
            or (scope.get("kind") == "resource" and str(scope.get("resource_id")) in resources)
        )) for scope in scopes
    )
