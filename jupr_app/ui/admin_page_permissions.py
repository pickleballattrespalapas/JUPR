from __future__ import annotations

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_PLAYERS,
    PERMISSION_MANAGE_SUBSCRIPTIONS,
    PERMISSION_MANAGE_TOURNAMENTS,
    PERMISSION_RUN_REPLAY,
    PERMISSION_VIEW_AUDIT_LOG,
    has_permission,
)

ADMIN_PAGE_PERMISSION_MATRIX: dict[str, tuple[str, ...]] = {
    "league_manager": (PERMISSION_MANAGE_MATCHES, PERMISSION_ENTER_SCORES),
    "match_uploader": (PERMISSION_ENTER_SCORES,),
    "match_log": (PERMISSION_MANAGE_MATCHES, PERMISSION_ENTER_SCORES),
    "player_editor": (PERMISSION_MANAGE_PLAYERS,),
    "admin_tools": (PERMISSION_VIEW_AUDIT_LOG,),
    "tournaments": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "tournament_manager": (PERMISSION_MANAGE_TOURNAMENTS,),
    "tournament_ops": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "tournament_live": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "player_updates_admin": (PERMISSION_MANAGE_SUBSCRIPTIONS,),
    "weekly_recap_admin": (PERMISSION_RUN_REPLAY,),
    "badge_debug": (PERMISSION_RUN_REPLAY,),
    "badge_audit": (PERMISSION_RUN_REPLAY,),
    "match_canonical_audit": (PERMISSION_RUN_REPLAY,),
    "top_players_printable": (PERMISSION_RUN_REPLAY,),
}


def is_admin_page_available_for_role(page_key: str, role: str) -> bool:
    required_permissions = ADMIN_PAGE_PERMISSION_MATRIX.get(str(page_key or "").strip().lower())
    if not required_permissions:
        return True
    return any(has_permission(role, permission) for permission in required_permissions)
