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
    "admin_guide": (PERMISSION_VIEW_AUDIT_LOG,),
    "admin_tools": (PERMISSION_VIEW_AUDIT_LOG,),
    "badge_audit": (PERMISSION_VIEW_AUDIT_LOG,),
    "badge_debug": (PERMISSION_VIEW_AUDIT_LOG,),
    "challenge_ladder_admin": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_MATCHES),
    "jupr_live_admin": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "league_manager": (PERMISSION_MANAGE_MATCHES, PERMISSION_ENTER_SCORES),
    "league_printout": (PERMISSION_ENTER_SCORES, PERMISSION_MANAGE_MATCHES),
    "match_canonical_audit": (PERMISSION_VIEW_AUDIT_LOG,),
    "match_log": (PERMISSION_MANAGE_MATCHES, PERMISSION_ENTER_SCORES),
    "match_uploader": (PERMISSION_ENTER_SCORES,),
    "moneyball": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "player_editor": (PERMISSION_MANAGE_PLAYERS,),
    "player_updates_admin": (PERMISSION_MANAGE_SUBSCRIPTIONS,),
    "theme_qa": (PERMISSION_RUN_REPLAY,),
    "top_players_printable": (PERMISSION_RUN_REPLAY,),
    "tournament_live": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "tournament_manager": (PERMISSION_MANAGE_TOURNAMENTS,),
    "tournament_ops": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "tournament_registration_admin": (PERMISSION_MANAGE_TOURNAMENTS,),
    "tournaments": (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
    "weekly_recap_admin": (PERMISSION_RUN_REPLAY,),
}


def is_admin_page_available_for_role(page_key: str, role: str, *, is_admin_only: bool = True) -> bool:
    normalized_key = str(page_key or "").strip().lower()
    required_permissions = ADMIN_PAGE_PERMISSION_MATRIX.get(normalized_key)
    if not required_permissions:
        return not is_admin_only
    return any(has_permission(role, permission) for permission in required_permissions)
