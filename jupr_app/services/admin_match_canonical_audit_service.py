from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.match_canonical_audit import build_match_canonical_audit
from jupr_app.domain.match_canonical_migration import normalize_legacy_matches_for_canonical

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_APPLY = "APPLY NORMALIZE"


def is_admin_match_canonical_audit_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT", "").strip().lower() in TRUTHY


def build_admin_match_canonical_audit_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    enabled = is_admin_match_canonical_audit_enabled()
    return {
        "ok": True,
        "enabled": enabled,
        "status": "ready" if enabled and supabase is not None else "disabled",
        "club_id": str(club_id),
        "required_permissions": {"audit": "view_audit_log", "apply": "manage_matches"},
        "options_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/options" if enabled else None,
        "audit_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/run" if enabled else None,
        "normalize_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/normalize" if enabled else None,
        "confirmation_text": CONFIRM_APPLY,
    }


def _load_ctx(supabase: Any, *, club_id: str, match_limit: int = 5000) -> Any:
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, str(club_id), match_limit=int(match_limit))
    return SimpleNamespace(
        supabase=supabase,
        club_id=str(club_id),
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_matches=df_matches,
        df_meta=df_meta,
        df_badges=df_badges,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        public_mode=False,
        admin_logged_in=True,
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
    )


def build_admin_match_canonical_options(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=5000)
    players = []
    df_players = ctx.df_players_all
    if df_players is not None and not df_players.empty and "id" in df_players.columns:
        for _, row in df_players.sort_values("name" if "name" in df_players.columns else "id").iterrows():
            try:
                pid = int(row.get("id"))
            except Exception:
                continue
            players.append({"player_id": pid, "player_name": str(row.get("name") or f"Player {pid}")})
    leagues = []
    df_matches = ctx.df_matches
    if df_matches is not None and not df_matches.empty and "league" in df_matches.columns:
        leagues = sorted(
            str(item).strip()
            for item in df_matches["league"].dropna().astype(str).tolist()
            if str(item).strip()
        )
        leagues = sorted(set(leagues))
    return {"ok": True, "players": players, "leagues": leagues, "schema_degraded": bool(ctx.schema_degraded), "schema_degraded_reason": ctx.schema_degraded_reason}


def run_admin_match_canonical_audit(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    league_id: str | None = None,
    limit: int = 1200,
) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=max(int(limit or 1200), 5000))
    report = build_match_canonical_audit(
        ctx,
        club_id=str(club_id),
        player_id=int(player_id),
        league_id=str(league_id).strip() if league_id else None,
        limit=max(100, min(int(limit or 1200), 5000)),
    )
    return {"ok": True, "mode": "match_canonical_audit", "report": report, "schema_degraded": bool(ctx.schema_degraded), "schema_degraded_reason": ctx.schema_degraded_reason}


def run_admin_match_canonical_normalize(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    match_ids: list[int] | None,
    dry_run: bool,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_match_canonical_audit",
) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    if not bool(dry_run) and str(confirmation_text or "").strip().upper() != CONFIRM_APPLY:
        raise ValueError(f"Type {CONFIRM_APPLY} to apply canonical normalization updates.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=5000)
    result = normalize_legacy_matches_for_canonical(
        supabase,
        ctx=ctx,
        club_id=str(club_id),
        player_id=int(player_id),
        match_ids=[int(mid) for mid in (match_ids or [])],
        dry_run=bool(dry_run),
    )
    if not bool(dry_run):
        payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="match_canonical_normalize_admin",
            entity_type="matches",
            entity_id="canonical_normalize",
            before_json={"player_id": int(player_id), "match_ids": match_ids or []},
            after_json={
                "source_client": "fastapi/nextjs",
                "source_page": source,
                "applied_update_count": int(result.get("applied_update_count") or 0),
                "applied_match_ids": result.get("applied_match_ids") or [],
                "proposed_update_count": int(result.get("proposed_update_count") or 0),
                "manual_review_count": len(result.get("manual_review_needed") or []),
            },
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, payload)
    return {"ok": True, "mode": "match_canonical_normalize", "result": result}
