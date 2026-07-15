from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import is_admin_player_editor_enabled

CONFIRM_MERGE = "MERGE"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
PLAYER_COLUMNS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, *, field: str = "value") -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_league_rows(supabase: Any, *, club_id: str, player_id: int) -> list[dict[str, Any]]:
    return _safe_rows(
        supabase.table("league_ratings")
        .select("id,league_name,player_id")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .execute()
    )


def _match_reference_counts(supabase: Any, *, club_id: str, player_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in PLAYER_COLUMNS:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id")
            .eq("club_id", str(club_id))
            .eq(column, int(player_id))
            .execute()
        )
        counts[column] = len(rows)
    counts["total"] = sum(counts.values())
    return counts


def _social_identity_counts(supabase: Any, *, club_id: str, source_player_id: int, target_player_id: int) -> dict[str, int]:
    source_rows = _safe_rows(supabase.table("club_people").select("id").eq("club_id", str(club_id)).eq("linked_player_id", int(source_player_id)).execute())
    target_rows = _safe_rows(supabase.table("club_people").select("id").eq("club_id", str(club_id)).eq("linked_player_id", int(target_player_id)).execute())
    return {"source_linked": len(source_rows), "target_linked": len(target_rows)}


def build_admin_player_merge_preview(supabase: Any, *, club_id: str, source_player_id: Any, target_player_id: Any) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    src_id = _safe_int(source_player_id, field="source_player_id")
    dst_id = _safe_int(target_player_id, field="target_player_id")
    if src_id == dst_id:
        raise ValueError("source and target players must be different")
    source = _fetch_player(supabase, club_id=str(club_id), player_id=src_id)
    target = _fetch_player(supabase, club_id=str(club_id), player_id=dst_id)
    if not source:
        raise ValueError("source player not found")
    if not target:
        raise ValueError("target player not found")
    source_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=src_id)
    target_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=dst_id)
    target_names = {str(row.get("league_name") or "") for row in target_leagues}
    move_ids: list[int] = []
    delete_ids: list[int] = []
    conflicts: list[str] = []
    for row in source_leagues:
        league_name = str(row.get("league_name") or "")
        rid = _safe_int(row.get("id"), field="league_rating_id")
        if league_name in target_names:
            conflicts.append(league_name)
            delete_ids.append(rid)
        else:
            move_ids.append(rid)
    return {
        "ok": True,
        "mode": "player_merge_preview",
        "source_player": {"id": src_id, "name": _clean_text(source.get("name"), limit=160)},
        "target_player": {"id": dst_id, "name": _clean_text(target.get("name"), limit=160)},
        "match_reference_counts": _match_reference_counts(supabase, club_id=str(club_id), player_id=src_id),
        "league_rating_plan": {
            "source_rows": source_leagues,
            "target_rows": target_leagues,
            "move_ids": move_ids,
            "delete_ids": delete_ids,
            "conflicts": sorted(set(conflicts)),
        },
        "social_identity_counts": _social_identity_counts(supabase, club_id=str(club_id), source_player_id=src_id, target_player_id=dst_id),
        "warnings": ["After executing a merge, run Replay History ALL to rebuild rating history from the rewired match rows."],
    }


def execute_admin_player_merge(
    supabase: Any,
    *,
    club_id: str,
    source_player_id: Any,
    target_player_id: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_player_editor_merge",
) -> dict[str, Any]:
    if str(confirmation_text or "").strip().upper() != CONFIRM_MERGE:
        raise ValueError(f"Type {CONFIRM_MERGE} to merge player records.")
    preview = build_admin_player_merge_preview(supabase, club_id=str(club_id), source_player_id=source_player_id, target_player_id=target_player_id)
    src_id = int(preview["source_player"]["id"])
    dst_id = int(preview["target_player"]["id"])
    src_name = str(preview["source_player"].get("name") or f"#{src_id}")
    dst_name = str(preview["target_player"].get("name") or f"#{dst_id}")

    match_updates: dict[str, int] = {}
    for column in PLAYER_COLUMNS:
        rows = _safe_rows(
            supabase.table("matches")
            .update({column: dst_id})
            .eq("club_id", str(club_id))
            .eq(column, src_id)
            .execute()
        )
        match_updates[column] = len(rows)

    plan = preview["league_rating_plan"]
    deleted_league_rows: list[dict[str, Any]] = []
    for rid in plan.get("delete_ids") or []:
        deleted_league_rows.extend(_safe_rows(supabase.table("league_ratings").delete().eq("club_id", str(club_id)).eq("id", int(rid)).execute()))
    moved_league_rows: list[dict[str, Any]] = []
    for rid in plan.get("move_ids") or []:
        moved_league_rows.extend(_safe_rows(supabase.table("league_ratings").update({"player_id": dst_id}).eq("club_id", str(club_id)).eq("id", int(rid)).execute()))

    source_social_rows = _safe_rows(supabase.table("club_people").select("id").eq("club_id", str(club_id)).eq("linked_player_id", src_id).execute())
    target_social_rows = _safe_rows(supabase.table("club_people").select("id").eq("club_id", str(club_id)).eq("linked_player_id", dst_id).execute())
    social_patch = {"linked_player_id": None if target_social_rows else dst_id}
    social_rows = []
    for row in source_social_rows:
        social_rows.extend(_safe_rows(supabase.table("club_people").update(social_patch).eq("club_id", str(club_id)).eq("id", str(row.get("id"))).execute()))

    inactive_name = f"{src_name} (MERGED into {dst_name} #{dst_id})"
    source_player_rows = _safe_rows(
        supabase.table("players")
        .update({"active": False, "inactive_at": _now_iso(), "name": inactive_name[:160]})
        .eq("club_id", str(club_id))
        .eq("id", src_id)
        .execute()
    )

    after_preview = build_admin_player_merge_preview(supabase, club_id=str(club_id), source_player_id=src_id, target_player_id=dst_id)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="merge_player_editor_players_admin",
        entity_type="players",
        entity_id=f"{src_id}->{dst_id}",
        before_json=preview,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "source_player_id": src_id,
            "target_player_id": dst_id,
            "match_updates": match_updates,
            "moved_league_rating_count": len(moved_league_rows),
            "deleted_conflicting_league_rating_count": len(deleted_league_rows),
            "social_identity_rows_updated": len(social_rows),
            "source_player": source_player_rows[0] if source_player_rows else None,
            "post_merge_preview": after_preview,
            "requires_replay": True,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings = ["Merge completed. Run Replay History ALL before relying on derived ratings/history."]
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "player_merge_execute",
        "source_player_id": src_id,
        "target_player_id": dst_id,
        "match_updates": match_updates,
        "league_rating_plan": plan,
        "moved_league_rating_count": len(moved_league_rows),
        "deleted_conflicting_league_rating_count": len(deleted_league_rows),
        "social_identity_rows_updated": len(social_rows),
        "requires_replay": True,
        "warnings": warnings,
    }
