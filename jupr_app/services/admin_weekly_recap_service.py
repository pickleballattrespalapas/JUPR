from __future__ import annotations

import os
from copy import deepcopy
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.recaps.weekly_recap import (
    DEFAULT_SPOTLIGHT_DESCRIPTIONS,
    SPOTLIGHT_DEFAULT_ORDER,
    compute_weekly_recap,
    get_spotlight_candidates,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_GENERATE = "GENERATE RECAP"
CONFIRM_SAVE = "SAVE RECAP"
CONFIRM_PUBLISH = "PUBLISH RECAP"
CONFIRM_UNPUBLISH = "UNPUBLISH RECAP"
ADMIN_WEEKLY_RECAP_SELECT = "*"
MAX_RECAP_DAYS = 60


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_weekly_recap_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first_row(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _parse_date(value: Any, *, field: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    raw = _clean_text(value, limit=20)
    if not raw:
        raise ValueError(f"{field} is required")
    try:
        return date.fromisoformat(raw[:10])
    except Exception as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc


def _date_range(start: Any, end: Any) -> tuple[date, date]:
    start_date = _parse_date(start, field="week_start")
    end_date = _parse_date(end, field="week_end")
    if end_date < start_date:
        raise ValueError("week_end must be on or after week_start")
    if (end_date - start_date).days + 1 > MAX_RECAP_DAYS:
        raise ValueError(f"Weekly recap date range cannot exceed {MAX_RECAP_DAYS} days")
    return start_date, end_date


def _default_award(award_key: str, order: int) -> dict[str, Any]:
    return {
        "players": [],
        "description": DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(award_key, ""),
        "order": order,
        "include": True,
    }


def normalize_spotlight_overrides(overrides: dict[str, Any] | None, generated_spotlight: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    normalized = {key: _default_award(key, idx + 1) for idx, key in enumerate(SPOTLIGHT_DEFAULT_ORDER)}
    for idx, item in enumerate(generated_spotlight or []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key not in normalized:
            continue
        normalized[key] = {
            "players": list(item.get("candidate_ids") or item.get("players") or []),
            "description": item.get("description") or DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(str(key), ""),
            "order": int(item.get("order") or (idx + 1)),
            "include": bool(item.get("include", True)),
        }
    for key, value in (overrides or {}).items():
        if key not in normalized or not isinstance(value, dict):
            continue
        normalized[key]["players"] = list(value.get("players") or normalized[key]["players"])
        normalized[key]["description"] = _clean_text(value.get("description", normalized[key]["description"]), limit=1000)
        normalized[key]["order"] = int(value.get("order") or normalized[key]["order"])
        normalized[key]["include"] = bool(value.get("include", normalized[key]["include"]))
    return normalized


def apply_weekly_recap_edits(generated_json: dict[str, Any] | None, edits_json: dict[str, Any] | None, candidates: dict[str, list[dict[str, Any]]] | None) -> dict[str, Any]:
    recap = deepcopy(generated_json or {})
    if not recap:
        return recap
    edits = dict(edits_json or {})
    looking_ahead = edits.get("looking_ahead")
    if isinstance(looking_ahead, list):
        recap["looking_ahead"] = [_clean_text(item, limit=240) for item in looking_ahead if _clean_text(item, limit=240)]
    candidate_maps = {
        key: {item.get("candidate_id"): item for item in items if isinstance(item, dict)}
        for key, items in (candidates or {}).items()
    }
    generated_spotlight = recap.get("spotlight", []) or []
    overrides = normalize_spotlight_overrides(edits.get("spotlight_overrides", {}), generated_spotlight)
    updated = []
    for key, config in overrides.items():
        if not config.get("include", True):
            continue
        selected_ids = list(config.get("players") or [])[:3]
        selected_options = [candidate_maps.get(key, {}).get(candidate_id) for candidate_id in selected_ids]
        selected_options = [item for item in selected_options if item]
        if not selected_options:
            fallback = (candidates or {}).get(key) or []
            selected_options = [item for item in fallback[:3] if isinstance(item, dict)]
        if not selected_options:
            continue
        updated.append(
            {
                "key": key,
                "label": selected_options[0].get("label", key),
                "players": [item.get("display", "") for item in selected_options if item.get("display")],
                "candidate_ids": [item.get("candidate_id") for item in selected_options if item.get("candidate_id")],
                "description": config.get("description") or DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(key, ""),
                "order": int(config.get("order", 999)),
                "include": True,
            }
        )
    updated.sort(key=lambda item: int(item.get("order", 999)))
    recap["spotlight"] = updated
    return recap


def _row_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": row.get("id"),
        "club_id": row.get("club_id"),
        "week_start": str(row.get("week_start") or ""),
        "week_end": str(row.get("week_end") or ""),
        "status": str(row.get("status") or "draft"),
        "generated_json": row.get("generated_json") or {},
        "edits_json": row.get("edits_json") or {},
        "final_json": row.get("final_json") or {},
        "published_at": row.get("published_at"),
        "published_by": row.get("published_by"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _fetch_recap_row(supabase: Any, *, club_id: str, week_start: str) -> dict[str, Any] | None:
    return _first_row(
        supabase.table("weekly_recaps")
        .select(ADMIN_WEEKLY_RECAP_SELECT)
        .eq("club_id", str(club_id))
        .eq("week_start", str(week_start))
        .limit(1)
        .execute()
    )


def _upsert_recap_row(supabase: Any, *, club_id: str, week_start: str, payload: dict[str, Any]) -> dict[str, Any]:
    before = _fetch_recap_row(supabase, club_id=str(club_id), week_start=str(week_start))
    clean_payload = {k: v for k, v in payload.items() if v is not None or k in {"published_at", "published_by"}}
    if before:
        row = _first_row(
            supabase.table("weekly_recaps")
            .update({**clean_payload, "updated_at": _now_iso()})
            .eq("club_id", str(club_id))
            .eq("week_start", str(week_start))
            .execute()
        )
        return row or {**before, **clean_payload}
    insert_payload = {"id": str(uuid4()), "created_at": _now_iso(), "updated_at": _now_iso(), **clean_payload}
    row = _first_row(supabase.table("weekly_recaps").insert(insert_payload).execute())
    return row or insert_payload


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    week_start: str,
    before_json: dict[str, Any] | None = None,
    after_json: dict[str, Any] | None = None,
    source: str,
) -> list[str]:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=action_type,
        entity_type="weekly_recap",
        entity_id=str(week_start),
        before_json=before_json or {},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, **(after_json or {})},
        source_page=source,
        flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return [write.warning] if write.warning else []


def _recap_context(supabase: Any, *, club_id: str) -> SimpleNamespace:
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
    ) = load_data(supabase, str(club_id))
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
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
        public_mode=False,
    )


def _candidates_for_row(supabase: Any, *, club_id: str, week_start: str, week_end: str, tz_name: str) -> dict[str, list[dict[str, Any]]]:
    start_date, end_date = _date_range(week_start, week_end)
    ctx = _recap_context(supabase, club_id=str(club_id))
    return get_spotlight_candidates(ctx, start_date=start_date, end_date=end_date, include_tournaments=True, tz_name=tz_name)


def build_admin_weekly_recap_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "list_endpoint": None,
            "warnings": ["Next Weekly Recap Admin is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP on FastAPI."],
        }
    count = 0
    published_count = 0
    if supabase is not None:
        try:
            rows = _safe_rows(supabase.table("weekly_recaps").select("id,status").eq("club_id", str(club_id)).execute())
            count = len(rows)
            published_count = len([row for row in rows if str(row.get("status") or "") == "published"])
        except Exception:
            count = 0
            published_count = 0
    return {
        "enabled": True,
        "status": "ready_for_weekly_recap_admin",
        "list_endpoint": "/admin/clubs/{club_id}/weekly-recap/recaps",
        "generate_endpoint": "/admin/clubs/{club_id}/weekly-recap/generate",
        "recap_count": count,
        "published_count": published_count,
        "warnings": [],
    }


def list_admin_weekly_recaps(supabase: Any, *, club_id: str, limit: int = 50) -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        raise PermissionError("Next Weekly Recap Admin is disabled.")
    try:
        rows = _safe_rows(
            supabase.table("weekly_recaps")
            .select("id,club_id,week_start,week_end,status,published_at,published_by,created_at,updated_at")
            .eq("club_id", str(club_id))
            .order("week_start", desc=True)
            .limit(max(1, min(int(limit or 50), 200)))
            .execute()
        )
    except Exception:
        rows = _safe_rows(supabase.table("weekly_recaps").select("*").eq("club_id", str(club_id)).execute())
    rows.sort(key=lambda row: str(row.get("week_start") or ""), reverse=True)
    return {"ok": True, "mode": "weekly_recap_list", "recaps": [_row_payload(row) for row in rows], "count": len(rows)}


def get_admin_weekly_recap(supabase: Any, *, club_id: str, week_start: str, include_candidates: bool = True, tz_name: str = "America/Mazatlan") -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        raise PermissionError("Next Weekly Recap Admin is disabled.")
    row = _fetch_recap_row(supabase, club_id=str(club_id), week_start=str(week_start))
    if row is None:
        raise ValueError("weekly recap not found")
    candidates = {}
    if include_candidates:
        try:
            candidates = _candidates_for_row(supabase, club_id=str(club_id), week_start=str(row.get("week_start")), week_end=str(row.get("week_end")), tz_name=tz_name)
        except Exception:
            candidates = {}
    return {"ok": True, "mode": "weekly_recap_detail", "recap": _row_payload(row), "candidates": candidates}


def generate_admin_weekly_recap(
    supabase: Any,
    *,
    club_id: str,
    week_start: str,
    week_end: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    tz_name: str = "America/Mazatlan",
    source: str = "next_weekly_recap_generate",
) -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        raise PermissionError("Next Weekly Recap Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_GENERATE:
        raise ValueError(f"Type {CONFIRM_GENERATE} to generate a weekly recap draft.")
    start_date, end_date = _date_range(week_start, week_end)
    before = _fetch_recap_row(supabase, club_id=str(club_id), week_start=start_date.isoformat())
    ctx = _recap_context(supabase, club_id=str(club_id))
    recap = compute_weekly_recap(ctx, start_date=start_date, end_date=end_date, include_tournaments=True, tz_name=tz_name)
    payload = {
        "club_id": str(club_id),
        "week_start": start_date.isoformat(),
        "week_end": str(recap.get("end_date") or end_date.isoformat()),
        "status": "draft",
        "generated_json": recap,
        "edits_json": {},
        "final_json": recap,
        "published_at": None,
        "published_by": None,
    }
    row = _upsert_recap_row(supabase, club_id=str(club_id), week_start=start_date.isoformat(), payload=payload)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="generate_weekly_recap_admin",
        week_start=start_date.isoformat(),
        before_json={"recap": _row_payload(before)} if before else {},
        after_json={"recap": _row_payload(row)},
        source=source,
    )
    candidates = get_spotlight_candidates(ctx, start_date=start_date, end_date=end_date, include_tournaments=True, tz_name=tz_name)
    return {"ok": True, "mode": "weekly_recap_generate", "recap": _row_payload(row), "candidates": candidates, "warnings": warnings}


def save_admin_weekly_recap(
    supabase: Any,
    *,
    club_id: str,
    week_start: str,
    edits_json: dict[str, Any] | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    tz_name: str = "America/Mazatlan",
    source: str = "next_weekly_recap_save",
) -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        raise PermissionError("Next Weekly Recap Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SAVE:
        raise ValueError(f"Type {CONFIRM_SAVE} to save the weekly recap draft.")
    before = _fetch_recap_row(supabase, club_id=str(club_id), week_start=str(week_start))
    if before is None:
        raise ValueError("weekly recap not found")
    edits = dict(edits_json or {})
    candidates = _candidates_for_row(supabase, club_id=str(club_id), week_start=str(before.get("week_start")), week_end=str(before.get("week_end")), tz_name=tz_name)
    final_json = apply_weekly_recap_edits(before.get("generated_json") or {}, edits, candidates)
    payload = {
        "club_id": str(club_id),
        "week_start": str(before.get("week_start")),
        "week_end": str(before.get("week_end")),
        "status": "draft",
        "generated_json": before.get("generated_json") or {},
        "edits_json": edits,
        "final_json": final_json,
        "published_at": None,
        "published_by": None,
    }
    row = _upsert_recap_row(supabase, club_id=str(club_id), week_start=str(before.get("week_start")), payload=payload)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="save_weekly_recap_admin",
        week_start=str(before.get("week_start")),
        before_json={"recap": _row_payload(before)},
        after_json={"recap": _row_payload(row)},
        source=source,
    )
    return {"ok": True, "mode": "weekly_recap_save", "recap": _row_payload(row), "candidates": candidates, "warnings": warnings}


def publish_admin_weekly_recap(
    supabase: Any,
    *,
    club_id: str,
    week_start: str,
    action: str,
    edits_json: dict[str, Any] | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    tz_name: str = "America/Mazatlan",
    source: str = "next_weekly_recap_publish",
) -> dict[str, Any]:
    if not is_admin_weekly_recap_enabled():
        raise PermissionError("Next Weekly Recap Admin is disabled.")
    clean_action = _clean_text(action, limit=40).lower() or "publish"
    required = CONFIRM_UNPUBLISH if clean_action == "unpublish" else CONFIRM_PUBLISH
    if _clean_text(confirmation_text, limit=80).upper() != required:
        raise ValueError(f"Type {required} to {clean_action} the weekly recap.")
    before = _fetch_recap_row(supabase, club_id=str(club_id), week_start=str(week_start))
    if before is None:
        raise ValueError("weekly recap not found")
    edits = dict(edits_json if edits_json is not None else (before.get("edits_json") or {}))
    candidates = _candidates_for_row(supabase, club_id=str(club_id), week_start=str(before.get("week_start")), week_end=str(before.get("week_end")), tz_name=tz_name)
    final_json = apply_weekly_recap_edits(before.get("generated_json") or {}, edits, candidates)
    is_publish = clean_action != "unpublish"
    payload = {
        "club_id": str(club_id),
        "week_start": str(before.get("week_start")),
        "week_end": str(before.get("week_end")),
        "status": "published" if is_publish else "draft",
        "generated_json": before.get("generated_json") or {},
        "edits_json": edits,
        "final_json": final_json,
        "published_at": _now_iso() if is_publish else None,
        "published_by": str(actor_email or "") if is_publish else None,
    }
    row = _upsert_recap_row(supabase, club_id=str(club_id), week_start=str(before.get("week_start")), payload=payload)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="publish_weekly_recap_admin" if is_publish else "unpublish_weekly_recap_admin",
        week_start=str(before.get("week_start")),
        before_json={"recap": _row_payload(before)},
        after_json={"recap": _row_payload(row)},
        source=source,
    )
    return {"ok": True, "mode": "weekly_recap_publish" if is_publish else "weekly_recap_unpublish", "recap": _row_payload(row), "candidates": candidates, "warnings": warnings}
