from __future__ import annotations

import os
from typing import Any

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_replay_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_REPLAY")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _fetch_league_metadata(supabase: Any, *, club_id: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select("league_name,k_factor,is_active,status")
            .eq("club_id", str(club_id))
            .execute()
        )
        return pd.DataFrame(rows), warnings
    except Exception as exc:
        warnings.append(f"Could not load leagues_metadata: {exc.__class__.__name__}")
        return pd.DataFrame(columns=["league_name", "k_factor", "is_active", "status"]), warnings


def _fallback_league_names(supabase: Any, *, club_id: str) -> list[str]:
    names: set[str] = set()
    for table_name, column in (("league_ratings", "league_name"), ("matches", "league")):
        try:
            rows = _safe_rows(supabase.table(table_name).select(column).eq("club_id", str(club_id)).execute())
        except Exception:
            continue
        for row in rows:
            value = _clean_text(row.get(column), limit=120)
            if value and value.upper() != "OVERALL" and value != "POPUP":
                names.add(value)
    return sorted(names)


def build_admin_replay_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_replay_enabled():
        return {
            "enabled": False,
            "status": "streamlit_fallback",
            "apply_endpoint": None,
            "options": [FULL_RESET_LABEL],
            "default_target_reset": FULL_RESET_LABEL,
            "confirmation_text": "REPLAY",
            "warnings": ["Next replay is disabled. Use Streamlit Admin Tools until JUPR_ENABLE_NEXT_ADMIN_REPLAY is enabled for the pilot."],
            "safety_rules": _safety_rules(),
        }

    df_meta, warnings = _fetch_league_metadata(supabase, club_id=str(club_id))
    league_names: list[str] = []
    if not df_meta.empty and "league_name" in df_meta.columns:
        league_names = sorted(
            {
                _clean_text(value, limit=120)
                for value in df_meta["league_name"].dropna().tolist()
                if _clean_text(value, limit=120) and _clean_text(value, limit=120).upper() != "OVERALL"
            }
        )
    if not league_names:
        league_names = _fallback_league_names(supabase, club_id=str(club_id))
    return {
        "enabled": True,
        "status": "replay_enabled",
        "apply_endpoint": "/admin/clubs/{club_id}/replay-history",
        "options": [FULL_RESET_LABEL, *league_names],
        "default_target_reset": FULL_RESET_LABEL,
        "confirmation_text": "REPLAY",
        "warnings": warnings,
        "safety_rules": _safety_rules(),
    }


def _safety_rules() -> list[str]:
    return [
        "Replay runs server-side through FastAPI and the Python replay_history domain function.",
        "League replay rewrites snapshots and rebuilds league_ratings for the selected league.",
        "Full reset also updates overall player ratings, wins, losses, and matches played.",
        "Replay requires Supabase JWT authorization with run_replay permission.",
        "Type REPLAY to confirm; keep Streamlit fallback available during pilot validation.",
    ]


def run_admin_replay_history(
    supabase: Any,
    *,
    club_id: str,
    target_reset: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_replay_history",
    confirmation_text: str = "",
) -> dict[str, Any]:
    if not is_admin_replay_enabled():
        raise PermissionError("Next replay is disabled.")
    if str(confirmation_text or "").strip().upper() != "REPLAY":
        raise ValueError("Type REPLAY to confirm replay.")
    target = _clean_text(target_reset or FULL_RESET_LABEL, limit=160) or FULL_RESET_LABEL
    df_meta, warnings = _fetch_league_metadata(supabase, club_id=str(club_id))
    before_json = {"target_reset": target, "source_client": "fastapi/nextjs", "source_page": source}
    result = replay_history(supabase=supabase, club_id=str(club_id), df_meta=df_meta, target_reset=target)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="replay_history",
        entity_type="replay_history",
        entity_id=target,
        before_json=before_json,
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "result": result},
        note=f"Replay History from Next/FastAPI for {target}",
        source_page=source,
        flagged_for_review=True,
    )
    audit_result = write_admin_activity_log(supabase, audit_payload)
    if audit_result.warning:
        warnings.append(audit_result.warning)
    if not audit_result.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "replayed", "target_reset": target, "result": result, "warnings": warnings}
