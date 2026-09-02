from __future__ import annotations

import os
from typing import Any

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.replay_history import FULL_RESET_LABEL
from jupr_app.services.replay_service import run_replay_with_job_tracking

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
            .select("league_name,k_factor,is_active,status,match_format")
            .eq("club_id", str(club_id))
            .execute()
        )
        return pd.DataFrame(rows), warnings
    except Exception as exc:
        warnings.append(f"Could not load leagues_metadata: {exc.__class__.__name__}")
        return pd.DataFrame(
            columns=[
                "league_name",
                "k_factor",
                "is_active",
                "status",
                "match_format",
            ]
        ), warnings


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


def _recent_replay_jobs(supabase: Any, *, club_id: str, limit: int = 20) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table("replay_jobs")
            .select(
                "id,target_reset,status,actor_email,source,created_at,started_at,"
                "finished_at,error_text,attempt_count,lease_expires_at"
            )
            .eq("club_id", str(club_id))
            .order("created_at", desc=True)
            .limit(max(1, min(int(limit), 50)))
            .execute()
        )
    except Exception as exc:
        return [], f"Replay job history is unavailable: {exc.__class__.__name__}"
    return [
        {
            "id": str(row.get("id") or ""),
            "target_reset": _clean_text(row.get("target_reset"), limit=160),
            "status": _clean_text(row.get("status") or "unknown", limit=40),
            "actor_email": _clean_text(row.get("actor_email"), limit=240),
            "source": _clean_text(row.get("source"), limit=120),
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "error_text": _clean_text(row.get("error_text"), limit=500),
            "attempt_count": int(row.get("attempt_count") or 0),
            "lease_expires_at": row.get("lease_expires_at"),
        }
        for row in rows
    ], None


def build_admin_replay_status(
    supabase: Any | None,
    *,
    club_id: str,
    include_recent_jobs: bool = False,
) -> dict[str, Any]:
    if not is_admin_replay_enabled():
        return {
            "enabled": False,
            "status": "streamlit_fallback",
            "apply_endpoint": None,
            "options": [FULL_RESET_LABEL],
            "default_target_reset": FULL_RESET_LABEL,
            "confirmation_text": "REPLAY",
            "warnings": ["Next replay is disabled. Use Streamlit Admin Tools until JUPR_ENABLE_NEXT_ADMIN_REPLAY is enabled for the pilot."],
            "recent_jobs": [],
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
    recent_jobs: list[dict[str, Any]] = []
    if include_recent_jobs:
        recent_jobs, jobs_warning = _recent_replay_jobs(
            supabase, club_id=str(club_id)
        )
        if jobs_warning:
            warnings.append(jobs_warning)
    return {
        "enabled": True,
        "status": "replay_enabled",
        "apply_endpoint": "/admin/clubs/{club_id}/replay-history",
        "options": [FULL_RESET_LABEL, *league_names],
        "default_target_reset": FULL_RESET_LABEL,
        "confirmation_text": "REPLAY",
        "warnings": warnings,
        "recent_jobs": recent_jobs,
        "safety_rules": _safety_rules(),
    }


def _safety_rules() -> list[str]:
    return [
        "Replay runs server-side through FastAPI and the Python replay_history domain function.",
        "Every replay creates a durable replay_jobs record; client retries reuse an idempotency key and workers hold expiring leases.",
        "League replay rewrites snapshots and reconciles match-backed ratings without removing roster-only membership.",
        "Full reset updates overall player aggregates and rebuilds replay-managed singles rows from their preserved legacy baseline.",
        "Replay requires Supabase JWT authorization with run_replay permission.",
        "Confirm the accessible Yes/No dialog; keep Streamlit fallback available during pilot validation.",
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
    idempotency_key: str | None = None,
    retry_failed: bool = False,
    worker_id: str | None = None,
    lease_seconds: int = 1800,
    replay_job_id: str | None = None,
) -> dict[str, Any]:
    if not is_admin_replay_enabled():
        raise PermissionError("Next replay is disabled.")
    if str(confirmation_text or "").strip().upper() != "REPLAY":
        raise ValueError("Type REPLAY to confirm replay.")
    target = _clean_text(target_reset or FULL_RESET_LABEL, limit=160) or FULL_RESET_LABEL
    df_meta, warnings = _fetch_league_metadata(supabase, club_id=str(club_id))
    before_json = {"target_reset": target, "source_client": "fastapi/nextjs", "source_page": source}
    tracked = run_replay_with_job_tracking(
        supabase=supabase,
        club_id=str(club_id),
        df_meta=df_meta,
        target_reset=target,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        idempotency_key=idempotency_key,
        source=source,
        retry_failed=retry_failed,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        replay_job_id=replay_job_id,
    )
    result = dict(tracked.get("result") or {})
    tracked_status = str(tracked.get("job_status"))
    job_succeeded = tracked_status == "succeeded"
    singles_replay_incomplete = (
        job_succeeded
        and target == FULL_RESET_LABEL
        and result.get("singles_replay_supported") is not True
    )
    if singles_replay_incomplete:
        incomplete_message = (
            "Full replay did not attest replay-managed singles recovery. "
            "Treat this job as incomplete and keep the prior recovery path available."
        )
        # The leased replay service validates this before its compare-and-set
        # finish. This branch is defensive for alternate/test implementations;
        # never rewrite a completed job without the lease that produced it.
        tracked_status = "invalid_result"
        job_succeeded = False
        warnings.append(incomplete_message)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="replay_history",
        entity_type="replay_history",
        entity_id=target,
        before_json=before_json,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "job_id": tracked.get("job_id"),
            "job_status": tracked_status,
            "idempotent_replay": bool(tracked.get("idempotent_replay")),
            "result": result,
        },
        note=f"Replay History from Next/FastAPI for {target}",
        source_page=source,
        flagged_for_review=True,
    )
    audit_result = write_admin_activity_log(supabase, audit_payload)
    if audit_result.warning:
        warnings.append(audit_result.warning)
    if not audit_result.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": job_succeeded and not singles_replay_incomplete,
        "mode": (
            "replay_incomplete"
            if singles_replay_incomplete
            else "replayed"
            if job_succeeded
            else "replay_in_progress"
        ),
        "target_reset": target,
        "job_id": tracked.get("job_id"),
        "job_status": tracked_status,
        "idempotent_replay": bool(tracked.get("idempotent_replay")),
        "result": result,
        "warnings": warnings,
    }
