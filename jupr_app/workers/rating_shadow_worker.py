from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any

from jupr_app.data.client import make_supabase
from jupr_app.domain.rating_model_comparison import (
    compare_jupr_with_bayesian_shadow,
)


_MATCH_COLUMNS = ",".join(
    (
        "id",
        "club_id",
        "date",
        "league",
        "deleted_at",
        "rating_scope",
        "match_format",
        "match_type",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
        "score_t1",
        "score_t2",
        "t1_p1_r",
        "t1_p2_r",
        "t2_p1_r",
        "t2_p2_r",
        "rating_bonus_elo",
    )
)


def _enabled(name: str) -> bool:
    return str(os.getenv(name) or "").strip().casefold() in {"1", "true", "yes"}


def _require_shadow_worker_enabled() -> None:
    if not _enabled("JUPR_ENABLE_RATING_SHADOW_WORKER"):
        raise ValueError("Rating shadow worker is disabled")
    environment = str(os.getenv("JUPR_ENV") or "").strip().casefold()
    if environment == "production" and not _enabled(
        "JUPR_ENABLE_PRODUCTION_RATING_SHADOW"
    ):
        raise ValueError(
            "Production rating shadow is disabled and requires separate approval"
        )


def _resolve_supabase_config() -> tuple[str, str]:
    url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url:
        raise ValueError("Missing required environment variable: SUPABASE_URL")
    if not service_role:
        raise ValueError(
            "Missing SUPABASE_SERVICE_ROLE_KEY. The shadow worker is server-only."
        )
    return url, service_role


def _fetch_matches(
    supabase: Any,
    *,
    club_id: str,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = (
            supabase.table("matches")
            .select(_MATCH_COLUMNS)
            .eq("club_id", club_id)
            .order("date")
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = [dict(row) for row in (response.data or [])]
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def _worker_summary(report: dict[str, Any], *, matches_read: int) -> dict[str, Any]:
    return {
        "ok": True,
        "purpose": "private_shadow_benchmark_only",
        "matches_read": matches_read,
        "selection": {
            "candidate_count": report["selection"]["candidate_count"],
            "selected_parameters": report["selection"]["selected_parameters"],
            "validation_window": report["selection"]["validation_window"],
        },
        "validation": report["validation"],
        "holdout": report["holdout"],
        "full_history": report["full_history"],
        "official_policy_checks": report["official_policy_checks"],
        "guardrails": report["guardrails"],
    }


def run_rating_shadow_worker(
    club_id: str,
    *,
    validation_start: str,
    validation_end: str,
    holdout_start: str,
    holdout_end: str | None = None,
    excluded_leagues: tuple[str, ...] = (),
) -> dict[str, Any]:
    _require_shadow_worker_enabled()
    url, service_role = _resolve_supabase_config()
    supabase = make_supabase(url, service_role)
    run_id: str | None = None

    created = (
        supabase.table("worker_run_log")
        .insert(
            {
                "worker_name": "rating_shadow_worker",
                "club_id": club_id,
                "status": "started",
                "summary_json": {},
            }
        )
        .execute()
        .data
        or []
    )
    run_id = str((created[0] or {}).get("id")) if created else None

    try:
        rows = _fetch_matches(supabase, club_id=club_id)
        excluded = {value.strip().casefold() for value in excluded_leagues if value.strip()}
        if excluded:
            rows = [
                row
                for row in rows
                if str(row.get("league") or "").strip().casefold() not in excluded
            ]
        report = compare_jupr_with_bayesian_shadow(
            rows,
            validation_start=validation_start,
            validation_end=validation_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
        )
        result = _worker_summary(report, matches_read=len(rows))
        result["excluded_leagues"] = sorted(excluded)
        if run_id:
            (
                supabase.table("worker_run_log")
                .update(
                    {
                        "status": "success",
                        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "summary_json": result,
                    }
                )
                .eq("id", run_id)
                .execute()
            )
        return result
    except Exception as exc:  # noqa: BLE001
        if run_id:
            try:
                (
                    supabase.table("worker_run_log")
                    .update(
                        {
                            "status": "failed",
                            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "error_text": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    .eq("id", run_id)
                    .execute()
                )
            except Exception:  # noqa: BLE001
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the private Bayesian rating shadow.")
    parser.add_argument("--club-id", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--holdout-start", required=True)
    parser.add_argument("--holdout-end")
    parser.add_argument("--exclude-league", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        result = run_rating_shadow_worker(
            args.club_id,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
            excluded_leagues=tuple(args.exclude_league),
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
