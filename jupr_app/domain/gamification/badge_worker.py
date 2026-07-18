from __future__ import annotations

from datetime import datetime, timezone
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pandas as pd
from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_player
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.domain.gamification.badge_queue import ack_badge_eval, dequeue_badge_eval
from jupr_app.domain.player_activity import add_activity_columns


def process_badge_eval_queue(
    supabase: Any,
    club_id: str,
    *,
    max_jobs: int = 10,
    time_budget_seconds: int = 5,
    ctx: Any | None = None,
    match_limit: int = 5000,
) -> dict[str, int]:
    clean_club_id = str(club_id or "").strip()
    if not clean_club_id:
        raise ValueError("club_id is required to process the badge evaluation queue")
    deadline = time.monotonic() + float(time_budget_seconds)
    processed = 0
    errored = 0

    while (processed + errored) < max_jobs:
        if time.monotonic() >= deadline:
            break
        job = dequeue_badge_eval(supabase, club_id=clean_club_id)
        if not job:
            break
        try:
            _process_job_with_retry(supabase, job, ctx=ctx, match_limit=match_limit)
            ack_badge_eval(supabase, job_id=str(job.get("id")), status="done")
            processed += 1
        except Exception as exc:  # noqa: BLE001 - worker should record failures
            error = f"{type(exc).__name__}: {exc}"
            details = f"{error}\n{traceback.format_exc()}".strip()
            ack_badge_eval(
                supabase,
                job_id=str(job.get("id")),
                status="error",
                error=details[:2000],
            )
            errored += 1
        if time.monotonic() >= deadline:
            break

    return {"processed": processed, "errored": errored}


def process_badge_eval_queue_until_empty(
    supabase: Any,
    club_id: str,
    *,
    max_total_jobs: int = 500,
    batch_max_jobs: int = 10,
    per_batch_time_budget_seconds: float = 2.0,
    max_wall_clock_seconds: float = 90.0,
    max_errors: int = 10,
    progress_cb: Callable[[dict[str, int | float | str]], None] | None = None,
) -> dict[str, int | float | str]:
    started = time.monotonic()
    drain_deadline = started + float(max_wall_clock_seconds)
    loops = 0
    total_processed = 0
    total_errored = 0
    stopped_reason = "max_wall_clock"
    error_only_loops = 0

    while True:
        now = time.monotonic()
        remaining = drain_deadline - now
        if remaining <= 0:
            break

        loops += 1
        batch_budget = min(float(per_batch_time_budget_seconds), max(0.1, remaining))
        batch = process_badge_eval_queue(
            supabase,
            club_id,
            max_jobs=batch_max_jobs,
            time_budget_seconds=batch_budget,
        )
        processed = int(batch.get("processed") or 0)
        errored = int(batch.get("errored") or 0)
        total_processed += processed
        total_errored += errored

        if errored > 0 and processed == 0:
            error_only_loops += 1
        else:
            error_only_loops = 0

        if progress_cb is not None:
            progress_cb(
                {
                    "loop": loops,
                    "processed": processed,
                    "errored": errored,
                    "total_processed": total_processed,
                    "total_errored": total_errored,
                    "duration_seconds": time.monotonic() - started,
                }
            )

        if processed == 0 and errored == 0:
            stopped_reason = "empty"
            break
        if (total_processed + total_errored) >= max_total_jobs:
            stopped_reason = "max_total_jobs"
            break
        if total_errored >= max_errors:
            stopped_reason = "max_errors"
            break
        if error_only_loops >= 3:
            stopped_reason = "error_circuit_breaker"
            break

    return {
        "total_processed": total_processed,
        "total_errored": total_errored,
        "loops": loops,
        "stopped_reason": stopped_reason,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _process_job_with_retry(
    supabase: Any,
    job: dict[str, Any],
    *,
    ctx: Any | None,
    match_limit: int,
) -> None:
    retry_delays = [0.5, 1.0]
    for attempt in range(len(retry_delays) + 1):
        try:
            _process_job(supabase, job, ctx=ctx, match_limit=match_limit)
            return
        except Exception as exc:  # noqa: BLE001 - worker retries transient read errors
            if attempt >= len(retry_delays) or not _is_transient_read_error(exc):
                raise
            time.sleep(retry_delays[attempt])


def _is_transient_read_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.ReadError):
        return True
    exc_type = type(exc)
    return exc_type.__name__ == "ReadError" and "http" in exc_type.__module__.lower()


def _process_job(
    supabase: Any,
    job: dict[str, Any],
    *,
    ctx: Any | None,
    match_limit: int,
) -> None:
    job_club_id = str(job.get("club_id") or "")
    context = _resolve_context(ctx, supabase, job_club_id, match_limit)
    event_type = str(job.get("event_type") or "")
    player_ids = [int(pid) for pid in (job.get("player_ids") or [])]
    context_id = str(job.get("context_id") or "overall")

    badge_ids = _badge_ids_for_trigger(context, event_type)
    if badge_ids and player_ids:
        _update_incremental_facts(supabase, job, player_ids, context_id)
        candidates = []
        for pid in player_ids:
            candidates.extend(
                [
                    c
                    for c in compute_candidates_for_player(job_club_id, pid, ctx=context)
                    if str(c.badge_id) in badge_ids
                ]
            )
        if candidates:
            upsert_player_badges(
                supabase,
                job_club_id,
                candidates,
                awarded_by="engine",
            )


def _resolve_context(ctx: Any | None, supabase: Any, club_id: str, match_limit: int) -> Any:
    if ctx is not None and str(getattr(ctx, "club_id", "") or "") == str(club_id):
        return ctx
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
    ) = load_live_badge_data(supabase, club_id, match_limit=match_limit)
    return SimpleNamespace(
        supabase=supabase,
        club_id=club_id,
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


def load_live_badge_data(supabase: Any, club_id: str, *, match_limit: int = 5000) -> tuple[Any, ...]:
    club_id = str(club_id)
    schema_degraded = False
    schema_degraded_reason = None

    p_resp = supabase.table("players").select("*").eq("club_id", club_id).execute()
    df_players_all = add_activity_columns(pd.DataFrame(p_resp.data or []))
    if not df_players_all.empty and "inactive_at" in df_players_all.columns:
        df_players_active = df_players_all[df_players_all["inactive_at"].isna()].copy()
    elif not df_players_all.empty and "active" in df_players_all.columns:
        df_players_active = df_players_all[df_players_all["active"] == True].copy()
    else:
        df_players_active = df_players_all.copy()

    l_resp = (
        supabase.table("league_ratings")
        .select("id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active")
        .eq("club_id", club_id)
        .execute()
    )
    df_leagues = pd.DataFrame(l_resp.data or [])

    m_resp = (
        supabase.table("matches")
        .select("*")
        .eq("club_id", club_id)
        .order("id", desc=True)
        .limit(int(match_limit))
        .execute()
    )
    df_matches = pd.DataFrame(m_resp.data or [])

    meta_resp = supabase.table("leagues_metadata").select("*").eq("club_id", club_id).execute()
    df_meta = pd.DataFrame(meta_resp.data or [])

    df_badges = _fetch_badges(supabase)
    df_player_badges, schema_degraded, schema_degraded_reason = _fetch_player_badges_live(supabase, club_id)

    if not df_players_all.empty and "id" in df_players_all.columns and "name" in df_players_all.columns:
        ids = pd.to_numeric(df_players_all["id"], errors="coerce").dropna().astype(int)
        names = df_players_all.loc[ids.index, "name"].astype(str)
        id_to_name = dict(zip(ids, names))
        name_to_id = dict(zip(names, ids))
    else:
        id_to_name, name_to_id = {}, {}

    return (
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
    )


def _fetch_badges(supabase: Any) -> pd.DataFrame:
    try:
        resp = supabase.table("badges").select(
            "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
            "tier,icon_key,scope,state,eval_triggers,created_at"
        ).execute()
    except APIError:
        resp = supabase.table("badges").select(
            "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
            "tier,icon_key,scope,created_at"
        ).execute()
    df_badges = pd.DataFrame(resp.data or [])
    if not df_badges.empty:
        if "state" not in df_badges.columns:
            df_badges["state"] = "live"
        if "eval_triggers" not in df_badges.columns:
            df_badges["eval_triggers"] = [["match_recorded", "match_updated"]] * len(df_badges)
    return df_badges


def _fetch_player_badges_live(supabase: Any, club_id: str) -> tuple[pd.DataFrame, bool, str | None]:
    base_cols = [
        "id",
        "club_id",
        "player_id",
        "badge_id",
        "earned_at",
        "context_type",
        "context_id",
        "match_id",
        "value_num",
        "value_json",
    ]
    optional_cols = ["awarded_by", "rule_version", "eval_run_id", "revoked_at", "revoked_by", "revoke_reason"]
    schema_degraded = False
    schema_degraded_reason = None
    try:
        resp = (
            supabase.table("player_badges")
            .select(",".join(base_cols + optional_cols))
            .eq("club_id", club_id)
            .execute()
        )
    except APIError:
        schema_degraded = True
        schema_degraded_reason = "player_badges optional provenance/revocation columns missing"
        resp = (
            supabase.table("player_badges")
            .select(",".join(base_cols))
            .eq("club_id", club_id)
            .execute()
        )
    df = pd.DataFrame(resp.data or [])
    for col in optional_cols:
        if col not in df.columns:
            df[col] = None
    return df, schema_degraded, schema_degraded_reason


def _badge_ids_for_trigger(ctx: Any, event_type: str) -> set[str]:
    df_badges = getattr(ctx, "df_badges", None)
    triggers_by_id: dict[str, list[str]] = {}
    if isinstance(df_badges, pd.DataFrame) and not df_badges.empty and "badge_id" in df_badges.columns:
        for row in df_badges.itertuples(index=False):
            badge_id = str(getattr(row, "badge_id", "") or "")
            triggers = getattr(row, "eval_triggers", None)
            triggers_by_id[badge_id] = _normalize_triggers(triggers)
    else:
        for badge in BADGE_DEFINITIONS:
            triggers_by_id[badge.badge_id] = list(badge.eval_triggers)
    return {badge_id for badge_id, triggers in triggers_by_id.items() if event_type in triggers}


def _normalize_triggers(value: Any) -> list[str]:
    if value is None:
        return ["match_recorded", "match_updated"]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [str(value)]
    return ["match_recorded", "match_updated"]


def _update_incremental_facts(
    supabase: Any,
    job: dict[str, Any],
    player_ids: list[int],
    context_id: str,
) -> None:
    if supabase is None:
        return
    event_type = str(job.get("event_type") or "")
    if event_type not in {"match_recorded", "match_updated"}:
        return
    for pid in player_ids:
        _increment_fact(supabase, job, pid, context_id, "matches_seen", 1)


def _increment_fact(
    supabase: Any,
    job: dict[str, Any],
    player_id: int,
    context_id: str,
    fact_key: str,
    delta: int,
) -> None:
    club_id = str(job.get("club_id") or "")
    resp = (
        supabase.table("player_badge_facts")
        .select("fact_value_num")
        .eq("club_id", club_id)
        .eq("player_id", int(player_id))
        .eq("context_id", context_id)
        .eq("fact_key", fact_key)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    current = 0
    if rows:
        try:
            current = int(rows[0].get("fact_value_num") or 0)
        except Exception:
            current = 0
    new_value = current + int(delta)
    supabase.table("player_badge_facts").upsert(
        {
            "club_id": club_id,
            "player_id": int(player_id),
            "context_id": context_id,
            "fact_key": fact_key,
            "fact_value_num": new_value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="club_id,player_id,context_id,fact_key",
    ).execute()
