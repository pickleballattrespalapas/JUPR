from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

from datetime import datetime, timezone
import time
from types import SimpleNamespace
from typing import Any

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_player
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.domain.gamification.badge_queue import ack_badge_eval, dequeue_badge_eval
from jupr_app.domain.gamification import v3_engine


def process_badge_eval_queue(
    supabase: Any,
    *,
    max_jobs: int = 10,
    time_budget_seconds: int = 5,
    ctx: Any | None = None,
    match_limit: int = 5000,
) -> dict[str, int]:
    started = time.time()
    processed = 0
    errored = 0

    while processed < max_jobs and (time.time() - started) < time_budget_seconds:
        job = dequeue_badge_eval(supabase)
        if not job:
            break
        try:
            job_club_id = str(job.get("club_id") or "")
            context = _resolve_context(ctx, supabase, job_club_id, match_limit)
            event_type = str(job.get("event_type") or "")
            player_ids = [int(pid) for pid in (job.get("player_ids") or [])]
            context_id = str(job.get("context_id") or "overall")

            badge_ids = _badge_ids_for_trigger(context, event_type)
            if badge_ids and player_ids:
                _update_incremental_facts(supabase, job, player_ids, context_id)
                v3_badge_ids: set[str] = set()
                if v3_engine.USE_BADGE_ENGINE_V3:
                    v3_badge_ids = _v3_badge_ids_with_conditions(supabase, badge_ids)
                    for pid in player_ids:
                        v3_context = _context_with_context_id(context, context_id)
                        if v3_badge_ids:
                            v3_engine.evaluate_badges_v3(pid, v3_context, allowed_badge_ids=v3_badge_ids)

                v2_badge_ids = badge_ids - v3_badge_ids
                if v2_badge_ids:
                    candidates = []
                    for pid in player_ids:
                        candidates.extend(
                            [
                                c
                                for c in compute_candidates_for_player(job_club_id, pid, ctx=context)
                                if str(c.badge_id) in v2_badge_ids
                            ]
                        )
                    if candidates:
                        upsert_player_badges(
                            supabase,
                            job_club_id,
                            candidates,
                            awarded_by="engine",
                        )
            ack_badge_eval(supabase, job_id=str(job.get("id")), status="done")
            processed += 1
        except Exception as exc:  # noqa: BLE001 - worker should record failures
            ack_badge_eval(supabase, job_id=str(job.get("id")), status="error", error=str(exc))
            errored += 1

    return {"processed": processed, "errored": errored}


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
    ) = load_data(supabase, club_id, match_limit=match_limit)
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


def _context_with_context_id(ctx: Any, context_id: str) -> Any:
    attrs = dict(vars(ctx)) if hasattr(ctx, "__dict__") else {}
    attrs["context_id"] = context_id
    return SimpleNamespace(**attrs)


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
    return {
        badge_id
        for badge_id, triggers in triggers_by_id.items()
        if event_type in triggers
    }


def _normalize_triggers(value: Any) -> list[str]:
    if value is None:
        return ["match_recorded", "match_updated"]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [str(value)]
    return ["match_recorded", "match_updated"]


def _v3_badge_ids_with_conditions(supabase: Any, badge_ids: set[str]) -> set[str]:
    if supabase is None or not badge_ids:
        return set()

    badge_rows = supabase.table("badges").select("badge_id,status").in_("badge_id", list(badge_ids)).execute().data or []
    eligible_status_badges = {
        str(row.get("badge_id") or "")
        for row in badge_rows
        if row.get("badge_id") and row.get("status") is not None
    }
    if not eligible_status_badges:
        return set()

    condition_rows = (
        supabase.table("badge_rule_conditions")
        .select("badge_id")
        .in_("badge_id", list(eligible_status_badges))
        .execute()
        .data
        or []
    )
    return {
        str(row.get("badge_id") or "")
        for row in condition_rows
        if row.get("badge_id")
    }


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
    sb_upsert(
        supabase,
        "player_badge_facts",
        {
            "club_id": club_id,
            "player_id": int(player_id),
            "context_id": context_id,
            "fact_key": fact_key,
            "fact_value_num": new_value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        conflict="club_id,player_id,context_id,fact_key",
    )
