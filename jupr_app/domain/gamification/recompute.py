from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badges_repo import upsert_player_badges


def run_badge_recompute(
    supabase: Any,
    *,
    club_id: str,
    mode: str = "append-only",
    league_id: str | None = None,
    context_id: str | None = None,
    player_id: int | None = None,
    badge_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    created_by: str | None = None,
    rule_version: str | None = None,
    revoke_reason: str | None = None,
    allow_strict_global: bool = False,
    match_limit: int = 5000,
    ctx: Any | None = None,
) -> dict[str, Any]:
    if mode not in {"dry-run", "append-only", "strict"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "strict" and not allow_strict_global:
        if not any([badge_id, player_id, context_id]):
            raise ValueError("Strict mode requires badge_id, player_id, or context_id unless allow_strict_global is set.")

    scope = _build_scope(
        club_id=club_id,
        league_id=league_id,
        context_id=context_id,
        player_id=player_id,
        badge_id=badge_id,
        since=since,
        until=until,
        mode=mode,
    )
    eval_run_id = _create_eval_run(supabase, created_by, mode, scope)
    _update_eval_run(supabase, eval_run_id, status="running", started_at=_now_iso())

    try:
        if ctx is None:
            ctx = _load_ctx(supabase, club_id, match_limit)
        ctx = _apply_match_filters(ctx, since=since, until=until)

        computed = list(
            compute_candidates_for_club(
                club_id=club_id,
                league_id=league_id,
                ctx=ctx,
            )
        )
        computed = _filter_candidates(
            computed,
            player_id=player_id,
            badge_id=badge_id,
            context_id=context_id,
        )

        existing_rows = _fetch_existing_awards(
            supabase,
            club_id=club_id,
            player_id=player_id,
            badge_id=badge_id,
            context_id=context_id,
        )
        existing_keys = {(row["player_id"], row["badge_id"], row["context_id"]) for row in existing_rows}
        computed_keys = {(c.player_id, str(c.badge_id), str(c.context_id)) for c in computed}

        rule_version = rule_version or _compute_rule_version()
        summary: dict[str, Any] = {
            "mode": mode,
            "scope": scope,
            "computed_count": len(computed_keys),
            "existing_count": len(existing_keys),
            "rule_version": rule_version,
        }

        if mode == "dry-run":
            summary["new_awards_count"] = len(computed_keys - existing_keys)
            summary["revoked_count"] = len(
                [row for row in existing_rows if row["key"] not in computed_keys and row.get("revoked_at") is None]
            )
            _update_eval_run(supabase, eval_run_id, status="completed", finished_at=_now_iso(), summary=summary)
            return summary

        created = upsert_player_badges(
            supabase,
            club_id,
            computed,
            awarded_by="recompute",
            rule_version=rule_version,
            eval_run_id=str(eval_run_id),
        )
        summary["new_awards_count"] = len(created)

        revoked_count = 0
        unrevoked_count = 0
        if mode == "strict":
            revoke_reason = revoke_reason or "strict recompute"
            for row in existing_rows:
                if row["key"] in computed_keys:
                    if row.get("revoked_at"):
                        _clear_revocation(supabase, row["id"])
                        unrevoked_count += 1
                    continue
                if row.get("revoked_at") is None:
                    _mark_revoked(supabase, row["id"], created_by, revoke_reason)
                    revoked_count += 1

        summary["revoked_count"] = revoked_count
        summary["unrevoked_count"] = unrevoked_count

        _update_eval_run(supabase, eval_run_id, status="completed", finished_at=_now_iso(), summary=summary)
        return summary
    except Exception as exc:
        _update_eval_run(
            supabase,
            eval_run_id,
            status="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )
        raise


def _build_scope(**kwargs: Any) -> dict[str, Any]:
    scope = {k: v for k, v in kwargs.items() if v is not None}
    return scope


def _load_ctx(supabase: Any, club_id: str, match_limit: int) -> Any:
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
    )


def _apply_match_filters(ctx: Any, *, since: str | None, until: str | None) -> Any:
    if not since and not until:
        return ctx
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return ctx
    df = df_matches.copy()
    df["date_dt"] = pd.to_datetime(df.get("date"), utc=True, errors="coerce")
    if since:
        since_dt = pd.to_datetime(since, utc=True, errors="coerce")
        df = df[df["date_dt"] >= since_dt]
    if until:
        until_dt = pd.to_datetime(until, utc=True, errors="coerce")
        df = df[df["date_dt"] <= until_dt]
    attrs = dict(vars(ctx)) if hasattr(ctx, "__dict__") else {}
    attrs["df_matches"] = df.drop(columns=["date_dt"])
    return SimpleNamespace(**attrs)


def _filter_candidates(
    candidates: Iterable[Any],
    *,
    player_id: int | None,
    badge_id: str | None,
    context_id: str | None,
) -> list[Any]:
    filtered = []
    for candidate in candidates:
        if player_id is not None and int(candidate.player_id) != int(player_id):
            continue
        if badge_id is not None and str(candidate.badge_id) != str(badge_id):
            continue
        if context_id is not None and str(candidate.context_id) != str(context_id):
            continue
        filtered.append(candidate)
    return filtered


def _fetch_existing_awards(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | None,
    badge_id: str | None,
    context_id: str | None,
) -> list[dict[str, Any]]:
    query = (
        supabase.table("player_badges")
        .select("id,player_id,badge_id,context_id,revoked_at")
        .eq("club_id", club_id)
    )
    if player_id is not None:
        query = query.eq("player_id", int(player_id))
    if badge_id is not None:
        query = query.eq("badge_id", str(badge_id))
    if context_id is not None:
        query = query.eq("context_id", str(context_id))
    resp = query.execute()
    rows = []
    for row in resp.data or []:
        key = (int(row.get("player_id")), str(row.get("badge_id")), str(row.get("context_id")))
        rows.append({**row, "key": key})
    return rows


def _mark_revoked(supabase: Any, row_id: str, revoked_by: str | None, revoke_reason: str) -> None:
    supabase.table("player_badges").update(
        {
            "revoked_at": _now_iso(),
            "revoked_by": revoked_by,
            "revoke_reason": revoke_reason,
        }
    ).eq("id", row_id).execute()


def _clear_revocation(supabase: Any, row_id: str) -> None:
    supabase.table("player_badges").update(
        {
            "revoked_at": None,
            "revoked_by": None,
            "revoke_reason": None,
        }
    ).eq("id", row_id).execute()


def _compute_rule_version() -> str:
    payload = [asdict(badge) for badge in BADGE_DEFINITIONS]
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()
    return f"badge_catalog:{digest}"


def _create_eval_run(supabase: Any, created_by: str | None, mode: str, scope: dict[str, Any]) -> str:
    resp = (
        supabase.table("badge_eval_runs")
        .insert(
            {
                "created_by": created_by,
                "mode": mode,
                "scope_json": scope,
                "status": "queued",
            }
        )
        .execute()
    )
    data = resp.data or []
    if not data:
        raise RuntimeError("Failed to create badge_eval_runs row.")
    return str(data[0].get("id"))


def _update_eval_run(
    supabase: Any,
    eval_run_id: str,
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {"status": status}
    if started_at is not None:
        payload["started_at"] = started_at
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if summary is not None:
        payload["summary_json"] = summary
    if error is not None:
        payload["error"] = error
    supabase.table("badge_eval_runs").update(payload).eq("id", eval_run_id).execute()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
