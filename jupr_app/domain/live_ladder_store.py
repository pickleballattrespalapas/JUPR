from __future__ import annotations

from typing import Any

import pandas as pd


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    safe = df.where(pd.notnull(df), None)
    rows = safe.to_dict("records")
    return [_to_builtin(row) for row in rows]


def get_active_session(ctx, club_id: str, league_name: str, week: str) -> dict[str, Any] | None:
    rows = (
        ctx.supabase.table("live_ladder_sessions")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("league_name", str(league_name))
        .eq("week", str(week))
        .eq("active", True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0] if rows else None


def get_or_create_session(
    ctx,
    club_id: str,
    league_name: str,
    week: str,
    total_rounds: int,
    ladder_round_num: int,
    state: str,
    players_per_court: int,
    court_sizes: list[int],
    ordered_player_ids: list[int],
) -> dict[str, Any]:
    existing = get_active_session(ctx, club_id=club_id, league_name=league_name, week=week)
    if existing:
        update_session_snapshot(
            ctx,
            str(existing.get("id")),
            total_rounds=int(total_rounds),
            ladder_round_num=int(ladder_round_num),
            state=str(state),
            players_per_court=int(players_per_court),
            court_sizes=[int(x) for x in (court_sizes or [])],
            ordered_player_ids=[int(x) for x in (ordered_player_ids or [])],
            active=True,
        )
        refreshed = (
            ctx.supabase.table("live_ladder_sessions")
            .select("*")
            .eq("id", str(existing.get("id")))
            .limit(1)
            .execute()
            .data
            or []
        )
        return refreshed[0] if refreshed else existing

    payload = {
        "club_id": str(club_id),
        "league_name": str(league_name),
        "week": str(week),
        "total_rounds": int(total_rounds),
        "ladder_round_num": int(ladder_round_num),
        "state": str(state),
        "players_per_court": int(players_per_court),
        "court_sizes": [int(x) for x in (court_sizes or [])],
        "ordered_player_ids": [int(x) for x in (ordered_player_ids or [])],
        "active": True,
    }
    rows = ctx.supabase.table("live_ladder_sessions").insert(payload).execute().data or []
    if not rows:
        raise RuntimeError("Unable to create live ladder session")
    return rows[0]


def update_session_snapshot(ctx, session_id: str, **fields) -> None:
    payload = _to_builtin(fields)
    if "court_sizes" in payload:
        payload["court_sizes"] = [int(x) for x in (payload.get("court_sizes") or [])]
    if "ordered_player_ids" in payload:
        payload["ordered_player_ids"] = [int(x) for x in (payload.get("ordered_player_ids") or [])]
    if "ladder_round_num" in payload and payload["ladder_round_num"] is not None:
        payload["ladder_round_num"] = int(payload["ladder_round_num"])
    if "players_per_court" in payload and payload["players_per_court"] is not None:
        payload["players_per_court"] = int(payload["players_per_court"])
    ctx.supabase.table("live_ladder_sessions").update(payload).eq("id", str(session_id)).execute()


def upsert_round_submission(
    ctx,
    session_id: str,
    round_num: int,
    status: str,
    schedule_sig: str | None,
    schedule_json: list[dict[str, Any]] | pd.DataFrame | None,
    results_json: list[dict[str, Any]] | pd.DataFrame | None,
    round_stats_json: list[dict[str, Any]] | pd.DataFrame | None,
    movement_preview_json: list[dict[str, Any]] | pd.DataFrame | None,
    ordered_ids_before: list[int],
    court_sizes: list[int],
    roster_ids: list[int],
    ordered_ids_after: list[int] | None = None,
) -> None:
    schedule_payload = _df_to_records(schedule_json) if isinstance(schedule_json, pd.DataFrame) else _to_builtin(schedule_json or [])
    results_payload = _df_to_records(results_json) if isinstance(results_json, pd.DataFrame) else _to_builtin(results_json or [])
    stats_payload = _df_to_records(round_stats_json) if isinstance(round_stats_json, pd.DataFrame) else _to_builtin(round_stats_json or [])
    movement_payload = _df_to_records(movement_preview_json) if isinstance(movement_preview_json, pd.DataFrame) else _to_builtin(movement_preview_json or [])

    payload = {
        "session_id": str(session_id),
        "round_num": int(round_num),
        "status": str(status),
        "schedule_sig": str(schedule_sig) if schedule_sig else None,
        "schedule_json": schedule_payload,
        "results_json": results_payload,
        "round_stats_json": stats_payload,
        "movement_preview_json": movement_payload,
        "ordered_ids_before": [int(x) for x in (ordered_ids_before or [])],
        "ordered_ids_after": [int(x) for x in (ordered_ids_after or [])] if ordered_ids_after else None,
        "court_sizes": [int(x) for x in (court_sizes or [])],
        "roster_ids": [int(x) for x in (roster_ids or [])],
    }

    ctx.supabase.table("live_ladder_rounds").upsert(payload, on_conflict="session_id,round_num").execute()


def confirm_round(ctx, session_id: str, round_num: int, ordered_ids_after: list[int]) -> None:
    ctx.supabase.table("live_ladder_rounds").upsert(
        {
            "session_id": str(session_id),
            "round_num": int(round_num),
            "status": "CONFIRMED",
            "ordered_ids_after": [int(x) for x in (ordered_ids_after or [])],
        },
        on_conflict="session_id,round_num",
    ).execute()


def load_resume_payload(ctx, session_id: str) -> dict[str, Any]:
    rows = (
        ctx.supabase.table("live_ladder_sessions")
        .select("*")
        .eq("id", str(session_id))
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise ValueError(f"Live ladder session not found: {session_id}")
    session = rows[0]
    round_rows = (
        ctx.supabase.table("live_ladder_rounds")
        .select("*")
        .eq("session_id", str(session_id))
        .order("round_num", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    latest = round_rows[0] if round_rows else None

    default_order = [int(x) for x in (session.get("ordered_player_ids") or [])]
    latest_order = [int(x) for x in (latest.get("ordered_ids_after") or [])] if latest else []
    restored_order = latest_order or default_order

    payload: dict[str, Any] = {
        "session_id": str(session.get("id")),
        "ladder_state": str(session.get("state") or "SETUP"),
        "ladder_round_num": int(session.get("ladder_round_num") or 1),
        "ladder_total_rounds": int(session.get("total_rounds") or 1),
        "live_ladder": {
            "ordered_player_ids": restored_order,
            "court_sizes": [int(x) for x in (session.get("court_sizes") or [])],
            "players_per_court": int(session.get("players_per_court") or 4),
        },
    }

    if latest and latest.get("movement_preview_json"):
        payload["ladder_movement_preview"] = _to_builtin(latest.get("movement_preview_json") or [])
    if latest and latest_order:
        payload["movement_applied_round"] = int(latest.get("round_num") or 0)
    return payload
