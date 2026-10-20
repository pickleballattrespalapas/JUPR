from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from jupr_app.domain.player_visibility import is_merged_player

INACTIVITY_DAYS = 14
INACTIVITY_THRESHOLD = timedelta(days=INACTIVITY_DAYS)
# Frame metadata preserves the database value behind the derived display flag.
# It must not become a player column exposed by API serialization.
PERSISTED_ACTIVE_BY_PLAYER_ID_ATTR = "persisted_active_by_player_id"


def coerce_utc_datetime(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def max_activity_time(current, candidate) -> datetime | None:
    cur_dt = coerce_utc_datetime(current)
    cand_dt = coerce_utc_datetime(candidate)
    if cur_dt is None:
        return cand_dt
    if cand_dt is None:
        return cur_dt
    return max(cur_dt, cand_dt)


def should_mark_inactive(
    last_game_at,
    created_at=None,
    *,
    now_utc: datetime | None = None,
    threshold: timedelta = INACTIVITY_THRESHOLD,
) -> bool:
    """Only a recorded game within the activity window establishes activity.

    Keep created_at in the signature for older callers; creating an account or
    registering for an event never establishes leaderboard activity.
    """
    now = now_utc or datetime.now(timezone.utc)
    baseline = coerce_utc_datetime(last_game_at)
    if baseline is None:
        return True
    return baseline > now or (now - baseline) >= threshold


def is_player_leaderboard_active(
    row: Mapping[str, Any], *, now_utc: datetime | None = None,
) -> bool:
    """Apply game recency as well as explicit exclusions at read time."""
    return (
        not is_merged_player(row)
        and row.get("active") is not False
        and row.get("is_active") is not False
        and not row.get("inactive_at")
        and not should_mark_inactive(row.get("last_game_at"), now_utc=now_utc)
    )


def add_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    data = df.copy()
    if "last_game_at" in data.columns:
        now = datetime.now(timezone.utc)
        data["active"] = [
            is_player_leaderboard_active(row, now_utc=now)
            for row in data.astype(object).where(pd.notna(data), None).to_dict("records")
        ]
    return data


def build_player_activity_update(
    existing_last_game_at, match_time, *, now_utc: datetime | None = None,
) -> dict:
    latest = max_activity_time(existing_last_game_at, match_time)
    if latest is None:
        return {}
    now = now_utc or datetime.now(timezone.utc)
    inactive = should_mark_inactive(latest, now_utc=now)
    return {
        "last_game_at": latest.isoformat(),
        "inactive_at": now.isoformat() if inactive else None,
        "active": not inactive,
    }


def build_recorded_game_activity_update(
    player: Mapping[str, Any], match_time, *, now_utc: datetime | None = None,
) -> dict:
    """Record scored play without undoing explicit visibility exclusions.

    The caller must supply the persisted active flag, not its display-time
    projection. An old automatic inactivity marker clears on a newer game;
    an exclusion after the latest game, or active=False without a marker,
    remains in force, as in the daily activity reconciler.
    """
    update = build_player_activity_update(player.get("last_game_at"), match_time, now_utc=now_utc)
    if not update:
        return update
    latest = coerce_utc_datetime(update["last_game_at"])
    exclusion = coerce_utc_datetime(player.get("inactive_at"))
    if (
        is_merged_player(player)
        or (player.get("active") is False and exclusion is None)
        or (exclusion is not None and exclusion > latest)
    ):
        update.update(active=player.get("active"), inactive_at=player.get("inactive_at"))
        if is_merged_player(player):
            update["active"] = False
    return update


def record_game_activity_with_cas(
    *, supabase, club_id: str, player_id: int, match_time, sb_retry,
) -> None:
    """Safely refresh activity for the legacy writer after its match insert.

    Unlike the atomic write-plan path, this caller cannot keep the pre-insert
    snapshot locked. Re-read and compare every activity/exclusion field so a
    concurrent game, administrator exclusion, or identity merge is preserved.
    """
    fields = ("last_game_at", "inactive_at", "active", "name")
    for _attempt in range(3):
        rows = sb_retry(
            lambda: supabase.table("players")
            .select("id,name,last_game_at,inactive_at,active")
            .eq("club_id", str(club_id)).eq("id", int(player_id))
            .limit(1).execute()
        ).data or []
        if len(rows) != 1:
            raise RuntimeError(f"Recorded-game activity requires an existing player {int(player_id)}.")
        current = rows[0]
        payload = build_recorded_game_activity_update(current, match_time)
        if not payload or all(current.get(key) == value for key, value in payload.items()):
            return

        def compare_and_swap():
            query = (supabase.table("players").update(payload)
                     .eq("club_id", str(club_id)).eq("id", int(player_id)))
            for field in fields:
                value = current.get(field)
                query = query.is_(field, "null") if value is None else query.eq(field, value)
            return query.execute()

        if sb_retry(compare_and_swap).data:
            return
    raise RuntimeError(
        f"Player {int(player_id)} changed during recorded-game activity refresh; "
        "the recorded match remains available for activity reconciliation."
    )


def recompute_last_game_at_for_players(
    *,
    supabase,
    club_id: str,
    player_ids: Iterable[int],
) -> None:
    """Recompute last_game_at for players after match deletions/voids."""
    for pid in {int(pid) for pid in player_ids if pid is not None}:
        resp = (
            supabase.table("matches")
            .select("date,score_t1,score_t2")
            .eq("club_id", str(club_id))
            .is_("deleted_at", None)
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .limit(50)
            .execute()
        )
        rows = resp.data or []
        df = pd.DataFrame(rows)
        if df.empty:
            latest = None
        else:
            df["score_t1"] = pd.to_numeric(df.get("score_t1", 0), errors="coerce").fillna(0).astype(int)
            df["score_t2"] = pd.to_numeric(df.get("score_t2", 0), errors="coerce").fillna(0).astype(int)
            df = df[(df["score_t1"] + df["score_t2"]) > 0].copy()
            if df.empty:
                latest = None
            else:
                df["date_dt"] = pd.to_datetime(df.get("date", None), errors="coerce", utc=True)
                latest = df["date_dt"].max()
                if pd.isna(latest):
                    latest = None
        now = datetime.now(timezone.utc)
        payload = build_player_activity_update(None, latest, now_utc=now) if latest is not None else {
            "last_game_at": None, "inactive_at": now.isoformat(), "active": False,
        }
        supabase.table("players").update(payload).eq("club_id", str(club_id)).eq("id", pid).execute()
