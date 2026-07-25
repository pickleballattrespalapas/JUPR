from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

from jupr_app.domain.player_activity import (
    coerce_utc_datetime,
    recompute_last_game_at_for_players,
)
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log


def _valid_singles_replay_baseline(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not {"rating", "wins", "losses", "matches_played"}.issubset(value):
        return False
    try:
        raw_rating = value["rating"]
        raw_counts = [value[key] for key in ("wins", "losses", "matches_played")]
        rating = float(raw_rating)
        counts = [float(raw) for raw in raw_counts]
    except (TypeError, ValueError):
        return False
    if (
        isinstance(raw_rating, bool)
        or any(isinstance(raw, bool) for raw in raw_counts)
        or not math.isfinite(rating)
        or any(not math.isfinite(count) for count in counts)
        or min(counts) < 0
        or any(not count.is_integer() for count in counts)
    ):
        return False
    last_game_at = value.get("last_game_at")
    return (
        last_game_at in (None, "")
        or coerce_utc_datetime(last_game_at) is not None
    )


def delete_rated_matches_with_replay(
    *,
    supabase,
    club_id: str,
    match_ids: Iterable[int],
    df_meta: Optional[pd.DataFrame],
    progress_cb: Optional[Callable[[float], None]] = None,
    actor: Optional[str] = None,
    actor_role: Optional[str] = None,
    source: Optional[str] = None,
    note: Optional[str] = None,
    flagged_for_review: bool = False,
) -> Dict[str, Any]:
    """Soft-delete rated matches, recompute player activity, then run FULL replay."""
    normalized_ids = sorted({int(mid) for mid in (match_ids or []) if mid is not None})
    if not normalized_ids:
        return {
            "deleted_count": 0,
            "deleted_ids": [],
            "affected_player_ids": [],
            "replay_result": None,
            "warning": "No match IDs supplied for delete.",
            "error": None,
            "replay_error": None,
            "recovery_required": False,
            "actor": actor,
        }

    try:
        rows_resp = (
            supabase.table("matches")
            .select(
                "id,match_type,match_format,rating_scope,"
                "t1_p1,t1_p2,t2_p1,t2_p2,singles_replay_managed"
            )
            .eq("club_id", str(club_id))
            .in_("id", normalized_ids)
            .execute()
        )
    except Exception as exc:
        raise ValueError(
            "Match recovery metadata is unavailable; no match was excluded."
        ) from exc
    before_rows = rows_resp.data or []
    existing_ids = sorted({int(r.get("id")) for r in before_rows if r.get("id") is not None})

    singles_rows = [
        row
        for row in before_rows
        if (
            str(row.get("match_format") or "").strip().lower() == "singles"
            or (
                row.get("t1_p2") is None
                and row.get("t2_p2") is None
                and "singles"
                in str(row.get("match_type") or "").strip().lower()
            )
        )
    ]
    if singles_rows:
        unmanaged_ids = [
            int(row["id"])
            for row in singles_rows
            if row.get("id") is not None
            and row.get("singles_replay_managed") is not True
        ]
        if unmanaged_ids:
            raise ValueError(
                "Legacy singles rows are not covered by deterministic replay and "
                f"cannot be excluded here: {unmanaged_ids[:10]}"
            )
        singles_player_ids = sorted(
            {
                int(value)
                for row in singles_rows
                for value in (row.get("t1_p1"), row.get("t2_p1"))
                if value is not None
            }
        )
        try:
            baseline_rows = (
                supabase.table("players")
                .select("id,singles_replay_baseline")
                .eq("club_id", str(club_id))
                .in_("id", singles_player_ids)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            raise ValueError(
                "Managed singles replay baseline is unavailable; no match was excluded."
            ) from exc
        baseline_ids = {
            int(row.get("id"))
            for row in baseline_rows
            if row.get("id") is not None
            and _valid_singles_replay_baseline(
                row.get("singles_replay_baseline")
            )
        }
        if baseline_ids != set(singles_player_ids):
            raise ValueError(
                "Managed singles replay baseline is incomplete; no match was excluded."
            )

    affected_player_ids: set[int] = set()
    for row in before_rows:
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            val = row.get(col)
            if val is None:
                continue
            try:
                affected_player_ids.add(int(val))
            except Exception:
                continue

    warning: str | None = None
    missing_ids = sorted(set(normalized_ids) - set(existing_ids))
    if missing_ids:
        warning = f"Some match IDs were not found and could not be deleted: {missing_ids[:10]}"

    recovery_errors: list[str] = []
    deleted_ids: list[int] = []
    if existing_ids:
        now_iso = datetime.now(timezone.utc).isoformat()
        update_result = (
            supabase.table("matches")
            .update(
                {
                    "deleted_at": now_iso,
                    "deleted_by": (actor or "admin"),
                    "deleted_source": (source or actor or "match_log"),
                    "delete_note": (note.strip() if isinstance(note, str) and note.strip() else None),
                    "updated_at": now_iso,
                    "updated_by": (actor or "admin"),
                }
            )
            .eq("club_id", str(club_id))
            .in_("id", existing_ids)
            .execute()
        )
        deleted_ids = sorted(
            {
                int(row.get("id"))
                for row in (getattr(update_result, "data", None) or [])
                if row.get("id") is not None
            }
        )
        if deleted_ids != existing_ids:
            recovery_errors.append(
                "Match exclusion did not attest every selected mutation "
                f"(expected {existing_ids}, updated {deleted_ids})."
            )
        log_result = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=actor or "admin",
                actor_role=actor_role or "",
                action_type="match_delete",
                entity_type="match",
                entity_id=",".join(str(mid) for mid in existing_ids),
                before_json=before_rows,
                after_json={
                    "requested_ids": existing_ids,
                    "deleted_ids": deleted_ids,
                    "deleted_at": now_iso,
                    "recovery_required": deleted_ids != existing_ids,
                },
                note=note,
                source_page=source or "match_log",
                flagged_for_review=flagged_for_review,
            ),
        )
        if log_result.warning:
            warning = (f"{warning} " if warning else "") + log_result.warning

    if affected_player_ids:
        try:
            recompute_last_game_at_for_players(
                supabase=supabase,
                club_id=str(club_id),
                player_ids=affected_player_ids,
            )
        except Exception as exc:  # noqa: BLE001
            recovery_errors.append(
                f"Player activity recovery failed after match exclusion: {exc}"
            )

    replay_result = None
    try:
        replay_result = replay_history(
            supabase=supabase,
            club_id=str(club_id),
            df_meta=df_meta,
            target_reset=FULL_RESET_LABEL,
            progress_cb=progress_cb,
        )
    except Exception as exc:  # noqa: BLE001
        recovery_errors.append(f"Rating replay failed after match exclusion: {exc}")
    if singles_rows and (
        recovery_errors
        or not isinstance(replay_result, dict)
        or replay_result.get("singles_replay_supported") is not True
    ):
        if not recovery_errors:
            recovery_errors.append(
                "Managed singles replay did not attest the required recovery projection."
            )

    replay_error = "; ".join(recovery_errors) or None

    return {
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids,
        "affected_player_ids": sorted(affected_player_ids),
        "replay_result": replay_result,
        "warning": warning,
        "error": None,
        "replay_error": replay_error,
        "recovery_required": bool(recovery_errors),
        "actor": actor,
    }
