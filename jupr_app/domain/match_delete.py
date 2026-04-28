from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

from jupr_app.domain.player_activity import recompute_last_game_at_for_players
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log


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
            "actor": actor,
        }

    rows_resp = (
        supabase.table("matches")
        .select("id,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", str(club_id))
        .in_("id", normalized_ids)
        .execute()
    )
    before_rows = rows_resp.data or []
    existing_ids = sorted({int(r.get("id")) for r in before_rows if r.get("id") is not None})

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

    if existing_ids:
        now_iso = datetime.now(timezone.utc).isoformat()
        (
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
                after_json={"deleted_ids": existing_ids, "deleted_at": now_iso},
                note=note,
                source_page=source or "match_log",
                flagged_for_review=flagged_for_review,
            ),
        )
        if log_result.warning:
            warning = (f"{warning} " if warning else "") + log_result.warning

    if affected_player_ids:
        recompute_last_game_at_for_players(
            supabase=supabase,
            club_id=str(club_id),
            player_ids=affected_player_ids,
        )

    replay_result = None
    replay_error = None
    try:
        replay_result = replay_history(
            supabase=supabase,
            club_id=str(club_id),
            df_meta=df_meta,
            target_reset=FULL_RESET_LABEL,
            progress_cb=progress_cb,
        )
    except Exception as exc:  # noqa: BLE001
        replay_error = str(exc)

    return {
        "deleted_count": len(existing_ids),
        "deleted_ids": existing_ids,
        "affected_player_ids": sorted(affected_player_ids),
        "replay_result": replay_result,
        "warning": warning,
        "error": None,
        "replay_error": replay_error,
        "actor": actor,
    }
