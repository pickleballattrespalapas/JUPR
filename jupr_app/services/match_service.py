from __future__ import annotations

from typing import Any

from jupr_app.domain.match_processing import process_matches
from jupr_app.services.context import ServiceContext
from jupr_app.services.result_types import ServiceResult


def submit_match_batch(
    ctx: ServiceContext,
    matches: list[dict[str, Any]],
    *,
    name_to_id: dict[str, int],
    df_players_all,
    df_leagues,
    df_meta,
) -> ServiceResult:
    try:
        result = process_matches(
            matches,
            supabase=ctx.supabase,
            club_id=ctx.club_id,
            name_to_id=name_to_id,
            df_players_all=df_players_all,
            df_leagues=df_leagues,
            df_meta=df_meta,
        )
    except Exception as exc:
        return ServiceResult.failure(str(exc))

    return ServiceResult.success(data=result)
