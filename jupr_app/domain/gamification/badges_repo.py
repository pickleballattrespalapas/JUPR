from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Iterable
from uuid import uuid4

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.copy_pack import get_badge_copy, pick_variant, render_template


logger = logging.getLogger(__name__)

PLAYER_BADGES_ON_CONFLICT = "club_id,player_id,badge_id,context_id"


def upsert_player_badges(
    supabase: Any,
    club_id: str,
    candidates: Iterable[BadgeCandidate],
) -> list[BadgeCandidate]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []

    existing = _fetch_existing_keys(supabase, club_id, candidate_list)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    created: list[BadgeCandidate] = []

    for candidate in candidate_list:
        context_id = candidate.context_id
        if context_id is None:
            context_id = str(candidate.badge_id)
        key = (int(candidate.player_id), str(candidate.badge_id), str(context_id))
        if key in existing:
            continue
        existing.add(key)
        value_json = dict(candidate.value_json or {})
        value_json.setdefault("badge_id", candidate.badge_id)
        if "tape_excerpt" not in value_json or not value_json.get("tape_excerpt"):
            value_json["tape_excerpt"] = _build_tape_excerpt(candidate, value_json)
        rows.append(
            {
                "id": str(uuid4()),
                "club_id": club_id,
                "player_id": int(candidate.player_id),
                "badge_id": candidate.badge_id,
                "earned_at": now,
                "context_type": candidate.context_type,
                "context_id": context_id,
                "match_id": candidate.match_id,
                "value_num": candidate.value_num,
                "value_json": value_json,
            }
        )
        created.append(candidate)

    if not rows:
        return []

    if logger.isEnabledFor(logging.INFO):
        sample_rows = [_redact_badge_row(row) for row in rows[:3]]
        logger.info(
            "player_badges upsert batch prepared",
            extra={
                "on_conflict": PLAYER_BADGES_ON_CONFLICT,
                "sample_rows": sample_rows,
                "total_rows": len(rows),
            },
        )

    chunk = 200
    for i in range(0, len(rows), chunk):
        supabase.table("player_badges").upsert(
            rows[i : i + chunk],
            on_conflict=PLAYER_BADGES_ON_CONFLICT,
        ).execute()
    return created


def _fetch_existing_keys(
    supabase: Any,
    club_id: str,
    candidates: list[BadgeCandidate],
) -> set[tuple[int, str, str]]:
    badge_ids = sorted({str(c.badge_id) for c in candidates})
    try:
        resp = (
            supabase.table("player_badges")
            .select("player_id,badge_id,context_id")
            .eq("club_id", club_id)
            .in_("badge_id", badge_ids)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch existing badge keys")
        return set()

    existing = set()
    for row in resp.data or []:
        try:
            player_id = int(row.get("player_id"))
        except Exception:
            continue
        badge_id = str(row.get("badge_id"))
        context_id = str(row.get("context_id") or badge_id)
        existing.add((player_id, badge_id, context_id))
    return existing


def _build_tape_excerpt(candidate: BadgeCandidate, data: dict[str, Any]) -> str:
    copy = get_badge_copy(candidate.badge_id)
    seed = f"{candidate.player_id}:{candidate.badge_id}:{candidate.context_id}"
    template = pick_variant(copy.get("tape_excerpts", []), seed)
    rendered = render_template(template, data | {"badge_name": copy.get("name", candidate.badge_id)})
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    if lines:
        return "\n".join(lines[:4])
    return ""


def _redact_badge_row(row: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(row)
    if "value_json" in redacted:
        value_json = redacted.get("value_json") or {}
        if isinstance(value_json, dict):
            redacted["value_json"] = {"keys": sorted(value_json.keys())}
    return redacted
