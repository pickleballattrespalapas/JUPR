from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
from typing import Any, Iterable
from uuid import uuid4

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.copy_pack import get_badge_copy, pick_variant, render_template


logger = logging.getLogger(__name__)
PLAYER_BADGES_CONFLICT_KEY = "club_id,player_id,badge_id,context_id"
_PLAYER_BADGES_CONTRACT_CHECKED = False


def ensure_player_badges_contract(supabase: Any) -> bool:
    expected_columns = ["club_id", "player_id", "badge_id", "context_id"]
    try:
        resp = (
            supabase.table("pg_indexes")
            .select("schemaname,tablename,indexname,indexdef")
            .eq("schemaname", "public")
            .eq("tablename", "player_badges")
            .execute()
        )
    except Exception:
        logger.exception("Failed to verify player_badges unique index; skipping contract check")
        return False

    rows = resp.data or []
    if _has_unique_index_on_columns(rows, expected_columns):
        return True

    logger.error(
        "player_badges unique contract missing or mismatched. Expected unique index on (%s).",
        ", ".join(expected_columns),
    )
    if _should_raise_on_schema_mismatch():
        raise RuntimeError("player_badges unique index contract missing")
    return False


def _has_unique_index_on_columns(rows: Iterable[dict[str, Any]], expected: list[str]) -> bool:
    expected_normalized = [col.strip().lower() for col in expected]
    for row in rows:
        indexdef = str(row.get("indexdef") or "")
        if "UNIQUE" not in indexdef.upper():
            continue
        columns = _extract_index_columns(indexdef)
        if columns == expected_normalized:
            return True
    return False


def _extract_index_columns(indexdef: str) -> list[str]:
    match = re.search(r"\(([^)]+)\)", indexdef)
    if not match:
        return []
    columns = []
    for col in match.group(1).split(","):
        col = col.strip().strip('"')
        if col:
            columns.append(col.lower())
    return columns


def _should_raise_on_schema_mismatch() -> bool:
    env = os.getenv("JUPR_ENV", "").lower()
    if env in {"dev", "development", "local"}:
        return True
    return os.getenv("BADGE_DEBUG") == "1" or os.getenv("JUPR_ADMIN") == "1"


def upsert_player_badges(
    supabase: Any,
    club_id: str,
    candidates: Iterable[BadgeCandidate],
    *,
    awarded_by: str = "engine",
    rule_version: str | None = None,
    eval_run_id: str | None = None,
) -> list[BadgeCandidate]:
    global _PLAYER_BADGES_CONTRACT_CHECKED
    candidate_list = list(candidates)
    if not candidate_list:
        return []

    if not _PLAYER_BADGES_CONTRACT_CHECKED:
        ensure_player_badges_contract(supabase)
        _PLAYER_BADGES_CONTRACT_CHECKED = True

    existing = _fetch_existing_keys(supabase, club_id, candidate_list)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    created: list[BadgeCandidate] = []

    for candidate in candidate_list:
        if candidate.context_id is None or str(candidate.context_id).strip() == "":
            raise ValueError(f"Missing context_id for badge {candidate.badge_id} (player {candidate.player_id})")
        context_id = str(candidate.context_id)
        key = (int(candidate.player_id), str(candidate.badge_id), str(context_id))
        if key in existing:
            continue
        existing.add(key)
        value_json = dict(candidate.value_json or {})
        value_json.setdefault("badge_id", candidate.badge_id)
        if "tape_excerpt" not in value_json or not value_json.get("tape_excerpt"):
            value_json["tape_excerpt"] = _build_tape_excerpt(candidate, value_json)
        if "tape_title" not in value_json or not value_json.get("tape_title"):
            value_json["tape_title"] = _build_tape_title(candidate, value_json)
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
                "awarded_by": awarded_by,
                "rule_version": rule_version,
                "eval_run_id": eval_run_id,
            }
        )
        created.append(candidate)

    if not rows:
        return []

    chunk = 200
    for i in range(0, len(rows), chunk):
        supabase.table("player_badges").upsert(
            rows[i : i + chunk],
            on_conflict=PLAYER_BADGES_CONFLICT_KEY,
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
        context_id = row.get("context_id")
        if context_id is None:
            continue
        existing.add((player_id, badge_id, str(context_id)))
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


def _build_tape_title(candidate: BadgeCandidate, data: dict[str, Any]) -> str:
    copy = get_badge_copy(candidate.badge_id)
    highlight = copy.get("highlight", {}) if isinstance(copy, dict) else {}
    titles = highlight.get("titles", []) if isinstance(highlight, dict) else []
    seed = f"{candidate.player_id}:{candidate.badge_id}:{candidate.context_id}:title"
    template = pick_variant(titles, seed)
    rendered = render_template(template, data | {"badge_name": copy.get("name", candidate.badge_id)})
    return rendered or str(data.get("badge_name") or candidate.badge_id)
