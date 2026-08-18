from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.leagues import (
    TOP_PERFORMER_BADGE_IDS,
    build_top_performer_badge_candidates,
    compute_top_performer_awards_for_config,
    mint_top_performer_badges,
    normalize_league_status,
)
from jupr_app.domain.league_analytics import (
    award_category_catalog,
    compute_league_player_analytics,
    compute_team_league_analytics,
)
from jupr_app.services.admin_league_manager_service import is_admin_league_manager_enabled

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_CLOSE_LEAGUE = "CLOSE LEAGUE"
CONFIRM_FREEZE_AWARDS = "FREEZE LEAGUE AWARDS"
CONFIRM_MINT_AWARDS = "MINT AWARDS"
CONFIRM_ARCHIVE_LEAGUE = "ARCHIVE LEAGUE"
AWARDS_WORKFLOW_VERSION = 3
AWARDS_WRITE_FLAG = "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE"
TOP_PERFORMER_BADGE_SEED_MIGRATION = "supabase/migrations/20260720014744_seed_top_performer_badges.sql"
REQUIRED_TOP_PERFORMER_BADGE_IDS = tuple(sorted(set(TOP_PERFORMER_BADGE_IDS.values())))
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
_WORKFLOW_RANK = {
    "not_started": 0,
    "frozen": 1,
    "previewed": 2,
    "overrides_confirmed": 3,
    "minting": 4,
    "mint_failed": 4,
    "minted": 5,
    "archived": 6,
}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_league_awards_write_enabled() -> bool:
    return is_admin_league_manager_enabled() and _truthy_env(AWARDS_WRITE_FLAG)


def _require_write_gate() -> None:
    if os.getenv("JUPR_ENV", "").strip().lower() == "production":
        raise PermissionError("League Awards writes are staging-only and disabled in production.")
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if not _truthy_env(AWARDS_WRITE_FLAG):
        raise PermissionError(f"League Awards writes are disabled. Enable {AWARDS_WRITE_FLAG} for the staging pilot.")
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise PermissionError("League Awards writes require the FastAPI SUPABASE_SERVICE_ROLE_KEY.")


def require_admin_league_awards_write() -> None:
    """Fail closed before a League Awards route performs auth or data access."""

    _require_write_gate()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        data = resp.data or []
        if isinstance(data, dict):
            return [dict(data)]
        return [dict(row) for row in data]
    except Exception:
        return []


def _first_row(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _award_catalog_for_league(
    meta_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return only award measures that exist for this league's format."""

    league_type = _clean_text(meta_row.get("league_type"), limit=40).casefold()
    match_format = _clean_text(meta_row.get("match_format"), limit=20).casefold()
    is_team = league_type in {"team", "team league", "team_league"}
    unavailable: set[str] = set()
    if not is_team:
        unavailable.update(
            {"team_champion", "team_wins", "team_point_differential"}
        )
    if match_format == "singles":
        unavailable.update({"best_partnership", "partner_variety"})
    return [
        dict(row)
        for row in award_category_catalog()
        if str(row.get("key")) not in unavailable
    ]


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_error(exc: Exception) -> str:
    message = _clean_text(str(exc), limit=300)
    return message or exc.__class__.__name__


def _clean_idempotency_key(value: Any) -> str:
    key = str(value or "").strip()
    if not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError("idempotency_key must be 8-160 characters using letters, numbers, dot, underscore, colon, or hyphen.")
    return key


def _fetch_table_rows(
    supabase: Any,
    table_name: str,
    *,
    club_id: str,
    filters: Mapping[str, Any] | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = supabase.table(table_name).select("*").eq("club_id", str(club_id))
        for field, value in dict(filters or {}).items():
            query = query.eq(field, value)
        ranged = getattr(query, "range", None)
        if callable(ranged):
            batch = _safe_rows(ranged(offset, offset + page_size - 1).execute())
        else:
            batch = _safe_rows(query.execute())
        rows.extend(batch)
        if not callable(ranged) or len(batch) < page_size:
            return rows
        offset += page_size


def _fetch_players_by_id(
    supabase: Any, *, club_id: str, player_ids: set[int]
) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    query = (
        supabase.table("players")
        .select("*")
        .eq("club_id", str(club_id))
        .in_("id", sorted(player_ids))
    )
    return _safe_rows(query.execute())


def _fetch_league_meta_row(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any] | None:
    clean_league = _clean_text(league_name, limit=120)
    try:
        return _first_row(
            supabase.table("leagues_metadata")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("league_name", clean_league)
            .limit(1)
            .execute()
        )
    except Exception:
        rows = _fetch_table_rows(supabase, "leagues_metadata", club_id=str(club_id))
        return next((row for row in rows if _clean_text(row.get("league_name"), limit=120) == clean_league), None)


def _id_to_name(players: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in players:
        try:
            player_id = int(row.get("id"))
        except Exception:
            continue
        name = _clean_text(row.get("name"), limit=160)
        if name:
            result[player_id] = name
    return result


def _award_inputs(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    metadata: Mapping[str, Any] | None = None,
    league_rows: Sequence[Mapping[str, Any]] | None = None,
    match_rows: Sequence[Mapping[str, Any]] | None = None,
    player_rows: Sequence[Mapping[str, Any]] | None = None,
    team_rows: Sequence[Mapping[str, Any]] | None = None,
    fixture_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[int, str]]:
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    meta_row = dict(metadata) if metadata is not None else _fetch_league_meta_row(
        supabase, club_id=str(club_id), league_name=clean_league
    )
    if not meta_row:
        raise ValueError("league not found")
    if _clean_text(meta_row.get("league_name"), limit=120) != clean_league:
        raise ValueError("league metadata does not match league_name")

    meta_rows = [meta_row]
    resolved_league_rows = (
        [dict(row) for row in league_rows]
        if league_rows is not None
        else _fetch_table_rows(
            supabase,
            "league_ratings",
            club_id=str(club_id),
            filters={"league_name": clean_league},
        )
    )
    resolved_match_rows = (
        [dict(row) for row in match_rows]
        if match_rows is not None
        else _fetch_table_rows(
            supabase,
            "matches",
            club_id=str(club_id),
            filters={"league": clean_league},
        )
    )
    resolved_team_rows = (
        [dict(row) for row in team_rows]
        if team_rows is not None
        else _fetch_table_rows(
            supabase,
            "team_league_teams",
            club_id=str(club_id),
            filters={"league_name": clean_league},
        )
    )
    resolved_fixture_rows = (
        [dict(row) for row in fixture_rows]
        if fixture_rows is not None
        else _fetch_table_rows(
            supabase,
            "team_league_fixtures",
            club_id=str(club_id),
            filters={"league_name": clean_league},
        )
    )
    player_ids = {
        int(value)
        for row in resolved_league_rows
        for value in [row.get("player_id")]
        if value not in (None, "")
    }
    for row in resolved_match_rows:
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            try:
                player_ids.add(int(row.get(key)))
            except Exception:
                pass
    resolved_player_rows = (
        [dict(row) for row in player_rows]
        if player_rows is not None
        else _fetch_players_by_id(
            supabase, club_id=str(club_id), player_ids=player_ids
        )
    )
    id_to_name = _id_to_name(resolved_player_rows)

    df_meta = pd.DataFrame(meta_rows)
    df_leagues = pd.DataFrame(resolved_league_rows)
    awards_config = _json_value(meta_row.get("awards_config"), {}) or {}
    awards = compute_top_performer_awards_for_config(
        df_leagues,
        df_meta,
        id_to_name,
        clean_league,
        awards_config=awards_config if isinstance(awards_config, dict) else {},
    )
    schedule_config = _json_value(meta_row.get("schedule_config"), {}) or {}
    expected_weeks: int | None = None
    if isinstance(schedule_config, dict):
        for key in ("weeks", "week_count", "total_weeks"):
            try:
                candidate = int(schedule_config.get(key) or 0)
            except Exception:
                candidate = 0
            if candidate > 0:
                expected_weeks = candidate
                break
    regular_weeks = [
        int(row.get("week_number") or 0)
        for row in resolved_fixture_rows
        if str(row.get("phase") or "") == "regular"
    ]
    if expected_weeks is None and regular_weeks:
        expected_weeks = max(regular_weeks)
    player_analytics = compute_league_player_analytics(
        resolved_match_rows,
        club_id=str(club_id),
        league_name=clean_league,
        players=resolved_player_rows,
        league_ratings=resolved_league_rows,
        match_format=_clean_text(meta_row.get("match_format"), limit=20),
        expected_weeks=expected_weeks,
    )
    if normalize_league_status(meta_row) not in {"ended", "archived"}:
        active_member_ids = {
            int(row["player_id"])
            for row in resolved_league_rows
            if row.get("player_id") not in (None, "")
            and row.get("is_active") is not False
            and not row.get("inactive_at")
        }
        player_analytics["players"] = [
            row
            for row in player_analytics["players"]
            if int(row.get("player_id") or 0) in active_member_ids
        ]
    team_analytics = compute_team_league_analytics(
        resolved_fixture_rows, resolved_team_rows
    )
    analytics_context = {
        "catalog": _award_catalog_for_league(meta_row),
        "measurable_player_stats": list(
            player_analytics["players"][0].keys()
            if player_analytics["players"]
            else []
        ),
        "player_analytics": player_analytics["players"],
        "team_analytics": team_analytics["teams"],
        "provenance": player_analytics["provenance"],
        "expected_weeks": expected_weeks,
    }
    configured = _computed_configured_awards(
        analytics_context,
        awards_config if isinstance(awards_config, dict) else {},
    )
    analytics_context["award_progress"] = list(configured or [])
    if configured is not None:
        awards = configured
    meta_row = {**meta_row, "_analytics": analytics_context}
    return meta_row, awards, df_meta, df_leagues, id_to_name


def _computed_configured_awards(
    analytics: Mapping[str, Any], awards_config: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    categories = awards_config.get("categories")
    if not isinstance(categories, Mapping) or not categories:
        return None
    catalog = {row["key"]: row for row in analytics.get("catalog", [])}
    result: list[dict[str, Any]] = []
    for category_key, raw_config in categories.items():
        config = dict(raw_config or {}) if isinstance(raw_config, Mapping) else {}
        if config.get("enabled") is False:
            continue
        spec = catalog.get(str(category_key))
        if not spec:
            continue
        recipients = (
            analytics.get("team_analytics", [])
            if spec["recipient_type"] == "team"
            else analytics.get("player_analytics", [])
        )
        minimum_metric = str(spec.get("minimum_metric") or "games")
        minimum = int(config.get("minimum") or config.get("min_games") or 0)
        qualified = [
            dict(row)
            for row in recipients
            if row.get(spec["metric"]) is not None
            and float(row.get(minimum_metric) or 0) >= minimum
        ]
        qualified.sort(
            key=lambda row: (
                -float(row.get(spec["metric"]) or 0),
                str(
                    row.get("player_name")
                    or row.get("team_name")
                    or ""
                ).lower(),
                str(row.get("player_id") or row.get("team_id") or ""),
            )
        )
        depth = max(1, min(3, int(config.get("depth") or 1)))
        if not qualified:
            continue
        cutoff = float(qualified[min(depth, len(qualified)) - 1][spec["metric"]])
        winners = [
            row
            for row in qualified
            if float(row[spec["metric"]]) >= cutoff
        ][:12]
        for row in winners:
            metric_value = row.get(spec["metric"])
            numeric_metric = float(metric_value)
            rank = 1 + sum(
                float(other[spec["metric"]]) > numeric_metric
                for other in winners
            )
            is_co_winner = (
                sum(
                    float(other[spec["metric"]]) == numeric_metric
                    for other in winners
                )
                > 1
            )
            recipient_type = str(spec["recipient_type"])
            recipient_id = (
                row.get("team_id")
                if recipient_type == "team"
                else row.get("player_id")
            )
            if recipient_id in (None, ""):
                continue
            award_key = (
                f"{category_key}:{rank}:{recipient_type}:{recipient_id}"
            )
            result.append(
                {
                    "award_key": award_key,
                    "category_key": str(category_key),
                    "category_label": str(spec["label"]),
                    "recipient_type": recipient_type,
                    "player_id": row.get("player_id"),
                    "team_id": row.get("team_id"),
                    "player_name": row.get("player_name"),
                    "team_name": row.get("team_name"),
                    "recipient_name": row.get("player_name")
                    or row.get("team_name"),
                    "metric_value": metric_value,
                    "metric_display": _metric_display(spec, row, metric_value),
                    "rank": rank,
                    "is_co_winner": is_co_winner,
                    "min_games": minimum,
                    "minimum_metric": minimum_metric,
                }
            )
    return result


def _metric_display(
    spec: Mapping[str, Any], row: Mapping[str, Any], metric_value: Any
) -> str:
    format_name = str(spec.get("format") or "")
    try:
        number = float(metric_value)
    except Exception:
        return str(metric_value)
    if format_name == "percent":
        return f"{number * 100:.1f}%"
    if format_name in {"rating", "signed_rating"}:
        prefix = "+" if format_name == "signed_rating" and number > 0 else ""
        return f"{prefix}{number:.3f}"
    if format_name in {"integer", "signed_integer"}:
        prefix = "+" if format_name == "signed_integer" and number > 0 else ""
        return f"{prefix}{int(number)}"
    if format_name in {"decimal", "signed_decimal"}:
        prefix = "+" if format_name == "signed_decimal" and number > 0 else ""
        return f"{prefix}{number:.2f}"
    if format_name == "team_record":
        return f"{int(row.get('wins') or 0)}-{int(row.get('losses') or 0)}"
    return str(metric_value)


def _award_identity(award: Mapping[str, Any]) -> str:
    stored = _clean_text(award.get("award_key"), limit=240)
    if stored:
        return stored
    category = _clean_text(award.get("category_key"), limit=80)
    rank = int(award.get("rank") or 1)
    return f"{category}:{rank}"


def _eligible_players(df_leagues: pd.DataFrame, id_to_name: Mapping[int, str], league_name: str) -> list[dict[str, Any]]:
    if df_leagues.empty or "league_name" not in df_leagues.columns or "player_id" not in df_leagues.columns:
        return []
    rows = df_leagues[df_leagues["league_name"].fillna("").astype(str).str.strip() == str(league_name)].copy()
    player_ids: set[int] = set()
    for value in rows["player_id"].dropna().tolist():
        try:
            player_ids.add(int(value))
        except Exception:
            continue
    return [
        {"player_id": player_id, "player_name": id_to_name.get(player_id, f"Player {player_id}")}
        for player_id in sorted(player_ids, key=lambda item: (str(id_to_name.get(item, "")).lower(), item))
    ]


def _league_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "league_name": _clean_text(row.get("league_name"), limit=120),
        "status": normalize_league_status(row),
        "is_active": bool(row.get("is_active", False)),
        "min_games": row.get("min_games"),
        "league_type": row.get("league_type"),
        "match_format": row.get("match_format"),
        "awards_config": _json_value(row.get("awards_config"), {}) or {},
        "awards_config_version": int(row.get("awards_config_version") or 0),
        "ended_at": row.get("ended_at"),
        "ended_by": row.get("ended_by"),
    }


def _new_workflow(meta_row: Mapping[str, Any]) -> dict[str, Any]:
    league_status = normalize_league_status(meta_row)
    status = "archived" if league_status == "archived" else ("frozen" if league_status == "ended" else "not_started")
    return {
        "version": AWARDS_WORKFLOW_VERSION,
        "status": status,
        "revision": 0,
        "frozen_at": meta_row.get("ended_at") if status in {"frozen", "archived"} else None,
        "frozen_by": meta_row.get("ended_by") if status in {"frozen", "archived"} else None,
        "preview": None,
        "final_awards": [],
        "override_notes": {},
        "mint": {"status": "not_started", "attempt_count": 0, "attempts": []},
        "archive": {"status": "archived" if status == "archived" else "not_started"},
        "operations": {},
        "updated_at": None,
    }


def _workflow_from_meta(meta_row: Mapping[str, Any]) -> dict[str, Any]:
    end_awards = _json_value(meta_row.get("end_awards"), {}) or {}
    stored = end_awards.get("workflow") if isinstance(end_awards, dict) else None
    if not isinstance(stored, dict) or int(stored.get("version") or 0) != AWARDS_WORKFLOW_VERSION:
        workflow = _new_workflow(meta_row)
        legacy_awards = end_awards.get("top_performers") if isinstance(end_awards, dict) else None
        if isinstance(legacy_awards, list) and legacy_awards:
            workflow["preview"] = {
                "awards": [dict(row) for row in legacy_awards if isinstance(row, dict)],
                "award_count": len(legacy_awards),
                "fingerprint": _fingerprint(legacy_awards),
                "generated_at": meta_row.get("ended_at"),
                "legacy_import": True,
            }
            workflow["final_awards"] = [dict(row) for row in legacy_awards if isinstance(row, dict)]
        return workflow
    workflow = _new_workflow(meta_row)
    workflow.update(dict(stored))
    workflow["version"] = AWARDS_WORKFLOW_VERSION
    workflow["revision"] = max(0, int(workflow.get("revision") or 0))
    workflow["operations"] = dict(workflow.get("operations") or {})
    workflow["override_notes"] = dict(workflow.get("override_notes") or {})
    workflow["final_awards"] = [dict(row) for row in (workflow.get("final_awards") or []) if isinstance(row, dict)]
    mint = dict(workflow.get("mint") or {})
    mint["attempt_count"] = max(0, int(mint.get("attempt_count") or 0))
    mint["attempts"] = list(mint.get("attempts") or [])[-10:]
    workflow["mint"] = mint
    workflow["archive"] = dict(workflow.get("archive") or {})
    return workflow


def _workflow_awards(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    final_awards = workflow.get("final_awards") or []
    if final_awards:
        return [dict(row) for row in final_awards if isinstance(row, dict)]
    preview = workflow.get("preview") or {}
    return [dict(row) for row in preview.get("awards", []) if isinstance(row, dict)] if isinstance(preview, dict) else []


def _persist_workflow(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    meta_row: Mapping[str, Any],
    workflow: dict[str, Any],
    lifecycle_patch: Mapping[str, Any] | None = None,
    source: str,
) -> dict[str, Any]:
    next_workflow = dict(workflow)
    next_workflow["version"] = AWARDS_WORKFLOW_VERSION
    next_workflow["revision"] = int(workflow.get("revision") or 0) + 1
    next_workflow["updated_at"] = _now_iso()
    final_awards = _workflow_awards(next_workflow)
    end_awards = {
        "schema_version": AWARDS_WORKFLOW_VERSION,
        "top_performers": final_awards,
        "source": source,
        "generated_at": ((next_workflow.get("preview") or {}).get("generated_at") if isinstance(next_workflow.get("preview"), dict) else None),
        "workflow": next_workflow,
    }
    update_payload = {"end_awards": end_awards, **dict(lifecycle_patch or {})}
    analytics = dict(meta_row.get("_analytics") or {})
    source_snapshot = dict(
        next_workflow.get("frozen_snapshot")
        or {
            "analytics": analytics,
            "awards_config": _json_value(meta_row.get("awards_config"), {}) or {},
            "awards_config_version": int(meta_row.get("awards_config_version") or 0),
        }
    )
    preview_fingerprint = str(
        (next_workflow.get("preview") or {}).get("fingerprint")
        or _fingerprint(final_awards)
    )
    result_fingerprint = _fingerprint(final_awards)
    records = _award_records(
        final_awards,
        source_snapshot=source_snapshot,
        override_notes=dict(next_workflow.get("override_notes") or {}),
    )
    rpc_method = getattr(supabase, "rpc", None)
    if callable(rpc_method):
        response = rpc_method(
            "league_awards_apply_workflow_v2",
            {
                "p_club_id": str(club_id),
                "p_league_name": str(league_name),
                "p_expected_workflow_revision": int(workflow.get("revision") or 0),
                "p_next_workflow": next_workflow,
                "p_lifecycle_patch": dict(lifecycle_patch or {}),
                "p_preview_fingerprint": preview_fingerprint,
                "p_result_fingerprint": result_fingerprint,
                "p_records": records,
                "p_source_snapshot": source_snapshot,
                "p_finalized": str(next_workflow.get("status") or "")
                in {"minted", "archived"},
                "p_actor_email": str(
                    next_workflow.get("updated_by")
                    or next_workflow.get("frozen_by")
                    or "unknown"
                ),
                "p_actor_role": "admin",
                "p_source": str(source),
            },
        ).execute()
        receipt = _first_row(response)
        if not receipt or not bool(receipt.get("committed")):
            raise RuntimeError(
                "League Awards returned no atomic workflow receipt."
            )
        updated = dict(receipt.get("league") or {})
    else:
        # The repository's in-memory unit-test adapter predates RPC support.
        updated = _first_row(
            supabase.table("leagues_metadata")
            .update(update_payload)
            .eq("club_id", str(club_id))
            .eq("league_name", str(league_name))
            .execute()
        )
    if not updated:
        raise RuntimeError("League Awards state write did not match a league row; no success was recorded.")
    workflow.clear()
    workflow.update(next_workflow)
    return {**dict(meta_row), **updated, **update_payload}


def _award_records(
    awards: list[dict[str, Any]],
    *,
    source_snapshot: Mapping[str, Any],
    override_notes: Mapping[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for award in awards:
        category = _clean_text(award.get("category_key"), limit=80)
        rank = int(award.get("rank") or 1)
        recipient_type = str(award.get("recipient_type") or "player")
        recipient_name = _clean_text(
            award.get("recipient_name")
            or award.get("player_name")
            or award.get("team_name"),
            limit=160,
        )
        award_identity = _award_identity(award)
        reason = _clean_text(
            override_notes.get(award_identity)
            or override_notes.get(f"{category}:{rank}"),
            limit=500,
        )
        computed_player_id = award.get(
            "computed_player_id", award.get("player_id")
        )
        computed_team_id = award.get(
            "computed_team_id", award.get("team_id")
        )
        computed_recipient_name = _clean_text(
            award.get("computed_recipient_name") or recipient_name,
            limit=160,
        )
        computed_metric_value = award.get(
            "computed_metric_value", award.get("metric_value")
        )
        records.append(
            {
                "award_key": award_identity,
                "category_key": category,
                "category_label": _clean_text(
                    award.get("category_label"), limit=160
                ),
                "recipient_type": recipient_type,
                "player_id": award.get("player_id")
                if recipient_type == "player"
                else None,
                "team_id": award.get("team_id")
                if recipient_type == "team"
                else None,
                "recipient_name": recipient_name,
                "placement": rank,
                "is_co_winner": bool(award.get("is_co_winner")),
                "metric_value": award.get("metric_value"),
                "computed_metric_value": computed_metric_value,
                "computed_player_id": computed_player_id,
                "computed_team_id": computed_team_id,
                "computed_recipient_name": computed_recipient_name,
                "metric_display": _clean_text(
                    award.get("metric_display"), limit=240
                ),
                "manual_label": None,
                "is_override": bool(reason)
                or computed_player_id != award.get("player_id")
                or computed_team_id != award.get("team_id"),
                "override_reason": reason or None,
                "public_visible": True,
                "source_snapshot": dict(source_snapshot),
            }
        )
    return records


def _operation_replay(workflow: Mapping[str, Any], *, action: str, idempotency_key: str, request_payload: Any) -> bool:
    existing = (workflow.get("operations") or {}).get(action) if isinstance(workflow.get("operations"), dict) else None
    if not isinstance(existing, dict) or existing.get("idempotency_key") != idempotency_key:
        return False
    if existing.get("request_fingerprint") != _fingerprint(request_payload):
        raise ValueError(f"idempotency_key was already used for a different {action} request.")
    return existing.get("status") == "completed"


def _record_operation(
    workflow: dict[str, Any],
    *,
    action: str,
    idempotency_key: str,
    request_payload: Any,
    status: str,
    error: str | None = None,
) -> None:
    operations = dict(workflow.get("operations") or {})
    operations[action] = {
        "idempotency_key": idempotency_key,
        "request_fingerprint": _fingerprint(request_payload),
        "status": status,
        "updated_at": _now_iso(),
        "error": error,
    }
    workflow["operations"] = operations


def _write_audit(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    action: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    source: str,
    note: str | None = None,
) -> str | None:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=f"league_awards_{action}_admin",
        entity_type="league",
        entity_id=str(league_name),
        before_json=dict(before),
        after_json={"source_client": "fastapi/nextjs", **dict(after)},
        note=note,
        source_page=source,
        flagged_for_review=True,
    )
    result = write_admin_activity_log(supabase, payload)
    if not result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return result.warning


def _require_audit_intent(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    action: str,
    workflow: Mapping[str, Any],
    idempotency_key: str,
    source: str,
) -> None:
    """Fail before mutation when the staging/production audit contract is strict."""
    if not _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        return
    _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=str(league_name),
        actor_email=actor_email,
        actor_role=actor_role,
        action=f"{action}_intent",
        before={"wizard": dict(workflow)},
        after={"requested_action": action, "idempotency_key_fingerprint": _fingerprint(idempotency_key)},
        source=source,
    )


def _badge_definition_readiness(supabase: Any) -> dict[str, Any]:
    """Report whether every Python-authoritative award badge can be minted.

    The lookup deliberately fails closed. A missing or unreadable definition must
    never be interpreted as permission to attempt a partial mint.
    """
    try:
        rows = _safe_rows(
            supabase.table("badges")
            .select("badge_id")
            .in_("badge_id", list(REQUIRED_TOP_PERFORMER_BADGE_IDS))
            .execute()
        )
        found = {str(row.get("badge_id") or "") for row in rows}
        missing = [badge_id for badge_id in REQUIRED_TOP_PERFORMER_BADGE_IDS if badge_id not in found]
        return {
            "badge_definitions_ready": not missing,
            "badge_definition_count": len(REQUIRED_TOP_PERFORMER_BADGE_IDS) - len(missing),
            "badge_definition_required_count": len(REQUIRED_TOP_PERFORMER_BADGE_IDS),
            "missing_badge_ids": missing,
            "badge_seed_migration": TOP_PERFORMER_BADGE_SEED_MIGRATION,
        }
    except Exception as exc:
        return {
            "badge_definitions_ready": False,
            "badge_definition_count": 0,
            "badge_definition_required_count": len(REQUIRED_TOP_PERFORMER_BADGE_IDS),
            "missing_badge_ids": list(REQUIRED_TOP_PERFORMER_BADGE_IDS),
            "badge_seed_migration": TOP_PERFORMER_BADGE_SEED_MIGRATION,
            "badge_definition_readiness_error": _safe_error(exc),
        }


def _response(
    meta_row: Mapping[str, Any],
    workflow: Mapping[str, Any],
    *,
    supabase: Any,
    mode: str,
    eligible_players: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    idempotent_replay: bool = False,
) -> dict[str, Any]:
    awards = _workflow_awards(workflow)
    mint = dict(workflow.get("mint") or {})
    analytics = dict(meta_row.get("_analytics") or {})
    return {
        "ok": True,
        "mode": mode,
        "league": _league_payload(dict(meta_row)),
        "league_name": _clean_text(meta_row.get("league_name"), limit=120),
        "awards": awards,
        "award_count": len(awards),
        "eligible_players": list(eligible_players or []),
        "wizard": dict(workflow),
        "writes_enabled": is_admin_league_awards_write_enabled(),
        "service_role_ready": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
        "badge_candidate_count": int(mint.get("verified_count") or 0),
        "badge_expected_count": int(mint.get("expected_count") or 0),
        "badge_verified_count": int(mint.get("verified_count") or 0),
        "idempotent_replay": bool(idempotent_replay),
        "warnings": list(warnings or []),
        "award_catalog": analytics.get(
            "catalog", _award_catalog_for_league(meta_row)
        ),
        "measurable_player_stats": analytics.get(
            "measurable_player_stats", []
        ),
        "player_analytics": analytics.get("player_analytics", []),
        "team_analytics": analytics.get("team_analytics", []),
        "award_progress": analytics.get("award_progress", []),
        "award_progress_count": len(analytics.get("award_progress", [])),
        "provenance": (
            (workflow.get("frozen_snapshot") or {}).get("provenance")
            if workflow.get("frozen_snapshot")
            else analytics.get("provenance", {})
        ),
        "expected_weeks": analytics.get("expected_weeks"),
        "awards_config_version": int(
            meta_row.get("awards_config_version") or 0
        ),
        **_badge_definition_readiness(supabase),
    }


def _enabled_public_award_specs(
    meta_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    awards_config = _json_value(meta_row.get("awards_config"), {}) or {}
    if not isinstance(awards_config, Mapping):
        return []
    categories = awards_config.get("categories")
    if not isinstance(categories, Mapping) or not categories:
        return []
    catalog = {
        str(row.get("key")): dict(row)
        for row in _award_catalog_for_league(meta_row)
    }
    enabled: list[dict[str, Any]] = []
    for category_key, raw_config in categories.items():
        config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
        spec = catalog.get(str(category_key))
        if spec and config.get("enabled") is not False:
            enabled.append(spec)
    return enabled


def _has_explicit_award_categories(meta_row: Mapping[str, Any]) -> bool:
    """Whether a league intentionally saved award category choices.

    Older leagues predate the draft-only award editor.  They still have valid
    default top-performer calculations, so an entirely absent ``categories``
    key must not hide their public awards.  An explicitly saved categories
    object, including an empty one, remains authoritative.
    """

    awards_config = _json_value(meta_row.get("awards_config"), {}) or {}
    return isinstance(awards_config, Mapping) and "categories" in awards_config


def get_public_league_award_progress(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    metadata: Mapping[str, Any] | None = None,
    league_rows: Sequence[Mapping[str, Any]] | None = None,
    match_rows: Sequence[Mapping[str, Any]] | None = None,
    player_rows: Sequence[Mapping[str, Any]] | None = None,
    team_rows: Sequence[Mapping[str, Any]] | None = None,
    fixture_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return current eligible awards without exposing workflow internals.

    Explicit award choices always win.  Leagues created before that editor (or
    with no saved category choices) use the established top-performer defaults
    so public standings do not silently lose valid award leaders.
    """

    try:
        meta_row = dict(metadata) if metadata is not None else _fetch_league_meta_row(
            supabase,
            club_id=str(club_id),
            league_name=str(league_name),
        )
        if not meta_row:
            return {"awards": [], "award_count": 0}
        enabled_specs = _enabled_public_award_specs(meta_row)
        explicit_categories = _has_explicit_award_categories(meta_row)
        if explicit_categories and not enabled_specs:
            return {"awards": [], "award_count": 0}

        # A legacy/default league uses the existing individual top-performer
        # calculation, which needs the same player inputs as configured awards.
        needs_player_analytics = (not explicit_categories) or any(
            spec.get("recipient_type") == "player" for spec in enabled_specs
        )
        needs_team_analytics = any(
            spec.get("recipient_type") == "team" for spec in enabled_specs
        )
        if not needs_player_analytics:
            league_rows = ()
            match_rows = ()
            player_rows = ()
        league_type = _clean_text(meta_row.get("league_type"), limit=40).casefold()
        if not needs_team_analytics and league_type not in {
            "team",
            "team league",
            "team_league",
        }:
            team_rows = ()
            fixture_rows = ()

        meta_row, computed_awards, _df_meta, _df_leagues, _id_map = _award_inputs(
            supabase,
            club_id=str(club_id),
            league_name=str(league_name),
            metadata=meta_row,
            league_rows=league_rows,
            match_rows=match_rows,
            player_rows=player_rows,
            team_rows=team_rows,
            fixture_rows=fixture_rows,
        )
        analytics = dict(meta_row.get("_analytics") or {})
        rows = list(analytics.get("award_progress") or [])
        if not explicit_categories:
            rows = list(computed_awards or [])
    except Exception:
        rows = []
    safe_rows = [
        {
            "category_key": row.get("category_key"),
            "category_label": row.get("category_label"),
            "recipient_type": row.get("recipient_type"),
            "player_id": row.get("player_id"),
            "team_id": row.get("team_id"),
            # Legacy top-performer rows use ``player_name`` while configured
            # category rows use ``recipient_name``.  The public projection is
            # deliberately one shape, so preserve either without exposing any
            # private player fields.
            "recipient_name": row.get("recipient_name")
            or row.get("player_name")
            or row.get("team_name"),
            "metric_value": row.get("metric_value"),
            "metric_display": row.get("metric_display"),
            "rank": row.get("rank"),
            "is_co_winner": row.get("is_co_winner"),
            "min_games": row.get("min_games"),
            "minimum_metric": row.get("minimum_metric"),
        }
        for row in rows
    ]
    return {"awards": safe_rows, "award_count": len(safe_rows)}


def get_admin_league_awards_wizard(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    meta_row, _computed, _df_meta, df_leagues, id_map = _award_inputs(
        supabase, club_id=str(club_id), league_name=str(league_name)
    )
    workflow = _workflow_from_meta(meta_row)
    return _response(
        meta_row,
        workflow,
        supabase=supabase,
        mode="league_awards_wizard",
        eligible_players=_eligible_players(df_leagues, id_map, _clean_text(league_name, limit=120)),
    )


def preview_admin_league_awards(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any]:
    """Read-only compatibility preview. Persisted preview is a separate guarded POST action."""
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    meta_row, awards, _df_meta, df_leagues, id_map = _award_inputs(
        supabase, club_id=str(club_id), league_name=str(league_name)
    )
    workflow = _workflow_from_meta(meta_row)
    payload = _response(
        meta_row,
        workflow,
        supabase=supabase,
        mode="league_awards_preview",
        eligible_players=_eligible_players(df_leagues, id_map, _clean_text(league_name, limit=120)),
    )
    payload["awards"] = awards
    payload["award_count"] = len(awards)
    payload["persisted"] = False
    return payload


def save_admin_league_awards_config(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    awards_config: Mapping[str, Any],
    expected_config_version: int,
    actor_email: str,
    actor_role: str,
    source: str = "next_league_manager_awards_config",
) -> dict[str, Any]:
    _require_write_gate()
    clean_league = _clean_text(league_name, limit=120)
    config = json.loads(_canonical_json(dict(awards_config or {})))
    categories = config.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("awards_config.categories must be an object.")
    meta_row = _fetch_league_meta_row(
        supabase, club_id=str(club_id), league_name=clean_league
    )
    if not meta_row:
        raise ValueError("league not found")
    known = {row["key"] for row in _award_catalog_for_league(meta_row)}
    unknown = sorted(set(categories) - known)
    if unknown:
        raise ValueError(
            "Unknown award categories: " + ", ".join(unknown)
        )
    for key, raw in categories.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Award category {key} must be an object.")
        depth = int(raw.get("depth") or 1)
        minimum = int(raw.get("minimum") or raw.get("min_games") or 0)
        if depth not in {1, 2, 3} or not 0 <= minimum <= 1000:
            raise ValueError(
                f"Award category {key} has an invalid depth or minimum."
            )
    if len(_canonical_json(config).encode("utf-8")) > 50000:
        raise ValueError("Awards configuration is too large.")

    rpc_method = getattr(supabase, "rpc", None)
    if callable(rpc_method):
        receipt = _first_row(
            rpc_method(
                "league_awards_save_config_v1",
                {
                    "p_club_id": str(club_id),
                    "p_league_name": clean_league,
                    "p_expected_config_version": int(
                        expected_config_version
                    ),
                    "p_awards_config": config,
                    "p_actor_email": _clean_text(actor_email, limit=320),
                    "p_actor_role": _clean_text(actor_role, limit=80),
                    "p_source": _clean_text(source, limit=160),
                },
            ).execute()
        )
        if not receipt or not receipt.get("committed"):
            raise RuntimeError(
                "Awards configuration returned no durable commit receipt."
            )
    else:
        updated = _first_row(
            supabase.table("leagues_metadata")
            .update(
                {
                    "awards_config": config,
                    "awards_config_version": int(expected_config_version) + 1,
                }
            )
            .eq("club_id", str(club_id))
            .eq("league_name", clean_league)
            .execute()
        )
        if not updated:
            raise RuntimeError("Awards configuration did not match a league.")
    return get_admin_league_awards_wizard(
        supabase, club_id=str(club_id), league_name=clean_league
    )


def freeze_admin_league_awards(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    idempotency_key: str,
    source: str = "next_league_manager_awards_freeze",
) -> dict[str, Any]:
    _require_write_gate()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_FREEZE_AWARDS:
        raise ValueError(f"Type {CONFIRM_FREEZE_AWARDS} to freeze the league for award review.")
    key = _clean_idempotency_key(idempotency_key)
    clean_league = _clean_text(league_name, limit=120)
    meta_row, computed_awards, _df_meta, df_leagues, id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    workflow = _workflow_from_meta(meta_row)
    request_payload = {"confirmation_text": CONFIRM_FREEZE_AWARDS}
    if _operation_replay(workflow, action="freeze", idempotency_key=key, request_payload=request_payload):
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_freeze", eligible_players=_eligible_players(df_leagues, id_map, clean_league), idempotent_replay=True)
    if _WORKFLOW_RANK.get(str(workflow.get("status")), -1) >= _WORKFLOW_RANK["frozen"]:
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_freeze", eligible_players=_eligible_players(df_leagues, id_map, clean_league), idempotent_replay=True)
    league_status = normalize_league_status(meta_row)
    if league_status not in {"active", "paused", "ended"}:
        raise ValueError("Only an active or paused league can enter the awards freeze workflow.")

    _require_audit_intent(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="freeze",
        workflow=workflow,
        idempotency_key=key,
        source=source,
    )

    frozen_at = meta_row.get("ended_at") or _now_iso()
    before_workflow = dict(workflow)
    workflow.update(
        {
            "status": "frozen",
            "frozen_at": frozen_at,
            "frozen_by": str(actor_email or ""),
            "preview": None,
            "final_awards": [],
            "override_notes": {},
            "mint": {"status": "not_started", "attempt_count": 0, "attempts": []},
            "archive": {"status": "not_started"},
            "frozen_snapshot": {
                "frozen_at": frozen_at,
                "awards": computed_awards,
                "awards_config": _json_value(
                    meta_row.get("awards_config"), {}
                )
                or {},
                "awards_config_version": int(
                    meta_row.get("awards_config_version") or 0
                ),
                "player_analytics": dict(
                    meta_row.get("_analytics") or {}
                ).get("player_analytics", []),
                "team_analytics": dict(
                    meta_row.get("_analytics") or {}
                ).get("team_analytics", []),
                "provenance": dict(meta_row.get("_analytics") or {}).get(
                    "provenance", {}
                ),
                "expected_weeks": dict(
                    meta_row.get("_analytics") or {}
                ).get("expected_weeks"),
            },
        }
    )
    _record_operation(workflow, action="freeze", idempotency_key=key, request_payload=request_payload, status="completed")
    updated = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=meta_row,
        workflow=workflow,
        lifecycle_patch={"is_active": False, "status": "ended", "ended_at": frozen_at, "ended_by": str(actor_email or "")},
        source=source,
    )
    warnings: list[str] = []
    audit_warning = _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="freeze",
        before={"league": _league_payload(meta_row), "wizard": before_workflow},
        after={"league": _league_payload(updated), "wizard": workflow},
        source=source,
    )
    if audit_warning:
        warnings.append(audit_warning)
    return _response(updated, workflow, supabase=supabase, mode="league_awards_freeze", eligible_players=_eligible_players(df_leagues, id_map, clean_league), warnings=warnings)


def persist_admin_league_awards_preview(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    idempotency_key: str,
    source: str = "next_league_manager_awards_preview_persist",
) -> dict[str, Any]:
    _require_write_gate()
    key = _clean_idempotency_key(idempotency_key)
    clean_league = _clean_text(league_name, limit=120)
    meta_row, awards, _df_meta, df_leagues, id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    workflow = _workflow_from_meta(meta_row)
    frozen_snapshot = dict(workflow.get("frozen_snapshot") or {})
    if isinstance(frozen_snapshot.get("awards"), list):
        awards = [
            dict(row)
            for row in frozen_snapshot["awards"]
            if isinstance(row, dict)
        ]
    request_payload = {"league_name": clean_league}
    if _operation_replay(workflow, action="preview", idempotency_key=key, request_payload=request_payload):
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_preview_persist", eligible_players=_eligible_players(df_leagues, id_map, clean_league), idempotent_replay=True)
    if _WORKFLOW_RANK.get(str(workflow.get("status")), -1) < _WORKFLOW_RANK["frozen"]:
        raise ValueError("Freeze the league before persisting an award preview.")
    if int((workflow.get("mint") or {}).get("attempt_count") or 0) > 0:
        raise ValueError("Award preview is locked after a mint attempt. Recover or retry the existing mint instead.")
    if str(workflow.get("status")) == "archived":
        raise ValueError("Archived league awards are read-only.")

    _require_audit_intent(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="preview",
        workflow=workflow,
        idempotency_key=key,
        source=source,
    )

    generated_at = _now_iso()
    before_workflow = dict(workflow)
    workflow.update(
        {
            "status": "previewed",
            "preview": {
                "awards": awards,
                "award_count": len(awards),
                "fingerprint": _fingerprint(awards),
                "generated_at": generated_at,
            },
            "final_awards": [],
            "override_notes": {},
            "mint": {"status": "not_started", "attempt_count": 0, "attempts": []},
        }
    )
    _record_operation(workflow, action="preview", idempotency_key=key, request_payload=request_payload, status="completed")
    updated = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=meta_row,
        workflow=workflow,
        source=source,
    )
    warnings: list[str] = []
    audit_warning = _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="preview",
        before={"wizard": before_workflow},
        after={"wizard": workflow, "award_count": len(awards)},
        source=source,
    )
    if audit_warning:
        warnings.append(audit_warning)
    return _response(updated, workflow, supabase=supabase, mode="league_awards_preview_persist", eligible_players=_eligible_players(df_leagues, id_map, clean_league), warnings=warnings)


def save_admin_league_award_overrides(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    overrides: list[dict[str, Any]],
    preview_fingerprint: str,
    actor_email: str,
    actor_role: str,
    idempotency_key: str,
    source: str = "next_league_manager_awards_overrides",
) -> dict[str, Any]:
    _require_write_gate()
    key = _clean_idempotency_key(idempotency_key)
    clean_league = _clean_text(league_name, limit=120)
    meta_row, _computed, _df_meta, df_leagues, id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    workflow = _workflow_from_meta(meta_row)
    normalized_overrides = [dict(item) for item in overrides if isinstance(item, dict)]
    request_payload = {"overrides": normalized_overrides, "preview_fingerprint": str(preview_fingerprint or "")}
    if _operation_replay(workflow, action="overrides", idempotency_key=key, request_payload=request_payload):
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_overrides", eligible_players=_eligible_players(df_leagues, id_map, clean_league), idempotent_replay=True)
    if str(workflow.get("status")) not in {"previewed", "overrides_confirmed"}:
        raise ValueError("Persist an award preview before confirming overrides.")
    if int((workflow.get("mint") or {}).get("attempt_count") or 0) > 0:
        raise ValueError("Award overrides are locked after a mint attempt.")
    preview = dict(workflow.get("preview") or {})
    stored_fingerprint = str(preview.get("fingerprint") or "")
    if not stored_fingerprint or str(preview_fingerprint or "") != stored_fingerprint:
        raise ValueError("Award preview is stale. Reload the persisted preview before saving overrides.")

    _require_audit_intent(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="overrides",
        workflow=workflow,
        idempotency_key=key,
        source=source,
    )

    preview_awards = [
        dict(row)
        for row in preview.get("awards", [])
        if isinstance(row, dict)
    ]
    awards_by_key: dict[str, dict[str, Any]] = {}
    keys_by_legacy_position: defaultdict[
        tuple[str, int], list[str]
    ] = defaultdict(list)
    for award in preview_awards:
        award_identity = _award_identity(award)
        if award_identity in awards_by_key:
            raise ValueError(
                "The persisted award preview has duplicate award identities."
            )
        awards_by_key[award_identity] = award
        keys_by_legacy_position[
            (
                str(award.get("category_key") or ""),
                int(award.get("rank") or 1),
            )
        ].append(award_identity)
    eligible = _eligible_players(df_leagues, id_map, clean_league)
    eligible_ids = {int(row["player_id"]) for row in eligible}
    overrides_by_key: dict[str, dict[str, Any]] = {}
    for item in normalized_overrides:
        category_key = _clean_text(item.get("category_key"), limit=80)
        try:
            rank = int(item.get("rank") or 1)
            player_id = int(item.get("player_id"))
        except Exception as exc:
            raise ValueError("Each award override requires a valid category_key, rank, and player_id.") from exc
        award_identity = _clean_text(item.get("award_key"), limit=240)
        if not award_identity:
            legacy_matches = keys_by_legacy_position.get(
                (category_key, rank), []
            )
            if len(legacy_matches) > 1:
                raise ValueError(
                    f"Award {category_key}:{rank} has co-winners; "
                    "award_key is required."
                )
            award_identity = legacy_matches[0] if legacy_matches else ""
        if award_identity not in awards_by_key:
            raise ValueError(
                f"Unknown award override {award_identity or f'{category_key}:{rank}'}."
            )
        target_award = awards_by_key[award_identity]
        if (
            str(target_award.get("category_key") or "") != category_key
            or int(target_award.get("rank") or 1) != rank
        ):
            raise ValueError(
                "Award override identity does not match its category and rank."
            )
        if str(target_award.get("recipient_type") or "player") != "player":
            raise ValueError(
                f"{category_key}:{rank} is a team award and cannot be reassigned to a player."
            )
        if player_id not in eligible_ids:
            raise ValueError(f"Player {player_id} is not in this league's award roster.")
        if award_identity in overrides_by_key:
            raise ValueError(f"Duplicate award override {award_identity}.")
        overrides_by_key[award_identity] = {
            "player_id": player_id,
            "reason": _clean_text(item.get("reason"), limit=500),
        }

    final_awards: list[dict[str, Any]] = []
    override_notes: dict[str, str] = {}
    frozen_snapshot = dict(workflow.get("frozen_snapshot") or {})
    frozen_player_rows = frozen_snapshot.get("player_analytics")
    if not isinstance(frozen_player_rows, list):
        frozen_player_rows = dict(meta_row.get("_analytics") or {}).get(
            "player_analytics", []
        )
    frozen_players = {
        int(row["player_id"]): dict(row)
        for row in frozen_player_rows
        if isinstance(row, Mapping) and row.get("player_id") is not None
    }
    catalog_by_key = {
        str(row["key"]): dict(row)
        for row in award_category_catalog()
    }
    selected_recipients: set[tuple[str, int]] = set()
    for award_identity, award in awards_by_key.items():
        if str(award.get("recipient_type") or "player") != "player":
            team_award = dict(award)
            team_award.setdefault("award_key", award_identity)
            final_awards.append(team_award)
            continue
        selected = overrides_by_key.get(
            award_identity,
            {"player_id": int(award.get("player_id")), "reason": ""},
        )
        selected_player_id = int(selected["player_id"])
        original_player_id = int(award.get("player_id"))
        reason = str(selected.get("reason") or "").strip()
        category_key = str(award.get("category_key") or "")
        rank = int(award.get("rank") or 1)
        if selected_player_id != original_player_id and len(reason) < 8:
            raise ValueError(
                "Override reason of at least 8 characters is required for "
                f"{category_key} rank {rank}."
            )
        recipient_key = (category_key, selected_player_id)
        if recipient_key in selected_recipients:
            raise ValueError(
                f"Player {selected_player_id} cannot receive duplicate "
                f"{category_key} winner records."
            )
        selected_recipients.add(recipient_key)
        updated_award = dict(award)
        updated_award.setdefault("award_key", award_identity)
        updated_award.setdefault("computed_player_id", original_player_id)
        updated_award.setdefault("computed_team_id", award.get("team_id"))
        updated_award.setdefault(
            "computed_recipient_name",
            award.get("recipient_name") or award.get("player_name"),
        )
        updated_award.setdefault(
            "computed_metric_value", award.get("metric_value")
        )
        updated_award["player_id"] = selected_player_id
        selected_name = id_map.get(
            selected_player_id, f"Player {selected_player_id}"
        )
        updated_award["player_name"] = selected_name
        updated_award["recipient_name"] = selected_name
        if selected_player_id != original_player_id:
            metric_name = str(
                catalog_by_key.get(category_key, {}).get("metric") or ""
            )
            selected_metric = frozen_players.get(
                selected_player_id, {}
            ).get(metric_name)
            updated_award["metric_value"] = selected_metric
            updated_award["metric_display"] = (
                str(selected_metric)
                if selected_metric is not None
                else "Not measurable"
            )
        final_awards.append(updated_award)
        if selected_player_id != original_player_id:
            override_notes[award_identity] = reason

    before_workflow = dict(workflow)
    workflow["status"] = "overrides_confirmed"
    workflow["final_awards"] = final_awards
    workflow["override_notes"] = override_notes
    workflow["overrides_confirmed_at"] = _now_iso()
    workflow["overrides_confirmed_by"] = str(actor_email or "")
    _record_operation(workflow, action="overrides", idempotency_key=key, request_payload=request_payload, status="completed")
    updated = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=meta_row,
        workflow=workflow,
        source=source,
    )
    warnings: list[str] = []
    audit_warning = _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="overrides",
        before={"wizard": before_workflow},
        after={"wizard": workflow, "override_count": len(override_notes)},
        source=source,
        note="; ".join(f"{key}: {value}" for key, value in sorted(override_notes.items())) or None,
    )
    if audit_warning:
        warnings.append(audit_warning)
    return _response(updated, workflow, supabase=supabase, mode="league_awards_overrides", eligible_players=eligible, warnings=warnings)


def _expected_badge_rows(*, club_id: str, league_name: str, awards: list[dict[str, Any]], ended_at: str | None, override_notes: Mapping[str, str]) -> list[dict[str, Any]]:
    candidates = build_top_performer_badge_candidates(
        awards,
        club_id=str(club_id),
        league_id=str(league_name),
        ended_at=ended_at,
        override_notes=override_notes,
    )
    expected: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        row = {
            "player_id": int(candidate.player_id),
            "badge_id": str(candidate.badge_id),
            "context_type": str(candidate.context_type),
            "context_id": str(candidate.context_id),
        }
        expected[
            (row["player_id"], row["badge_id"], row["context_type"], row["context_id"])
        ] = row
    return list(expected.values())


def _verify_badge_rows(supabase: Any, *, club_id: str, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not expected:
        return []
    badge_ids = sorted({str(row["badge_id"]) for row in expected})
    rows = _safe_rows(
        supabase.table("player_badges")
        .select("player_id,badge_id,context_type,context_id")
        .eq("club_id", str(club_id))
        .in_("badge_id", badge_ids)
        .execute()
    )
    found: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        try:
            badge_key = (
                int(row.get("player_id")),
                str(row.get("badge_id")),
                str(row.get("context_type")),
                str(row.get("context_id")),
            )
        except Exception:
            continue
        found[badge_key] = {
            "player_id": badge_key[0],
            "badge_id": badge_key[1],
            "context_type": badge_key[2],
            "context_id": badge_key[3],
        }
    missing = [
        row
        for row in expected
        if (
            int(row["player_id"]),
            str(row["badge_id"]),
            str(row["context_type"]),
            str(row["context_id"]),
        )
        not in found
    ]
    if missing:
        raise RuntimeError(f"Badge mint verification found {len(expected) - len(missing)} of {len(expected)} expected rows.")
    return [
        found[
            (
                int(row["player_id"]),
                str(row["badge_id"]),
                str(row["context_type"]),
                str(row["context_id"]),
            )
        ]
        for row in expected
    ]


def mint_admin_league_awards(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    idempotency_key: str,
    source: str = "next_league_manager_awards_mint",
) -> dict[str, Any]:
    _require_write_gate()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_MINT_AWARDS:
        raise ValueError(f"Type {CONFIRM_MINT_AWARDS} to mint and verify top performer badges.")
    key = _clean_idempotency_key(idempotency_key)
    clean_league = _clean_text(league_name, limit=120)
    meta_row, _computed, _df_meta, df_leagues, id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    workflow = _workflow_from_meta(meta_row)
    request_payload = {"confirmation_text": CONFIRM_MINT_AWARDS, "final_awards_fingerprint": _fingerprint(workflow.get("final_awards") or [])}
    eligible = _eligible_players(df_leagues, id_map, clean_league)

    awards = [dict(row) for row in workflow.get("final_awards", []) if isinstance(row, dict)]
    if str(workflow.get("status")) == "minted":
        expected = _expected_badge_rows(
            club_id=str(club_id),
            league_name=clean_league,
            awards=awards,
            ended_at=meta_row.get("ended_at") or workflow.get("frozen_at"),
            override_notes=dict(workflow.get("override_notes") or {}),
        )
        verified = _verify_badge_rows(supabase, club_id=str(club_id), expected=expected)
        workflow["mint"] = {**dict(workflow.get("mint") or {}), "verified_count": len(verified), "expected_count": len(expected)}
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_mint", eligible_players=eligible, idempotent_replay=True)
    if str(workflow.get("status")) not in {"overrides_confirmed", "minting", "mint_failed"}:
        raise ValueError("Confirm the persisted award preview and any overrides before minting badges.")

    badge_readiness = _badge_definition_readiness(supabase)
    if not badge_readiness["badge_definitions_ready"]:
        missing = ", ".join(badge_readiness["missing_badge_ids"])
        raise RuntimeError(
            "League Awards badge definitions are not ready; mint was not attempted. "
            f"Apply the reviewed equivalent of {TOP_PERFORMER_BADGE_SEED_MIGRATION}. "
            f"Missing badge IDs: {missing}."
        )

    _require_audit_intent(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="mint",
        workflow=workflow,
        idempotency_key=key,
        source=source,
    )

    before_workflow = dict(workflow)
    mint_state = dict(workflow.get("mint") or {})
    attempts = list(mint_state.get("attempts") or [])[-9:]
    attempt_number = int(mint_state.get("attempt_count") or 0) + 1
    started_at = _now_iso()
    attempts.append({"attempt": attempt_number, "idempotency_key": key, "status": "running", "started_at": started_at})
    mint_state.update(
        {
            "status": "running",
            "idempotency_key": key,
            "attempt_count": attempt_number,
            "attempts": attempts,
            "last_error": None,
            "started_at": started_at,
        }
    )
    workflow["status"] = "minting"
    workflow["mint"] = mint_state
    _record_operation(workflow, action="mint", idempotency_key=key, request_payload=request_payload, status="running")
    running_meta = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=meta_row,
        workflow=workflow,
        source=source,
    )
    ended_at = running_meta.get("ended_at") or workflow.get("frozen_at")
    expected = _expected_badge_rows(
        club_id=str(club_id),
        league_name=clean_league,
        awards=awards,
        ended_at=ended_at,
        override_notes=dict(workflow.get("override_notes") or {}),
    )
    try:
        created = mint_top_performer_badges(
            supabase,
            club_id=str(club_id),
            league_id=clean_league,
            awards=awards,
            ended_at=ended_at,
            override_notes=dict(workflow.get("override_notes") or {}),
        )
        verified = _verify_badge_rows(supabase, club_id=str(club_id), expected=expected)
    except Exception as exc:
        error = _safe_error(exc)
        attempts[-1] = {**attempts[-1], "status": "failed", "finished_at": _now_iso(), "error": error}
        mint_state.update(
            {
                "status": "failed",
                "attempts": attempts,
                "last_error": error,
                "expected_count": len(expected),
                "verified_count": 0,
            }
        )
        workflow["status"] = "mint_failed"
        workflow["mint"] = mint_state
        _record_operation(workflow, action="mint", idempotency_key=key, request_payload=request_payload, status="failed", error=error)
        try:
            _persist_workflow(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
                meta_row=running_meta,
                workflow=workflow,
                source=source,
            )
            _write_audit(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
                actor_email=actor_email,
                actor_role=actor_role,
                action="mint_failed",
                before={"wizard": before_workflow},
                after={"wizard": workflow, "error": error},
                source=source,
            )
        except Exception:
            pass
        raise RuntimeError("Badge mint could not be verified. The persisted workflow remains retryable and does not report success.") from exc

    finished_at = _now_iso()
    attempts[-1] = {**attempts[-1], "status": "verified", "finished_at": finished_at}
    mint_state.update(
        {
            "status": "verified",
            "attempts": attempts,
            "verified_at": finished_at,
            "expected_count": len(expected),
            "verified_count": len(verified),
            "created_candidate_count": len(created or []),
            "verified_rows": verified,
            "last_error": None,
        }
    )
    workflow["status"] = "minted"
    workflow["mint"] = mint_state
    _record_operation(workflow, action="mint", idempotency_key=key, request_payload=request_payload, status="completed")
    updated = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=running_meta,
        workflow=workflow,
        source=source,
    )
    warnings: list[str] = []
    audit_warning = _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="mint_verified",
        before={"wizard": before_workflow},
        after={"wizard": workflow, "expected_count": len(expected), "verified_count": len(verified)},
        source=source,
    )
    if audit_warning:
        warnings.append(audit_warning)
    return _response(updated, workflow, supabase=supabase, mode="league_awards_mint", eligible_players=eligible, warnings=warnings)


def _skip_badge_mint_for_legacy_close(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    meta_row: Mapping[str, Any],
    workflow: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    workflow["status"] = "minted"
    workflow["mint"] = {
        "status": "skipped_by_legacy_request",
        "attempt_count": 0,
        "attempts": [],
        "expected_count": 0,
        "verified_count": 0,
        "verified_at": _now_iso(),
    }
    return _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=str(league_name),
        meta_row=meta_row,
        workflow=workflow,
        source=source,
    )


def archive_admin_league_awards(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    idempotency_key: str,
    source: str = "next_league_manager_awards_archive",
) -> dict[str, Any]:
    _require_write_gate()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_ARCHIVE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_ARCHIVE_LEAGUE} to archive this completed awards workflow.")
    key = _clean_idempotency_key(idempotency_key)
    clean_league = _clean_text(league_name, limit=120)
    meta_row, _computed, _df_meta, df_leagues, id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    workflow = _workflow_from_meta(meta_row)
    request_payload = {"confirmation_text": CONFIRM_ARCHIVE_LEAGUE}
    eligible = _eligible_players(df_leagues, id_map, clean_league)
    if _operation_replay(workflow, action="archive", idempotency_key=key, request_payload=request_payload) or str(workflow.get("status")) == "archived":
        return _response(meta_row, workflow, supabase=supabase, mode="league_awards_archive", eligible_players=eligible, idempotent_replay=True)
    if str(workflow.get("status")) != "minted":
        raise ValueError("Mint and verify awards before archiving the league.")
    mint_status = str((workflow.get("mint") or {}).get("status") or "")
    if mint_status not in {"verified", "skipped_by_legacy_request"}:
        raise ValueError("Badge mint is not verified; archive remains blocked.")

    _require_audit_intent(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="archive",
        workflow=workflow,
        idempotency_key=key,
        source=source,
    )

    before_workflow = dict(workflow)
    archived_at = _now_iso()
    workflow["status"] = "archived"
    workflow["archive"] = {"status": "archived", "archived_at": archived_at, "archived_by": str(actor_email or "")}
    _record_operation(workflow, action="archive", idempotency_key=key, request_payload=request_payload, status="completed")
    updated = _persist_workflow(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        meta_row=meta_row,
        workflow=workflow,
        lifecycle_patch={"is_active": False, "status": "archived"},
        source=source,
    )
    warnings: list[str] = []
    audit_warning = _write_audit(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        action="archive",
        before={"league": _league_payload(meta_row), "wizard": before_workflow},
        after={"league": _league_payload(updated), "wizard": workflow},
        source=source,
    )
    if audit_warning:
        warnings.append(audit_warning)
    return _response(updated, workflow, supabase=supabase, mode="league_awards_archive", eligible_players=eligible, warnings=warnings)


def close_admin_league_and_award(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    award_badges: bool = True,
    source: str = "next_league_manager_awards_close",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Compatibility close call backed by the persisted wizard.

    New clients should use the explicit freeze/preview/overrides/mint/archive actions.
    """
    _require_write_gate()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CLOSE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_CLOSE_LEAGUE} to close this league and award top performers.")
    clean_league = _clean_text(league_name, limit=120)
    base_key = str(idempotency_key or "").strip()
    if not base_key:
        base_key = f"legacy-{_fingerprint([str(club_id), clean_league, str(actor_email).lower()])[:32]}"
    _clean_idempotency_key(base_key)
    freeze_admin_league_awards(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        confirmation_text=CONFIRM_FREEZE_AWARDS,
        idempotency_key=f"{base_key}:freeze",
        source=source,
    )
    persisted = persist_admin_league_awards_preview(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        actor_email=actor_email,
        actor_role=actor_role,
        idempotency_key=f"{base_key}:preview",
        source=source,
    )
    preview_fingerprint = str(((persisted.get("wizard") or {}).get("preview") or {}).get("fingerprint") or "")
    confirmed = save_admin_league_award_overrides(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        overrides=[],
        preview_fingerprint=preview_fingerprint,
        actor_email=actor_email,
        actor_role=actor_role,
        idempotency_key=f"{base_key}:overrides",
        source=source,
    )
    if award_badges:
        result = mint_admin_league_awards(
            supabase,
            club_id=str(club_id),
            league_name=clean_league,
            actor_email=actor_email,
            actor_role=actor_role,
            confirmation_text=CONFIRM_MINT_AWARDS,
            idempotency_key=f"{base_key}:mint",
            source=source,
        )
    else:
        workflow = dict(confirmed.get("wizard") or {})
        meta_row = _fetch_league_meta_row(supabase, club_id=str(club_id), league_name=clean_league)
        if not meta_row:
            raise RuntimeError("League disappeared while closing awards.")
        updated = _skip_badge_mint_for_legacy_close(
            supabase,
            club_id=str(club_id),
            league_name=clean_league,
            meta_row=meta_row,
            workflow=workflow,
            source=source,
        )
        result = _response(updated, workflow, supabase=supabase, mode="league_awards_mint_skipped")
    result["mode"] = "league_awards_close"
    return result
