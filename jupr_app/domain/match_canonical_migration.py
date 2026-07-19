from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from jupr_app.domain.match_canonical_audit import build_match_canonical_audit
from jupr_app.domain.match_filters import normalize_player_id, normalize_score

PLAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "t1_p1": ("team1_player1", "player1_id", "p1_id"),
    "t1_p2": ("team1_player2", "player2_id", "p2_id"),
    "t2_p1": ("team2_player1", "player3_id", "p3_id"),
    "t2_p2": ("team2_player2", "player4_id", "p4_id"),
}
SCORE_ALIASES: dict[str, tuple[str, ...]] = {
    "score_t1": ("team1_score", "score1", "score_a"),
    "score_t2": ("team2_score", "score2", "score_b"),
}


def normalize_legacy_matches_for_canonical(
    supabase,
    *,
    ctx,
    club_id: str,
    match_ids: list[int] | None = None,
    player_id: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    audit = build_match_canonical_audit(
        ctx,
        club_id=str(club_id),
        player_id=player_id,
        limit=None,
    )

    excluded = audit.get("excluded_only_in_profile", [])
    target_ids = {int(x) for x in (match_ids or [])}
    if target_ids:
        excluded = [row for row in excluded if int(row.get("match_id") or 0) in target_ids]

    by_id = _rows_by_id(getattr(ctx, "df_matches", pd.DataFrame()))
    proposals: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []

    for diag in excluded:
        mid = int(diag.get("match_id") or 0)
        source_row = by_id.get(mid)
        if source_row is None:
            manual_review.append({"match_id": mid, "reason": "row_not_found_in_ctx_df_matches"})
            continue

        patch, reasons = _propose_patch(source_row)
        if not patch:
            manual_review.append({"match_id": mid, "reason": "no_safe_normalization", "exclusion_reasons": diag.get("exclusion_reasons", [])})
            continue

        expected = {
            key: _json_scalar(source_row.get(key))
            for key in sorted(set(patch) | ({"updated_at"} if "updated_at" in source_row else set()))
        }
        proposals.append(
            {
                "match_id": mid,
                "patch": patch,
                "expected": expected,
                "normalization_reasons": reasons,
                "exclusion_reasons": diag.get("exclusion_reasons", []),
            }
        )

    applied = []
    if not dry_run:
        for proposal in proposals:
            mid = int(proposal["match_id"])
            patch = dict(proposal["patch"])
            (
                supabase.table("matches")
                .update(patch)
                .eq("club_id", str(club_id))
                .eq("id", mid)
                .execute()
            )
            applied.append(mid)

    fingerprint_payload = {
        "club_id": str(club_id),
        "player_id": int(player_id) if player_id is not None else None,
        "proposals": [
            {
                "match_id": int(row["match_id"]),
                "expected": row["expected"],
                "patch": row["patch"],
            }
            for row in proposals
        ],
    }
    preview_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "dry_run": bool(dry_run),
        "read_only": bool(dry_run),
        "club_id": str(club_id),
        "player_id": int(player_id) if player_id is not None else None,
        "requested_match_ids": sorted(target_ids),
        "candidate_count": len(excluded),
        "proposed_update_count": len(proposals),
        "applied_update_count": len(applied),
        "applied_match_ids": applied,
        "proposals": proposals,
        "manual_review_needed": manual_review,
        "preview_fingerprint": preview_fingerprint,
    }


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _rows_by_id(df_matches: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if df_matches is None or df_matches.empty or "id" not in df_matches.columns:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for _, row in df_matches.iterrows():
        try:
            mid = int(row.get("id"))
        except Exception:
            continue
        out[mid] = row.to_dict()
    return out


def _propose_patch(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    patch: dict[str, Any] = {}
    reasons: list[str] = []

    for target, aliases in PLAYER_ALIASES.items():
        canonical = normalize_player_id(row.get(target))
        if canonical:
            continue
        alias_value = _first_normalized_player(row, aliases)
        if alias_value:
            patch[target] = int(alias_value)
            reasons.append(f"filled_{target}_from_legacy_alias")

    for target, aliases in SCORE_ALIASES.items():
        canonical = normalize_score(row.get(target))
        if canonical > 0:
            continue
        alias_score = _first_normalized_score(row, aliases)
        if alias_score > 0:
            patch[target] = int(alias_score)
            reasons.append(f"filled_{target}_from_legacy_alias")

    match_type = str(row.get("match_type") or "").strip()
    normalized_match_type = _normalize_match_type(match_type)
    if normalized_match_type and normalized_match_type != match_type:
        patch["match_type"] = normalized_match_type
        reasons.append("normalized_match_type")

    context_type = str(row.get("context_type") or "").strip()
    normalized_context_type = _normalize_context_type(context_type)
    if normalized_context_type != context_type:
        patch["context_type"] = normalized_context_type
        reasons.append("normalized_context_type")

    league = str(row.get("league") or "").strip()
    if not league:
        league_fallback = _infer_league(row)
        if league_fallback:
            patch["league"] = league_fallback
            reasons.append("inferred_league")

    if _is_legacy_tournament_mismatch(row):
        patch.setdefault("context_type", "")
        patch.setdefault("match_type", "League")
        reasons.append("cleared_legacy_tournament_marker")

    return patch, reasons


def _first_normalized_player(row: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        pid = normalize_player_id(row.get(alias))
        if pid:
            return pid
    return None


def _first_normalized_score(row: dict[str, Any], aliases: tuple[str, ...]) -> int:
    for alias in aliases:
        score = normalize_score(row.get(alias))
        if score > 0:
            return score
    return 0


def _normalize_match_type(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        return ""
    if key in {"popup", "pop up", "pop-up"}:
        return "PopUp"
    if key in {"tournament", "tourny", "tourney"}:
        return "Tournament"
    if key in {"league", "league night", "regular", "ranked"}:
        return "League"
    return str(raw).strip()


def _normalize_context_type(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        return ""
    if key in {"event", "popup", "pop up", "pop-up"}:
        return "EVENT"
    if key in {"tournament", "tourny", "tourney"}:
        return "TOURNAMENT"
    if key in {"league", "ladder"}:
        return "LEAGUE"
    return str(raw).strip()


def _infer_league(row: dict[str, Any]) -> str:
    for key in ("league_name", "league_id"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    context_type = str(row.get("context_type") or "").strip().upper()
    context_id = str(row.get("context_id") or "").strip()
    if context_type == "LEAGUE" and context_id:
        return context_id.split(":")[0]
    return ""


def _is_legacy_tournament_mismatch(row: dict[str, Any]) -> bool:
    tournament_id = row.get("tournament_id")
    if tournament_id is not None and str(tournament_id).strip() != "":
        return False

    match_type = str(row.get("match_type") or "").strip().upper()
    context_type = str(row.get("context_type") or "").strip().upper()
    league = str(row.get("league") or "").strip()
    score_ok = normalize_score(row.get("score_t1")) + normalize_score(row.get("score_t2")) > 0
    players_ok = normalize_player_id(row.get("t1_p1")) and normalize_player_id(row.get("t2_p1"))

    looks_tournament = match_type == "TOURNAMENT" or context_type == "TOURNAMENT"
    return bool(looks_tournament and league and score_ok and players_ok)
