from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from typing import Any

import pandas as pd

from jupr_app.domain.gamification.badge_registry import registry
from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.evaluators import build_evaluation_context
from jupr_app.domain.match_filters import MatchFilterAudit, MatchFilterAuditStep, apply_match_filters_with_audit
from jupr_app.domain.player_aggregate_audit import compute_player_aggregate_reconciliation


@dataclass
class BadgeDebugReport:
    club_id: str
    league_id: str | None
    player_id: int
    badge_id: str
    matches_raw: list[str] = field(default_factory=list)
    matches_filtered: list[str] = field(default_factory=list)
    filter_audit_steps: list[MatchFilterAuditStep] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def build_badge_debug_report(
    ctx: Any,
    club_id: str,
    league_id: str | None,
    player_id: int,
    badge_id: str,
    *,
    limit_matches: int | None = None,
    filtered_matches: pd.DataFrame | None = None,
    match_audit: MatchFilterAudit | None = None,
) -> BadgeDebugReport:
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None:
        df_matches = pd.DataFrame()

    raw_matches = df_matches
    if limit_matches is not None and not df_matches.empty:
        raw_matches = df_matches.head(int(limit_matches)).copy()

    if filtered_matches is None or match_audit is None:
        context_filters = {"club_id": club_id, "exclude_popups": True}
        if league_id:
            context_filters["league_name"] = league_id
        filtered_matches, match_audit = apply_match_filters_with_audit(
            raw_matches, context_filters
        )

    evaluation = build_evaluation_context(
        ctx, club_id, league_id, as_of=None, df_matches_override=filtered_matches
    )
    report = BadgeDebugReport(
        club_id=str(club_id),
        league_id=league_id,
        player_id=int(player_id),
        badge_id=str(badge_id),
        matches_raw=list(match_audit.raw_match_ids),
        matches_filtered=list(match_audit.final_match_ids),
        filter_audit_steps=list(match_audit.steps),
    )

    spec = registry().get(str(badge_id))
    if spec is None:
        report.errors.append(f"Badge ID '{badge_id}' not found in registry.")
        return report

    try:
        for candidate in spec.evaluator(evaluation):
            if int(candidate.player_id) != int(player_id):
                continue
            report.candidates.append(_candidate_to_debug_row(candidate))
    except Exception:
        report.errors.append(traceback.format_exc())

    report.diagnostics = _build_badge_diagnostics(
        evaluation=evaluation,
        badge_id=str(badge_id),
        player_id=int(player_id),
    )
    report.diagnostics["player_aggregate_reconciliation"] = compute_player_aggregate_reconciliation(
        ctx,
        player_id=int(player_id),
        club_id=str(club_id),
        league_id=league_id,
        filtered_matches=filtered_matches,
        match_audit=match_audit,
    )

    return report


def _candidate_to_debug_row(candidate: BadgeCandidate) -> dict[str, Any]:
    match_id = candidate.match_id
    derived_match_id = _derive_match_id(candidate)
    if not match_id and derived_match_id:
        match_id = derived_match_id
    return {
        "badge_id": candidate.badge_id,
        "player_id": int(candidate.player_id),
        "club_id": candidate.club_id,
        "context_type": candidate.context_type,
        "context_id": candidate.context_id,
        "match_id": match_id,
        "value_json": candidate.value_json,
        "value_num": candidate.value_num,
    }


def _derive_match_id(candidate: BadgeCandidate) -> str | None:
    if candidate.match_id:
        return str(candidate.match_id)

    context_type = (candidate.context_type or "").lower()
    if "match" in context_type and candidate.context_id:
        return str(candidate.context_id)

    value_json = candidate.value_json
    if isinstance(value_json, dict):
        match_id = value_json.get("match_id")
        if match_id:
            return str(match_id)
    return None


def _build_badge_diagnostics(*, evaluation: Any, badge_id: str, player_id: int) -> dict[str, Any]:
    badge_id = str(badge_id)
    if badge_id == "high_roller":
        return _high_roller_diagnostics(evaluation=evaluation, player_id=player_id)
    if badge_id == "dedicated_participant_50":
        return _participant_threshold_diagnostics(
            evaluation=evaluation,
            player_id=player_id,
            badge_id=badge_id,
            threshold=50,
        )
    if badge_id == "lifetime_participant_200":
        return _participant_threshold_diagnostics(
            evaluation=evaluation,
            player_id=player_id,
            badge_id=badge_id,
            threshold=200,
        )
    return {}


def _high_roller_diagnostics(*, evaluation: Any, player_id: int) -> dict[str, Any]:
    facts = getattr(evaluation, "facts", pd.DataFrame())
    if facts is None:
        facts = pd.DataFrame()
    player_facts = facts[facts["player_id"] == int(player_id)].copy() if "player_id" in facts.columns else pd.DataFrame()
    player_wins = player_facts[player_facts["win"] == True].copy() if "win" in player_facts.columns else pd.DataFrame()
    if "match_id" in player_wins.columns:
        distinct_wins = int(player_wins["match_id"].dropna().nunique())
    else:
        distinct_wins = 0
    threshold = 100
    return {
        "player_id": int(player_id),
        "badge_id": "high_roller",
        "filtered_player_match_rows": int(len(player_facts)),
        "filtered_player_win_rows": int(len(player_wins)),
        "filtered_player_distinct_win_match_ids": distinct_wins,
        "threshold_required": threshold,
        "qualifies_boolean": bool(distinct_wins >= threshold),
    }


def _participant_threshold_diagnostics(
    *,
    evaluation: Any,
    player_id: int,
    badge_id: str,
    threshold: int,
) -> dict[str, Any]:
    counts = {}
    ctx = getattr(evaluation, "ctx", None)
    if ctx is not None:
        from jupr_app.domain.gamification.participation import compute_lifetime_games

        counts = compute_lifetime_games(ctx) or {}
    player_games = int(counts.get(int(player_id), 0))
    return {
        "player_id": int(player_id),
        "badge_id": str(badge_id),
        "filtered_player_distinct_match_ids": player_games,
        "threshold_required": int(threshold),
        "qualifies_boolean": bool(player_games >= int(threshold)),
    }
