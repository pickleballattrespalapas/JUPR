from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import json
import math
from types import SimpleNamespace
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_registry import active_badge_ids, registry
from jupr_app.domain.gamification.badge_types import BadgeCandidate, BadgeEvaluationContext
from jupr_app.domain.gamification.badges_repo import build_player_badge_rows
from jupr_app.domain.gamification.evaluators import build_evaluation_context


MATCH_EXCLUSION_BADGE_CONTRACT_VERSION = "jupr:match-exclusion-badges:v1"

# This is an intentionally frozen upper bound. Adding or removing a live badge
# requires a new contract version so an in-flight exclusion can finish against
# the exact badge scope captured before its match mutation.
MATCH_EXCLUSION_BADGE_IDS: tuple[str, ...] = (
    "blowout_artist",
    "bounce_back",
    "clean_sweep_week",
    "david_vs_goliath",
    "dedicated_participant_50",
    "draft_master",
    "first_win",
    "giant_slayer",
    "hall_of_fame_night",
    "high_roller",
    "hot_streak",
    "ice_in_veins",
    "iron_week",
    "legendary_upset",
    "level_up",
    "lifetime_participant_200",
    "marathon_month",
    "most_improved_monthly",
    "mountain_climber",
    "network_builder",
    "participant",
    "pickle_perfection",
    "rocket_start",
    "social_butterfly",
    "steady_hand",
    "swiss_army_knife",
    "untouchable",
    "upset_champion",
    "weekly_regular",
)

MATCH_EXCLUSION_BADGE_CLAIM_RPC = "claim_match_exclusion_badge_progress"
MATCH_EXCLUSION_BADGE_APPLY_RPC = "apply_match_exclusion_badge_reconciliation"
MATCH_EXCLUSION_BADGE_FAIL_RPC = "fail_match_exclusion_badge_progress"


def resolve_match_exclusion_badge_ids(
    supabase: Any,
    *,
    club_id: str,
    match_limit: int = 5000,
) -> list[str]:
    """Resolve and freeze the exact badge scope before a match exclusion."""

    clean_club_id = _required_text(club_id, "club_id")
    _assert_frozen_code_contract()
    ctx = _load_reconciliation_context(
        supabase,
        club_id=clean_club_id,
        match_limit=match_limit,
    )
    return list(_effective_catalog_badge_ids(ctx))


def build_match_exclusion_badge_plan(
    *,
    ctx: Any,
    club_id: str,
    player_id: int,
    badge_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Build one player's exact desired engine-owned badge rows."""

    clean_club_id = _required_text(club_id, "club_id")
    clean_player_id = _positive_player_id(player_id)
    clean_badge_ids = _validate_frozen_badge_ids(badge_ids)
    _assert_context_is_canonical_safe(ctx, clean_club_id)
    evaluation = _canonical_evaluation_context(ctx, clean_club_id)
    return _build_player_plan(
        evaluation=evaluation,
        club_id=clean_club_id,
        player_id=clean_player_id,
        badge_ids=clean_badge_ids,
    )


def apply_match_exclusion_badge_plan(
    supabase: Any,
    *,
    operation_id: str,
    progress_id: str,
    club_id: str,
    lease_token: str,
    worker_id: str,
    actor_email: str,
    desired_badges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically apply one claimed player's exact desired badge plan."""

    response = supabase.rpc(
        MATCH_EXCLUSION_BADGE_APPLY_RPC,
        {
            "p_operation_id": _required_text(operation_id, "operation_id"),
            "p_progress_id": _required_text(progress_id, "progress_id"),
            "p_club_id": _required_text(club_id, "club_id"),
            "p_lease_token": _required_text(lease_token, "lease_token"),
            "p_worker_id": _required_text(worker_id, "worker_id"),
            "p_desired_badges": [dict(row) for row in desired_badges],
            "p_actor_email": str(actor_email or "").strip(),
        },
    ).execute()
    result = _rpc_payload(response, "badge reconciliation apply")
    if not bool(result.get("ok")) or str(result.get("status") or "") != "succeeded":
        raise RuntimeError(
            "Atomic badge reconciliation did not report a succeeded progress row."
        )
    if str(result.get("operation_id") or "") != str(operation_id):
        raise RuntimeError("Atomic badge reconciliation returned the wrong operation.")
    if str(result.get("progress_id") or "") != str(progress_id):
        raise RuntimeError("Atomic badge reconciliation returned the wrong progress row.")
    if int(result.get("desired_count") or 0) != len(desired_badges):
        raise RuntimeError("Atomic badge reconciliation returned an incomplete desired-row count.")
    return result


def reconcile_match_exclusion_badges(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    player_ids: list[int],
    actor_email: str = "",
    match_limit: int = 5000,
) -> dict[str, Any]:
    """Claim and reconcile every pending player for one exclusion operation."""

    clean_club_id = _required_text(club_id, "club_id")
    clean_operation_id = _required_text(operation_id, "operation_id")
    expected_player_ids = sorted({_positive_player_id(value) for value in player_ids})
    if not expected_player_ids:
        raise ValueError("player_ids must contain at least one player")
    _assert_frozen_code_contract()

    worker_id = f"match-exclusion-badges:{uuid4()}"
    expected_player_set = set(expected_player_ids)
    processed_player_ids: list[int] = []
    operation_badge_ids: tuple[str, ...] | None = None
    context: Any | None = None
    evaluation: BadgeEvaluationContext | None = None
    inserted_count = 0
    updated_count = 0
    revoked_count = 0
    unchanged_count = 0
    desired_count = 0
    all_idempotent = True

    while len(processed_player_ids) < len(expected_player_ids):
        claim_response = supabase.rpc(
            MATCH_EXCLUSION_BADGE_CLAIM_RPC,
            {
                "p_operation_id": clean_operation_id,
                "p_club_id": clean_club_id,
                "p_worker_id": worker_id,
                "p_lease_seconds": 120,
                "p_retry_failed": True,
            },
        ).execute()
        claim = _rpc_payload(claim_response, "badge reconciliation claim")
        if not bool(claim.get("ok")):
            raise RuntimeError(
                str(claim.get("message") or claim.get("code") or "Badge reconciliation claim failed.")
            )
        if not bool(claim.get("claimed")):
            claim_contract_version = str(
                claim.get("badge_contract_version") or ""
            )
            if (
                claim_contract_version
                and claim_contract_version
                != MATCH_EXCLUSION_BADGE_CONTRACT_VERSION
            ):
                raise RuntimeError(
                    "Badge progress uses an unsupported reconciliation contract version."
                )
            no_claim_badge_ids = claim.get("badge_ids") or claim.get(
                "badge_allowlist"
            )
            if no_claim_badge_ids is not None:
                operation_badge_ids = _validate_frozen_badge_ids(
                    no_claim_badge_ids
                )
            break

        progress_id = _required_text(claim.get("progress_id"), "progress_id")
        lease_token = _required_text(claim.get("lease_token"), "lease_token")
        claimed_player_id = _positive_player_id(claim.get("player_id"))
        try:
            if str(claim.get("operation_id") or "") != clean_operation_id:
                raise RuntimeError("Badge progress claim returned the wrong operation.")
            if str(claim.get("club_id") or "") != clean_club_id:
                raise RuntimeError("Badge progress claim returned the wrong club.")
            if claimed_player_id not in expected_player_set:
                raise RuntimeError(
                    f"Badge progress claimed unexpected player {claimed_player_id}."
                )
            if claimed_player_id in processed_player_ids:
                raise RuntimeError(
                    f"Badge progress claimed player {claimed_player_id} more than once."
                )

            contract_version = str(claim.get("badge_contract_version") or "")
            if contract_version != MATCH_EXCLUSION_BADGE_CONTRACT_VERSION:
                raise RuntimeError(
                    "Badge progress uses an unsupported reconciliation contract version."
                )
            claimed_badge_ids = _validate_frozen_badge_ids(
                claim.get("badge_ids") or claim.get("badge_allowlist") or []
            )
            if operation_badge_ids is None:
                operation_badge_ids = claimed_badge_ids
            elif claimed_badge_ids != operation_badge_ids:
                raise RuntimeError(
                    "Badge progress rows disagree on the operation's frozen badge allowlist."
                )

            if context is None:
                context = _load_reconciliation_context(
                    supabase,
                    club_id=clean_club_id,
                    match_limit=match_limit,
                )
                evaluation = _canonical_evaluation_context(context, clean_club_id)
            if evaluation is None:
                raise RuntimeError("Canonical badge evaluation context is unavailable.")

            desired_badges = _build_player_plan(
                evaluation=evaluation,
                club_id=clean_club_id,
                player_id=claimed_player_id,
                badge_ids=claimed_badge_ids,
            )
            applied = apply_match_exclusion_badge_plan(
                supabase,
                operation_id=clean_operation_id,
                progress_id=progress_id,
                club_id=clean_club_id,
                lease_token=lease_token,
                worker_id=worker_id,
                actor_email=actor_email,
                desired_badges=desired_badges,
            )
        except Exception as exc:
            _fail_progress(
                supabase,
                operation_id=clean_operation_id,
                progress_id=progress_id,
                club_id=clean_club_id,
                lease_token=lease_token,
                worker_id=worker_id,
                error_text=str(exc),
            )
            raise

        current_desired_count = int(applied.get("desired_count") or 0)
        current_inserted_count = int(applied.get("inserted_count") or 0)
        current_updated_count = int(applied.get("updated_count") or 0)
        desired_count += current_desired_count
        inserted_count += current_inserted_count
        updated_count += current_updated_count
        revoked_count += int(applied.get("revoked_count") or 0)
        unchanged_count += max(
            0,
            current_desired_count - current_inserted_count - current_updated_count,
        )
        all_idempotent = all_idempotent and bool(applied.get("idempotent"))
        processed_player_ids.append(claimed_player_id)

    return {
        "ok": True,
        "operation_id": clean_operation_id,
        "club_id": clean_club_id,
        "contract_version": MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
        "player_ids": expected_player_ids,
        "processed_player_ids": sorted(processed_player_ids),
        "badge_ids": list(operation_badge_ids or ()),
        "desired_count": desired_count,
        "awarded_count": inserted_count,
        "inserted_count": inserted_count,
        "updated_count": updated_count,
        "revoked_count": revoked_count,
        "unchanged_count": unchanged_count,
        "idempotent": all_idempotent if processed_player_ids else True,
    }


def _load_reconciliation_context(
    supabase: Any,
    *,
    club_id: str,
    match_limit: int,
) -> Any:
    if int(match_limit) <= 0:
        raise ValueError("match_limit must be positive")
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, club_id, match_limit=int(match_limit))
    ctx = SimpleNamespace(
        supabase=supabase,
        club_id=club_id,
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_matches=df_matches,
        df_meta=df_meta,
        df_badges=df_badges,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        public_mode=False,
        admin_logged_in=True,
        schema_degraded=bool(schema_degraded),
        schema_degraded_reason=schema_degraded_reason,
    )
    _assert_context_is_canonical_safe(ctx, club_id)
    _assert_complete_active_match_load(
        supabase,
        club_id=club_id,
        df_matches=df_matches,
        match_limit=int(match_limit),
    )
    return ctx


def _assert_complete_active_match_load(
    supabase: Any,
    *,
    club_id: str,
    df_matches: pd.DataFrame,
    match_limit: int,
) -> None:
    response = (
        supabase.table("matches")
        .select("id", count="exact")
        .eq("club_id", club_id)
        .is_("deleted_at", None)
        .limit(1)
        .execute()
    )
    exact_count = getattr(response, "count", None)
    if exact_count is None:
        raise RuntimeError(
            "Badge reconciliation could not prove the active-match row count."
        )
    exact_count = int(exact_count)
    if exact_count > int(match_limit):
        raise RuntimeError(
            "Badge reconciliation active-match history exceeds match_limit; "
            "increase the bounded limit before any exclusion write."
        )
    if exact_count != len(df_matches.index):
        raise RuntimeError(
            "Badge reconciliation active-match context is incomplete or truncated "
            f"(loaded={len(df_matches.index)}, exact={exact_count})."
        )


def _assert_context_is_canonical_safe(ctx: Any, club_id: str) -> None:
    if str(getattr(ctx, "club_id", "") or "") != str(club_id):
        raise RuntimeError("Badge reconciliation context belongs to the wrong club.")
    if bool(getattr(ctx, "schema_degraded", False)):
        reason = str(getattr(ctx, "schema_degraded_reason", "") or "unknown schema gap")
        raise RuntimeError(
            f"Badge reconciliation requires the complete canonical schema: {reason}"
        )
    df_matches = getattr(ctx, "df_matches", None)
    if isinstance(df_matches, pd.DataFrame) and not df_matches.empty:
        if "deleted_at" not in df_matches.columns:
            raise RuntimeError(
                "Badge reconciliation requires matches.deleted_at in the canonical match load."
            )
        deleted = df_matches["deleted_at"]
        if bool((~(deleted.isna() | deleted.astype(str).str.strip().eq(""))).any()):
            raise RuntimeError(
                "Badge reconciliation canonical data unexpectedly contains soft-deleted matches."
            )


def _effective_catalog_badge_ids(ctx: Any) -> tuple[str, ...]:
    df_badges = getattr(ctx, "df_badges", None)
    if (
        not isinstance(df_badges, pd.DataFrame)
        or df_badges.empty
        or "badge_id" not in df_badges.columns
    ):
        raise RuntimeError("Badge reconciliation requires the current badge catalog.")
    rows_by_id: dict[str, Any] = {}
    for row in df_badges.itertuples(index=False):
        badge_id = str(getattr(row, "badge_id", "") or "").strip()
        if not badge_id:
            continue
        if badge_id in rows_by_id:
            raise RuntimeError(f"Badge catalog contains duplicate badge_id {badge_id}.")
        rows_by_id[badge_id] = row

    resolved: list[str] = []
    for badge_id in MATCH_EXCLUSION_BADGE_IDS:
        row = rows_by_id.get(badge_id)
        if row is None:
            continue
        if str(getattr(row, "state", "") or "").strip().lower() != "live":
            continue
        if not _catalog_row_is_active(getattr(row, "is_active", True)):
            continue
        if "match_updated" not in _normalize_triggers(
            getattr(row, "eval_triggers", None)
        ):
            continue
        if str(getattr(row, "badge_status", "live") or "").strip().lower() != "live":
            continue
        if (
            str(getattr(row, "badge_award_timing", "live") or "")
            .strip()
            .lower()
            != "live"
        ):
            continue
        resolved.append(badge_id)
    if not resolved:
        raise RuntimeError(
            "Badge reconciliation found no live match_updated badges to freeze."
        )
    return tuple(resolved)


def _canonical_evaluation_context(ctx: Any, club_id: str) -> BadgeEvaluationContext:
    evaluation = build_evaluation_context(
        ctx,
        club_id=club_id,
        league_id=None,
        as_of=None,
    )
    canonical_facts = evaluation.facts_canonical
    if canonical_facts is None:
        canonical_facts = evaluation.facts
    return replace(
        evaluation,
        facts=canonical_facts,
        matches=canonical_facts,
        facts_canonical=canonical_facts,
        facts_hybrid=canonical_facts,
    )


def _build_player_plan(
    *,
    evaluation: BadgeEvaluationContext,
    club_id: str,
    player_id: int,
    badge_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    specs = registry()
    candidates: list[BadgeCandidate] = []
    for badge_id in badge_ids:
        spec = specs.get(badge_id)
        if spec is None:
            raise RuntimeError(
                f"Frozen badge {badge_id} no longer has a registered evaluator."
            )
        for candidate in spec.evaluator(evaluation):
            if int(candidate.player_id) == int(player_id):
                candidates.append(candidate)

    desired_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        row = _candidate_plan_row(
            candidate,
            club_id=club_id,
            player_id=player_id,
            badge_ids=set(badge_ids),
        )
        key = (
            player_id,
            row["badge_id"],
            row["context_type"],
            row["context_id"],
        )
        if key in desired_by_key:
            raise RuntimeError(
                "Canonical badge evaluation produced duplicate "
                f"(player_id, badge_id, context_id) key {key}."
            )
        desired_by_key[key] = row
    return [
        desired_by_key[key]
        for key in sorted(
            desired_by_key,
            key=lambda item: (item[1], item[2], item[3]),
        )
    ]


def _candidate_plan_row(
    candidate: BadgeCandidate,
    *,
    club_id: str,
    player_id: int,
    badge_ids: set[str],
) -> dict[str, Any]:
    if str(candidate.club_id) != str(club_id):
        raise RuntimeError("Badge evaluator returned a candidate for the wrong club.")
    if int(candidate.player_id) != int(player_id):
        raise RuntimeError("Badge evaluator returned a candidate for the wrong player.")
    badge_id = str(candidate.badge_id or "").strip()
    if badge_id not in badge_ids:
        raise RuntimeError("Badge evaluator returned a candidate outside the frozen allowlist.")
    context_type = str(candidate.context_type or "").strip()
    context_id = str(candidate.context_id or "").strip()
    if not context_type or not context_id:
        raise RuntimeError(
            f"Badge {badge_id} produced an incomplete canonical context key."
        )
    projected = build_player_badge_rows(
        club_id,
        [candidate],
        awarded_by="engine",
        rule_version=MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
    )[0]
    row = {
        field: _json_safe(projected.get(field))
        for field in (
            "badge_id",
            "context_type",
            "context_id",
            "match_id",
            "value_num",
            "value_json",
            "rule_version",
        )
    }
    json.dumps(row, sort_keys=True, allow_nan=False)
    return row


def _validate_frozen_badge_ids(values: Iterable[Any]) -> tuple[str, ...]:
    requested = [str(value or "").strip() for value in values]
    if any(not value for value in requested):
        raise RuntimeError("Frozen badge allowlist contains a blank badge ID.")
    if len(requested) != len(set(requested)):
        raise RuntimeError("Frozen badge allowlist contains duplicate badge IDs.")
    unexpected = sorted(set(requested) - set(MATCH_EXCLUSION_BADGE_IDS))
    if unexpected:
        raise RuntimeError(
            "Frozen badge allowlist exceeds the supported v1 contract: "
            + ", ".join(unexpected)
        )
    requested_set = set(requested)
    return tuple(
        badge_id for badge_id in MATCH_EXCLUSION_BADGE_IDS if badge_id in requested_set
    )


def _assert_frozen_code_contract() -> None:
    code_ids = set(active_badge_ids()) & set(registry())
    frozen_ids = set(MATCH_EXCLUSION_BADGE_IDS)
    if code_ids != frozen_ids:
        missing = sorted(code_ids - frozen_ids)
        retired = sorted(frozen_ids - code_ids)
        raise RuntimeError(
            "Match-exclusion badge contract drifted; publish a new version before "
            f"writes. missing={missing}, no_longer_live={retired}"
        )


def _fail_progress(
    supabase: Any,
    *,
    operation_id: str,
    progress_id: str,
    club_id: str,
    lease_token: str,
    worker_id: str,
    error_text: str,
) -> None:
    try:
        response = supabase.rpc(
            MATCH_EXCLUSION_BADGE_FAIL_RPC,
            {
                "p_operation_id": operation_id,
                "p_progress_id": progress_id,
                "p_club_id": club_id,
                "p_lease_token": lease_token,
                "p_worker_id": worker_id,
                "p_error_text": str(error_text or "badge reconciliation failed")[:2000],
            },
        ).execute()
        result = _rpc_payload(response, "badge reconciliation failure")
        if not bool(result.get("ok")):
            raise RuntimeError("Badge progress failure RPC did not acknowledge the error.")
    except Exception as fail_exc:
        raise RuntimeError(
            "Badge reconciliation failed and its progress row could not be marked failed."
        ) from fail_exc


def _rpc_payload(response: Any, label: str) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    if isinstance(data, str):
        try:
            decoded = json.loads(data)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            return decoded
    raise RuntimeError(f"Atomic {label} RPC returned no JSON object.")


def _normalize_triggers(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except Exception:
            decoded = [value]
        value = decoded
    if isinstance(value, (tuple, list, set)):
        return {
            str(item or "").strip()
            for item in value
            if str(item or "").strip()
        }
    return set()


def _catalog_row_is_active(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _required_text(value: Any, field: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field} is required")
    return clean


def _positive_player_id(value: Any) -> int:
    try:
        player_id = int(value)
    except Exception as exc:
        raise ValueError("player_ids must contain integer identities") from exc
    if player_id <= 0:
        raise ValueError("player_ids must contain positive integer identities")
    return player_id


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, str):
        return value
    return str(value)
