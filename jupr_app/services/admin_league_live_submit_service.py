from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.league_live_publish import (
    LeagueLivePublishError,
    build_league_live_publish_request,
    build_rating_review,
    rows_to_safe_csv,
    stable_payload_fingerprint,
)
from jupr_app.domain.player_ops import safe_add_player
from jupr_app.services.admin_league_live_service import (
    LeagueLiveConflictError,
    LeagueLivePersistenceError,
    _as_dict,
    _as_list,
    _clean_text,
    _fetch_session_row,
    _first_row,
    _league_match_format,
    _league_match_structure,
    _safe_int,
    _safe_rows,
    build_admin_league_live_round_plan,
    ensure_admin_league_live_submit_enabled,
    get_admin_league_live_session,
    save_admin_league_live_round,
)
from jupr_app.services.admin_match_uploader_service import submit_admin_match_uploader_batch


CONFIRM_SUBMIT_ROUND = "SUBMIT LEAGUE ROUND"
CONFIRM_RECONCILE_ROUND = "RECONCILE LEAGUE ROUND"
CONFIRM_COMPENSATE_ROUND = "VERIFY LEAGUE COMPENSATION"
CONFIRM_CREATE_GUEST = "CREATE LIVE GUEST"
IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _idempotency_key(value: Any) -> str:
    clean = str(value or "").strip()
    if not IDEMPOTENCY_PATTERN.fullmatch(clean):
        raise ValueError(
            "idempotency_key must be 8-160 characters using letters, numbers, dot, underscore, colon, or hyphen."
        )
    return clean


def _required_audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    before_json: dict[str, Any] | None = None,
    after_json: dict[str, Any] | None = None,
    source: str,
    after_domain_mutation: bool = False,
) -> None:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=str(action_type),
        entity_type=str(entity_type),
        entity_id=str(entity_id),
        before_json=before_json or {},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, **(after_json or {})},
        source_page=source,
        flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok:
        if after_domain_mutation:
            raise LeagueLivePersistenceError(
                "League Live completion audit is unavailable after a domain mutation. The operation may have "
                "completed; reload its durable status and reconcile. Do not blindly republish."
            )
        raise LeagueLivePersistenceError(
            "League Live requires a durable audit intent before the requested write; no new match or session mutation was attempted by this request."
        )


def ensure_admin_league_live_publish_schema_ready(supabase: Any) -> None:
    """Read-only PostgREST preflight; this must run before audit or domain mutation."""
    checks = (
        ("matches", "id,club_id,context_type,context_id,deleted_at"),
        ("league_live_rounds", "id,operation_key"),
        (
            "league_live_publish_operations",
            "id,status,idempotency_key,request_fingerprint,match_context_ids,published_match_ids,completion_audited_at",
        ),
        ("league_live_guest_players", "id,status,idempotency_key,request_fingerprint,player_id"),
    )
    for table_name, columns in checks:
        try:
            supabase.table(table_name).select(columns).limit(1).execute()
        except Exception as exc:
            raise LeagueLivePersistenceError(
                "League Live submit schema is not ready. Apply the canonical order-20 domain migration and "
                "order-21 publish reconciliation migration to staging before enabling writes."
            ) from exc


def _player_ids(matches: list[dict[str, Any]]) -> list[int]:
    return sorted(
        {
            int(value)
            for row in matches
            for value in (row.get("t1_p1"), row.get("t1_p2"), row.get("t2_p1"), row.get("t2_p2"))
            if _safe_int(value) is not None
        }
    )


def _fetch_players(supabase: Any, *, club_id: str, player_ids: list[int]) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    try:
        return _safe_rows(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .in_("id", player_ids)
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to read back League Live player ratings.") from exc


def _fetch_publish_operation(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    idempotency_key: str | None = None,
) -> dict[str, Any] | None:
    try:
        query = (
            supabase.table("league_live_publish_operations")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .eq("round_number", int(round_number))
        )
        if idempotency_key:
            query = query.eq("idempotency_key", str(idempotency_key))
        return _first_row(query.limit(1).execute())
    except Exception as exc:
        raise LeagueLivePersistenceError(
            "League Live publish coordination storage is unavailable; no matches were published."
        ) from exc


def _update_publish_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    payload = {**patch, "updated_at": _now_iso()}
    try:
        updated = _first_row(
            supabase.table("league_live_publish_operations")
            .update(payload)
            .eq("club_id", str(club_id))
            .eq("id", str(operation_id))
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to persist League Live publish recovery state.") from exc
    if updated is None:
        raise LeagueLivePersistenceError("League Live publish recovery state did not persist.")
    return updated


def _published_matches(
    supabase: Any,
    *,
    club_id: str,
    context_ids: list[str],
) -> list[dict[str, Any]]:
    if not context_ids:
        return []
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("context_type", "league_live_session")
            .in_("context_id", context_ids)
            .is_("deleted_at", None)
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to verify published League Live match rows.") from exc
    by_context = {str(row.get("context_id") or ""): row for row in rows}
    return [by_context[context_id] for context_id in context_ids if context_id in by_context]


def _operation_public_payload(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(operation.get("id") or ""),
        "session_id": str(operation.get("session_id") or ""),
        "round_number": _safe_int(operation.get("round_number"), 1) or 1,
        "idempotency_key": str(operation.get("idempotency_key") or ""),
        "request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "plan_operation_key": str(operation.get("plan_operation_key") or ""),
        "status": str(operation.get("status") or "intent"),
        "match_context_ids": _as_list(operation.get("match_context_ids")),
        "published_match_ids": _as_list(operation.get("published_match_ids")),
        "attempt_count": _safe_int(operation.get("attempt_count"), 0) or 0,
        "error_text": operation.get("error_text"),
        "completed_at": operation.get("completed_at"),
        "completion_audited_at": operation.get("completion_audited_at"),
        "created_at": operation.get("created_at"),
        "updated_at": operation.get("updated_at"),
    }


def list_admin_league_live_publish_operations(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    try:
        rows = _safe_rows(
            supabase.table("league_live_publish_operations")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to load League Live publish history.") from exc
    return sorted(
        (_operation_public_payload(row) for row in rows),
        key=lambda row: int(row.get("round_number") or 0),
    )


def _completed_response(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    operation: dict[str, Any],
    idempotent_replay: bool,
) -> dict[str, Any]:
    detail = get_admin_league_live_session(supabase, club_id=str(club_id), session_id=str(session_id))
    request = _as_dict(operation.get("request_json"))
    round_number = int(operation.get("round_number") or 1)
    saved_round = next(
        (row for row in detail.get("rounds", []) if int(row.get("round_number") or 0) == round_number),
        None,
    )
    rating_review = _as_dict(_as_dict(operation.get("result_json")).get("rating_review"))
    return {
        "ok": True,
        "mode": "league_live_round_publish",
        "idempotent_replay": bool(idempotent_replay),
        "operation_key": str(operation.get("plan_operation_key") or request.get("expected_operation_key") or ""),
        "publish_operation": _operation_public_payload(operation),
        "session": detail["session"],
        "round": saved_round,
        "rounds": detail.get("rounds", []),
        "courts": detail.get("courts", []),
        "rating_review": rating_review,
        "published_match_ids": _as_list(operation.get("published_match_ids")),
        "recovery": {
            "session_detail": f"/admin/clubs/{{club_id}}/league-manager/live-sessions/{session_id}",
            "match_log": "/admin/match-log",
            "replay_history": "/admin/replay-history",
        },
        "warnings": list(rating_review.get("warnings") or []),
    }


def _complete_audit_if_needed(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    operation: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if operation.get("completion_audited_at"):
        return operation
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="complete_league_live_round_publish_admin",
        entity_type="league_live_publish_operation",
        entity_id=str(operation.get("id") or session_id),
        after_json={
            "session_id": str(session_id),
            "round_number": int(operation.get("round_number") or 1),
            "published_match_ids": _as_list(operation.get("published_match_ids")),
            "request_fingerprint": str(operation.get("request_fingerprint") or ""),
            "audit_marker": (
                f"league-live-publish-complete:{operation.get('id')}:"
                f"{operation.get('request_fingerprint')}"
            ),
        },
        source=source,
        after_domain_mutation=True,
    )
    try:
        return _update_publish_operation(
            supabase,
            club_id=str(club_id),
            operation_id=str(operation.get("id")),
            patch={"completion_audited_at": _now_iso(), "updated_by": str(actor_email or "")},
        )
    except LeagueLivePersistenceError as exc:
        raise LeagueLivePersistenceError(
            "League Live may have completed and its completion audit may already exist, but the durable audit marker "
            "did not persist. Reload and retry reconciliation; duplicate audits share the operation ID/fingerprint. "
            "Do not republish matches."
        ) from exc


def _finish_round_state(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    operation: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    reconciliation: bool,
) -> dict[str, Any]:
    request = _as_dict(operation.get("request_json"))
    context_ids = [str(value) for value in _as_list(operation.get("match_context_ids")) if str(value)]
    published = _published_matches(supabase, club_id=str(club_id), context_ids=context_ids)
    if len(published) != len(context_ids):
        updated = _update_publish_operation(
            supabase,
            club_id=str(club_id),
            operation_id=str(operation.get("id")),
            patch={
                "status": "recovery_required",
                "error_text": f"Only {len(published)} of {len(context_ids)} deterministic match contexts were verified.",
                "published_match_ids": [row.get("id") for row in published],
                "updated_by": str(actor_email or ""),
            },
        )
        raise LeagueLivePersistenceError(
            f"League Live publish requires recovery: {updated.get('error_text')} Stop and inspect Match Log/Replay History."
        )

    operation = _update_publish_operation(
        supabase,
        club_id=str(club_id),
        operation_id=str(operation.get("id")),
        patch={
            "status": "reconciling" if reconciliation else "published",
            "published_match_ids": [row.get("id") for row in published],
            "error_text": None,
            "updated_by": str(actor_email or ""),
        },
    )
    try:
        saved = save_admin_league_live_round(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id),
            round_number=int(operation.get("round_number") or request.get("round_number") or 1),
            round_label=request.get("round_label"),
            match_date=request.get("match_date"),
            preview=_as_dict(request.get("preview")),
            matches=_as_list(request.get("matches")),
            movement_overrides=_as_list(request.get("movement_overrides")),
            override_reason=request.get("override_reason"),
            roster_change=_as_dict(request.get("roster_change")) or None,
            bench_player_ids=_as_list(request.get("bench_player_ids")),
            bench_override_reason=request.get("bench_override_reason"),
            expected_updated_at=str(request.get("expected_updated_at") or ""),
            expected_operation_key=str(request.get("expected_operation_key") or ""),
            submitted_match_count=len(published),
            submitted_match_ids=[row.get("id") for row in published],
            courts=_as_list(request.get("courts")),
            advance_after_save=True,
            actor_email=actor_email,
            actor_role=actor_role,
            confirmation_text="SAVE ROUND",
            source="next_league_live_round_reconcile" if reconciliation else "next_league_live_round_publish",
        )
    except Exception as exc:
        _update_publish_operation(
            supabase,
            club_id=str(club_id),
            operation_id=str(operation.get("id")),
            patch={
                "status": "published",
                "error_text": f"Official matches exist but League Live state is not reconciled: {exc}",
                "updated_by": str(actor_email or ""),
            },
        )
        raise LeagueLivePersistenceError(
            "Official matches were published, but League Live state was not reconciled. Do not publish again; use Reconcile round or Match Log/Replay History."
        ) from exc

    expected_player_ids = _player_ids(_as_list(request.get("matches")))
    before_rows = _as_list(operation.get("rating_before_json"))
    after_rows = _fetch_players(supabase, club_id=str(club_id), player_ids=expected_player_ids)
    rating_review = build_rating_review(
        before_rows=before_rows,
        after_rows=after_rows,
        expected_player_ids=expected_player_ids,
        published_match_count=len(published),
    )
    result_json = {
        "round_id": _as_dict(saved.get("round")).get("id"),
        "session_updated_at": _as_dict(saved.get("session")).get("updated_at"),
        "rating_review": rating_review,
        "reconciled": bool(reconciliation),
    }
    operation = _update_publish_operation(
        supabase,
        club_id=str(club_id),
        operation_id=str(operation.get("id")),
        patch={
            "status": "completed",
            "rating_after_json": after_rows,
            "result_json": result_json,
            "completed_at": _now_iso(),
            "error_text": None,
            "updated_by": str(actor_email or ""),
        },
    )
    operation = _complete_audit_if_needed(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        operation=operation,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
    )
    return _completed_response(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        operation=operation,
        idempotent_replay=bool(reconciliation or saved.get("idempotent_replay")),
    )


def submit_admin_league_live_round_publish(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    matches: list[dict[str, Any]],
    expected_match_count: int,
    expected_updated_at: str,
    expected_operation_key: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    round_label: str | None = None,
    match_date: str | None = None,
    preview: dict[str, Any] | None = None,
    courts: list[dict[str, Any]] | None = None,
    movement_overrides: list[dict[str, Any]] | None = None,
    override_reason: str | None = None,
    roster_change: dict[str, Any] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    source: str = "next_league_live_round_submit",
) -> dict[str, Any]:
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SUBMIT_ROUND:
        raise ValueError(f"Type {CONFIRM_SUBMIT_ROUND} to publish every scored match in this round.")
    key = _idempotency_key(idempotency_key)
    session = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if session is None:
        raise ValueError("league live session not found")
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    operation = _fetch_publish_operation(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=safe_round,
    )
    if safe_round != (_safe_int(session.get("current_round"), 1) or 1):
        recoverable_states = {"completed", "published", "reconciling", "recovery_required"}
        if not operation or str(operation.get("status")) not in recoverable_states:
            raise LeagueLiveConflictError("This round is no longer current. Reload before publishing.")
    match_structure = _league_match_structure(
        supabase,
        club_id=str(club_id),
        league_name=str(session.get("league_name") or ""),
    )
    match_format = _league_match_format(
        supabase,
        club_id=str(club_id),
        league_name=str(session.get("league_name") or ""),
    )
    request = build_league_live_publish_request(
        session_id=str(session_id),
        round_number=safe_round,
        league_name=str(session.get("league_name") or ""),
        week_tag=str(session.get("week_tag") or ""),
        match_date=str(match_date or ""),
        matches=list(matches or []),
        expected_match_count=int(expected_match_count),
        expected_updated_at=str(expected_updated_at),
        expected_operation_key=str(expected_operation_key),
        round_label=round_label,
        preview=preview,
        courts=courts,
        movement_overrides=movement_overrides,
        override_reason=override_reason,
        roster_change=roster_change,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
        match_structure=match_structure,
    )
    if operation:
        if str(operation.get("idempotency_key")) != key:
            raise LeagueLiveConflictError("This round already has a different publish idempotency key.")
        if str(operation.get("request_fingerprint")) != str(request["request_fingerprint"]):
            raise LeagueLiveConflictError("The publish idempotency key was reused with a different round request.")
        if str(operation.get("status")) == "completed":
            operation = _complete_audit_if_needed(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                operation=operation,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
            return _completed_response(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                operation=operation,
                idempotent_replay=True,
            )
        if str(operation.get("status")) == "compensated":
            raise LeagueLiveConflictError(
                "This publish operation was compensated. Start a new session or round instead of reusing its key."
            )
        if str(operation.get("status")) in {"published", "reconciling", "recovery_required"}:
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="reconcile_league_live_round_publish_intent_admin",
                entity_type="league_live_publish_operation",
                entity_id=str(operation.get("id")),
                before_json={"status": operation.get("status")},
                after_json={"request_fingerprint": request["request_fingerprint"]},
                source=source,
            )
            return _finish_round_state(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                operation=operation,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                reconciliation=True,
            )
    if str(session.get("updated_at") or "") != str(request["expected_updated_at"]):
        raise LeagueLiveConflictError("League Live session changed. Reload and preview Python movement before publishing.")
    verified_plan = build_admin_league_live_round_plan(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=safe_round,
        expected_updated_at=str(request["expected_updated_at"]),
        matches=_as_list(request.get("matches")),
        courts=_as_list(request.get("courts")),
        movement_overrides=_as_list(request.get("movement_overrides")),
        override_reason=request.get("override_reason"),
        roster_change=_as_dict(request.get("roster_change")) or None,
        bench_player_ids=_as_list(request.get("bench_player_ids")),
        bench_override_reason=request.get("bench_override_reason"),
    )
    if str(verified_plan.get("operation_key") or "") != str(request["expected_operation_key"]):
        raise LeagueLiveConflictError("League Live plan changed. Preview Python movement again before publishing.")

    if operation is None:
        operation_id = str(uuid4())
        _required_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="submit_league_live_round_intent_admin",
            entity_type="league_live_publish_operation",
            entity_id=operation_id,
            before_json={"session_updated_at": session.get("updated_at")},
            after_json={
                "round_number": safe_round,
                "expected_match_count": int(expected_match_count),
                "idempotency_key_fingerprint": stable_payload_fingerprint(key),
                "request_fingerprint": request["request_fingerprint"],
                "plan_operation_key": request["expected_operation_key"],
                "audit_marker": f"league-live-publish-intent:{operation_id}:{request['request_fingerprint']}",
            },
            source=source,
        )
        player_ids = _player_ids(request["matches"])
        rating_before = _fetch_players(supabase, club_id=str(club_id), player_ids=player_ids)
        now = _now_iso()
        payload = {
            "id": operation_id,
            "club_id": str(club_id),
            "session_id": str(session_id),
            "round_number": safe_round,
            "idempotency_key": key,
            "request_fingerprint": request["request_fingerprint"],
            "plan_operation_key": request["expected_operation_key"],
            "expected_session_updated_at": request["expected_updated_at"],
            "status": "intent",
            "request_json": request,
            "match_context_ids": request["match_context_ids"],
            "published_match_ids": [],
            "rating_before_json": rating_before,
            "rating_after_json": [],
            "result_json": {},
            "attempt_count": 0,
            "created_by": str(actor_email or ""),
            "updated_by": str(actor_email or ""),
            "created_at": now,
            "updated_at": now,
        }
        try:
            operation = _first_row(supabase.table("league_live_publish_operations").insert(payload).execute())
        except Exception as exc:
            raise LeagueLivePersistenceError(
                "Unable to persist the League Live publish intent; no matches were published."
            ) from exc
        if operation is None:
            raise LeagueLivePersistenceError("League Live publish intent did not persist; no matches were published.")
    else:
        verified_contexts = _published_matches(
            supabase,
            club_id=str(club_id),
            context_ids=request["match_context_ids"],
        )
        if len(verified_contexts) == len(request["match_context_ids"]):
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="recover_league_live_round_publish_response_loss_admin",
                entity_type="league_live_publish_operation",
                entity_id=str(operation.get("id")),
                before_json={"status": operation.get("status"), "attempt_count": operation.get("attempt_count")},
                after_json={
                    "request_fingerprint": request["request_fingerprint"],
                    "verified_match_context_count": len(verified_contexts),
                    "audit_marker": (
                        f"league-live-publish-response-loss:{operation.get('id')}:"
                        f"{request['request_fingerprint']}"
                    ),
                },
                source=source,
            )
            return _finish_round_state(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                operation=operation,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                reconciliation=True,
            )
        if verified_contexts:
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="detect_partial_league_live_round_publish_admin",
                entity_type="league_live_publish_operation",
                entity_id=str(operation.get("id")),
                before_json={"status": operation.get("status"), "attempt_count": operation.get("attempt_count")},
                after_json={
                    "request_fingerprint": request["request_fingerprint"],
                    "verified_match_context_count": len(verified_contexts),
                    "expected_match_context_count": len(request["match_context_ids"]),
                    "audit_marker": (
                        f"league-live-publish-partial:{operation.get('id')}:"
                        f"{request['request_fingerprint']}"
                    ),
                },
                source=source,
            )
            _update_publish_operation(
                supabase,
                club_id=str(club_id),
                operation_id=str(operation.get("id")),
                patch={
                    "status": "recovery_required",
                    "published_match_ids": [row.get("id") for row in verified_contexts],
                    "error_text": (
                        f"Only {len(verified_contexts)} of {len(request['match_context_ids'])} deterministic "
                        "match contexts were verified before retry."
                    ),
                    "updated_by": str(actor_email or ""),
                },
            )
            raise LeagueLivePersistenceError(
                "League Live detected a partial prior publish before retry. No match was republished; inspect Match Log/Replay History."
            )
        _required_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="retry_league_live_round_publish_intent_admin",
            entity_type="league_live_publish_operation",
            entity_id=str(operation.get("id")),
            before_json={"status": operation.get("status"), "attempt_count": operation.get("attempt_count")},
            after_json={
                "request_fingerprint": request["request_fingerprint"],
                "audit_marker": (
                    f"league-live-publish-retry:{operation.get('id')}:"
                    f"{request['request_fingerprint']}"
                ),
            },
            source=source,
        )

    if operation is None:
        raise LeagueLivePersistenceError("League Live publish operation is unavailable.")
    operation = _update_publish_operation(
        supabase,
        club_id=str(club_id),
        operation_id=str(operation.get("id")),
        patch={
            "status": "publishing",
            "attempt_count": int(operation.get("attempt_count") or 0) + 1,
            "error_text": None,
            "updated_by": str(actor_email or ""),
        },
    )
    try:
        submit_admin_match_uploader_batch(
            supabase,
            club_id=str(club_id),
            matches=request["matches"],
            actor_email=actor_email,
            actor_role=actor_role,
            idempotency_key=f"league-live:{key}",
            match_format=match_format,
            source="next_league_live_all_match_publish",
        )
    except Exception as exc:
        published = _published_matches(
            supabase,
            club_id=str(club_id),
            context_ids=request["match_context_ids"],
        )
        if len(published) != len(request["match_context_ids"]):
            status = "recovery_required" if published else "retryable"
            _update_publish_operation(
                supabase,
                club_id=str(club_id),
                operation_id=str(operation.get("id")),
                patch={
                    "status": status,
                    "published_match_ids": [row.get("id") for row in published],
                    "error_text": str(exc)[:1000],
                    "updated_by": str(actor_email or ""),
                },
            )
            recovery = "Inspect Match Log/Replay History before retrying." if published else "Retry with the same idempotency key."
            raise LeagueLivePersistenceError(f"League Live match publish did not complete. {recovery}") from exc

    return _finish_round_state(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        operation=operation,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        reconciliation=False,
    )


def reconcile_admin_league_live_round_publish(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_round_reconcile",
) -> dict[str, Any]:
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_RECONCILE_ROUND:
        raise ValueError(f"Type {CONFIRM_RECONCILE_ROUND} to reconcile an interrupted publish.")
    operation = _fetch_publish_operation(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=max(1, int(round_number)),
    )
    if operation is None:
        raise ValueError("No durable publish operation exists for this round.")
    if str(operation.get("status")) == "completed":
        operation = _complete_audit_if_needed(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id),
            operation=operation,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
        )
        return _completed_response(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id),
            operation=operation,
            idempotent_replay=True,
        )
    if str(operation.get("status")) not in {"published", "reconciling", "recovery_required"}:
        raise LeagueLiveConflictError(
            "No complete set of official matches is available to reconcile. Retry the original publish with its idempotency key."
        )
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="reconcile_league_live_round_publish_intent_admin",
        entity_type="league_live_publish_operation",
        entity_id=str(operation.get("id")),
        before_json={"status": operation.get("status"), "error_text": operation.get("error_text")},
        after_json={
            "round_number": int(operation.get("round_number") or round_number),
            "request_fingerprint": str(operation.get("request_fingerprint") or ""),
            "audit_marker": (
                f"league-live-publish-reconcile:{operation.get('id')}:"
                f"{operation.get('request_fingerprint')}"
            ),
        },
        source=source,
    )
    return _finish_round_state(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        operation=operation,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        reconciliation=True,
    )


def compensate_admin_league_live_round_publish(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    recovery_reference: str,
    reason: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_round_compensate",
) -> dict[str, Any]:
    """Record operator-verified recovery after Match Log/Replay History removed every live context."""
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_COMPENSATE_ROUND:
        raise ValueError(f"Type {CONFIRM_COMPENSATE_ROUND} after Match Log and Replay History recovery.")
    reason_text = _clean_text(reason, limit=500)
    reference = _clean_text(recovery_reference, limit=240)
    if len(reason_text) < 10:
        raise ValueError("Compensation requires an operator reason of at least 10 characters.")
    if len(reference) < 8:
        raise ValueError("Compensation requires a Match Log or Replay History recovery reference.")
    operation = _fetch_publish_operation(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=max(1, int(round_number)),
    )
    if operation is None:
        raise ValueError("No durable publish operation exists for this round.")
    if str(operation.get("status")) == "compensated":
        return {
            "ok": True,
            "mode": "league_live_round_compensated",
            "idempotent_replay": True,
            "publish_operation": _operation_public_payload(operation),
            "warnings": [],
        }
    if str(operation.get("status")) not in {"published", "reconciling", "recovery_required"}:
        raise LeagueLiveConflictError(
            "Compensation is only available for an interrupted publish with verified official match recovery."
        )
    contexts = [str(value) for value in _as_list(operation.get("match_context_ids")) if str(value)]
    remaining = _published_matches(supabase, club_id=str(club_id), context_ids=contexts)
    if remaining:
        raise LeagueLiveConflictError(
            f"{len(remaining)} active League Live match context(s) remain. Finish Match Log and Replay History recovery first."
        )
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="compensate_league_live_round_publish_admin",
        entity_type="league_live_publish_operation",
        entity_id=str(operation.get("id")),
        before_json={"status": operation.get("status"), "published_match_ids": operation.get("published_match_ids")},
        after_json={
            "reason": reason_text,
            "recovery_reference": reference,
            "verified_active_contexts": 0,
            "request_fingerprint": str(operation.get("request_fingerprint") or ""),
            "audit_marker": (
                f"league-live-publish-compensate:{operation.get('id')}:"
                f"{operation.get('request_fingerprint')}"
            ),
        },
        source=source,
    )
    result = _as_dict(operation.get("result_json"))
    result["compensation"] = {
        "reason": reason_text,
        "recovery_reference": reference,
        "verified_at": _now_iso(),
        "verified_by": str(actor_email or ""),
    }
    operation = _update_publish_operation(
        supabase,
        club_id=str(club_id),
        operation_id=str(operation.get("id")),
        patch={
            "status": "compensated",
            "published_match_ids": [],
            "result_json": result,
            "error_text": None,
            "completed_at": _now_iso(),
            "updated_by": str(actor_email or ""),
        },
    )
    return {
        "ok": True,
        "mode": "league_live_round_compensated",
        "idempotent_replay": False,
        "publish_operation": _operation_public_payload(operation),
        "warnings": [],
    }


def create_admin_league_live_guest(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    guest_name: str,
    starting_jupr: float,
    reason: str,
    expected_updated_at: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_guest_create",
) -> dict[str, Any]:
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE_GUEST:
        raise ValueError(f"Type {CONFIRM_CREATE_GUEST} to create a guest player record.")
    key = _idempotency_key(idempotency_key)
    name = _clean_text(guest_name, limit=160)
    reason_text = _clean_text(reason, limit=500)
    if len(name) < 2:
        raise ValueError("Guest name must contain at least two characters.")
    if len(reason_text) < 10:
        raise ValueError("Guest creation requires an operator reason of at least 10 characters.")
    try:
        jupr = float(starting_jupr)
    except Exception as exc:
        raise ValueError("Guest starting JUPR must be between 1.0 and 7.0.") from exc
    if jupr < 1.0 or jupr > 7.0:
        raise ValueError("Guest starting JUPR must be between 1.0 and 7.0.")
    session = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if session is None:
        raise ValueError("league live session not found")
    if str(session.get("updated_at") or "") != str(expected_updated_at or ""):
        raise LeagueLiveConflictError("League Live session changed. Reload before creating a guest.")
    fingerprint = stable_payload_fingerprint(
        {"session_id": str(session_id), "guest_name": name, "starting_jupr": round(jupr, 2), "reason": reason_text}
    )
    existing = _first_row(
        supabase.table("league_live_guest_players")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("idempotency_key", key)
        .limit(1)
        .execute()
    )
    if existing:
        if str(existing.get("request_fingerprint")) != fingerprint:
            raise LeagueLiveConflictError("Guest idempotency key was reused with different values.")
        player_id = _safe_int(existing.get("player_id"))
        if str(existing.get("status")) == "completed" and player_id is not None:
            player = _first_row(
                supabase.table("players")
                .select("*")
                .eq("club_id", str(club_id))
                .eq("id", int(player_id))
                .limit(1)
                .execute()
            )
            if player:
                return {"ok": True, "mode": "league_live_guest_create", "idempotent_replay": True, "player": _guest_player_payload(player), "guest_operation_id": str(existing.get("id")), "warnings": []}
    else:
        duplicate_player = _first_row(
            supabase.table("players")
            .select("id,name")
            .eq("club_id", str(club_id))
            .eq("name", name)
            .limit(1)
            .execute()
        )
        if duplicate_player is not None:
            raise LeagueLiveConflictError(
                "A club player already uses this name. Select that player for the substitution instead of creating a guest."
            )
        guest_operation_id = str(uuid4())
        _required_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="create_league_live_guest_intent_admin",
            entity_type="league_live_guest",
            entity_id=guest_operation_id,
            after_json={
                "session_id": str(session_id),
                "guest_name": name,
                "starting_jupr": jupr,
                "reason": reason_text,
                "idempotency_key_fingerprint": stable_payload_fingerprint(key),
                "request_fingerprint": fingerprint,
                "audit_marker": f"league-live-guest-intent:{guest_operation_id}:{fingerprint}",
            },
            source=source,
        )
        now = _now_iso()
        existing = _first_row(
            supabase.table("league_live_guest_players")
            .insert(
                {
                    "id": guest_operation_id,
                    "club_id": str(club_id),
                    "session_id": str(session_id),
                    "player_id": None,
                    "idempotency_key": key,
                    "request_fingerprint": fingerprint,
                    "guest_name": name,
                    "starting_jupr": jupr,
                    "reason": reason_text,
                    "status": "intent",
                    "created_by": str(actor_email or ""),
                    "updated_by": str(actor_email or ""),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            .execute()
        )
        if existing is None:
            raise LeagueLivePersistenceError("Guest intent did not persist; no player was created.")

    player = _first_row(
        supabase.table("players")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("name", name)
        .limit(1)
        .execute()
    )
    if player is None:
        ok, error = safe_add_player(
            supabase=supabase,
            club_id=str(club_id),
            name=name,
            rating_jupr=jupr,
        )
        if not ok:
            _update_guest_recovery(supabase, club_id=str(club_id), operation_id=str(existing.get("id")), error_text=str(error), actor_email=actor_email)
            raise LeagueLivePersistenceError(f"Guest player creation requires recovery: {error}")
        player = _first_row(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("name", name)
            .limit(1)
            .execute()
        )
    if player is None or _safe_int(player.get("id")) is None:
        _update_guest_recovery(supabase, club_id=str(club_id), operation_id=str(existing.get("id")), error_text="Created guest could not be read back.", actor_email=actor_email)
        raise LeagueLivePersistenceError("Guest player may have been created but could not be read back. Use Player Editor before retrying.")
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="complete_league_live_guest_create_admin",
        entity_type="league_live_guest",
        entity_id=str(existing.get("id")),
        after_json={
            "session_id": str(session_id),
            "player_id": int(player["id"]),
            "reason": reason_text,
            "request_fingerprint": fingerprint,
            "audit_marker": f"league-live-guest-complete:{existing.get('id')}:{fingerprint}",
        },
        source=source,
        after_domain_mutation=True,
    )
    completed = _first_row(
        supabase.table("league_live_guest_players")
        .update(
            {
                "player_id": int(player["id"]),
                "status": "completed",
                "error_text": None,
                "completed_at": _now_iso(),
                "updated_at": _now_iso(),
                "updated_by": str(actor_email or ""),
            }
        )
        .eq("club_id", str(club_id))
        .eq("id", str(existing.get("id")))
        .execute()
    )
    if completed is None:
        raise LeagueLivePersistenceError("Guest was created but its recovery record did not finalize. Use Player Editor before retrying.")
    return {"ok": True, "mode": "league_live_guest_create", "idempotent_replay": False, "player": _guest_player_payload(player), "guest_operation_id": str(completed.get("id")), "warnings": []}


def _guest_player_payload(player: dict[str, Any]) -> dict[str, Any]:
    rating = player.get("rating")
    rating_jupr: float | None = None
    try:
        if rating is not None:
            rating_jupr = float(rating) / 400.0
    except Exception:
        rating_jupr = None
    return {
        "id": int(player["id"]),
        "name": str(player.get("name") or f"Player {player['id']}"),
        "rating": rating,
        "rating_jupr": rating_jupr,
    }


def _update_guest_recovery(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    error_text: str,
    actor_email: str,
) -> None:
    try:
        (
            supabase.table("league_live_guest_players")
            .update(
                {
                    "status": "recovery_required",
                    "error_text": str(error_text)[:1000],
                    "updated_at": _now_iso(),
                    "updated_by": str(actor_email or ""),
                }
            )
            .eq("club_id", str(club_id))
            .eq("id", str(operation_id))
            .execute()
        )
    except Exception:
        pass


def build_admin_league_live_export(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    export_kind: str,
) -> dict[str, Any]:
    ensure_admin_league_live_submit_enabled()
    ensure_admin_league_live_publish_schema_ready(supabase)
    detail = get_admin_league_live_session(supabase, club_id=str(club_id), session_id=str(session_id))
    kind = str(export_kind or "matches").strip().lower()
    operations = list_admin_league_live_publish_operations(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
    )
    if kind == "matches":
        contexts = [
            str(value)
            for operation in operations
            for value in _as_list(operation.get("match_context_ids"))
            if str(value)
        ]
        rows = _published_matches(supabase, club_id=str(club_id), context_ids=contexts)
    elif kind == "ratings":
        raw_operations = _safe_rows(
            supabase.table("league_live_publish_operations")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
        rows = []
        for operation in raw_operations:
            review = _as_dict(_as_dict(operation.get("result_json")).get("rating_review"))
            for row in _as_list(review.get("rows")):
                rows.append({"round_number": int(operation.get("round_number") or 0), **dict(row)})
    elif kind == "roster":
        rows = [dict(row) for row in _as_list(detail["session"].get("roster_json"))]
    elif kind == "rounds":
        rows = [dict(row) for row in detail.get("rounds", [])]
    else:
        raise ValueError("export kind must be matches, ratings, roster, or rounds")
    safe_league = re.sub(r"[^A-Za-z0-9._-]+", "-", str(detail["session"].get("league_name") or "league-live")).strip("-") or "league-live"
    return {
        "ok": True,
        "mode": "league_live_export",
        "kind": kind,
        "filename": f"{safe_league}-{kind}.csv",
        "content_type": "text/csv; charset=utf-8",
        "row_count": len(rows),
        "rows": rows,
        "csv_text": rows_to_safe_csv(rows),
    }
