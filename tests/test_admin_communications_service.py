from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jupr_app.domain.notifications.player_profile_update_repo import (
    StaleCommunicationsStateError,
    claim_communications_admin_operation,
    complete_communications_admin_operation,
    create_outbox_row,
    replace_verified_subscriber_atomic,
    retry_outbox_rows_guarded,
)
from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.domain.notifications import player_update_sender as sender
from jupr_app.services import admin_communications_service as communications
from jupr_app.services import admin_player_updates_service as updates
from tests.test_admin_match_log_service import FakeSupabase


def test_retry_uncertain_email_requires_stronger_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "list_outbox_rows",
        lambda *_args, **_kwargs: [{"id": "outbox-1", "send_status": "sending", "row_version": 3}],
    )
    monkeypatch.setattr(communications, "_audit", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        communications,
        "retry_outbox_rows_guarded",
        lambda *_args, **_kwargs: {"requested": 1, "reset_to_pending": 1, "stale": 0, "stale_ids": [], "rows": []},
    )

    with pytest.raises(ValueError, match="RETRY UNCERTAIN EMAILS"):
        communications.retry_outbox_rows(
            object(),
            club_id="club",
            items=[{"id": "outbox-1", "expected_row_version": 3}],
            confirmation_text="RETRY PLAYER UPDATES",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    result = communications.retry_outbox_rows(
        object(),
        club_id="club",
        items=[{"id": "outbox-1", "expected_row_version": 3}],
        confirmation_text="RETRY UNCERTAIN EMAILS",
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test",
    )
    assert result["reset_to_pending"] == 1


def test_stale_claim_prevents_selected_email_delivery(monkeypatch) -> None:
    outbox = {
        "id": "outbox-1",
        "subscription_id": "subscription-1",
        "week_start": "2026-07-01",
        "week_end": "2026-07-07",
        "send_status": "pending",
        "row_version": 4,
    }
    monkeypatch.setattr(sender, "list_outbox_rows", lambda *_args, **_kwargs: [outbox])
    monkeypatch.setattr(
        sender,
        "claim_outbox_row_for_send",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StaleCommunicationsStateError("stale")),
    )
    monkeypatch.setattr(sender, "get_email_mode", lambda: "dry_run")

    result = updates.send_pending_player_update_emails_for_range(
        SimpleNamespace(supabase=object(), club_id="club"),
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 7),
        outbox_items=[{"id": "outbox-1", "expected_row_version": 3}],
        actor_email="admin@example.com",
    )

    assert result == {
        "attempted": 1,
        "sent": 0,
        "skipped": 0,
        "errors": 0,
        "stale": 1,
        "uncertain": 0,
        "email_mode": "dry_run",
    }


def test_next_live_delivery_requires_dedicated_gate(monkeypatch) -> None:
    monkeypatch.setattr(sender, "list_outbox_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(sender, "get_email_mode", lambda: "live")
    monkeypatch.delenv("JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL", raising=False)

    with pytest.raises(PermissionError, match="Live Player Updates email is disabled"):
        updates.send_pending_player_update_emails_for_range(
            SimpleNamespace(supabase=object(), club_id="club"),
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 7),
        )


def test_post_smtp_finalize_failure_stays_uncertain_not_retryable_error(monkeypatch) -> None:
    outbox = {
        "id": "outbox-1",
        "subscription_id": "subscription-1",
        "player_id": 7,
        "email": "verified@example.com",
        "week_start": "2026-07-01",
        "week_end": "2026-07-07",
        "send_status": "pending",
        "row_version": 1,
    }
    status_calls: list[dict] = []
    monkeypatch.setenv("JUPR_STAGING_EMAIL_REDIRECT_TO", "sink@example.com")
    monkeypatch.setattr(sender, "list_outbox_rows", lambda *_args, **_kwargs: [outbox])
    monkeypatch.setattr(sender, "claim_outbox_row_for_send", lambda *_args, **_kwargs: {**outbox, "send_status": "sending", "row_version": 2, "delivery_attempt_id": "11111111-1111-1111-1111-111111111111"})
    monkeypatch.setattr(sender, "_safe_subscription", lambda *_args, **_kwargs: {"id": "subscription-1", "request_status": "active", "preferences_json": {"send_only_if_changed": False}})
    monkeypatch.setattr(sender, "_safe_digest_for_week", lambda *_args, **_kwargs: {"final_json": {"summary": {}, "links": {}}})
    monkeypatch.setattr(sender, "ensure_unsubscribe_token", lambda *_args, **_kwargs: "unsubscribe-token")
    monkeypatch.setattr(sender, "render_player_digest_chart_png", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sender, "build_player_update_email_subject", lambda *_args, **_kwargs: "Update")
    monkeypatch.setattr(sender, "build_player_update_email_html", lambda *_args, **_kwargs: "<p>Update</p>")
    monkeypatch.setattr(sender, "build_player_update_email_text", lambda *_args, **_kwargs: "Update")
    monkeypatch.setattr(sender, "get_email_mode", lambda: "staging_redirect")
    monkeypatch.setattr(sender, "send_email_with_inline_chart", lambda **_kwargs: "provider-message")

    def fail_finalize(*_args, **kwargs):
        status_calls.append(dict(kwargs))
        raise RuntimeError("database response lost after SMTP")

    monkeypatch.setattr(sender, "update_outbox_status", fail_finalize)
    result = updates.send_pending_player_update_emails_for_range(
        SimpleNamespace(supabase=object(), club_id="club"),
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 7),
        actor_email="admin@example.com",
    )

    assert result["uncertain"] == 1
    assert result["errors"] == 1
    assert len(status_calls) == 1
    assert status_calls[0]["send_status"] == "sent"


def test_queue_operation_key_is_idempotent() -> None:
    supabase = FakeSupabase({"player_profile_update_outbox": []})
    kwargs = {
        "subscription_id": "subscription-1",
        "club_id": "club",
        "player_id": 7,
        "week_start": date(2026, 7, 1),
        "week_end": date(2026, 7, 7),
        "email": "verified@example.com",
        "operation_key": "3c013b83-1c65-4a26-80df-444247438d01",
    }

    first = create_outbox_row(supabase, **kwargs)
    second = create_outbox_row(supabase, **kwargs)

    assert first == second
    assert len(supabase.tables["player_profile_update_outbox"]) == 1


def test_atomic_replacement_rpc_carries_version_and_normalized_email() -> None:
    calls: list[tuple[str, dict]] = []

    class RpcQuery:
        def execute(self):
            return SimpleNamespace(data={"id": "new-subscription", "request_status": "active", "row_version": 1})

    class RpcSupabase:
        def rpc(self, name, payload):
            calls.append((name, payload))
            return RpcQuery()

    result = replace_verified_subscriber_atomic(
        RpcSupabase(),
        club_id="club",
        old_subscription_id="old-subscription",
        new_email=" New.Verified@Example.COM ",
        new_request_note="confirmed by operator",
        verified_by="admin@example.com",
        admin_note="staging fixture",
        expected_row_version=8,
        operation_key="4d4f621a-bb22-438d-8bad-2707a5926859",
    )

    assert result["id"] == "new-subscription"
    assert calls[0][0] == "replace_verified_update_subscription"
    assert calls[0][1]["p_expected_row_version"] == 8
    assert calls[0][1]["p_new_email_normalized"] == "new.verified@example.com"


def test_atomic_replacement_rejects_invalid_email_before_rpc() -> None:
    class RpcSupabase:
        def rpc(self, *_args, **_kwargs):
            raise AssertionError("invalid email must not reach the replacement RPC")

    with pytest.raises(ValueError, match="valid email"):
        replace_verified_subscriber_atomic(
            RpcSupabase(),
            club_id="club",
            old_subscription_id="old-subscription",
            new_email="not-an-email",
            new_request_note=None,
            verified_by="admin@example.com",
            admin_note=None,
            expected_row_version=8,
            operation_key="4d4f621a-bb22-438d-8bad-2707a5926859",
        )


def test_sending_retry_requires_strong_confirmation_and_expired_lease() -> None:
    row = {
        "id": "outbox-lease",
        "club_id": "club",
        "send_status": "sending",
        "row_version": 3,
        "last_attempt_at": datetime.now(timezone.utc).isoformat(),
    }
    supabase = FakeSupabase({"player_profile_update_outbox": [row]})
    item = [{"id": "outbox-lease", "expected_row_version": 3}]

    with pytest.raises(ValueError, match="RETRY UNCERTAIN EMAILS"):
        retry_outbox_rows_guarded(supabase, club_id="club", items=item)
    with pytest.raises(ValueError, match="30-minute send lease"):
        retry_outbox_rows_guarded(supabase, club_id="club", items=item, allow_uncertain=True)

    supabase.tables["player_profile_update_outbox"][0]["last_attempt_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=31)
    ).isoformat()
    result = retry_outbox_rows_guarded(supabase, club_id="club", items=item, allow_uncertain=True)
    assert result["reset_to_pending"] == 1


def test_operation_key_fingerprint_replays_only_identical_request() -> None:
    supabase = FakeSupabase({"communications_admin_operations": []})
    key = "be8a077d-182e-4774-8d79-37ef345c7f5d"
    request = {"old_subscription_id": "old", "new_email_normalized": "safe@example.com", "actor_email": "admin@example.com"}
    claim_communications_admin_operation(
        supabase,
        club_id="club",
        operation_key=key,
        operation_type="replace_verified_subscriber",
        request_json=request,
    )
    complete_communications_admin_operation(
        supabase,
        club_id="club",
        operation_key=key,
        result_json={"subscription": {"id": "replacement"}},
    )
    replay = claim_communications_admin_operation(
        supabase,
        club_id="club",
        operation_key=key,
        operation_type="replace_verified_subscriber",
        request_json=request,
    )
    assert replay["status"] == "completed"
    assert replay["result_json"]["subscription"]["id"] == "replacement"
    with pytest.raises(ValueError, match="different communications request"):
        claim_communications_admin_operation(
            supabase,
            club_id="club",
            operation_key=key,
            operation_type="replace_verified_subscriber",
            request_json={**request, "actor_email": "other@example.com"},
        )
    assert len(supabase.tables["communications_admin_operations"]) == 1


def test_replacement_response_loss_replays_without_second_creation(monkeypatch) -> None:
    key = "d7fb969a-9986-43c9-a78f-412d33f42a86"
    supabase = FakeSupabase(
        {
            "communications_admin_operations": [],
            "player_profile_update_subscriptions": [
                {"id": "old", "club_id": "club", "request_status": "active", "row_version": 8, "player_id": 7, "email": "old@example.com"}
            ],
        }
    )
    created: list[dict] = []
    audit_attempts = 0

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(communications, "_required_audit_intent", lambda *_args, **_kwargs: None)

    def replace_once(*_args, **_kwargs):
        if not created:
            created.append({"id": "replacement", "club_id": "club", "player_id": 7, "email": "new@example.com", "row_version": 1})
        return created[0]

    def completion_audit(*_args, **_kwargs):
        nonlocal audit_attempts
        audit_attempts += 1
        if audit_attempts == 1:
            raise RuntimeError("response lost after committed replacement")
        return []

    monkeypatch.setattr(communications, "replace_verified_subscriber_atomic", replace_once)
    monkeypatch.setattr(communications, "_audit", completion_audit)
    kwargs = dict(
        club_id="club",
        subscription_id="old",
        expected_row_version=8,
        new_email="new@example.com",
        request_note="verified",
        admin_note="keep history",
        confirmation_text="REPLACE VERIFIED SUBSCRIBER",
        operation_key=key,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test",
    )
    with pytest.raises(RuntimeError, match="response lost"):
        communications.replace_active_subscription(supabase, **kwargs)
    response = communications.replace_active_subscription(supabase, **kwargs)
    assert response["subscription"]["id"] == "replacement"
    assert len(created) == 1
    with pytest.raises(ValueError, match="different communications request"):
        communications.replace_active_subscription(supabase, **{**kwargs, "actor_email": "other@example.com"})
    assert len(created) == 1


def test_required_audit_messages_distinguish_intent_from_outcome(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr(
        communications,
        "write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="unavailable"),
    )
    with pytest.raises(RuntimeError, match="nothing was changed or sent"):
        communications._required_audit_intent(
            object(),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="club_owner",
            action_type="send_selected_player_updates_admin",
            entity_type="player_update_outbox",
            entity_id="op",
            reviewed_scope={"items": []},
            source="test",
        )
    with pytest.raises(RuntimeError, match="may have completed"):
        communications._audit(
            object(),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="club_owner",
            action_type="send_selected_player_updates_admin",
            entity_type="player_update_outbox",
            entity_id="op",
            after_json={"phase": "complete"},
            source="test",
        )


def test_queue_operation_claim_happens_only_after_required_intent(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "get_communications_admin_operation",
        lambda *_args, **_kwargs: events.append("get") or None,
    )
    monkeypatch.setattr(
        communications,
        "_required_audit_intent",
        lambda *_args, **_kwargs: events.append("intent"),
    )
    monkeypatch.setattr(
        communications,
        "claim_communications_admin_operation",
        lambda *_args, **_kwargs: events.append("claim") or {"status": "started"},
    )
    monkeypatch.setattr(communications, "_build_ctx", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        communications,
        "generate_and_queue_digest_for_player",
        lambda *_args, **_kwargs: events.append("queue") or {"queued": 1, "failed": 0},
    )
    monkeypatch.setattr(communications, "_audit", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(communications, "complete_communications_admin_operation", lambda *_args, **_kwargs: None)

    communications.queue_player_digests(
        object(),
        club_id="club",
        start_date="2026-07-01",
        end_date="2026-07-07",
        player_id=7,
        only_players_with_matches=False,
        confirmation_text="QUEUE PLAYER UPDATES",
        operation_key="659139ef-3cd7-450f-b454-661957e25c91",
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test",
    )

    assert events == ["get", "intent", "claim", "queue"]


def test_queue_intent_failure_leaves_operation_and_target_unchanged(monkeypatch) -> None:
    supabase = FakeSupabase({"communications_admin_operations": [], "player_profile_update_outbox": []})
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "_required_audit_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("audit unavailable")),
    )
    monkeypatch.setattr(
        communications,
        "generate_and_queue_digest_for_player",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("queue must not run")),
    )

    with pytest.raises(RuntimeError, match="audit unavailable"):
        communications.queue_player_digests(
            supabase,
            club_id="club",
            start_date="2026-07-01",
            end_date="2026-07-07",
            player_id=7,
            only_players_with_matches=False,
            confirmation_text="QUEUE PLAYER UPDATES",
            operation_key="7b127864-10c8-4f46-a3da-0c15d67628b6",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert supabase.tables["communications_admin_operations"] == []
    assert supabase.tables["player_profile_update_outbox"] == []


def test_started_replacement_validates_before_intent_without_reclaim(monkeypatch) -> None:
    events: list[str] = []
    operation = {
        "club_id": "club",
        "operation_key": "3e6101c7-e141-40c2-9f1a-b68d24163ebf",
        "operation_type": "replace_verified_subscriber",
        "status": "started",
        "request_json": {
            "old_subscription_id": "old",
            "new_email_normalized": "safe@example.com",
            "request_note": "verified",
            "admin_note": "keep history",
            "actor_email": "admin@example.com",
        },
    }
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "get_communications_admin_operation",
        lambda *_args, **_kwargs: events.append("get") or operation,
    )

    def validate(*_args, **_kwargs):
        events.append("validate")
        return operation

    monkeypatch.setattr(communications, "validate_communications_admin_operation", validate)
    monkeypatch.setattr(
        communications,
        "get_subscription",
        lambda *_args, **_kwargs: events.append("read_subscription")
        or {"id": "old", "club_id": "club", "player_id": 7, "email": "old@example.com", "row_version": 8},
    )
    monkeypatch.setattr(
        communications,
        "_required_audit_intent",
        lambda *_args, **_kwargs: events.append("intent")
        or (_ for _ in ()).throw(RuntimeError("audit unavailable")),
    )
    monkeypatch.setattr(
        communications,
        "claim_communications_admin_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("started operation must not be reclaimed")),
    )
    monkeypatch.setattr(
        communications,
        "replace_verified_subscriber_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("replacement must not run")),
    )

    with pytest.raises(RuntimeError, match="audit unavailable"):
        communications.replace_active_subscription(
            object(),
            club_id="club",
            subscription_id="old",
            expected_row_version=8,
            new_email="safe@example.com",
            request_note="verified",
            admin_note="keep history",
            confirmation_text="REPLACE VERIFIED SUBSCRIBER",
            operation_key="3e6101c7-e141-40c2-9f1a-b68d24163ebf",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert events == ["get", "validate", "read_subscription", "intent"]


def test_retry_mutation_failure_records_distinct_failure_audit(monkeypatch) -> None:
    failures: list[dict] = []
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "list_outbox_rows",
        lambda *_args, **_kwargs: [{"id": "outbox-1", "send_status": "error", "row_version": 4}],
    )
    monkeypatch.setattr(communications, "_required_audit_intent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        communications,
        "retry_outbox_rows_guarded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("retry mutation failed")),
    )
    monkeypatch.setattr(communications, "_audit_failure", lambda *_args, **kwargs: failures.append(kwargs))

    with pytest.raises(RuntimeError, match="retry mutation failed"):
        communications.retry_outbox_rows(
            object(),
            club_id="club",
            items=[{"id": "outbox-1", "expected_row_version": 4}],
            confirmation_text="RETRY PLAYER UPDATES",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert failures[0]["action_type"] == "retry_player_update_outbox_admin"
    assert failures[0]["reviewed_scope"]["items"][0]["id"] == "outbox-1"


def test_delete_mutation_failure_records_distinct_failure_audit(monkeypatch) -> None:
    failures: list[dict] = []
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(communications, "_required_audit_intent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        communications,
        "delete_pending_outbox_rows_guarded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("delete mutation failed")),
    )
    monkeypatch.setattr(communications, "_audit_failure", lambda *_args, **kwargs: failures.append(kwargs))

    with pytest.raises(RuntimeError, match="delete mutation failed"):
        communications.delete_outbox_rows(
            object(),
            club_id="club",
            items=[{"id": "outbox-1", "expected_row_version": 4}],
            confirmation_text="DELETE QUEUED UPDATES",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert failures[0]["action_type"] == "delete_player_update_outbox_admin"
    assert failures[0]["reviewed_scope"]["items"][0]["expected_row_version"] == 4


def test_replace_mutation_failure_records_distinct_failure_audit(monkeypatch) -> None:
    failures: list[dict] = []
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(communications, "get_communications_admin_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        communications,
        "get_subscription",
        lambda *_args, **_kwargs: {
            "id": "old",
            "club_id": "club",
            "player_id": 7,
            "email": "old@example.com",
            "request_status": "active",
            "row_version": 8,
        },
    )
    monkeypatch.setattr(communications, "_required_audit_intent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(communications, "claim_communications_admin_operation", lambda *_args, **_kwargs: {"status": "started"})
    monkeypatch.setattr(
        communications,
        "replace_verified_subscriber_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("replacement mutation failed")),
    )
    monkeypatch.setattr(communications, "_audit_failure", lambda *_args, **kwargs: failures.append(kwargs))

    with pytest.raises(RuntimeError, match="replacement mutation failed"):
        communications.replace_active_subscription(
            object(),
            club_id="club",
            subscription_id="old",
            expected_row_version=8,
            new_email="new@example.com",
            request_note="verified",
            admin_note="keep history",
            confirmation_text="REPLACE VERIFIED SUBSCRIBER",
            operation_key="73301c90-1411-47d7-adab-37a8743702b4",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert failures[0]["action_type"] == "replace_verified_subscriber_admin"
    assert failures[0]["reviewed_scope"]["subscription_id"] == "old"
    assert failures[0]["reviewed_scope"]["new_email_masked"] == "n***@example.com"


def test_deactivate_mutation_failure_records_distinct_failure_audit(monkeypatch) -> None:
    failures: list[dict] = []
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setattr(
        communications,
        "get_subscription",
        lambda *_args, **_kwargs: {
            "id": "subscription-1",
            "club_id": "club",
            "player_id": 7,
            "email": "verified@example.com",
            "request_status": "active",
            "row_version": 3,
        },
    )
    monkeypatch.setattr(communications, "_required_audit_intent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        communications,
        "mark_unsubscribed_guarded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("deactivate mutation failed")),
    )
    monkeypatch.setattr(communications, "_audit_failure", lambda *_args, **kwargs: failures.append(kwargs))

    with pytest.raises(RuntimeError, match="deactivate mutation failed"):
        communications.deactivate_active_subscription(
            object(),
            club_id="club",
            subscription_id="subscription-1",
            expected_row_version=3,
            confirmation_text="UNSUBSCRIBE VERIFIED SUBSCRIBER",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test",
        )

    assert failures[0]["action_type"] == "deactivate_verified_subscriber_admin"
    assert failures[0]["reviewed_scope"] == {"subscription_id": "subscription-1", "expected_row_version": 3}
