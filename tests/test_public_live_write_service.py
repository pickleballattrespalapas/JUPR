from __future__ import annotations

import pytest

from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError,
    PublicLiveRateLimitError,
    PublicLiveRecoveryRequiredError,
    claim_public_live_completion_executor,
)
from jupr_app.services.public_live_write_service import (
    PublicLiveSessionError,
    advance_public_live_session,
    build_public_live_export,
    complete_public_live_session,
    create_public_live_session,
    create_public_round_robin_session,
    substitute_public_live_participant,
    update_public_round_robin_scores,
)


TOKEN_SECRET = "public-live-test-secret-that-is-long-enough"
REQUESTER_HASH = "a" * 64


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, owner: "FakeSupabase", table_name: str):
        self.owner = owner
        self.table_name = table_name
        self.action = "select"
        self.payload = None
        self.filters: list[tuple[str, str, object]] = []
        self.row_limit: int | None = None

    def select(self, *_args, **_kwargs):
        self.action = "select"
        return self

    def insert(self, payload, **_kwargs):
        self.action = "insert"
        self.payload = dict(payload)
        return self

    def update(self, payload, **_kwargs):
        self.action = "update"
        self.payload = dict(payload)
        return self

    def eq(self, key, value):
        self.filters.append((str(key), "eq", value))
        return self

    def gte(self, key, value):
        self.filters.append((str(key), "gte", value))
        return self

    def limit(self, value, *_args, **_kwargs):
        self.row_limit = int(value)
        return self

    def _matches(self, row):
        for key, operation, value in self.filters:
            if operation == "eq" and row.get(key) != value:
                return False
            if operation == "gte" and str(row.get(key) or "") < str(value):
                return False
        return True

    def execute(self):
        rows = self.owner.tables.setdefault(self.table_name, [])
        if self.action == "insert":
            if self.table_name == "public_live_operations":
                key = str(self.payload.get("operation_key") or "")
                if any(str(row.get("operation_key") or "") == key for row in rows):
                    raise RuntimeError("duplicate operation")
            if self.table_name == "live_sessions":
                creation_key = self.payload.get("creation_operation_key")
                if creation_key and any(row.get("club_id") == self.payload.get("club_id") and row.get("creation_operation_key") == creation_key for row in rows):
                    raise RuntimeError("duplicate creation")
            created = dict(self.payload)
            rows.append(created)
            return FakeResponse([dict(created)])
        selected = [row for row in rows if self._matches(row)]
        if self.action == "update":
            updated = []
            for row in selected:
                row.update(dict(self.payload))
                updated.append(dict(row))
            if self.table_name in self.owner.fail_after_update_tables:
                self.owner.fail_after_update_tables.remove(self.table_name)
                raise RuntimeError("simulated response loss after update")
            return FakeResponse(updated)
        if self.row_limit is not None:
            selected = selected[: self.row_limit]
        return FakeResponse([dict(row) for row in selected])


class FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {
            "live_sessions": [],
            "public_live_operations": [],
            "players": [],
        }
        self.fail_after_update_tables: set[str] = set()

    def table(self, table_name):
        return FakeQuery(self, str(table_name))

    def rpc(self, function_name, payload):
        owner = self

        class FakeRpc:
            def execute(self):
                assert function_name == "claim_public_live_completion_executor"
                for operation in owner.tables["public_live_operations"]:
                    if operation.get("club_id") != payload.get("p_club_id"):
                        continue
                    if operation.get("operation_key") != payload.get("p_operation_key"):
                        continue
                    if operation.get("action") != "complete":
                        continue
                    if operation.get("status") not in {"intent", "running", "applied", "recovery_required"}:
                        continue
                    if operation.get("executor_token"):
                        return FakeResponse([])
                    operation["executor_token"] = payload.get("p_executor_token")
                    operation["lease_expires_at"] = "2099-01-01T00:00:00+00:00"
                    operation["status"] = "running"
                    return FakeResponse([dict(operation)])
                return FakeResponse([])

        return FakeRpc()


def _create_round_robin(supabase: FakeSupabase, *, key: str = "create-key-0001", mode: str = "quick"):
    return create_public_live_session(
        supabase,
        club_id="tres_palapas",
        event_name="Public Test RR",
        event_type="round_robin",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
        live_mode=mode,
        total_rounds=1,
        court_sizes=None,
        host_name="Test Host" if mode == "club_social" else None,
        skill_levels=["3.5"] if mode == "club_social" else None,
        idempotency_key=key,
        requester_hash=REQUESTER_HASH,
        token_secret=TOKEN_SECRET,
    )


def _score_all(supabase: FakeSupabase, created: dict, *, key: str = "scores-key-0001"):
    session = created["session"]
    scores = [
        {"match_id": match["id"], "score_a": 11, "score_b": 8}
        for round_data in session["rounds"]
        for match in round_data["matches"]
    ]
    return update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=session["session_key"],
        edit_token=created["edit_token"],
        expected_version=session["version"],
        idempotency_key=key,
        requester_hash=REQUESTER_HASH,
        scores=scores,
    )


def test_public_round_robin_create_and_score_update_hashes_token():
    supabase = FakeSupabase()
    created = create_public_round_robin_session(
        supabase,
        club_id="tres_palapas",
        event_name="Public Test RR",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
        idempotency_key="create-key-0001",
        requester_hash=REQUESTER_HASH,
        token_secret=TOKEN_SECRET,
    )
    row = supabase.tables["live_sessions"][0]
    assert created["edit_token"] not in str(row)
    assert len(row["edit_token_hash"]) == 64

    match_id = created["session"]["rounds"][0]["matches"][0]["id"]
    updated = update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=1,
        idempotency_key="scores-key-0001",
        requester_hash=REQUESTER_HASH,
        scores=[{"match_id": match_id, "score_a": 11, "score_b": 8}],
    )
    assert updated["session"]["version"] == 2
    assert updated["session"]["rounds"][0]["matches"][0]["score_a"] == 11
    assert updated["session"]["standings"]


def test_public_create_retry_recovers_same_session_and_token():
    supabase = FakeSupabase()
    first = _create_round_robin(supabase)
    second = _create_round_robin(supabase)
    assert second["idempotent_replay"] is True
    assert second["session"]["session_key"] == first["session"]["session_key"]
    assert second["edit_token"] == first["edit_token"]
    assert len(supabase.tables["live_sessions"]) == 1


def test_public_create_ledger_fingerprints_names_without_retaining_them():
    supabase = FakeSupabase()
    _create_round_robin(supabase, key="privacy-safe-create-ledger")
    operation = supabase.tables["public_live_operations"][0]

    assert operation["request_fingerprint"]
    assert operation["request_json"]["participant_count"] == 4
    assert "participant_names" not in operation["request_json"]
    assert "Amy" not in str(operation["request_json"])
    assert "Amy" not in str(operation["result_json"])
    assert "edit_token" not in str(operation)


def test_invalid_create_fails_before_persisting_intent():
    supabase = FakeSupabase()

    with pytest.raises(PublicLiveSessionError, match="Host / Submitter Name"):
        create_public_live_session(
            supabase,
            club_id="tres_palapas",
            event_name="Invalid Social",
            event_type="round_robin",
            participant_names=["Amy", "Brooke", "Chris", "Dana"],
            live_mode="club_social",
            total_rounds=1,
            court_sizes=None,
            host_name=None,
            skill_levels=["3.5"],
            idempotency_key="invalid-create-host",
            requester_hash=REQUESTER_HASH,
            token_secret=TOKEN_SECRET,
        )

    assert supabase.tables["live_sessions"] == []
    assert supabase.tables["public_live_operations"] == []


def test_club_social_links_selected_current_player_and_rejects_unresolved_duplicate():
    supabase = FakeSupabase()
    supabase.tables["players"] = [
        {"id": 101, "club_id": "tres_palapas", "name": "Amy", "is_active": True},
        {"id": 202, "club_id": "other_club", "name": "Brooke", "is_active": True},
    ]

    created = create_public_live_session(
        supabase,
        club_id="tres_palapas",
        event_name="Linked Social",
        event_type="round_robin",
        participant_names=["Amy", "Guest One", "Guest Two", "Guest Three"],
        live_mode="club_social",
        total_rounds=1,
        court_sizes=None,
        host_name="Test Host",
        skill_levels=["3.5"],
        participant_player_ids={"Amy": 101},
        idempotency_key="linked-social-create",
        requester_hash=REQUESTER_HASH,
        token_secret=TOKEN_SECRET,
    )
    amy = next(row for row in created["session"]["participants"] if row["name"] == "Amy")
    assert amy["player_id"] == 101

    fresh = FakeSupabase()
    fresh.tables["players"] = [{"id": 101, "club_id": "tres_palapas", "name": "Amy"}]
    with pytest.raises(PublicLiveSessionError, match="current-player search"):
        create_public_live_session(
            fresh,
            club_id="tres_palapas",
            event_name="Unlinked Social",
            event_type="round_robin",
            participant_names=["Amy", "Guest One", "Guest Two", "Guest Three"],
            live_mode="club_social",
            total_rounds=1,
            court_sizes=None,
            host_name="Test Host",
            skill_levels=["3.5"],
            participant_player_ids={},
            idempotency_key="unlinked-social-create",
            requester_hash=REQUESTER_HASH,
            token_secret=TOKEN_SECRET,
        )
    assert fresh.tables["live_sessions"] == []
    assert fresh.tables["public_live_operations"][-1]["status"] == "rejected"


def test_public_score_stale_version_and_wrong_token_change_nothing():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase)
    match_id = created["session"]["rounds"][0]["matches"][0]["id"]
    with pytest.raises(PermissionError):
        update_public_round_robin_scores(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token="wrong",
            expected_version=1,
            idempotency_key="scores-key-wrong",
            requester_hash=REQUESTER_HASH,
            scores=[{"match_id": match_id, "score_a": 11, "score_b": 8}],
        )
    with pytest.raises(PublicLiveConflictError):
        update_public_round_robin_scores(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=9,
            idempotency_key="scores-key-stale",
            requester_hash=REQUESTER_HASH,
            scores=[{"match_id": match_id, "score_a": 11, "score_b": 8}],
        )
    assert supabase.tables["live_sessions"][0]["version"] == 1


def test_public_score_response_loss_reconciles_from_row_marker():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase)
    match_id = created["session"]["rounds"][0]["matches"][0]["id"]
    supabase.fail_after_update_tables.add("live_sessions")
    result = update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=1,
        idempotency_key="scores-key-loss",
        requester_hash=REQUESTER_HASH,
        scores=[{"match_id": match_id, "score_a": 11, "score_b": 7}],
    )
    assert result["session"]["version"] == 2
    replay = update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=1,
        idempotency_key="scores-key-loss",
        requester_hash=REQUESTER_HASH,
        scores=[{"match_id": match_id, "score_a": 11, "score_b": 7}],
    )
    assert replay["idempotent_replay"] is True
    assert replay["session"]["version"] == 2


def test_public_league_scores_advance_and_guest_substitution():
    supabase = FakeSupabase()
    created = create_public_live_session(
        supabase,
        club_id="tres_palapas",
        event_name="Public Ladder",
        event_type="league_ladder",
        participant_names=[f"Player {index}" for index in range(1, 9)],
        live_mode="quick",
        total_rounds=2,
        court_sizes=[4, 4],
        host_name=None,
        skill_levels=None,
        idempotency_key="league-create-01",
        requester_hash=REQUESTER_HASH,
        token_secret=TOKEN_SECRET,
    )
    first_match = created["session"]["rounds"][0]["matches"][0]
    substituted = substitute_public_live_participant(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=1,
        idempotency_key="league-sub-0001",
        requester_hash=REQUESTER_HASH,
        scope="game",
        round_number=1,
        original_participant_id=created["session"]["participants"][0]["id"],
        substitute_name="Guest Player",
        match_id=first_match["id"],
    )
    substitution_operation = supabase.tables["public_live_operations"][-1]
    assert "Guest Player" not in str(substitution_operation["result_json"])
    assert "session" not in substitution_operation["result_json"]
    scores = [
        {"match_id": match["id"], "score_a": 11, "score_b": 8}
        for match in substituted["session"]["rounds"][0]["matches"]
    ]
    scored = update_public_round_robin_scores(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=2,
        idempotency_key="league-scores-01",
        requester_hash=REQUESTER_HASH,
        scores=scores,
    )
    advanced = advance_public_live_session(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=scored["session"]["version"],
        idempotency_key="league-advance-1",
        requester_hash=REQUESTER_HASH,
    )
    assert advanced["advanced_to_round"] == 2
    assert advanced["session"]["current_round"] == 2
    assert advanced["session"]["substitutions"][0]["substitute_name"] == "Guest Player"


def test_invalid_mutation_is_rejected_without_changing_session():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="invalid-mutation-create")

    with pytest.raises(PublicLiveSessionError, match="no longer belong"):
        update_public_round_robin_scores(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key="invalid-score-row",
            requester_hash=REQUESTER_HASH,
            scores=[{"match_id": "not-this-session", "score_a": 11, "score_b": 8}],
        )

    assert supabase.tables["live_sessions"][0]["version"] == 1
    operation = supabase.tables["public_live_operations"][-1]
    assert operation["action"] == "scores"
    assert operation["status"] == "rejected"


def test_incomplete_completion_is_rate_limited_and_rejected_without_session_write():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="incomplete-completion-create")
    with pytest.raises(PublicLiveSessionError, match="Complete every scheduled score"):
        complete_public_live_session(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key="incomplete-completion",
            requester_hash=REQUESTER_HASH,
        )

    assert supabase.tables["public_live_operations"][-1]["status"] == "rejected"
    assert supabase.tables["live_sessions"][0]["status"] == "active"


def test_stale_mutation_is_recorded_as_rejected():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="stale-rejected-create")
    match_id = created["session"]["rounds"][0]["matches"][0]["id"]

    with pytest.raises(PublicLiveConflictError, match="changed after it was loaded"):
        update_public_round_robin_scores(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=99,
            idempotency_key="stale-rejected-score",
            requester_hash=REQUESTER_HASH,
            scores=[{"match_id": match_id, "score_a": 11, "score_b": 8}],
        )

    operation = supabase.tables["public_live_operations"][-1]
    assert operation["status"] == "rejected"
    assert operation["error_text"] == "stale authoritative version"


def test_club_social_completion_is_durable_and_export_is_formula_safe():
    supabase = FakeSupabase()
    created = create_public_live_session(
        supabase,
        club_id="tres_palapas",
        event_name="Social Test",
        event_type="round_robin",
        participant_names=["=Amy", "Brooke", "Chris", "Dana"],
        live_mode="club_social",
        total_rounds=1,
        court_sizes=None,
        host_name="Host",
        skill_levels=["3.5"],
        idempotency_key="social-create-1",
        requester_hash=REQUESTER_HASH,
        token_secret=TOKEN_SECRET,
    )
    scored = _score_all(supabase, created, key="social-scores-1")
    calls = []

    def submitter(_supabase, **kwargs):
        calls.append(kwargs)
        return {"status": "pending", "saved_rounds": ["rr"], "event_id": "social-1"}

    completed = complete_public_live_session(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=scored["session"]["version"],
        idempotency_key="social-complete-1",
        requester_hash=REQUESTER_HASH,
        social_submitter=submitter,
    )
    assert completed["session"]["status"] == "completed"
    assert completed["session"]["social"]["submission_status"] == "pending"
    assert len(calls) == 1
    exported = build_public_live_export(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        export_format="csv",
    )
    assert "'=Amy" in exported["content"]
    assert created["edit_token"] not in exported["content"]


@pytest.mark.parametrize(
    ("row_patch", "reason"),
    [
        ({"status": "abandoned"}, "non-public status"),
        ({"status": "active", "expires_at": "2000-01-01T00:00:00+00:00"}, "expired session"),
    ],
)
def test_public_live_export_rejects_rows_hidden_from_public_detail(row_patch, reason):
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key=f"hidden-export-{reason.replace(' ', '-')}")
    supabase.tables["live_sessions"][0].update(row_patch)

    with pytest.raises(PublicLiveSessionError, match="Live session not found"):
        build_public_live_export(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            export_format="json",
        )


def test_club_social_rejects_substitution_before_moderation_attribution_can_diverge():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="social-no-substitution", mode="club_social")
    participant_id = created["session"]["participants"][0]["id"]

    with pytest.raises(PublicLiveSessionError, match="Club Social substitutions are not supported"):
        substitute_public_live_participant(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key="social-substitution-rejected",
            requester_hash=REQUESTER_HASH,
            scope="round",
            round_number=1,
            original_participant_id=participant_id,
            substitute_name="Guest Replacement",
        )

    assert supabase.tables["live_sessions"][0]["version"] == 1
    assert supabase.tables["public_live_operations"][-1]["status"] == "rejected"


def test_club_social_completion_reservation_blocks_other_writes_and_same_key_recovers():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="social-create-recovery", mode="club_social")
    scored = _score_all(supabase, created, key="social-scores-recovery")
    attempts = {"count": 0}

    def flaky_submitter(_supabase, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated moderation response loss")
        return {"status": "pending", "saved_rounds": ["rr"], "event_id": "social-recovered"}

    with pytest.raises(PublicLiveRecoveryRequiredError):
        complete_public_live_session(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=scored["session"]["version"],
            idempotency_key="social-complete-recovery",
            requester_hash=REQUESTER_HASH,
            social_submitter=flaky_submitter,
        )
    reserved = supabase.tables["live_sessions"][0]
    assert reserved["pending_operation_action"] == "complete"
    match_id = scored["session"]["rounds"][0]["matches"][0]["id"]
    with pytest.raises(PublicLiveRecoveryRequiredError):
        update_public_round_robin_scores(
            supabase,
            club_id="tres_palapas",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=reserved["version"],
            idempotency_key="scores-during-completion",
            requester_hash=REQUESTER_HASH,
            scores=[{"match_id": match_id, "score_a": 11, "score_b": 9}],
        )
    recovered = complete_public_live_session(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=scored["session"]["version"],
        idempotency_key="social-complete-recovery",
        requester_hash=REQUESTER_HASH,
        social_submitter=flaky_submitter,
    )
    assert recovered["session"]["status"] == "completed"
    assert supabase.tables["live_sessions"][0]["pending_operation_key"] is None
    assert attempts["count"] == 2


def test_club_social_completion_allows_only_one_concurrent_executor():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="social-create-single-executor", mode="club_social")
    scored = _score_all(supabase, created, key="social-scores-single-executor")
    operation_key = "single-executor-complete"
    from jupr_app.services.public_live_operation_service import begin_public_live_operation

    operation, _ = begin_public_live_operation(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        action="complete",
        idempotency_key=operation_key,
        requester_hash=REQUESTER_HASH,
        expected_version=scored["session"]["version"],
        request_payload={},
    )
    claim_public_live_completion_executor(
        supabase,
        club_id="tres_palapas",
        operation_key_value=operation["operation_key"],
    )
    with pytest.raises(PublicLiveRecoveryRequiredError, match="already being reconciled"):
        claim_public_live_completion_executor(
            supabase,
            club_id="tres_palapas",
            operation_key_value=operation["operation_key"],
        )


def test_club_social_final_response_loss_reconciles_without_duplicate_submit():
    supabase = FakeSupabase()
    created = _create_round_robin(supabase, key="social-create-final-loss", mode="club_social")
    scored = _score_all(supabase, created, key="social-scores-final-loss")
    calls = []

    def submitter(_supabase, **kwargs):
        calls.append(kwargs)
        supabase.fail_after_update_tables.add("live_sessions")
        return {"status": "pending", "saved_rounds": ["rr"], "event_id": "social-final-loss"}

    result = complete_public_live_session(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=scored["session"]["version"],
        idempotency_key="social-complete-final-loss",
        requester_hash=REQUESTER_HASH,
        social_submitter=submitter,
    )
    assert result["session"]["status"] == "completed"
    replay = complete_public_live_session(
        supabase,
        club_id="tres_palapas",
        session_key=created["session"]["session_key"],
        edit_token=created["edit_token"],
        expected_version=scored["session"]["version"],
        idempotency_key="social-complete-final-loss",
        requester_hash=REQUESTER_HASH,
        social_submitter=submitter,
    )
    assert replay["idempotent_replay"] is True
    assert len(calls) == 1


def test_public_create_rate_limit_is_durable(monkeypatch):
    supabase = FakeSupabase()
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR", "1")
    _create_round_robin(supabase, key="rate-create-01")
    with pytest.raises(PublicLiveRateLimitError):
        _create_round_robin(supabase, key="rate-create-02")
