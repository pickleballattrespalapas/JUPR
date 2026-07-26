from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services.admin_challenge_ladder_service import (
    _published_match_relation,
    _run_atomic_match_side_effects,
    accept_admin_challenge_ladder_challenge,
    add_admin_challenge_ladder_roster_player,
    create_admin_challenge_ladder_challenge,
    get_admin_challenge_ladder_tier_movement_review,
    move_admin_challenge_ladder_roster_player,
    prepare_admin_challenge_ladder_result_atomic_plan,
    preview_admin_challenge_ladder_result,
    preview_admin_challenge_ladder_result_for_challenge,
    preview_admin_challenge_ladder_tier_roster_replacement,
    recover_admin_challenge_ladder_operation_result,
    record_admin_challenge_ladder_forfeit,
    record_admin_challenge_ladder_pass,
    record_admin_challenge_ladder_result,
    replace_admin_challenge_ladder_tier_roster,
    save_admin_challenge_ladder_player_overrides,
)
from jupr_app.domain.challenge_ladder import month_key_utc
from jupr_app.services.admin_live_ladder_operation_service import (
    deterministic_operation_key,
    replay_durable_admin_operation_if_present,
    stable_request_fingerprint,
)
from services.api.admin_challenge_ladder_routes import install_admin_challenge_ladder_routes


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.insert_payload = None
        self.upsert_payload = None
        self.update_payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = deepcopy(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        if self.insert_payload is not None:
            row = {"id": len(rows) + 100, **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[dict(row)])
        if self.upsert_payload is not None:
            payloads = self.upsert_payload if isinstance(self.upsert_payload, list) else [self.upsert_payload]
            saved = []
            for payload in payloads:
                existing = next(
                    (
                        row
                        for row in rows
                        if str(row.get("club_id")) == str(payload.get("club_id"))
                        and str(row.get("player_id")) == str(payload.get("player_id"))
                    ),
                    None,
                )
                if existing is None:
                    existing = {"id": len(rows) + 100, **payload}
                    rows.append(existing)
                else:
                    existing.update(payload)
                saved.append(dict(existing))
            return SimpleNamespace(data=saved)
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in scoped])


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [
                {"club_id": "club", "id": 1, "name": "Defender"},
                {"club_id": "club", "id": 2, "name": "Challenger"},
                {"club_id": "club", "id": 3, "name": "Partner A"},
                {"club_id": "club", "id": 4, "name": "Partner B"},
                {"club_id": "club", "id": 5, "name": "Partner C"},
                {"club_id": "club", "id": 6, "name": "Partner D"},
                {"club_id": "club", "id": 7, "name": "Roster Candidate"},
            ],
            "ladder_settings": [{"club_id": "club", "challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7}],
            "ladder_roster": [
                {"club_id": "club", "id": 10, "player_id": 1, "tier_id": "ADV", "rank": 1, "is_active": True},
                {"club_id": "club", "id": 11, "player_id": 2, "tier_id": "ADV", "rank": 2, "is_active": True},
                {"club_id": "club", "id": 12, "player_id": 3, "tier_id": "ADV", "rank": 3, "is_active": True},
                {"club_id": "club", "id": 13, "player_id": 4, "tier_id": "ADV", "rank": 4, "is_active": True},
                {"club_id": "club", "id": 14, "player_id": 5, "tier_id": "ADV", "rank": 5, "is_active": True},
                {"club_id": "club", "id": 15, "player_id": 6, "tier_id": "ADV", "rank": 6, "is_active": True},
            ],
            "ladder_challenges": [{"club_id": "club", "id": 100, "challenger_id": 2, "defender_id": 1, "tier_id": "ADV", "status": "PENDING_ACCEPTANCE", "created_at": "2026-01-01T00:00:00Z"}],
            "ladder_pass_usage": [],
            "ladder_player_flags": [],
            "matches": [],
            "admin_activity_log": [],
            "live_ladder_admin_operations": [],
        }
        self.rpc_calls = []

    def table(self, name):
        return FakeQuery(self.storage, name)

    def rpc(self, name, params):
        supabase = self

        class FakeRpc:
            def execute(self):
                assert name in {
                    "admin_apply_challenge_ladder_result_atomic_v1",
                    "admin_finalize_challenge_ladder_result_v1",
                    "admin_finalize_challenge_ladder_forfeit_v1",
                }
                supabase.rpc_calls.append(
                    {
                        "name": name,
                        "params": deepcopy(params),
                        "matches_before": deepcopy(supabase.storage["matches"]),
                    }
                )
                challenge = next(
                    row
                    for row in supabase.storage["ladder_challenges"]
                    if str(row["club_id"]) == str(params["p_club_id"])
                    and int(row["id"]) == int(params["p_challenge_id"])
                )
                challenger = next(
                    row
                    for row in supabase.storage["ladder_roster"]
                    if int(row["player_id"]) == int(challenge["challenger_id"])
                )
                defender = next(
                    row
                    for row in supabase.storage["ladder_roster"]
                    if int(row["player_id"]) == int(challenge["defender_id"])
                )
                challenger_before = int(challenger["rank"])
                defender_before = int(defender["rank"])
                if name == "admin_finalize_challenge_ladder_forfeit_v1":
                    forfeited = int(params["p_forfeited_by_id"])
                    winner_id = (
                        int(challenge["defender_id"])
                        if forfeited == int(challenge["challenger_id"])
                        else int(challenge["challenger_id"])
                    )
                else:
                    winner_id = int(params["p_winner_id"])
                official_matches = None
                if name == "admin_apply_challenge_ladder_result_atomic_v1":
                    match_rows = deepcopy(params["p_match_rows"])
                    contexts = {
                        str(params["p_match_context_a"]),
                        str(params["p_match_context_b"]),
                    }
                    if any(
                        str(row.get("context_id")) in contexts
                        for row in supabase.storage["matches"]
                    ):
                        raise RuntimeError(
                            "an official challenge match context already exists"
                        )
                    inserted_ids = {}
                    for slot, row in zip(("a", "b"), match_rows, strict=True):
                        saved = {
                            "id": 501 if slot == "a" else 502,
                            **row,
                            "club_id": params["p_club_id"],
                            "deleted_at": None,
                        }
                        supabase.storage["matches"].append(saved)
                        inserted_ids[slot] = saved["id"]
                    official_matches = {
                        "inserted": 2,
                        "skipped": False,
                        "atomic": True,
                        "match_ids": inserted_ids,
                        "match_context_ids": [
                            params["p_match_context_a"],
                            params["p_match_context_b"],
                        ],
                    }
                swapped = winner_id == int(challenge["challenger_id"])
                if swapped:
                    challenger["rank"], defender["rank"] = defender_before, challenger_before
                rank_change = {
                    "swapped": swapped,
                    "challenger": {
                        "player_id": challenge["challenger_id"],
                        "before": challenger_before,
                        "after": int(challenger["rank"]),
                    },
                    "defender": {
                        "player_id": challenge["defender_id"],
                        "before": defender_before,
                        "after": int(defender["rank"]),
                    },
                }
                public_result = deepcopy(params.get("p_public_result_json"))
                if name == "admin_apply_challenge_ladder_result_atomic_v1":
                    public_result = {
                        "version": 1,
                        "match_ids": official_matches["match_ids"],
                    }
                if public_result is not None:
                    public_result["rank_change"] = rank_change
                challenge.update(
                    {
                        "status": "FORFEITED" if name == "admin_finalize_challenge_ladder_forfeit_v1" else "COMPLETED",
                        "winner_id": winner_id,
                        "completed_at": params["p_completed_at"],
                        "resolution_notes": params.get("p_resolution_notes"),
                        "public_result_json": public_result,
                        "updated_at": params["p_completed_at"],
                    }
                )
                if name == "admin_finalize_challenge_ladder_forfeit_v1":
                    challenge["forfeit_by"] = params["p_forfeited_by_id"]
                    challenge["forfeit_reason"] = params["p_forfeit_reason"]
                receipt = {
                    "ok": True,
                    "core_committed": True,
                    "challenge": deepcopy(challenge),
                    "rank_result": rank_change,
                    "public_result_json": public_result,
                    "post_processors": {
                        "status": (
                            "pending"
                            if name == "admin_apply_challenge_ladder_result_atomic_v1"
                            else "complete"
                        )
                    },
                }
                if official_matches is not None:
                    receipt["official_matches"] = official_matches
                return SimpleNamespace(data=receipt)

        return FakeRpc()


def _planned_match_row(payload: dict) -> dict:
    return {
        "club_id": "club",
        "date": payload["date"],
        "league": payload["league"],
        "t1_p1": payload["t1_p1"],
        "t1_p2": payload["t1_p2"],
        "t2_p1": payload["t2_p1"],
        "t2_p2": payload["t2_p2"],
        "score_t1": payload["s1"],
        "score_t2": payload["s2"],
        "elo_delta": 1.0,
        "match_type": payload["match_type"],
        "week_tag": "",
        "t1_p1_r": 3.0,
        "t1_p2_r": 3.0,
        "t2_p1_r": 3.0,
        "t2_p2_r": 3.0,
        "t1_p1_r_end": 3.1,
        "t1_p2_r_end": 3.1,
        "t2_p1_r_end": 2.9,
        "t2_p2_r_end": 2.9,
        "context_type": payload["context_type"],
        "context_id": payload["context_id"],
        "rating_scope": "",
        "match_format": "doubles",
    }


def _planned_rating_updates() -> tuple[list[dict], list[dict], list[dict]]:
    player_updates = []
    league_updates = []
    for player_id in (1, 2, 3, 4):
        won = player_id in {2, 3}
        player_updates.append(
            {
                "player_id": player_id,
                "rating_mode": "doubles",
                "expected": {
                    "rating": 3.0,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "last_game_at": None,
                    "inactive_at": None,
                    "active": True,
                },
                "after": {
                    "rating": 3.1 if won else 2.9,
                    "wins": 2 if won else 0,
                    "losses": 0 if won else 2,
                    "matches_played": 2,
                    "last_game_at": "2026-01-02T00:00:00Z",
                    "inactive_at": None,
                    "active": True,
                },
            }
        )
        league_updates.append(
            {
                "player_id": player_id,
                "league_name": "OVERALL",
                "expected": None,
                "after": {
                    "rating": 3.1 if won else 2.9,
                    "wins": 2 if won else 0,
                    "losses": 0 if won else 2,
                    "matches_played": 2,
                    "starting_rating": 3.0,
                    "is_active": True,
                    "inactive_at": None,
                },
            }
        )
    return (
        player_updates,
        league_updates,
        [{"league_name": "OVERALL", "expected": None}],
    )


def _atomic_result_core(
    supabase: FakeSupabase,
    *,
    operation_key: str = "a" * 64,
    publish_official_matches: bool = True,
    preview_fingerprint: str = "preview-1",
) -> dict:
    challenge = deepcopy(supabase.storage["ladder_challenges"][0])
    context_a = f"{operation_key[:8]}-context-a"
    context_b = f"{operation_key[:8]}-context-b"
    match_payloads = [
        {
            "date": "2026-01-02T00:00:00Z",
            "league": "OVERALL",
            "match_type": "ChallengeLadder",
            "is_popup": False,
            "context_type": "challenge_ladder",
            "context_id": context_a,
            "t1_p1": 2,
            "t1_p2": 3,
            "t2_p1": 1,
            "t2_p2": 4,
            "s1": 22,
            "s2": 11,
        },
        {
            "date": "2026-01-02T00:00:00Z",
            "league": "OVERALL",
            "match_type": "ChallengeLadder",
            "is_popup": False,
            "context_type": "challenge_ladder",
            "context_id": context_b,
            "t1_p1": 2,
            "t1_p2": 4,
            "t2_p1": 1,
            "t2_p2": 3,
            "s1": 22,
            "s2": 10,
        },
    ]
    match_rows = [_planned_match_row(payload) for payload in match_payloads]
    player_updates, league_updates, metadata_expectations = (
        _planned_rating_updates()
    )
    core = {
        "version": 1,
        "challenge_expected": {
            key: challenge.get(key)
            for key in (
                "id",
                "club_id",
                "challenger_id",
                "defender_id",
                "tier_id",
                "status",
                "updated_at",
                "winner_id",
                "completed_at",
                "public_result_json",
            )
        },
        "tier_roster_expected": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "player_id",
                    "tier_id",
                    "rank",
                    "is_active",
                    "updated_at",
                )
            }
            for row in supabase.storage["ladder_roster"]
            if row.get("tier_id") == challenge.get("tier_id")
        ],
        "winner_id": 2,
        "completed_at": "2026-01-02T00:00:00Z",
        "resolution_notes": "Next result publish. Winner: 2.",
        "publish_official_matches": publish_official_matches,
        "match_context_ids": (
            [context_a, context_b] if publish_official_matches else []
        ),
        "match_payloads": match_payloads if publish_official_matches else [],
        "write_plan": {
            "match_rows": match_rows if publish_official_matches else [],
            "player_updates": player_updates if publish_official_matches else [],
            "league_rating_updates": (
                league_updates if publish_official_matches else []
            ),
            "league_metadata_expectations": (
                metadata_expectations if publish_official_matches else []
            ),
        },
        "side_effect_context": {
            "affected_player_ids": [1, 2, 3, 4],
            "successful_match_dates": ["2026-01-02T00:00:00Z"],
            "has_badge_eligible_match": True,
            "match_payloads": match_payloads if publish_official_matches else [],
        },
        "preview": {
            "final_winner_id": 2,
            "winner_summary": "Challenger wins both matches.",
            "scores": {
                "match_a": {"score_t1": 22, "score_t2": 11},
                "match_b": {"score_t1": 22, "score_t2": 10},
            },
        },
        "preview_fingerprint": preview_fingerprint,
    }
    core["plan_fingerprint"] = stable_request_fingerprint(core)
    return core


def _committed_result_operation(
    supabase: FakeSupabase,
    *,
    operation_key: str | None = None,
) -> tuple[dict, dict]:
    client_request = {
        "expected_version": "version-1",
        "idempotency_key": "result-recovery-1",
        "preview_fingerprint": "preview-1",
    }
    resolved_operation_key = operation_key or deterministic_operation_key(
        club_id="club",
        surface="challenge_ladder",
        operation_type="publish_result",
        entity_id="100",
        idempotency_key=client_request["idempotency_key"],
    )
    core = _atomic_result_core(
        supabase,
        operation_key=resolved_operation_key,
        preview_fingerprint=client_request["preview_fingerprint"],
    )
    rank_change = {
        "swapped": True,
        "challenger": {"player_id": 2, "before": 2, "after": 1},
        "defender": {"player_id": 1, "before": 1, "after": 2},
    }
    public_result = {
        "version": 1,
        "match_ids": {"a": 501, "b": 502},
        "rank_change": rank_change,
    }
    challenge = supabase.storage["ladder_challenges"][0]
    challenge.update(
        {
            "status": "COMPLETED",
            "winner_id": 2,
            "completed_at": core["completed_at"],
            "resolution_notes": core["resolution_notes"],
            "public_result_json": deepcopy(public_result),
            "updated_at": core["completed_at"],
        }
    )
    roster = {int(row["player_id"]): row for row in supabase.storage["ladder_roster"]}
    roster[2]["rank"], roster[1]["rank"] = 1, 2
    for slot, row in zip(("a", "b"), core["write_plan"]["match_rows"], strict=True):
        supabase.storage["matches"].append(
            {
                "id": 501 if slot == "a" else 502,
                "club_id": "club",
                "deleted_at": None,
                **deepcopy(row),
            }
        )
    operation = {
        "operation_key": resolved_operation_key,
        "club_id": "club",
        "surface": "challenge_ladder",
        "operation_type": "publish_result",
        "entity_id": "100",
        "idempotency_key": client_request["idempotency_key"],
        "request_fingerprint": stable_request_fingerprint(client_request),
        "expected_version": "version-1",
        "status": "mutated",
        "attempt_count": 1,
        "request_json": {
            "client_request": deepcopy(client_request),
            "atomic_core": deepcopy(core),
        },
        "result_json": {
            "ok": True,
            "core_committed": True,
            "mode": "challenge_ladder_result_core",
            "challenge": deepcopy(challenge),
            "rank_result": rank_change,
            "public_result_json": deepcopy(public_result),
            "official_matches": {
                "inserted": 2,
                "skipped": False,
                "atomic": True,
                "match_ids": {"a": 501, "b": 502},
                "match_context_ids": deepcopy(core["match_context_ids"]),
            },
            "post_processors": {"status": "pending"},
            "plan_fingerprint": core["plan_fingerprint"],
            "side_effect_context": deepcopy(core["side_effect_context"]),
        },
        "recovery_json": {
            "match_context_ids": deepcopy(core["match_context_ids"]),
        },
        "error_text": None,
        "completed_at": core["completed_at"],
        "created_at": core["completed_at"],
        "updated_at": core["completed_at"],
    }
    return operation, client_request


def test_create_challenge_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    try:
        create_admin_challenge_ladder_challenge(FakeSupabase(), club_id="club", challenger_id=2, defender_id=1, tier_id="ADV", ledger_ref=None, override=False, start_clock=False, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="CREATE")
    except ValueError as exc:
        assert "CREATE LADDER CHALLENGE" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_create_and_accept_challenge(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = []
    created = create_admin_challenge_ladder_challenge(supabase, club_id="club", challenger_id=2, defender_id=1, tier_id="ADV", ledger_ref="ledger", override=False, start_clock=True, actor_email="admin@example.com", actor_role="club_owner", confirmation_text="CREATE LADDER CHALLENGE")
    assert created["challenge"]["status"] == "PENDING_ACCEPTANCE"
    accepted = accept_admin_challenge_ladder_challenge(supabase, club_id="club", challenge_id=created["challenge"]["id"], actor_email="admin@example.com", actor_role="club_owner", confirmation_text="ACCEPT LADDER CHALLENGE")
    assert accepted["challenge"]["status"] == "ACCEPTED_SCHEDULING"


def test_create_challenge_rejects_locked_players_without_override(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    try:
        create_admin_challenge_ladder_challenge(
            supabase,
            club_id="club",
            challenger_id=2,
            defender_id=1,
            tier_id="ADV",
            ledger_ref=None,
            override=False,
            start_clock=False,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="CREATE LADDER CHALLENGE",
        )
    except ValueError as exc:
        assert "status: Locked" in str(exc)
    else:
        raise AssertionError("expected locked-player eligibility rejection")
    assert len(supabase.storage["ladder_challenges"]) == 1


def test_create_challenge_uses_shared_protected_and_cooldown_policy(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = [
        {
            "club_id": "club",
            "id": 100,
            "challenger_id": 1,
            "defender_id": 2,
            "tier_id": "ADV",
            "status": "COMPLETED",
            "created_at": "2099-01-01T00:00:00Z",
            "completed_at": "2099-01-02T00:00:00Z",
            "winner_id": 2,
        }
    ]

    created = create_admin_challenge_ladder_challenge(
        supabase,
        club_id="club",
        challenger_id=2,
        defender_id=1,
        tier_id="ADV",
        ledger_ref=None,
        override=False,
        start_clock=False,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="CREATE LADDER CHALLENGE",
    )

    assert created["challenge"]["challenger_id"] == 2
    assert created["challenge"]["defender_id"] == 1


def test_create_challenge_override_returns_copyable_notice(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    monkeypatch.setenv("LADDER_ADMIN_NAME", "Director Dana")
    monkeypatch.setenv("LADDER_ADMIN_CONTACT", "dana@example.com")
    supabase = FakeSupabase()

    created = create_admin_challenge_ladder_challenge(
        supabase,
        club_id="club",
        challenger_id=2,
        defender_id=1,
        tier_id="ADV",
        challenger_contact="challenger@example.com",
        ledger_ref="front desk 42",
        override=True,
        start_clock=False,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="CREATE LADDER CHALLENGE",
    )

    assert created["challenge"]["id"] == 101
    assert "challenger@example.com" in created["notice"]["email_full"]
    assert "Director Dana" in created["notice"]["email_full"]
    assert "dana@example.com" in created["notice"]["sms"]
    assert "front desk 42" in created["notice"]["email_full"]
    assert "Challenge ID: #101" in created["notice"]["email_full"]


def test_preview_defender_holds_exact_tie():
    preview = preview_admin_challenge_ladder_result(
        challenger_id=2,
        defender_id=1,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 9], [7, 11]],
        match_b_games=[[9, 11], [11, 7]],
    )
    assert preview["final_winner_id"] == 1


def test_preview_rejects_four_distinct_partners_instead_of_cross_swap():
    try:
        preview_admin_challenge_ladder_result(
            challenger_id=2,
            defender_id=1,
            partner_a_challenger_id=3,
            partner_a_defender_id=4,
            partner_b_challenger_id=5,
            partner_b_defender_id=6,
            match_a_games=[[11, 9], [11, 7]],
            match_b_games=[[11, 8], [11, 6]],
        )
    except ValueError as exc:
        assert "same two swing partners" in str(exc)
    else:
        raise AssertionError("expected the four-distinct-partner layout to fail")


def test_challenge_result_preview_is_read_only_and_reports_rank_effect(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    before = deepcopy(supabase.storage)

    result = preview_admin_challenge_ladder_result_for_challenge(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=True,
    )

    assert result["mode"] == "challenge_ladder_result_preview"
    assert result["preview"]["final_winner_id"] == 2
    assert result["rank_result"] == {"would_swap": True, "reason": "challenger win"}
    assert result["partner_names"]["a_chal"] == "Partner A"
    assert result["would_publish_official_matches"] is True
    assert supabase.storage == before


def test_challenge_result_preview_rejects_player_from_another_club(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"

    try:
        preview_admin_challenge_ladder_result_for_challenge(
            supabase,
            club_id="club",
            challenge_id=100,
            partner_a_challenger_id=3,
            partner_a_defender_id=4,
            partner_b_challenger_id=4,
            partner_b_defender_id=999,
            match_a_games=[[11, 5], [11, 6]],
            match_b_games=[[11, 4], [11, 6]],
            match_date="2026-01-02T00:00:00Z",
            winner_override="computed",
            publish_official_matches=True,
        )
    except ValueError as exc:
        assert "players in this club" in str(exc)
    else:
        raise AssertionError("expected club-scoped partner validation")


def test_challenge_result_preview_route_requires_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    app = FastAPI()
    install_admin_challenge_ladder_routes(app, get_supabase_client=lambda: supabase)

    contract_path = "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview"
    path = "/admin/clubs/club/challenge-ladder/challenges/100/result/preview"
    assert "post" in app.openapi()["paths"][contract_path]
    response = TestClient(app).post(
        path,
        json={
            "partner_a_challenger_id": 3,
            "partner_a_defender_id": 4,
            "partner_b_challenger_id": 4,
            "partner_b_defender_id": 3,
            "match_a_games": [[11, 5], [11, 6]],
            "match_b_games": [[11, 4], [11, 6]],
            "match_date": "2026-01-02T00:00:00Z",
        },
    )

    assert response.status_code == 401
    assert supabase.storage["admin_activity_log"] == []


def test_forfeit_swaps_rank_when_defender_forfeits(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    result = record_admin_challenge_ladder_forfeit(
        supabase,
        club_id="club",
        challenge_id=100,
        forfeited_by_id=1,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="RECORD LADDER FORFEIT",
        operation_key="c" * 64,
    )
    assert result["challenge"]["winner_id"] == 2
    assert result["challenge"]["status"] == "FORFEITED"
    assert supabase.storage["ladder_challenges"][0]["public_result_json"] is None
    ranks = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}
    assert ranks[2] == 1
    assert ranks[1] == 2
    assert supabase.storage["matches"] == []
    assert [call["name"] for call in supabase.rpc_calls] == [
        "admin_finalize_challenge_ladder_forfeit_v1"
    ]
    assert supabase.rpc_calls[0]["params"]["p_operation_key"] == "c" * 64


def test_monthly_pass_closes_challenge_without_rank_change(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    ranks_before = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}

    result = record_admin_challenge_ladder_pass(
        supabase,
        club_id="club",
        challenge_id=100,
        player_id=2,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="RECORD LADDER PASS",
    )

    assert result["challenge"]["status"] == "CANCELED"
    assert result["pass_usage"]["player_id"] == 2
    assert result["pass_usage"]["month_key"] == month_key_utc(datetime.now(timezone.utc))
    assert {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]} == ranks_before
    assert supabase.storage["admin_activity_log"][-1]["action_type"] == "challenge_pass_used"


def test_monthly_pass_rejects_duplicate_usage_in_utc_month(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_pass_usage"].append(
        {
            "club_id": "club",
            "player_id": 2,
            "month_key": month_key_utc(datetime.now(timezone.utc)),
            "used_at": "2026-07-01T00:00:00Z",
            "challenge_id": 99,
        }
    )

    try:
        record_admin_challenge_ladder_pass(
            supabase,
            club_id="club",
            challenge_id=100,
            player_id=2,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="RECORD LADDER PASS",
        )
    except ValueError as exc:
        assert "already used" in str(exc)
    else:
        raise AssertionError("expected duplicate monthly-pass rejection")
    assert supabase.storage["ladder_challenges"][0]["status"] == "PENDING_ACCEPTANCE"


def test_roster_add_appends_new_player_to_selected_tier(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = add_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=7,
        tier_id="INT",
        admin_note="approved placement",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="ADD LADDER PLAYER",
    )

    assert result["reactivated"] is False
    assert result["roster"]["player_name"] == "Roster Candidate"
    assert result["roster"]["tier_id"] == "INT"
    assert result["roster"]["rank"] == 1
    assert supabase.storage["admin_activity_log"][-1]["entity_type"] == "ladder_roster"


def test_roster_add_reactivates_at_bottom(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    inactive = supabase.storage["ladder_roster"][-1]
    inactive.update({"is_active": False, "left_at": "2026-06-01T00:00:00Z"})

    result = add_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=6,
        tier_id="ADV",
        admin_note=None,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="ADD LADDER PLAYER",
    )

    assert result["reactivated"] is True
    assert result["roster"]["rank"] == 6
    assert result["roster"]["is_active"] is True


def test_tier_roster_replacement_preview_is_read_only_and_blocks_open_challenges(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    result = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Challenger", "Defender", "Roster Candidate"],
    )

    assert result["can_apply"] is False
    assert result["summary"]["current_count"] == 6
    assert result["summary"]["proposed_count"] == 3
    assert result["summary"]["removed_count"] == 4
    assert [row["player_id"] for row in result["proposed_roster"]] == [2, 1, 7]
    assert result["open_challenge_blockers"][0]["challenge_id"] == 100
    assert supabase.storage == before


def test_tier_roster_replacement_preview_rejects_duplicate_and_missing_names(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    for ranked_names, expected in [
        (["Defender", "Defender"], "Duplicate names"),
        (["Defender", "Not A Club Player"], "Create these club players"),
    ]:
        try:
            preview_admin_challenge_ladder_tier_roster_replacement(
                supabase,
                club_id="club",
                tier_id="ADV",
                ranked_names=ranked_names,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("expected replacement roster name validation error")


def test_tier_roster_replacement_rejects_stale_preview(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = []
    preview = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Defender", "Challenger", "Partner A", "Partner B", "Partner C", "Partner D"],
    )
    supabase.storage["ladder_roster"][0]["rank"] = 2

    try:
        replace_admin_challenge_ladder_tier_roster(
            supabase,
            club_id="club",
            tier_id="ADV",
            ranked_player_ids=[row["player_id"] for row in preview["proposed_roster"]],
            preview_fingerprint=preview["preview_fingerprint"],
            admin_note=None,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="REPLACE LADDER TIER",
        )
    except ValueError as exc:
        assert "changed after preview" in str(exc)
    else:
        raise AssertionError("expected stale tier-roster preview rejection")
    assert supabase.storage["admin_activity_log"] == []


def test_tier_roster_replacement_applies_reviewed_order_and_recompresses_source(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"] = []
    supabase.storage["ladder_roster"][-1].update({"tier_id": "INT", "rank": 2})
    supabase.storage["ladder_roster"].append(
        {"club_id": "club", "id": 16, "player_id": 7, "tier_id": "INT", "rank": 1, "is_active": True}
    )
    preview = preview_admin_challenge_ladder_tier_roster_replacement(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_names=["Challenger", "Defender", "Roster Candidate"],
    )

    result = replace_admin_challenge_ladder_tier_roster(
        supabase,
        club_id="club",
        tier_id="ADV",
        ranked_player_ids=[row["player_id"] for row in preview["proposed_roster"]],
        preview_fingerprint=preview["preview_fingerprint"],
        admin_note="annual roster reset",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="REPLACE LADDER TIER",
    )

    assert [(row["player_id"], row["rank"]) for row in result["roster"]] == [(2, 1), (1, 2), (7, 3)]
    assert result["summary"]["moved_from_other_tier_count"] == 1
    assert result["recompressed_player_ids"] == [6]
    inactive_ids = {
        row["player_id"]
        for row in supabase.storage["ladder_roster"]
        if row.get("is_active") is False
    }
    assert {3, 4, 5}.issubset(inactive_ids)
    source_player = next(row for row in supabase.storage["ladder_roster"] if row["player_id"] == 6)
    assert source_player["tier_id"] == "INT"
    assert source_player["rank"] == 1
    audit = supabase.storage["admin_activity_log"][-1]
    assert audit["action_type"] == "roster_replace_tier"
    assert audit["before_json"]["rows"]
    assert audit["after_json"]["preview_fingerprint"] == preview["preview_fingerprint"]


def test_roster_move_appends_and_recompresses_previous_tier(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = move_admin_challenge_ladder_roster_player(
        supabase,
        club_id="club",
        player_id=4,
        destination_tier="INT",
        recompress_old=True,
        admin_note="reviewed movement",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="MOVE LADDER PLAYER",
    )

    assert result["previous_tier"] == "ADV"
    assert result["roster"]["tier_id"] == "INT"
    assert result["roster"]["rank"] == 1
    assert result["recompressed_count"] == 2
    advanced = sorted(
        (row for row in supabase.storage["ladder_roster"] if row["tier_id"] == "ADV" and row["is_active"]),
        key=lambda row: row["rank"],
    )
    assert [row["rank"] for row in advanced] == [1, 2, 3, 4, 5]


def test_player_overrides_insert_normalizes_utc_and_writes_audit(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    result = save_admin_challenge_ladder_player_overrides(
        supabase,
        club_id="club",
        player_id=2,
        vacation_until="2026-08-01T12:00:00-05:00",
        reinstate_required=True,
        reinstate_notes="Contact ladder director",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE LADDER OVERRIDES",
    )

    assert result["player_flags"]["vacation_until"] == "2026-08-01T17:00:00+00:00"
    assert result["player_flags"]["reinstate_required"] is True
    assert supabase.storage["ladder_player_flags"][0]["player_id"] == 2
    assert supabase.storage["admin_activity_log"][-1]["entity_type"] == "ladder_player_flags"


def test_player_overrides_update_can_clear_values_without_touching_tier_flags(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_player_flags"].append(
        {
            "club_id": "club",
            "player_id": 2,
            "vacation_until": "2026-08-01T17:00:00+00:00",
            "reinstate_required": True,
            "reinstate_notes": "Old note",
            "tier_move_flag": True,
            "tier_move_dest_tier": "INT",
            "tier_move_count": 10,
        }
    )

    result = save_admin_challenge_ladder_player_overrides(
        supabase,
        club_id="club",
        player_id=2,
        vacation_until=None,
        reinstate_required=False,
        reinstate_notes="",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE LADDER OVERRIDES",
    )

    stored = supabase.storage["ladder_player_flags"][0]
    assert stored["vacation_until"] is None
    assert stored["reinstate_required"] is False
    assert stored["reinstate_notes"] is None
    assert stored["tier_move_flag"] is True
    assert result["player_flags"]["tier_move_dest_tier"] == "INT"


def test_player_overrides_reject_naive_date_time(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()

    try:
        save_admin_challenge_ladder_player_overrides(
            supabase,
            club_id="club",
            player_id=2,
            vacation_until="2026-08-01T12:00:00",
            reinstate_required=False,
            reinstate_notes=None,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="SAVE LADDER OVERRIDES",
        )
    except ValueError as exc:
        assert "UTC offset or Z" in str(exc)
    else:
        raise AssertionError("expected timezone validation error")
    assert supabase.storage["ladder_player_flags"] == []


def _tier_movement_match(match_id, date, end_rating):
    return {
        "club_id": "club",
        "id": match_id,
        "date": date,
        "t1_p1": 2,
        "t1_p2": 3,
        "t2_p1": 4,
        "t2_p2": 5,
        "t1_p1_r_end": end_rating,
        "t1_p2_r_end": 1600,
        "t2_p1_r_end": 1600,
        "t2_p2_r_end": 1600,
        "deleted_at": None,
    }


def test_tier_movement_review_reports_ten_match_trigger(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["matches"] = [
        _tier_movement_match(index, f"2026-07-{index:02d}T12:00:00Z", 1400)
        for index in range(1, 11)
    ]

    result = get_admin_challenge_ladder_tier_movement_review(supabase, club_id="club")

    assert result["summary"]["trigger_count"] == 1
    assert result["summary"]["required_consecutive_matches"] == 10
    assert result["triggers"] == [
        {
            "player_id": 2,
            "player_name": "Challenger",
            "current_tier": "ADV",
            "destination_tier": "INT",
            "consecutive_match_count": 10,
            "latest_match_at": "2026-07-10T12:00:00+00:00",
        }
    ]


def test_tier_movement_review_resets_after_current_tier_match(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["matches"] = [
        *[_tier_movement_match(index, f"2026-07-{index:02d}T12:00:00Z", 1400) for index in range(1, 11)],
        _tier_movement_match(11, "2026-07-11T12:00:00Z", 1600),
    ]

    result = get_admin_challenge_ladder_tier_movement_review(supabase, club_id="club")

    assert result["summary"]["trigger_count"] == 0
    assert result["triggers"] == []


def test_new_ladder_operation_routes_require_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    app = FastAPI()
    install_admin_challenge_ladder_routes(app, get_supabase_client=lambda: supabase)
    openapi = app.openapi()["paths"]
    before = deepcopy(supabase.storage)

    requests = [
        (
            "/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/pass",
            "/admin/clubs/club/challenge-ladder/challenges/100/pass",
            {"player_id": 2, "confirmation_text": "RECORD LADDER PASS"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster",
            "/admin/clubs/club/challenge-ladder/roster",
            {"player_id": 7, "tier_id": "INT", "confirmation_text": "ADD LADDER PLAYER"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/move",
            "/admin/clubs/club/challenge-ladder/roster/4/move",
            {"destination_tier": "INT", "confirmation_text": "MOVE LADDER PLAYER"},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier/preview",
            "/admin/clubs/club/challenge-ladder/roster/replace-tier/preview",
            {"tier_id": "ADV", "ranked_names": ["Defender"]},
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier",
            "/admin/clubs/club/challenge-ladder/roster/replace-tier",
            {
                "tier_id": "ADV",
                "ranked_player_ids": [1],
                "preview_fingerprint": "not-a-real-preview",
                "confirmation_text": "REPLACE LADDER TIER",
            },
            "post",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides",
            "/admin/clubs/club/challenge-ladder/roster/2/overrides",
            {"vacation_until": None, "reinstate_required": False, "confirmation_text": "SAVE LADDER OVERRIDES"},
            "put",
        ),
        (
            "/admin/clubs/{club_id}/challenge-ladder/tier-movement-review",
            "/admin/clubs/club/challenge-ladder/tier-movement-review",
            None,
            "get",
        ),
    ]
    for contract_path, path, payload, method in requests:
        assert method in openapi[contract_path]
        response = TestClient(app).request(method.upper(), path, json=payload)
        assert response.status_code == 401
    assert supabase.storage == before


def test_atomic_result_plan_is_read_only_exact_and_fingerprinted(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    preview = preview_admin_challenge_ladder_result_for_challenge(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=True,
    )
    before = deepcopy(supabase.storage)
    captured = {}
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service.load_data",
        lambda *_args, **_kwargs: (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            {},
            {},
            False,
            None,
        ),
    )

    def build_plan(payloads, **kwargs):
        captured["payloads"] = deepcopy(payloads)
        captured["build_write_plan_only"] = kwargs.get("build_write_plan_only")
        player_updates, league_updates, metadata_expectations = (
            _planned_rating_updates()
        )
        return {
            "inserted": 2,
            "skipped_incomplete": 0,
            "skipped_empty": 0,
            "skipped_unrated": 0,
            "write_plan": {
                "match_rows": [_planned_match_row(payload) for payload in payloads],
                "player_updates": player_updates,
                "league_rating_updates": league_updates,
                "league_metadata_expectations": metadata_expectations,
            },
            "side_effect_context": {
                "affected_player_ids": [1, 2, 3, 4],
                "successful_match_dates": ["2026-01-02T00:00:00Z"],
                "has_badge_eligible_match": True,
                "match_payloads": deepcopy(payloads),
            },
        }

    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service.process_matches",
        build_plan,
    )

    prepared = prepare_admin_challenge_ladder_result_atomic_plan(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=True,
        expected_preview_fingerprint=preview["preview_fingerprint"],
        operation_key="d" * 64,
    )

    core = prepared["atomic_core"]
    unsigned = {key: value for key, value in core.items() if key != "plan_fingerprint"}
    assert captured["build_write_plan_only"] is True
    assert len(captured["payloads"]) == 2
    assert len(set(core["match_context_ids"])) == 2
    assert core["match_context_ids"] == [
        row["context_id"] for row in core["write_plan"]["match_rows"]
    ]
    assert len(core["write_plan"]["player_updates"]) == 4
    assert len(core["write_plan"]["league_rating_updates"]) == 4
    assert len(core["write_plan"]["league_metadata_expectations"]) == 1
    assert core["challenge_expected"]["status"] == "ACCEPTED_SCHEDULING"
    assert core["tier_roster_expected"] == sorted(
        core["tier_roster_expected"], key=lambda row: row["id"]
    )
    assert core["preview_fingerprint"] == preview["preview_fingerprint"]
    assert core["plan_fingerprint"] == stable_request_fingerprint(unsigned)
    assert supabase.storage == before


def test_result_rejects_tampered_atomic_plan_before_any_rpc(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    atomic_core = _atomic_result_core(supabase)
    atomic_core["winner_id"] = 1

    with pytest.raises(RuntimeError, match="plan fingerprint is invalid"):
        record_admin_challenge_ladder_result(
            supabase,
            club_id="club",
            challenge_id=100,
            partner_a_challenger_id=3,
            partner_a_defender_id=4,
            partner_b_challenger_id=4,
            partner_b_defender_id=3,
            match_a_games=[[11, 5], [11, 6]],
            match_b_games=[[11, 4], [11, 6]],
            match_date="2026-01-02T00:00:00Z",
            winner_override="computed",
            publish_official_matches=True,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH LADDER RESULT",
            expected_preview_fingerprint="preview-1",
            publish_context_prefix="a" * 64,
            atomic_core=atomic_core,
        )

    assert supabase.rpc_calls == []
    assert supabase.storage["matches"] == []


def test_atomic_post_processing_uses_operation_key_as_badge_dedupe_identity(
    monkeypatch,
):
    supabase = FakeSupabase()
    core = _atomic_result_core(supabase)
    badge_calls = []
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service.run_badge_side_effects",
        lambda **kwargs: badge_calls.append(deepcopy(kwargs))
        or {"mode": "queue", "processed": 1, "errored": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service.queue_player_updates",
        lambda **_kwargs: {"mode": "queued", "failed": 0},
    )

    result = _run_atomic_match_side_effects(
        supabase,
        club_id="club",
        operation_key="7" * 64,
        write_plan=core["write_plan"],
        side_effect_context=core["side_effect_context"],
    )

    assert result["badge_summary"]["mode"] == "queue"
    assert len(badge_calls) == 1
    assert badge_calls[0]["dedupe_match_id"] == "7" * 64
    assert badge_calls[0]["affected_players"] == {1, 2, 3, 4}


def test_played_result_publishes_matches_and_swaps(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    operation_key = "a" * 64
    atomic_core = _atomic_result_core(supabase, operation_key=operation_key)
    side_effect_calls = []
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **kwargs: side_effect_calls.append(deepcopy(kwargs))
        or {
            "badge_summary": {"mode": "inline", "ok": True},
            "player_update_queue": {"mode": "inline", "ok": True},
        },
    )
    result = record_admin_challenge_ladder_result(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=True,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH LADDER RESULT",
        expected_preview_fingerprint="preview-1",
        publish_context_prefix=operation_key,
        atomic_core=atomic_core,
    )
    assert result["official_matches"]["inserted"] == 2
    assert result["official_matches"]["atomic"] is True
    assert result["challenge"]["winner_id"] == 2
    ranks = {row["player_id"]: row["rank"] for row in supabase.storage["ladder_roster"]}
    assert ranks[2] == 1
    assert ranks[1] == 2
    assert result["public_result_json"] == {
        "version": 1,
        "match_ids": {"a": 501, "b": 502},
        "rank_change": {
            "swapped": True,
            "challenger": {"player_id": 2, "before": 2, "after": 1},
            "defender": {"player_id": 1, "before": 1, "after": 2},
        },
    }
    assert "context" not in str(result["public_result_json"]).lower()
    assert [call["name"] for call in supabase.rpc_calls] == [
        "admin_apply_challenge_ladder_result_atomic_v1"
    ]
    assert supabase.rpc_calls[0]["matches_before"] == []
    params = supabase.rpc_calls[0]["params"]
    assert params["p_atomic_core"] == atomic_core
    assert params["p_plan_fingerprint"] == atomic_core["plan_fingerprint"]
    assert params["p_match_rows"] == atomic_core["write_plan"]["match_rows"]
    assert params["p_player_updates"] == atomic_core["write_plan"]["player_updates"]
    assert params["p_match_context_a"] == atomic_core["match_context_ids"][0]
    assert params["p_match_context_b"] == atomic_core["match_context_ids"][1]
    assert len(side_effect_calls) == 1
    assert side_effect_calls[0]["operation_key"] == operation_key


def test_played_result_rejects_any_existing_context_including_soft_deleted(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    operation_key = "b" * 64
    atomic_core = _atomic_result_core(supabase, operation_key=operation_key)
    supabase.storage["matches"].append(
        {
            "id": 499,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "context_id": atomic_core["match_context_ids"][0],
            "deleted_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("post-processors must not run after a rejected atomic core")
        ),
    )

    before_challenge = deepcopy(supabase.storage["ladder_challenges"][0])
    before_ranks = deepcopy(supabase.storage["ladder_roster"])
    before_matches = deepcopy(supabase.storage["matches"])
    try:
        record_admin_challenge_ladder_result(
            supabase,
            club_id="club",
            challenge_id=100,
            partner_a_challenger_id=3,
            partner_a_defender_id=4,
            partner_b_challenger_id=4,
            partner_b_defender_id=3,
            match_a_games=[[11, 5], [11, 6]],
            match_b_games=[[11, 4], [11, 6]],
            match_date="2026-01-02T00:00:00Z",
            winner_override="computed",
            publish_official_matches=True,
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH LADDER RESULT",
            expected_preview_fingerprint="preview-1",
            publish_context_prefix=operation_key,
            atomic_core=atomic_core,
        )
    except RuntimeError as exc:
        assert "failed closed" in str(exc)
    else:
        raise AssertionError("expected soft-deleted context collision to fail closed")

    assert supabase.storage["ladder_challenges"][0] == before_challenge
    assert supabase.storage["ladder_roster"] == before_ranks
    assert supabase.storage["matches"] == before_matches
    assert [call["name"] for call in supabase.rpc_calls] == [
        "admin_apply_challenge_ladder_result_atomic_v1"
    ]


def test_unrated_ladder_result_uses_only_atomic_rank_challenge_finalizer(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "1")
    supabase = FakeSupabase()
    supabase.storage["ladder_challenges"][0]["status"] = "ACCEPTED_SCHEDULING"
    operation_key = "e" * 64
    atomic_core = _atomic_result_core(
        supabase,
        operation_key=operation_key,
        publish_official_matches=False,
    )

    result = record_admin_challenge_ladder_result(
        supabase,
        club_id="club",
        challenge_id=100,
        partner_a_challenger_id=3,
        partner_a_defender_id=4,
        partner_b_challenger_id=4,
        partner_b_defender_id=3,
        match_a_games=[[11, 5], [11, 6]],
        match_b_games=[[11, 4], [11, 6]],
        match_date="2026-01-02T00:00:00Z",
        winner_override="computed",
        publish_official_matches=False,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH LADDER RESULT",
        expected_preview_fingerprint="preview-1",
        publish_context_prefix=operation_key,
        atomic_core=atomic_core,
    )

    assert result["official_matches"] == {"inserted": 0, "skipped": True}
    assert result["public_result_json"] is None
    assert result["match_context_ids"] == []
    assert supabase.storage["matches"] == []
    assert [call["name"] for call in supabase.rpc_calls] == [
        "admin_finalize_challenge_ladder_result_v1"
    ]
    assert supabase.rpc_calls[0]["params"]["p_public_result_json"] is None


def test_committed_result_receipt_recovers_post_processors_without_core_rpc(
    monkeypatch,
):
    supabase = FakeSupabase()
    operation, _client_request = _committed_result_operation(supabase)
    side_effect_calls = []
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda _supabase, **kwargs: side_effect_calls.append(deepcopy(kwargs))
        or {
            "badge_summary": {"mode": "inline", "ok": True},
            "player_update_queue": {"mode": "inline", "failed": 0},
        },
    )

    recovered = recover_admin_challenge_ladder_operation_result(
        supabase,
        operation=operation,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_response_loss_recovery",
    )

    assert recovered is not None
    assert recovered["mode"] == "challenge_ladder_result"
    assert recovered["challenge"]["status"] == "COMPLETED"
    assert recovered["official_matches"]["match_ids"] == {"a": 501, "b": 502}
    assert recovered["official_matches"]["badge_summary"]["ok"] is True
    assert recovered["public_result_json"]["rank_change"]["swapped"] is True
    assert len(side_effect_calls) == 1
    assert side_effect_calls[0]["operation_key"] == operation["operation_key"]
    assert side_effect_calls[0]["write_plan"] == operation["request_json"][
        "atomic_core"
    ]["write_plan"]
    assert supabase.rpc_calls == []


def test_durable_result_replay_recovers_once_then_never_calls_core_or_posts_again(
    monkeypatch,
):
    supabase = FakeSupabase()
    operation, client_request = _committed_result_operation(supabase)
    supabase.storage["live_ladder_admin_operations"].append(operation)
    side_effect_calls = []

    def recover(row):
        return recover_admin_challenge_ladder_operation_result(
            supabase,
            operation=row,
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test_result_replay",
        )

    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: side_effect_calls.append("posts")
        or {
            "badge_summary": {"mode": "inline", "ok": True},
            "player_update_queue": {"mode": "inline", "failed": 0},
        },
    )

    first = replay_durable_admin_operation_if_present(
        supabase,
        club_id="club",
        surface="challenge_ladder",
        operation_type="publish_result",
        entity_id="100",
        idempotency_key=client_request["idempotency_key"],
        request_payload=client_request,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_result_replay",
        recover_incomplete=recover,
    )
    second = replay_durable_admin_operation_if_present(
        supabase,
        club_id="club",
        surface="challenge_ladder",
        operation_type="publish_result",
        entity_id="100",
        idempotency_key=client_request["idempotency_key"],
        request_payload=client_request,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_result_replay",
        recover_incomplete=recover,
    )

    assert first is not None and second is not None
    assert first["idempotent_replay"] is True
    assert second["idempotent_replay"] is True
    assert first["mode"] == second["mode"] == "challenge_ladder_result"
    assert side_effect_calls == ["posts"]
    assert supabase.rpc_calls == []
    assert supabase.storage["live_ladder_admin_operations"][0]["status"] == "completed"


def test_failed_result_post_processor_keeps_core_receipt_retryable_without_core_rpc(
    monkeypatch,
):
    supabase = FakeSupabase()
    operation, client_request = _committed_result_operation(supabase)
    supabase.storage["live_ladder_admin_operations"].append(operation)
    attempts = []

    def recover(row):
        return recover_admin_challenge_ladder_operation_result(
            supabase,
            operation=row,
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test_post_retry",
        )

    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: attempts.append("failed")
        or (_ for _ in ()).throw(RuntimeError("badge processing did not complete")),
    )
    with pytest.raises(RuntimeError, match="badge processing"):
        replay_durable_admin_operation_if_present(
            supabase,
            club_id="club",
            surface="challenge_ladder",
            operation_type="publish_result",
            entity_id="100",
            idempotency_key=client_request["idempotency_key"],
            request_payload=client_request,
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test_post_retry",
            recover_incomplete=recover,
        )
    assert operation["status"] == "mutated"
    assert operation["result_json"]["core_committed"] is True
    assert supabase.rpc_calls == []

    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: attempts.append("complete")
        or {
            "badge_summary": {"mode": "inline", "ok": True},
            "player_update_queue": {"mode": "inline", "failed": 0},
        },
    )
    recovered = replay_durable_admin_operation_if_present(
        supabase,
        club_id="club",
        surface="challenge_ladder",
        operation_type="publish_result",
        entity_id="100",
        idempotency_key=client_request["idempotency_key"],
        request_payload=client_request,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_post_retry",
        recover_incomplete=recover,
    )

    assert recovered is not None
    assert recovered["mode"] == "challenge_ladder_result"
    assert attempts == ["failed", "complete"]
    assert operation["status"] == "completed"
    assert supabase.rpc_calls == []


def test_forfeit_core_receipt_recovers_without_request_plan_or_side_effects(
    monkeypatch,
):
    supabase = FakeSupabase()
    completed_at = "2026-01-02T00:00:00Z"
    challenge = supabase.storage["ladder_challenges"][0]
    challenge.update(
        {
            "status": "FORFEITED",
            "winner_id": 2,
            "forfeit_by": 1,
            "forfeit_reason": "Forfeit",
            "completed_at": completed_at,
            "public_result_json": None,
            "updated_at": completed_at,
        }
    )
    roster = {int(row["player_id"]): row for row in supabase.storage["ladder_roster"]}
    roster[2]["rank"], roster[1]["rank"] = 1, 2
    operation = {
        "operation_key": "9" * 64,
        "club_id": "club",
        "surface": "challenge_ladder",
        "operation_type": "record_forfeit",
        "entity_id": "100",
        "result_json": {
            "ok": True,
            "core_committed": True,
            "mode": "challenge_ladder_forfeit_core",
            "challenge": deepcopy(challenge),
            "rank_result": {
                "swapped": True,
                "challenger": {"player_id": 2, "before": 2, "after": 1},
                "defender": {"player_id": 1, "before": 1, "after": 2},
            },
            "public_result_json": None,
            "post_processors": {"status": "complete"},
        },
    }
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("forfeit recovery has no rating post-processors")
        ),
    )

    recovered = recover_admin_challenge_ladder_operation_result(
        supabase,
        operation=operation,
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_forfeit_response_loss",
    )

    assert recovered is not None
    assert recovered["mode"] == "challenge_ladder_forfeit"
    assert recovered["challenge"]["status"] == "FORFEITED"
    assert recovered["rank_result"]["challenger_new_rank"] == 1
    assert supabase.rpc_calls == []


def test_committed_receipt_fails_closed_if_public_match_was_soft_deleted(
    monkeypatch,
):
    supabase = FakeSupabase()
    operation, _client_request = _committed_result_operation(supabase)
    supabase.storage["matches"][0]["deleted_at"] = "2026-01-03T00:00:00Z"
    monkeypatch.setattr(
        "jupr_app.services.admin_challenge_ladder_service._run_atomic_match_side_effects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("posts must not run when relation verification fails")
        ),
    )

    with pytest.raises(RuntimeError, match="exactly one active row"):
        recover_admin_challenge_ladder_operation_result(
            supabase,
            operation=operation,
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test_deleted_relation_recovery",
        )

    assert supabase.rpc_calls == []


def test_public_match_relation_rejects_score_or_participant_drift() -> None:
    supabase = FakeSupabase()
    challenge = supabase.storage["ladder_challenges"][0]
    payloads = [
        {
            "context_id": "context-a",
            "t1_p2": 3,
            "t2_p2": 4,
            "s1": 22,
            "s2": 11,
        },
        {
            "context_id": "context-b",
            "t1_p2": 4,
            "t2_p2": 3,
            "s1": 22,
            "s2": 10,
        },
    ]
    supabase.storage["matches"] = [
        {
            "id": 501,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "context_id": "context-a",
            "deleted_at": None,
            "t1_p1": 2,
            "t1_p2": 3,
            "t2_p1": 1,
            "t2_p2": 4,
            "score_t1": 21,
            "score_t2": 11,
        },
        {
            "id": 502,
            "club_id": "club",
            "context_type": "challenge_ladder",
            "context_id": "context-b",
            "deleted_at": None,
            "t1_p1": 2,
            "t1_p2": 4,
            "t2_p1": 1,
            "t2_p2": 3,
            "score_t1": 22,
            "score_t2": 10,
        },
    ]

    try:
        _published_match_relation(
            supabase,
            club_id="club",
            challenge=challenge,
            payloads=payloads,
        )
    except RuntimeError as exc:
        assert "reviewed participants and score" in str(exc)
    else:
        raise AssertionError("expected exact published score mismatch to fail")
