from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.services.direct_match_entry_service import (
    DIRECT_MATCH_RPC,
    DirectMatchConflictError,
    DirectMatchRecoveryRequiredError,
    stable_direct_match_request,
    submit_atomic_direct_matches,
)


MATCH = {
    "date": "2026-07-26",
    "league": "Open",
    "match_type": "Live Match",
    "week_tag": "Week 1",
    "t1_p1": 1,
    "t1_p2": 2,
    "t2_p1": 3,
    "t2_p2": 4,
    "score_t1": 11,
    "score_t2": 7,
}

PLAYER_UPDATES = [
    {
        "player_id": player_id,
        "rating_mode": "doubles",
        "expected": {
            "rating": 1200,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "last_game_at": None,
            "inactive_at": None,
            "active": True,
        },
        "after": {
            "rating": 1210 if player_id < 3 else 1190,
            "wins": 1 if player_id < 3 else 0,
            "losses": 0 if player_id < 3 else 1,
            "matches_played": 1,
            "last_game_at": "2026-07-26T00:00:00+00:00",
            "inactive_at": None,
            "active": True,
        },
    }
    for player_id in (1, 2, 3, 4)
]

CALCULATED = {
    "inserted": 1,
    "skipped_incomplete": 0,
    "skipped_empty": 0,
    "skipped_unrated": 0,
    "write_plan": {
        "match_rows": [{**MATCH, "club_id": "club", "match_format": "doubles"}],
        "player_updates": PLAYER_UPDATES,
        "league_rating_updates": [],
        "league_metadata_expectations": [],
    },
    "side_effect_context": {
        "affected_player_ids": [1, 2, 3, 4],
        "successful_match_dates": ["2026-07-26T00:00:00+00:00"],
        "has_badge_eligible_match": True,
        "match_payloads": [{"date": "2026-07-26", "league": "Open"}],
    },
}


class RpcSupabase:
    def __init__(
        self,
        response=None,
        error: Exception | None = None,
        preflight_receipt: dict | None = None,
    ):
        self.response = response
        self.error = error
        self.preflight_receipt = preflight_receipt
        self.calls: list[tuple[str, dict]] = []

    def table(self, name):
        if name != "admin_direct_match_entry_operations":
            raise AssertionError(f"Unexpected table read: {name}")
        row = (
            {
                "request_fingerprint": self.preflight_receipt[
                    "request_fingerprint"
                ],
                "match_format": self.preflight_receipt["match_format"],
                "result_json": self.preflight_receipt,
            }
            if self.preflight_receipt
            else None
        )

        class Query:
            def select(self, *_args):
                return self

            def eq(self, *_args):
                return self

            def limit(self, *_args):
                return self

            def execute(self):
                return SimpleNamespace(data=[row] if row else [])

        return Query()

    def rpc(self, name, payload):
        self.calls.append((name, payload))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            execute=lambda: SimpleNamespace(data=dict(self.response or {}))
        )


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        [{"id": player_id, "name": f"Player {player_id}"} for player_id in range(1, 5)]
    )


def _receipt(*, idempotent: bool = False) -> dict:
    _, fingerprint = stable_direct_match_request(
        club_id="club",
        match_format="doubles",
        matches=[MATCH],
    )
    return {
        "ok": True,
        "committed": True,
        "idempotent": idempotent,
        "duplicate_request": False,
        "operation_id": "00000000-0000-4000-8000-000000000001",
        "idempotency_key": "score:operation-1",
        "request_fingerprint": fingerprint,
        "match_format": "doubles",
        "inserted": 1,
        "match_ids": ["101"],
        "result_summary": {
            "inserted": 1,
            "skipped_incomplete": 0,
            "skipped_empty": 0,
            "skipped_unrated": 0,
        },
        "player_updates": PLAYER_UPDATES,
    }


def _submit(monkeypatch, supabase, processor=CALCULATED):
    monkeypatch.setattr(
        "jupr_app.services.direct_match_entry_service.process_matches",
        processor
        if callable(processor)
        else lambda *_args, **_kwargs: processor,
    )
    return submit_atomic_direct_matches(
        supabase,
        club_id="club",
        matches=[MATCH],
        match_format="doubles",
        idempotency_key="score:operation-1",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        source="test",
        name_to_id={},
        df_players_all=_players(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )


def test_atomic_success_uses_one_rpc_and_runs_postprocessors_once(monkeypatch):
    supabase = RpcSupabase(_receipt())
    calls = {"badges": 0, "queue": 0}
    monkeypatch.setattr(
        "jupr_app.services.direct_match_entry_service.run_badge_side_effects",
        lambda **_kwargs: calls.__setitem__("badges", calls["badges"] + 1)
        or {"mode": "queue"},
    )
    monkeypatch.setattr(
        "jupr_app.services.direct_match_entry_service.queue_player_updates",
        lambda **_kwargs: calls.__setitem__("queue", calls["queue"] + 1)
        or {"mode": "queued"},
    )

    result = _submit(monkeypatch, supabase)

    assert [name for name, _payload in supabase.calls] == [DIRECT_MATCH_RPC]
    assert result["match_write_committed"] is True
    assert result["result"]["inserted"] == 1
    assert result["feedback"]["latest_match_id"] == "101"
    assert result["feedback"]["ratings_updated"] is True
    assert calls == {"badges": 1, "queue": 1}


def test_exact_retry_returns_stored_result_without_postprocessors(monkeypatch):
    receipt = _receipt(idempotent=False)
    supabase = RpcSupabase(
        _receipt(idempotent=True),
        preflight_receipt=receipt,
    )
    monkeypatch.setattr(
        "jupr_app.services.direct_match_entry_service.run_badge_side_effects",
        lambda **_kwargs: pytest.fail("idempotent retry must not rerun badges"),
    )
    monkeypatch.setattr(
        "jupr_app.services.direct_match_entry_service.queue_player_updates",
        lambda **_kwargs: pytest.fail("idempotent retry must not requeue updates"),
    )

    result = _submit(
        monkeypatch,
        supabase,
        processor=lambda *_args, **_kwargs: pytest.fail(
            "stored retry must not rebuild a rating plan"
        ),
    )

    assert result["operation"]["idempotent"] is True
    assert result["result"]["badge_summary"]["mode"] == "idempotent_retry_skipped"
    assert result["feedback"]["affected_players"][0]["rating_after"] == 1210
    assert supabase.calls == []


def test_stale_conflict_is_explicit_and_never_falls_back_to_table_writes(
    monkeypatch,
):
    supabase = RpcSupabase(
        error=RuntimeError(
            "JUPR_DIRECT_MATCH_PLAYER_STALE: doubles state changed"
        )
    )

    with pytest.raises(DirectMatchConflictError, match="Nothing"):
        _submit(monkeypatch, supabase)

    assert [name for name, _payload in supabase.calls] == [DIRECT_MATCH_RPC]
    assert len(supabase.calls) == 1


def test_same_idempotency_key_with_changed_body_conflicts_before_planning(
    monkeypatch,
):
    receipt = _receipt()
    supabase = RpcSupabase(preflight_receipt=receipt)

    with pytest.raises(DirectMatchConflictError, match="different match"):
        submit_atomic_direct_matches(
            supabase,
            club_id="club",
            matches=[{**MATCH, "score_t2": 8}],
            match_format="doubles",
            idempotency_key="score:operation-1",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            source="test",
            name_to_id={},
            df_players_all=_players(),
            df_leagues=pd.DataFrame(),
            df_meta=pd.DataFrame(),
        )

    assert supabase.calls == []


def test_ambiguous_response_requires_exact_idempotent_retry(monkeypatch):
    supabase = RpcSupabase(error=TimeoutError("connection reset after request"))

    with pytest.raises(
        DirectMatchRecoveryRequiredError,
        match="idempotency key prevents duplicate",
    ):
        _submit(monkeypatch, supabase)

    assert [name for name, _payload in supabase.calls] == [DIRECT_MATCH_RPC]


def test_request_fingerprint_is_stable_and_binds_changed_scores():
    request_one, fingerprint_one = stable_direct_match_request(
        club_id="club",
        match_format="doubles",
        matches=[MATCH],
    )
    request_two, fingerprint_two = stable_direct_match_request(
        club_id="club",
        match_format="doubles",
        matches=[dict(MATCH)],
    )
    _changed, changed_fingerprint = stable_direct_match_request(
        club_id="club",
        match_format="doubles",
        matches=[{**MATCH, "score_t2": 8}],
    )

    assert request_one == request_two
    assert fingerprint_one == fingerprint_two
    assert changed_fingerprint != fingerprint_one
