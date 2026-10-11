"""Automatic legacy publish callers carry only their committed match IDs."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain import match_processing, singles_match_processing
from jupr_app.domain.matches.persistence import insert_match_chunks_with_rating_scope_fallback
from jupr_app.services import admin_challenge_ladder_service as ladder
from jupr_app.services import admin_tournament_match_publish_service as tournament
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_player_update_queueing import _Supabase, _players_df


class _InsertReceipts:
    def __init__(self, *, omit_optional=False, bad_receipt=None):
        self.payloads = []
        self.next_id = 401
        self.omit_optional = omit_optional
        self.bad_receipt = bad_receipt

    def table(self, name):
        assert name == "matches"
        return self

    def insert(self, rows):
        self.payload = deepcopy(rows)
        return self

    def execute(self):
        if self.omit_optional and any("rating_scope" in row for row in self.payload):
            raise RuntimeError("column rating_scope does not exist in matches schema cache")
        self.payloads.append(self.payload)
        rows = []
        for payload in self.payload:
            rows.append({**payload, "id": self.next_id})
            self.next_id += 1
        if self.bad_receipt == "missing_id":
            rows[-1].pop("id")
        elif self.bad_receipt == "partial":
            rows.pop()
        elif self.bad_receipt == "duplicate":
            rows[-1]["id"] = rows[0]["id"]
        elif self.bad_receipt == "invalid":
            rows[-1]["id"] = True
        return SimpleNamespace(data=rows)


def test_insert_receipts_keep_exact_ids_across_chunks_and_optional_column_fallback():
    sb = _InsertReceipts(omit_optional=True)
    ids = insert_match_chunks_with_rating_scope_fallback(
        db_matches=[{"club_id": "club", "rating_scope": "overall"} for _ in range(301)],
        supabase=sb,
        sb_retry=lambda fn: fn(),
    )
    assert ids == list(range(401, 702))
    assert [len(chunk) for chunk in sb.payloads] == [300, 1]
    assert all("rating_scope" not in row for chunk in sb.payloads for row in chunk)


@pytest.mark.parametrize("bad_receipt", ["missing_id", "partial", "duplicate", "invalid"])
def test_ambiguous_insert_receipt_never_provides_a_partial_email_scope(bad_receipt):
    sb = _InsertReceipts(bad_receipt=bad_receipt)
    ids = insert_match_chunks_with_rating_scope_fallback(
        db_matches=[{"club_id": "club"}, {"club_id": "club"}],
        supabase=sb,
        sb_retry=lambda fn: fn(),
    )
    assert ids == []
    assert len(sb.payloads) == 1


@pytest.mark.parametrize("format_name", ["doubles", "singles"])
def test_legacy_processors_queue_actual_insert_ids_without_prior_same_day_matches(monkeypatch, format_name):
    sb = _Supabase()
    sb.tables["matches"] = [{"id": 900, "club_id": "club", "date": "2026-10-10"}]
    players = _players_df()
    players["singles_rating"] = 1200
    sb.tables["players"] = [{**row, "club_id": "club"} for row in players.to_dict("records")]
    module = singles_match_processing if format_name == "singles" else match_processing
    queue_calls = []
    badge_calls = []
    if format_name == "doubles":
        monkeypatch.setattr(module, "enqueue_badge_eval", lambda *args, **kwargs: badge_calls.append(kwargs) or {"queued": True})
        monkeypatch.setattr(module, "process_badge_eval_queue", lambda *args, **kwargs: {"processed": 1, "errored": 0})
    monkeypatch.setattr(
        module, "queue_player_updates_for_affected_subscribers",
        lambda *args, **kwargs: queue_calls.append(kwargs) or {"queued": 1, "failed": 0},
    )
    processor = module.process_singles_matches if format_name == "singles" else module.process_matches
    result = processor(
        [{
            "date": "2026-10-10", "league": "OVERALL", "match_format": format_name,
            "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 8,
        }],
        supabase=sb, club_id="club", name_to_id={},
        df_players_all=players, df_leagues=pd.DataFrame(), df_meta=pd.DataFrame(),
    )
    assert result["inserted"] == 1
    assert len(queue_calls) == 1
    assert queue_calls[0]["match_ids"] == [sb.tables["matches"][-1]["id"]]
    assert 900 not in queue_calls[0]["match_ids"]
    if format_name == "doubles":
        assert badge_calls[0]["match_id"] == str(sb.tables["matches"][-1]["id"])
        assert queue_calls[0]["badge_refs"] == [badge_calls[0]["match_id"]]
    else:
        assert queue_calls[0]["badge_refs"] == []


def _tournament_matches():
    return [
        {"id": 501, "club_id": "club", "tournament_id": "tour", "tournament_game_id": "game-a", "deleted_at": None},
        {"id": 502, "club_id": "club", "tournament_id": "tour", "tournament_game_id": "game-b", "deleted_at": None},
        {"id": 600, "club_id": "club", "tournament_id": "tour", "tournament_game_id": "other-game", "deleted_at": None},
        {"id": 601, "club_id": "other-club", "tournament_id": "tour", "tournament_game_id": "game-a", "deleted_at": None},
        {"id": 602, "club_id": "club", "tournament_id": "other-tour", "tournament_game_id": "game-b", "deleted_at": None},
    ]


def test_tournament_match_receipt_is_limited_to_club_tournament_and_exact_source_games():
    sb = FakeSupabase({"matches": _tournament_matches()})
    assert tournament._committed_tournament_match_ids(
        sb, club_id="club", tournament_id="tour", game_ids=["game-a", "game-b"]
    ) == [501, 502]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "deleted", "invalid_id"])
def test_tournament_match_receipt_fails_closed_for_incomplete_or_ambiguous_scope(failure):
    rows = _tournament_matches()
    if failure == "missing":
        rows.pop(0)
    elif failure == "duplicate":
        rows.append({**rows[0], "id": 599})
    elif failure == "deleted":
        rows[0]["deleted_at"] = "2026-10-10"
    else:
        rows[0]["id"] = None
    assert tournament._committed_tournament_match_ids(
        FakeSupabase({"matches": rows}), club_id="club", tournament_id="tour", game_ids=["game-a", "game-b"]
    ) == []


@pytest.mark.parametrize("badge_mode", ["queue", "inline_error", "error"])
def test_challenge_post_processors_forward_committed_receipt_scope(monkeypatch, badge_mode):
    calls = []
    monkeypatch.setattr(ladder, "run_badge_side_effects", lambda **kwargs: {"mode": badge_mode, "notification_badge_refs": [] if badge_mode.endswith("error") else ["operation-key"]})
    monkeypatch.setattr(ladder, "queue_player_updates", lambda **kwargs: calls.append(kwargs) or {"mode": "queued", "failed": 0})
    def post_process():
        return ladder._run_atomic_match_side_effects(
            object(), club_id="club", operation_key="operation-key",
            write_plan={"match_rows": [{"context_id": "operation-a"}, {"context_id": "operation-b"}], "player_updates": [{"player_id": 1, "expected": {"rating": 1200}, "after": {"rating": 1216}}]},
            side_effect_context={"affected_player_ids": [1, 2, 3, 4], "successful_match_dates": ["2026-10-10"]},
            match_ids=[501, 502],
        )
    if badge_mode.endswith("error"):
        with pytest.raises(RuntimeError, match="badge processing did not complete"):
            post_process()
        assert calls == []
        return
    post_process()
    assert len(calls) == 1
    assert calls[0]["match_ids"] == [501, 502]
    assert calls[0]["badge_refs"] == ["operation-key"]
    assert calls[0]["session_evidence"]["player_updates"][0]["after"]["rating"] == 1216


@pytest.mark.parametrize("badge_mode", ["queue", "inline_error", "error"])
def test_tournament_atomic_publish_forwards_committed_scope_rating_evidence_and_badge_job(monkeypatch, badge_mode):
    from tests.test_api_contract_admin_tournament_match_publish import (
        install_official_publish_prerequisites, match_publish_tables,
    )

    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH", "1")
    tables = match_publish_tables()
    install_official_publish_prerequisites(tables)
    sb = FakeSupabase(tables)
    plan = tournament.build_admin_tournament_official_publish_plan(
        sb, club_id="club", tournament_id="tour_1", draw_id="draw_1",
    )

    def commit(_supabase, **kwargs):
        rows = kwargs["write_plan"]["match_rows"]
        tables.setdefault("matches", []).extend({**row, "id": 501 + index, "deleted_at": None} for index, row in enumerate(rows))
        return {"ok": True, "inserted": len(rows)}

    calls = []
    badge_calls = []
    monkeypatch.setattr(tournament, "_apply_official_rating_plan_atomic", commit)
    monkeypatch.setattr(tournament, "run_badge_side_effects", lambda **kwargs: badge_calls.append(kwargs) or {"mode": badge_mode, "notification_badge_refs": [] if badge_mode.endswith("error") else [kwargs["dedupe_match_id"]]})
    monkeypatch.setattr(tournament, "queue_player_updates", lambda **kwargs: calls.append(kwargs) or {"mode": "queued", "failed": 0})
    monkeypatch.setattr(tournament, "auto_send_player_updates_for_match_payloads", lambda *args, **kwargs: {"mode": "disabled"})

    def publish():
        return tournament.publish_admin_tournament_draw_matches(
            sb, club_id="club", tournament_id="tour_1", draw_id="draw_1",
            actor_email="admin@example.com", actor_role="club_owner", confirmation_text="PUBLISH MATCHES",
            expected_plan=plan, guarded_operation_key="operation-key", guarded_request_fingerprint="request-fingerprint",
        )
    if badge_mode.endswith("error"):
        with pytest.raises(RuntimeError, match="post-processor failed"):
            publish()
        assert calls == []
        assert len(tables["matches"]) == 1
        return
    result = publish()

    assert result["match_count"] == 1
    assert calls[0]["match_ids"] == [501]
    assert len(calls[0]["session_evidence"]["player_updates"]) == 4
    assert calls[0]["badge_refs"] == ["operation-key"]
    assert badge_calls[0]["dedupe_match_id"] == "operation-key"


@pytest.mark.parametrize("enqueue_succeeded", [False, True])
@pytest.mark.parametrize("inline_succeeded", [False, True])
def test_legacy_badge_fallback_failure_never_queues_email_and_success_clears_job_refs(
    monkeypatch, enqueue_succeeded, inline_succeeded,
):
    sb = _Supabase()
    players = _players_df()
    sb.tables["players"] = [{**row, "club_id": "club"} for row in players.to_dict("records")]
    calls = []
    monkeypatch.setattr(match_processing, "enqueue_badge_eval", lambda *args, **kwargs: {"queued": enqueue_succeeded})
    monkeypatch.setattr(match_processing, "process_badge_eval_queue", lambda *args, **kwargs: {"processed": 0, "errored": 1})

    def inline(*args, **kwargs):
        if not inline_succeeded:
            raise RuntimeError("award computation failed")
        return {"mode": "inline", "awarded_count": 1}

    monkeypatch.setattr(match_processing, "run_live_badge_awards", inline)
    monkeypatch.setattr(match_processing, "queue_player_updates_for_affected_subscribers", lambda *args, **kwargs: calls.append(kwargs) or {"queued": 1, "failed": 0})
    result = match_processing.process_matches(
        [{"date": "2026-10-10", "league": "OVERALL", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "s1": 11, "s2": 8}],
        supabase=sb, club_id="club", name_to_id={}, df_players_all=players,
        df_leagues=pd.DataFrame(), df_meta=pd.DataFrame(),
    )
    assert result["inserted"] == len(sb.tables["matches"]) == 1
    assert all(row["matches_played"] == 1 for row in sb.tables["players"])
    if inline_succeeded:
        assert result["player_update_queue"]["mode"] == "queued"
        assert len(calls) == 1
        assert calls[0]["badge_refs"] == []
    else:
        assert result["player_update_queue"]["mode"] == "error"
        assert "award evaluation did not complete" in result["player_update_queue"]["error"]
        assert calls == []
