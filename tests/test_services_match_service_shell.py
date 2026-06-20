from jupr_app.services.context import ServiceContext
from jupr_app.services.match_service import submit_match_batch
from jupr_app.services.result_types import ServiceResult


def test_service_context_creation_is_worker_safe():
    ctx = ServiceContext(supabase=object(), club_id="club-123", source="streamlit")

    assert ctx.club_id == "club-123"
    assert ctx.source == "streamlit"
    assert ctx.actor_email is None


def test_service_result_helpers():
    ok = ServiceResult.success(data={"count": 1}, warnings=["heads-up"])
    assert ok.ok is True
    assert ok.data == {"count": 1}
    assert ok.warnings == ["heads-up"]
    assert ok.errors == []

    failed = ServiceResult.failure("boom")
    assert failed.ok is False
    assert failed.data == {}
    assert failed.warnings == []
    assert failed.errors == ["boom"]

    failed_many = ServiceResult.failure(["a", "b"], data={"step": 2}, warnings=["warn"])
    assert failed_many.errors == ["a", "b"]
    assert failed_many.data == {"step": 2}
    assert failed_many.warnings == ["warn"]


def test_submit_match_batch_wraps_process_matches(monkeypatch):
    captured = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        captured["kwargs"] = kwargs
        return {"processed": len(match_list), "club": kwargs["club_id"]}

    monkeypatch.setattr("jupr_app.services.match_service.process_matches", fake_process_matches)

    ctx = ServiceContext(supabase=object(), club_id="club-xyz")
    matches = [{"t1_p1": 1, "t2_p1": 2}]
    fake_retry = object()

    result = submit_match_batch(
        ctx,
        matches,
        name_to_id={"A": 1},
        df_players_all="players",
        df_leagues="leagues",
        df_meta="meta",
        sb_retry=fake_retry,
        default_k_factor=40,
        min_win_delta_elo=2.5,
        cap_loser_gain_elo=12.0,
    )

    assert result.ok is True
    assert result.errors == []
    assert result.data == {"processed": 1, "club": "club-xyz"}
    assert captured["match_list"] == matches
    assert captured["kwargs"]["club_id"] == "club-xyz"
    assert captured["kwargs"]["supabase"] is ctx.supabase
    assert captured["kwargs"]["sb_retry"] is fake_retry
    assert captured["kwargs"]["default_k_factor"] == 40
    assert captured["kwargs"]["min_win_delta_elo"] == 2.5
    assert captured["kwargs"]["cap_loser_gain_elo"] == 12.0
