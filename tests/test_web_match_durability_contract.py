from pathlib import Path


MATCH_PANEL = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
REPLAY_FORM = Path("apps/web/app/admin/replay-history/ReplayHistoryForm.tsx")
REPLAY_PAGE = Path("apps/web/app/admin/replay-history/page.tsx")


def test_match_log_closes_notes_bulk_and_recovery_gaps() -> None:
    source = MATCH_PANEL.read_text(encoding="utf-8")

    assert "Match notes" in source
    assert "Bulk stage visible matches" in source
    assert "Stage bulk changes" in source
    assert "Shift UTC date" in source
    assert "Replace player slot" in source
    assert "idempotency_key" in source
    assert "replay_target" in source
    assert "Mandatory replay recovery required" in source
    assert "Type RECOVER" in source
    assert "cannot be cleared. Choose a replacement player instead." in source


def test_replay_ui_exposes_durable_job_identity_and_history() -> None:
    form = REPLAY_FORM.read_text(encoding="utf-8")
    page = REPLAY_PAGE.read_text(encoding="utf-8")

    assert "idempotency_key" in form
    assert "result.job_id" in form
    assert "result.job_status" in form
    assert "Recent durable replay jobs" in page
    assert "data.recent_jobs" in page
