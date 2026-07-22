from pathlib import Path


MATCH_PANEL = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
MATCH_PAGE = Path("apps/web/app/admin/match-log/page.tsx")
MATCH_API = Path("apps/web/lib/adminMatchLogApi.ts")
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
    assert 'confirmationText="RECOVER"' in source
    assert 'title="Retry this mandatory replay?"' in source
    assert "cannot be cleared. Choose a replacement player instead." in source


def test_match_log_filters_are_selectable_clearable_and_keep_results_first() -> None:
    page = MATCH_PAGE.read_text(encoding="utf-8")
    api = MATCH_API.read_text(encoding="utf-8")

    assert '<select key={`league-${leagueParam || "all"}`} name="league"' in page
    assert '<select key={`week-${weekTagParam || "all"}`} name="week_tag"' in page
    assert '<option value="">All leagues</option>' in page
    assert '<option value="">All weeks</option>' in page
    assert '<input name="league"' not in page
    assert '<input name="week_tag"' not in page
    assert '<Link href="/admin/match-log"' in page
    assert ">Clear filters</Link>" in page
    assert "if (selected?.trim()) options.add(selected.trim());" in page
    assert "filter_options?:" in api
    assert page.index('data-testid="match-log-results"') < page.index("<h2>Duplicate scan</h2>")


def test_replay_ui_exposes_durable_job_identity_and_history() -> None:
    form = REPLAY_FORM.read_text(encoding="utf-8")
    page = REPLAY_PAGE.read_text(encoding="utf-8")

    assert "idempotency_key" in form
    assert "result.job_id" in form
    assert "result.job_status" in form
    assert "Recent durable replay jobs" in page
    assert "data.recent_jobs" in page
