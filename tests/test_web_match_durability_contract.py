from pathlib import Path


MATCH_PANEL = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
MATCH_BULK_EXCLUDE_PANEL = Path(
    "apps/web/app/admin/match-log/MatchLogBulkExcludePanel.tsx"
)
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


def test_match_log_forwards_secondary_recovery_context_filters() -> None:
    page = MATCH_PAGE.read_text(encoding="utf-8")
    api = MATCH_API.read_text(encoding="utf-8")

    assert "context_type?: string;" in page
    assert "context_id?: string;" in page
    assert "context_ids?: string;" in page
    assert "contextType: contextTypeParam" in page
    assert "contextIds: contextIdsParam" in page
    assert 'query.set("context_type", String(params.contextType))' in api
    assert 'query.set("context_id", String(params.contextId))' in api
    assert 'query.set("context_ids", contextIds)' in api
    assert 'name="context_ids"' in page
    assert "Advanced recovery context" in page
    assert "setRawData(null)" in page
    assert "dataScope === requestScope" in page
    assert 'key={`context-type-${contextTypeParam || "all"}`}' in page
    assert 'key={`context-ids-${contextIdsParam || "all"}`}' in page


def test_match_log_destructive_controls_follow_endpoint_availability() -> None:
    panel = MATCH_PANEL.read_text(encoding="utf-8")
    page = MATCH_PAGE.read_text(encoding="utf-8")
    api = MATCH_API.read_text(encoding="utf-8")
    bulk_exclude = MATCH_BULK_EXCLUDE_PANEL.read_text(encoding="utf-8")

    assert "exclude_endpoint?: string | null;" in api
    assert (
        "duplicateCleanupEnabled={Boolean(data.correction_plan.duplicate_cleanup_endpoint)}"
        in page
    )
    assert (
        "enabled={Boolean(data.correction_plan.exclude_endpoint)}" in page
    )
    assert "duplicateCleanupEnabled ? (" in panel
    assert "Destructive duplicate cleanup is disabled." in panel
    assert "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE" in bulk_exclude


def test_replay_ui_exposes_durable_job_identity_and_history() -> None:
    form = REPLAY_FORM.read_text(encoding="utf-8")
    page = REPLAY_PAGE.read_text(encoding="utf-8")

    assert "idempotency_key" in form
    assert "result.job_id" in form
    assert "result.job_status" in form
    assert "Recent durable replay jobs" in form
    assert "useAuthenticatedAutoLoad" in form
    assert "includeJobs: true" in form
    assert "cache: \"no-store\"" in Path("apps/web/lib/adminReplayApi.ts").read_text(encoding="utf-8")
    assert "data.recent_jobs" not in page
    assert "singles_replay_supported === true" in form
    assert 'result.mode === "replay_incomplete"' in form
    assert "if ((payload as AdminReplayResultResponse).ok) setIdempotencyKey" in form
    assert 'result.target_reset !== "ALL (Full System Reset)"' in form
