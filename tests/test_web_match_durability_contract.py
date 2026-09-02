from pathlib import Path


MATCH_PANEL = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
MATCH_BULK_EXCLUDE_PANEL = Path(
    "apps/web/app/admin/match-log/MatchLogBulkExcludePanel.tsx"
)
MATCH_PAGE = Path("apps/web/app/admin/match-log/MatchLogWorkspace.tsx")
MATCH_API = Path("apps/web/lib/adminMatchLogApi.ts")
MATCH_EXCLUSION_RECOVERY = Path(
    "apps/web/app/admin/match-log/MatchLogExclusionRecoveryPanel.tsx"
)
MATCH_QUICK_REPLAY_PANEL = Path(
    "apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx"
)
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
    assert 'operation.status === "pending_replay"' in source
    assert 'operation.status === "recovery_required"' in source
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
    assert "<Link href={modePath}" in page
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
    assert "Atomic duplicate soft-exclusion is disabled." in panel
    assert "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE" in bulk_exclude


def test_match_exclusion_ui_retains_exact_targets_and_operation_recovery() -> None:
    panel = MATCH_PANEL.read_text(encoding="utf-8")
    bulk_exclude = MATCH_BULK_EXCLUDE_PANEL.read_text(encoding="utf-8")
    recovery = MATCH_EXCLUSION_RECOVERY.read_text(encoding="utf-8")
    page = MATCH_PAGE.read_text(encoding="utf-8")
    api = MATCH_API.read_text(encoding="utf-8")

    assert "row_version?: number | null;" in api
    assert "AdminMatchExclusionTarget" in api
    assert "expected_row_version" in bulk_exclude
    assert "idempotency_key: idempotencyKey" in bulk_exclude
    assert "setIdempotencyKey(requestKey())" in bulk_exclude
    assert "targets," in panel
    assert "idempotency_key: duplicateIdempotencyKey" in panel
    assert "setDuplicateIdempotencyKey(requestKey())" in panel
    assert "Soft-exclude duplicate rows" in panel
    assert "Yes, delete duplicate rows" not in panel
    assert (
        "match-log/exclusions/${encodeURIComponent(activeOperation.id)}/recover"
        in recovery
    )
    assert 'confirmationText="RECOVER"' in recovery
    assert "It never repeats the soft exclusion." in recovery
    assert "recent_exclusion_operations" in page
    assert "exclusionOperation={exclusionOperation}" in page


def test_successful_match_log_mutations_refetch_parent_owned_data() -> None:
    page = MATCH_PAGE.read_text(encoding="utf-8")
    panel = MATCH_PANEL.read_text(encoding="utf-8")
    bulk_exclude = MATCH_BULK_EXCLUDE_PANEL.read_text(encoding="utf-8")
    recovery = MATCH_EXCLUSION_RECOVERY.read_text(encoding="utf-8")
    quick_replay = MATCH_QUICK_REPLAY_PANEL.read_text(encoding="utf-8")

    assert "const [reloadNonce, setReloadNonce] = useState(0);" in page
    assert "const handleMutationComplete = useCallback(() => {" in page
    handler = page[page.index("const handleMutationComplete"):page.index("useEffect(() =>", page.index("const handleMutationComplete"))]
    assert "mutationRefreshPending.current = true;" in handler
    assert "setRawData(null);" not in handler
    assert 'setDataScope("");' not in handler
    assert "setReplayData(null);" not in handler
    assert "setReloadNonce((current) => current + 1);" in page
    assert "getAdminMatchLog({" in page
    assert "getAdminReplayStatus(clubId, { accessToken, apiBase })" in page
    assert "Match and replay status refreshed from the server." in page
    assert "replay status could not be reloaded" in page
    assert "matchIdsParam,\n    reloadNonce,\n    selectedFilterParam" in page
    assert page.count("onMutationComplete={handleMutationComplete}") == 4

    for child in (panel, bulk_exclude, recovery, quick_replay):
        assert "onMutationComplete: () => void;" in child
        assert "onMutationComplete();" in child

    assert "router.refresh()" not in panel
    assert "router.refresh()" not in bulk_exclude
    assert 'from "next/navigation"' not in panel
    assert 'from "next/navigation"' not in bulk_exclude


def test_match_log_uses_focused_subpages_and_local_stage_feedback() -> None:
    page = MATCH_PAGE.read_text(encoding="utf-8")
    panel = MATCH_PANEL.read_text(encoding="utf-8")

    for section in ("edit", "bulk", "duplicates", "exclude", "social", "replay"):
        wrapper = Path(f"apps/web/app/admin/match-log/{section}/page.tsx")
        assert wrapper.exists()
        assert "MatchLogWorkspace" in wrapper.read_text(encoding="utf-8")

    assert 'mode: "review" | "edit" | "bulk" | "duplicates" | "exclude" | "social" | "replay"' in page
    assert 'aria-label="Match Log sections"' in page
    assert 'aria-current={item.mode === mode ? "page" : undefined}' in page
    assert "preserveFilters(item.path)" in page
    assert 'data-testid="match-log-staged-edits"' in panel
    assert "stagedEditsRef.current?.focus()" in panel
    assert "stagedEditsRef.current?.scrollIntoView" in panel
    assert 'role={messageTone === "success" ? "status" : "alert"}' in panel
    assert 'feedback("guided")' in panel
    assert 'feedback("bulk")' in panel
    assert 'feedback("apply")' in panel


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
    assert "if (!replayResult.ok) throw new Error(completionMessage);" in form
    assert "setIdempotencyKey(requestKey());" in form
    assert 'result.target_reset !== "ALL (Full System Reset)"' in form
