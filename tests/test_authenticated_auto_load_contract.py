from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_shared_hook_waits_for_auth_and_deduplicates_token_scope_pairs() -> None:
    hook = _read("apps/web/lib/useAuthenticatedAutoLoad.ts")

    assert 'if (!accessToken)' in hook
    assert 'loadedKeyRef.current = "";' in hook
    assert "loadRef.current = load;" in hook
    assert "const requestKey = `${accessToken}\\u0000${scopeKey}`;" in hook
    assert "if (loadedKeyRef.current === requestKey) return;" in hook
    assert "loadedKeyRef.current = requestKey;" in hook
    assert "void loadRef.current();" in hook
    assert "}, [accessToken, scopeKey]);" in hook
    assert "}, [accessToken, load]);" not in hook


def test_shared_request_guard_invalidates_late_selection_responses() -> None:
    hook = _read("apps/web/lib/useAuthenticatedAutoLoad.ts")

    assert 'export function useLatestRequestGuard(scopeKey = "", onScopeChange?: () => void)' in hook
    assert "currentScopeRef.current = scopeKey;" in hook
    assert "onScopeChangeRef.current?.();" in hook
    assert "generationRef.current += 1;" in hook
    assert "generationRef.current === generation && requestScopeRef.current === currentScopeRef.current" in hook
    assert "return useMemo(() => ({ begin, invalidate, isCurrent })" in hook


def test_named_operator_surfaces_auto_load_lists_and_keep_only_recovery_controls() -> None:
    expectations = {
        "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx": ("loadLeagues", "Refresh leagues"),
        "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx": ("loadLeagues", "Refresh leagues"),
        "apps/web/app/admin/top-players-printable/TopPlayersPrintablePanel.tsx": ("loadRankings", "Refresh rankings"),
        "apps/web/app/admin/tournament-live/TournamentLivePanel.tsx": ("loadTournaments", "Refresh tournaments"),
        "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx": ("loadTournaments", "Refresh tournaments"),
        "apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx": ("loadTournaments", "Refresh tournaments"),
        "apps/web/app/admin/tournaments/registrations/RegistrationManagementPanel.tsx": ("loadTournaments", "Refresh tournaments"),
        "apps/web/app/admin/tournaments/delete-draft/DeleteDraftPanel.tsx": ("loadTournaments", "Refresh draft tournaments"),
        "apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx": ("loadInitialWorkspace", "Refresh sessions"),
    }
    forbidden = (
        ">Load leagues<",
        ">Load tournaments<",
        '"Load leagues"',
        '"Load tournaments"',
        "Load selected",
        "Load ops snapshot",
        "Load prepared draws",
        "Open authoritative board",
    )

    for relative, (loader, recovery_label) in expectations.items():
        source = _read(relative)
        assert "useAuthenticatedAutoLoad(" in source, relative
        assert loader in source, relative
        assert recovery_label in source, relative
        for label in forbidden:
            assert label not in source, f"{relative}: {label}"


def test_admin_buttons_do_not_reintroduce_known_manual_prerequisites() -> None:
    admin_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "apps/web/app/admin").rglob("*.tsx")
    )
    forbidden_labels = (
        '"Load leagues"',
        '"Load tournaments"',
        '"Load draft tournaments"',
        '"Load requests"',
        '"Load selected"',
        '"Load sessions"',
        '"Load players"',
        '"Load options"',
        '"Load players/leagues"',
        '"Load Admin Tools"',
        '"Load review queue"',
        '"Load Challenge Ladder"',
        '"Load prepared draws"',
        '"Load ops snapshot"',
        '"Open authoritative board"',
    )

    for label in forbidden_labels:
        assert label not in admin_source, label


def test_secondary_operator_queues_auto_load_and_filter_changes_refetch() -> None:
    scoped = {
        "apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx": "filter",
        "apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx": '`${startDate}\\u0000${endDate}`',
        "apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx": 'filter || "all"',
        "apps/web/app/admin/tools/AdminToolsPanel.tsx": "socialSubmissionStatus",
    }
    unscoped = (
        "apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx",
        "apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx",
        "apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx",
        "apps/web/app/admin/players/PlayerEditorPanel.tsx",
        "apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx",
    )

    for relative, scope in scoped.items():
        source = _read(relative)
        assert "useAuthenticatedAutoLoad(" in source, relative
        assert scope in source, relative

    for relative in unscoped:
        assert "useAuthenticatedAutoLoad(" in _read(relative), relative


def test_selection_changes_clear_old_records_and_ignore_late_responses() -> None:
    guarded = (
        "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
        "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx",
        "apps/web/app/admin/tournament-live/TournamentLivePanel.tsx",
        "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
        "apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx",
        "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx",
        "apps/web/app/admin/tournaments/registrations/RegistrationManagementPanel.tsx",
        "apps/web/app/admin/players/PlayerEditorPanel.tsx",
        "apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx",
    )

    for relative in guarded:
        source = _read(relative)
        assert "useLatestRequestGuard" in source, relative
        assert ".begin()" in source, relative
        assert ".isCurrent(generation)" in source, relative

    tournament_live = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.tsx")
    assert "selectTournament(event.target.value)" in tournament_live
    assert "selectDraw(event.target.value)" in tournament_live
    assert "setSnapshot(null);" in tournament_live

    weekly = _read("apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx")
    assert "selectRecap(event.target.value)" in weekly
    assert "setSelectedRecap(null);" in weekly


def test_refresh_preserves_valid_tournament_selections_and_reloads_current_detail() -> None:
    expectations = {
        "apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx": (
            'setSelectedTournamentId("");',
            "await loadDetail(selectedBeforeRefresh)",
        ),
        "apps/web/app/admin/tournament-live/TournamentLivePanel.tsx": (
            'setSelectedTournamentId("");',
            "await loadDraws(selectedTournamentBeforeRefresh, selectedDrawBeforeRefresh)",
        ),
        "apps/web/app/admin/tournaments/registrations/RegistrationManagementPanel.tsx": (
            'setSelectedTournamentId("");',
            "await loadDetail(selectedBeforeRefresh, true)",
        ),
    }

    for relative, (selection_clear, reload_contract) in expectations.items():
        source = _read(relative)
        refresh_setup = source.split("async function loadTournaments", 1)[1].split("try {", 1)[0]
        assert selection_clear not in refresh_setup, relative
        assert "BeforeRefresh" in source, relative
        assert reload_contract in source, relative

    tournament_live = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.tsx")
    assert "preferredDrawId = selectedDrawId" in tournament_live
    assert "nextDraws.some((row) => row.id === preferredDrawId)" in tournament_live
    assert 'void loadDraws(tournamentId, "")' in tournament_live



def test_player_editor_workspace_ignores_logout_and_retry_races() -> None:
    source = _read("apps/web/app/admin/players/PlayerEditorPanel.tsx")

    assert "const workspaceRequest = useLatestRequestGuard(accessToken, clearProtectedPlayerState);" in source
    assert "const detailRequest = useLatestRequestGuard(accessToken);" in source
    assert "const socialRequest = useLatestRequestGuard(accessToken);" in source
    assert "const generation = workspaceRequest.begin();" in source
    assert "if (!workspaceRequest.isCurrent(generation)) return;" in source
    assert "if (workspaceRequest.isCurrent(generation)) setSaving(false);" in source
    assert "function clearProtectedPlayerState()" in source
    assert "playersRequest.invalidate();" in source
    assert "detailRequest.invalidate();" in source


def test_admin_tools_scopes_reads_independently_and_guards_every_secondary_action() -> None:
    source = _read("apps/web/app/admin/tools/AdminToolsPanel.tsx")

    assert source.count("useAuthenticatedAutoLoad(") >= 2
    assert "loadAdminToolsWorkspace" not in source
    assert 'flaggedOnly ? "flagged" : "all"' in source
    assert "socialSubmissionStatus" in source
    assert "const overviewRequest = useLatestRequestGuard(accessToken, clearProtectedAdminToolsState);" in source
    assert "const socialQueueRequest = useLatestRequestGuard(accessToken);" in source
    assert "const actionRequest = useLatestRequestGuard(accessToken);" in source
    assert source.count("const generation = actionRequest.begin();") >= 8
    assert source.count("if (!actionRequest.isCurrent(generation)) return") >= 8
    assert "overviewMessage" in source
    assert "socialQueueMessage" in source


def test_player_updates_auto_loads_date_ranges_and_ignores_old_session_responses() -> None:
    source = _read("apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx")

    assert "const workspaceScope = `${accessToken}\\u0000${startDate}\\u0000${endDate}`;" in source
    assert "const workspaceRequest = useLatestRequestGuard(workspaceScope, clearProtectedWorkspace);" in source
    assert "const actionRequest = useLatestRequestGuard(accessToken);" in source
    assert "useAuthenticatedAutoLoad(" in source
    assert "if (!workspaceRequest.isCurrent(generation)) return" in source
    assert "if (!actionRequest.isCurrent(generation)) return" in source
    assert "Loading the selected date range…" in source
    assert "Refresh workspace" in source
    assert "Reload workspace" not in source
    assert 'const [loadedWorkspaceScope, setLoadedWorkspaceScope] = useState("");' in source
    assert "const workspaceIsCurrentRange = Boolean(accessToken && workspace && loadedWorkspaceScope === workspaceScope);" in source
    assert "const currentWorkspace = workspaceIsCurrentRange ? workspace : null;" in source
    assert "const workspaceControlsDisabled = busy || workspaceLoading || !workspaceIsCurrentRange;" in source
    assert (
        "const mutationControlsDisabled = workspaceControlsDisabled || "
        "!status.mutations_enabled;"
    ) in source
    assert "setLoadedWorkspaceScope(requestedWorkspaceScope);" in source
    assert 'disabled={workspaceControlsDisabled}' in source
    assert source.count("disabled={mutationControlsDisabled ||") >= 6
    assert "setReplacementEmail(\"\");" in source
    assert "setReplacementNote(\"\");" in source
    assert "setPreview(null);" in source
    assert "{preview && workspaceIsCurrentRange ?" in source


def test_replay_history_ignores_old_token_responses_and_clears_protected_results() -> None:
    source = _read("apps/web/app/admin/replay-history/ReplayHistoryForm.tsx")

    assert "const replayRequest = useLatestRequestGuard(accessToken, clearProtectedReplayState);" in source
    assert "function clearProtectedReplayState()" in source
    assert "setResult(null);" in source
    assert "setIdempotencyKey(requestKey());" in source
    assert "const generation = replayRequest.begin();" in source
    assert "if (replayRequest.isCurrent(generation)) {" in source
    assert "if (replayRequest.isCurrent(generation)) setPending(false);" in source
    assert "disabled={pending}" in source


def test_league_live_clears_stale_session_state_before_detail_reads() -> None:
    source = _read("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")
    session_loader = source.split("async function loadSessionDetail", 1)[1].split("function selectSession", 1)[0]
    league_loader = source.split("async function loadLeagueDetail", 1)[1].split("function selectLeague", 1)[0]

    assert "const sessionDetailRequest = useLatestRequestGuard(accessToken);" in source
    assert "const leagueDetailRequest = useLatestRequestGuard(accessToken);" in source
    assert session_loader.index("clearPersistedSessionBinding(selectedSessionId);") < session_loader.index("await requestJson<LeagueLiveDetailResponse>")
    assert league_loader.index("clearPersistedSessionBinding();") < league_loader.index("await requestJson<AdminLeagueManagerDetailResponse>")
    assert "The previous league roster remains visible" not in source
    assert "!detail || loadedLeagueName !== leagueName || !rosterSuggestion || !sessionRoster.length" in source


def test_admin_session_revalidation_is_shared_and_fails_closed() -> None:
    source = _read("apps/web/lib/useAdminSession.ts")
    auth = _read("apps/web/lib/adminAuthClient.ts")

    assert 'import { useSyncExternalStore } from "react";' in source
    assert "let sharedSnapshot: AdminSessionState = serverSnapshot;" in source
    assert "let restoreRequest: Promise<void> | null = null;" in source
    assert "const listeners = new Set<() => void>();" in source
    assert "if (restoreRequest) return restoreRequest;" in source
    assert "if (!background) emit(snapshotFromSession(null, { loading: true }));" in source
    assert "emit(snapshotFromSession(authorized));" in source
    assert "snapshotFromSession(null" in source
    assert "stored && adminSessionIsFresh(stored) && stored.capabilities?.authorized" in source
    assert "if (eventSource === SHARED_SESSION_CHANGE_SOURCE) return;" in source
    assert "return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);" in source
    assert "const storageSnapshot =" in auth
    assert "const storageIsUnchanged =" in auth
    assert auth.count("if (!storageIsUnchanged()) return null;") >= 3
    assert "saveAdminSession(authorized, { changeSource: options.changeSource });" in auth
    assert "clearAdminSession({ changeSource: options.changeSource });" in auth

def test_secondary_reports_actions_and_recap_writes_ignore_old_token_responses() -> None:
    badge = _read("apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx")
    canonical = _read("apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx")
    ladder = _read("apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx")
    awards = _read("apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx")
    recap = _read("apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx")

    assert "const reportRequest = useLatestRequestGuard(accessToken);" in badge
    assert badge.count("const generation = reportRequest.begin();") >= 2
    assert badge.count("if (!reportRequest.isCurrent(generation)) return;") >= 2
    assert "const auditRequest = useLatestRequestGuard(accessToken);" in canonical
    assert "if (!auditRequest.isCurrent(generation)) return;" in canonical
    assert "const tierReviewRequest = useLatestRequestGuard(accessToken);" in ladder
    assert "if (!tierReviewRequest.isCurrent(generation)) return;" in ladder
    assert "const actionRequest = useLatestRequestGuard(accessToken);" in awards
    assert awards.count("if (!actionRequest.isCurrent(generation)) return;") >= 2
    assert "const writeRequest = useLatestRequestGuard(accessToken);" in recap
    assert recap.count("const generation = writeRequest.begin();") >= 3
    assert recap.count("if (!writeRequest.isCurrent(generation)) return") >= 5


def test_league_awards_browser_flow_uses_accessible_confirmation_dialogs() -> None:
    source = _read("apps/web/e2e/league-awards.staging.spec.ts")

    assert 'getByRole("button", { name: "Yes, freeze league" })' in source
    assert 'getByRole("button", { name: "Yes, mint and verify" })' in source
    assert 'getByRole("button", { name: "Yes, archive league" })' in source
    assert "getByPlaceholder" not in source
    assert "expect.poll(() => leagueListReads).toBe(1)" in source
    assert "expect.poll(() => awardsStateReads).toBe(1)" in source


def test_empty_failure_retry_and_reset_states_remain_explicit() -> None:
    sources = "\n".join(
        _read(relative)
        for relative in (
            "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
            "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx",
            "apps/web/app/admin/tournament-live/TournamentLivePanel.tsx",
            "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
            "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx",
            "apps/web/app/admin/players/PlayerEditorPanel.tsx",
            "apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx",
        )
    )

    assert "No leagues are available" in sources
    assert "No tournaments" in sources
    assert "No saved recaps" in sources
    assert "Unable to load" in sources
    assert "Retry" in sources
    assert "Refresh" in sources
    assert "setDetail(null)" in sources
