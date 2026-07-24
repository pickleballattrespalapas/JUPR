from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _async_function_body(source: str, name: str) -> str:
    marker = f"async function {name}"
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated async function {name}")


TOKEN_ACTIONS: dict[str, tuple[str, ...]] = {
    "apps/web/app/admin/tournaments/status/TournamentStatusPanel.tsx": (
        "submitAction",
    ),
    "apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx": (
        "saveBulkUpdate",
    ),
    "apps/web/app/admin/tournaments/delete-draft/DeleteDraftPanel.tsx": (
        "deleteDraft",
    ),
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx": (
        "saveTournamentEdit",
        "saveRegistrationEdit",
        "saveSelectionEdit",
    ),
    "apps/web/app/admin/tournaments/registrations/RegistrationManagementPanel.tsx": (
        "exportCsv",
        "previewBroadcast",
    ),
    "apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx": (
        "applyAction",
    ),
    "apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx": (
        "createSession",
        "updateSession",
        "saveScores",
        "advanceRound",
        "publishMatches",
        "reconcileOperation",
    ),
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx": (
        "createLeagueDraft",
        "duplicateLeagueDraft",
        "transitionLeagueLifecycle",
        "saveLeagueSettings",
        "previewLeagueSchedule",
        "saveRosterMembership",
    ),
    "apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx": (
        "createSession",
        "saveSessionSnapshot",
        "generatePreview",
        "previewPythonMovement",
        "submitRound",
        "createGuest",
        "reconcileRound",
        "verifyCompensation",
        "downloadExport",
    ),
    "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx": (
        "createDraw",
        "importRegistrations",
        "importBulkTeams",
        "saveTeams",
        "generateGames",
        "generatePlayoffs",
        "saveScore",
        "generatePodium",
        "awardPodium",
        "publishOfficialMatches",
        "previewResultsImport",
        "commitResultsImport",
    ),
    "apps/web/app/admin/tournament-live/TournamentLivePanel.tsx": (
        "executePending",
        "reconcileOperation",
    ),
    "apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx": (
        "normalize",
        "inspectOperation",
    ),
    "apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx": (
        "runRecompute",
        "revokeBadge",
        "updateBadgeState",
        "inspectBadgeOperation",
    ),
    "apps/web/app/admin/players/PlayerEditorPanel.tsx": (
        "createPlayer",
        "savePlayer",
        "saveLeagueRating",
        "saveSocialIdentity",
        "autoLinkSocialIdentities",
        "previewMerge",
        "executeMerge",
        "lookupMergeOperation",
        "attachReplayEvidence",
        "compensateMerge",
    ),
    "apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx": (
        "updateChallenge",
        "simpleAction",
        "createChallenge",
        "recordForfeit",
        "recordPass",
        "addRosterPlayer",
        "moveRosterPlayer",
        "previewRosterReplacement",
        "applyRosterReplacement",
        "savePlayerOverrides",
        "previewResult",
        "publishResult",
        "reconcileLastOperation",
    ),
}


def test_modified_admin_actions_are_scoped_to_the_current_access_token() -> None:
    for relative, action_names in TOKEN_ACTIONS.items():
        source = _source(relative)
        assert "const actionRequest = useLatestRequestGuard(accessToken" in source, relative
        for action_name in action_names:
            body = _async_function_body(source, action_name)
            assert "actionRequest.begin()" in body, f"{relative}:{action_name}"
            assert "actionRequest.isCurrent(" in body, f"{relative}:{action_name}"


def test_existing_shared_action_guards_remain_token_scoped() -> None:
    support = _source("apps/web/app/admin/support-requests/SupportRequestsPanel.tsx")
    support_action = _async_function_body(support, "saveStatus")
    assert "useLatestRequestGuard(accessToken, clearProtectedSupportRequests)" in support
    assert "requestsRequest.begin()" in support_action
    assert "requestsRequest.isCurrent(" in support_action

    player_updates = _source("apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx")
    shared_action = _async_function_body(player_updates, "runAction")
    assert "const actionRequest = useLatestRequestGuard(accessToken);" in player_updates
    assert "actionRequest.begin()" in shared_action
    assert "actionRequest.isCurrent(" in shared_action
    for wrapper in (
        "queueDigests",
        "sendSelected",
        "retrySelected",
        "deleteSelected",
        "replaceSubscriber",
        "deactivateSubscriber",
    ):
        assert "await runAction(" in _async_function_body(player_updates, wrapper)

    replay = _source("apps/web/app/admin/replay-history/ReplayHistoryForm.tsx")
    replay_action = _async_function_body(replay, "onSubmit")
    assert "useLatestRequestGuard(accessToken, clearProtectedReplayState)" in replay
    assert "replayRequest.begin()" in replay_action
    assert "replayRequest.isCurrent(" in replay_action

    for relative, request_name, actions in (
        (
            "apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx",
            "writeRequest",
            ("generateDraft", "saveDraft", "publishAction"),
        ),
        (
            "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx",
            "actionRequest",
            ("runAction", "saveOverrides"),
        ),
        (
            "apps/web/app/admin/tools/AdminToolsPanel.tsx",
            "actionRequest",
            (
                "saveRole",
                "runQueueWorker",
                "moderateSocialSubmission",
                "applyTournamentMatchBackfill",
                "runBadgeRecompute",
                "inspectOperation",
                "recoverTournamentBackfill",
            ),
        ),
    ):
        source = _source(relative)
        assert f"useLatestRequestGuard(accessToken" in source
        for action in actions:
            body = _async_function_body(source, action)
            assert f"{request_name}.begin()" in body, f"{relative}:{action}"
            assert f"{request_name}.isCurrent(" in body, f"{relative}:{action}"


def test_durable_key_derivation_cannot_repopulate_logout_state() -> None:
    for relative in (
        "apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx",
        "apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx",
    ):
        source = _source(relative)
        durable = _async_function_body(source, "durableFields")
        assert "setLastOperationKey" not in durable, relative
        assert "operationKey" in durable, relative


def test_admin_session_ignores_unrelated_storage_events() -> None:
    auth = _source("apps/web/lib/adminAuthClient.ts")
    hook = _source("apps/web/lib/useAdminSession.ts")
    card = _source("apps/web/app/admin/AdminSessionCard.tsx")

    assert 'export const ADMIN_SESSION_STORAGE_KEY = "jupr_admin_session_v1";' in auth
    assert "key === null || key === ADMIN_SESSION_STORAGE_KEY" in auth
    assert "adminSessionStorageEventIsRelevant(event.key)" in hook
    assert 'window.addEventListener("storage", handleStorage);' in hook
    assert 'window.addEventListener("storage", load);' not in hook
    assert "adminSessionStorageEventIsRelevant(event.key)" in card
    assert 'window.addEventListener("storage", handleStorage);' in card
