from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read_web(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_tournament_route_context_preserves_the_selected_day() -> None:
    helper = read_web("lib/tournamentRouteContext.ts")

    assert "dayId: string" in helper
    assert 'readValue(searchParams, "day")' in helper
    assert 'readValue(searchParams, "day_id")' in helper
    assert 'params.set("day", context.dayId)' in helper


def test_day_ops_transport_is_typed_and_isolated_from_the_draw_client() -> None:
    client = read_web("lib/adminTournamentDayOpsApi.ts")

    assert "AdminTournamentDayWorkspaceSnapshot" in client
    assert "AdminTournamentDayCourt" in client
    assert "AdminTournamentDayQueueEntry" in client
    assert "AdminTournamentDayGame" in client
    assert "AdminTournamentDayGameScoreInput" in client
    assert "AdminTournamentDayGameScore" in client
    assert "game_scores?: AdminTournamentDayGameScore[]" in client
    assert "game_scores?: AdminTournamentDayGameScoreInput[]" in client
    assert "individual_game_format?" in client
    assert "individual_game_target?" in client
    assert "individual_game_win_by_two?" in client
    assert "AdminTournamentDayCommandAction" in client
    for action in (
        "activate_day",
        "activate_draw",
        "pause_draw",
        "resume_draw",
        "auto_fill_courts",
        "assign_next_court",
        "assign_game_to_court",
        "reserve_game_for_court",
        "requeue_game",
        "move_game_to_court",
        "score_and_release",
        "correct_completed_score",
        "generate_playoffs",
        "close_day",
    ):
        assert action in client
    assert "/days/${encodeURIComponent(options.dayId)}" in client
    assert "`${dayBase(options)}/snapshot`" in client
    assert "`${dayBase(options)}/commands`" in client
    assert "client_idempotency_key" in client
    assert "day_run_version" in client
    assert "state_fingerprint" in client
    assert "close_day: AdminTournamentDayReadiness" in client
    assert "correct_completed_score: AdminTournamentDayReadiness" in client
    assert "correction_readiness: AdminTournamentDayReadiness" in client
    assert "assignments: AdminTournamentDayReadiness" in client
    assert "allowed_advance_counts: number[]" in client
    assert "default_advance_count: number | null" in client
    assert "advance_count?: number" in client
    assert "court_id?: string" in client
    assert "target_court_version?: string" in client
    assert 'action: AdminTournamentDayCommandAction | "reconcile"' in client
    assert "scope: {" in client
    assert "club_id: string" in client
    assert "Record<string, unknown>[]" not in client


def test_live_operations_base_is_the_canonical_day_console() -> None:
    page = read_web("app/admin/tournaments/live-operations/page.tsx")
    route = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspaceRoute.tsx")
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    nav = read_web("components/TournamentPhaseNav.tsx")

    assert "TournamentDayWorkspaceRoute" in page
    assert "TournamentLiveRoute" not in page
    assert "initialDayId={context.dayId}" in route
    assert "TournamentDayWorkspacePanel" in route
    assert 'aria-label="Tournament day scope"' in panel
    assert 'aria-labelledby="day-workspace-tab-board"' in panel
    assert 'aria-label="Tournament match queue"' in panel
    assert "snapshot.eligible_queue" in panel
    assert "Oldest ready first" in panel
    assert "oldestReadyQueue" in panel
    assert ".sort((left, right) => Number(left.priority || 0)" not in panel
    assert "#{entry.priority || entry.position}" not in panel
    assert "Held and blocked matches" in panel
    assert "Recovery required" in panel
    assert 'confirmationText="RECONCILE DAY OPERATIONS"' in panel
    assert "Day workspace" in nav


def test_day_console_tabs_render_one_clean_workspace_panel_at_a_time() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    css = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css")

    assert 'role="tablist"' in panel
    assert 'role="tab"' in panel
    assert "aria-selected={panelFocus === panel}" in panel
    assert "tabIndex={panelFocus === panel ? 0 : -1}" in panel
    for focus in ("board", "queue", "draws", "corrections"):
        assert f'panelFocus === "{focus}" ? (' in panel
        assert f'id="day-workspace-{focus}"' in panel
        assert f'aria-labelledby="day-workspace-tab-{focus}"' in panel
    assert 'document.getElementById(`day-workspace-${panel}`)?.focus()' not in panel
    assert 'button[aria-selected="true"]' in css
    assert '.viewNav button[aria-pressed="true"]' not in css


def test_day_console_places_global_actions_with_their_labeled_tab() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")

    assert 'panelFocus === "draws" && !dayStarted' in panel
    assert 'panelFocus === "draws" && dayStarted' in panel
    assert 'panelFocus === "corrections" && writesFrozen' in panel
    assert 'panelFocus === "corrections" ? (\n            <details className={styles.technicalDetails}' in panel
    assert 'setPanelFocus("queue")' not in panel
    assert 'panel: "queue"' in panel


def test_day_console_renders_authoritative_courts_and_progression_controls() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    css = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css")

    assert "snapshot.courts.map" in panel
    assert "rr_slot_number" not in panel
    assert "Activate draw" in panel
    assert "Pause draw" in panel
    assert "Resume draw" in panel
    assert "Fill available courts" not in panel
    assert "Send to next open court" in panel
    assert "Choose court" in panel
    assert "Move or remove" in panel
    assert "Return game to queue" in panel
    assert '"assign_next_court"' in panel
    assert '"assign_game_to_court"' in panel
    assert '"reserve_game_for_court"' in panel
    assert '"requeue_game"' in panel
    assert '"move_game_to_court"' in panel
    assert "target_court_version" in panel
    assert "Progression" in panel
    assert "score_and_release" in panel
    assert "generate_playoffs" in panel
    assert "Review playoff setup" in panel
    assert "Generate reviewed playoffs" in panel
    assert "playoff_configuration" in panel
    assert "seed_team_ids" in panel
    assert "round_scoring" in panel
    assert "Round-robin summary" in panel
    assert "Bracket preview" in panel
    assert "Close tournament day" in panel
    assert 'submitCommand("close_day"' in panel
    assert "CLOSE TOURNAMENT DAY" in read_web(
        "lib/tournamentDayWorkspaceState.mjs"
    )
    assert "Review score" not in panel
    assert "Confirm this score and release the court?" not in panel
    assert 'selectedScoreIsBestOfThree ? " series" : ""' in panel
    assert "InteractionDialog" in panel
    assert 'title={`Enter result · ${selectedScoreCourt.label}`}' in panel
    assert "Inline score and release" not in panel
    assert "data-autofocus" in panel
    assert "Non-play result" in panel
    assert "Use the non-played outcome command" in panel
    assert '"record_non_played_result"' in panel
    assert "Team that forfeited" in panel
    assert "Team that retired" in panel
    assert "Team that did not show" in panel
    assert "non_playing_team_id" in panel
    assert "Operator note" in panel and "(optional)" in panel
    assert "Previously played scores remain unchanged for player ratings" in panel
    assert "Completed games before retirement (optional)" in panel
    assert "validateBestOfThreeRetirementGameScores" in panel
    assert 'mode="retirement"' in panel
    assert "Completed rating games" in panel
    assert "game_scores: retirementGameScores" in panel
    assert "synthetic progression" in panel
    assert "Unusual score" in panel
    assert "unusual_score_acknowledgement" in panel
    assert "BestOfThreeScoreFields" in panel
    assert "Individual game scores" in panel
    assert "Game 3 appears only if the teams split the first two games" in panel
    assert 'aria-live="polite"' in panel
    assert "game_scores: validatedGameScores" in panel
    assert "draw.readiness.assignments" in panel
    assert "Court assignment evidence" in panel
    assert "entry.note ||" in panel
    assert "grid-template-columns: repeat(5, minmax(0, 1fr))" in css
    assert "@media (max-width: 1320px)" in css
    assert "@media (max-width: 640px)" in css
    assert "overflow-x: clip" in css


def test_day_console_visually_and_textually_marks_medal_matches_in_both_queues() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    css = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css")
    state = read_web("lib/tournamentDayWorkspaceState.mjs")
    client = read_web("lib/adminTournamentDayOpsApi.ts")

    assert "tournamentDayMedalMatchKind(game)" in panel
    assert panel.count("data-medal-match={medalKind || undefined}") == 2
    assert "Gold medal match" in panel
    assert "Bronze medal match" in panel
    assert ".goldMedalMatch" in css
    assert ".bronzeMedalMatch" in css
    assert ".goldMedalBadge" in css
    assert ".bronzeMedalBadge" in css
    assert '"PLAYOFF"' in state
    assert '"FINAL"' in state
    assert '"BRONZE"' in state
    assert "playoff_round?: string | null" in client


def test_day_court_assignment_is_manual_atomic_and_version_fenced() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    state = read_web("lib/tournamentDayWorkspaceState.mjs")

    assert 'submitCommand(\n      "assign_next_court"' in panel
    assert '? "reserve_game_for_court"\n      : "assign_game_to_court"' in panel
    assert "dayActionConfirmation(action)" in panel
    assert '"reserve_game_for_court"' in panel
    assert 'submitCommand(\n      "requeue_game"' in panel
    assert 'submitCommand(\n      "move_game_to_court"' in panel
    assert "queue_entry_version: entry.version" in panel
    assert "court_version: targetCourt.version" in panel
    assert "target_court_version: targetCourt.version" in panel
    assert "Court assignment controls closed because the queue or court board changed" in panel
    assert "reserved next for the selected occupied court" in panel
    assert "refills it from the server-ordered eligible queue" not in panel
    assert "refills available courts" not in panel
    assert "ASSIGN NEXT OPEN COURT" in state
    assert "ASSIGN GAME TO COURT" in state
    assert "RETURN GAME TO QUEUE" in state
    assert "MOVE GAME TO COURT" in state
    assert "readyActiveDrawQueue" in panel
    assert 'aria-label="Ready and court-reserved games from active draws"' in panel
    assert "courtBoardQueue.map" in panel
    assert "Next on this court" in panel
    assert "promotedReservationNotice" in panel
    assert "flex-wrap: nowrap" in read_web(
        "app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css"
    )


def test_day_console_corrects_only_server_authorized_completed_scores() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    route = read_web("app/admin/tournaments/live-operations/corrections/page.tsx")
    legacy = read_web("app/admin/tournament-live/TournamentLivePanel.tsx")
    state = read_web("lib/tournamentDayWorkspaceState.mjs")

    assert '"corrections"' in panel
    assert "readiness.correct_completed_score" in panel
    assert "game.correction_readiness" in panel
    assert "Correct completed score" in panel
    assert "Before correction" in panel
    assert "After correction" in panel
    assert "validateBestOfThreeCorrectionDraft" in panel
    assert "Corrected individual games" in panel
    assert "game_scores: selectedCorrectionValidatedGames" in panel
    assert "CORRECT COMPLETED SCORE" in state
    assert 'submitCommand(\n                          "correct_completed_score"' in panel
    assert "expected: expectedVersions({ draw_version: draw.version, game_version: game.version })" in panel
    assert "correctionEditor.expected" in panel
    assert "PLAYOFF_RESET_REQUIRED" in panel
    assert "redirect(tournamentRouteHref" in route
    assert 'panel: "corrections"' in route
    assert "context.dayId" in route
    assert "Day-owned completed scores must use the guarded tournament-day correction workspace" in legacy


def test_day_score_outcome_and_correction_editors_are_version_fenced_and_isolated() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")

    assert panel.count("expected: AdminTournamentDayCommandExpected") >= 3
    assert "reviewedGame: ReviewedGameTruth" in panel
    assert "gameScores: string" in panel
    assert "gameScores: JSON.stringify(normalizedGameScores(game.game_scores))" in panel
    assert "current.gameScores === reviewed.gameScores" in panel
    assert "reviewedAssignmentVersion" in panel
    assert "reviewedGameStillCurrent" in panel
    assert "expectedSnapshotChanged" in panel
    assert "Score editor closed because authoritative tournament-day state changed" in panel
    assert "Correction editor closed because the reviewed result" in panel
    assert "Non-played outcome editor closed because the reviewed matchup" in panel
    assert "scoreEditor.expected" in panel
    assert "correctionEditor.expected" in panel
    assert "outcomeEditor.expected" in panel
    assert "setCorrectionEditor(null);\n    setOutcomeEditor(null);" in panel
    assert "setScoreEditor(null);\n    setCorrectionEditor(null);" in panel
    assert 'id="day-non-play-entry-form"' in panel
    assert 'title={selectedOutcomeCourt ? `Enter result · ${selectedOutcomeCourt.label}` : "Record non-play result"}' in panel
    assert 'document.getElementById("non-played-outcome-editor")' not in panel
    assert "scrollIntoView" not in panel


def test_day_result_dialog_submits_directly_and_retains_uncertain_recovery() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    css = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css")
    non_play_submit = panel[
        panel.index("async function saveOutcome") : panel.index("if (sessionLoading)")
    ]

    assert 'id="day-score-entry-form"' in panel
    assert 'void saveScore()' in panel
    assert 'void saveOutcome()' in panel
    assert 'await resultAction.run(() => submitCommand(' in panel
    assert 'resultAction.recover(uncertainResult.onRecover)' in panel
    assert 'phase={resultAction.phase}' in panel
    assert 'className={`${styles.notice} ${styles.statusToast}`}' in panel
    assert ".statusToast" in css
    assert 'setPanelFocus("queue")' not in panel
    assert "score_a:" not in non_play_submit
    assert "score_b:" not in non_play_submit
    assert "game_scores: retirementGameScores" in non_play_submit
    assert "unusual_score_acknowledgement: outcomeEditor.unusualScoreAcknowledged" in non_play_submit


def test_day_snapshot_identity_is_checked_for_club_tournament_and_day() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")

    assert "payload.scope?.club_id" in panel
    assert "payload.scope?.tournament_id" in panel
    assert "payload.scope?.registration_day_id" in panel
    assert "assertWorkspaceSnapshotScope(payload, clubId, tournamentId, selectedDayId)" in panel
    assert "assertWorkspaceSnapshotScope(result.snapshot, clubId, tournamentId, command.dayId)" in panel
    assert "assertWorkspaceSnapshotScope(result.snapshot, clubId, tournamentId, selectedDayId)" in panel


def test_day_start_initializes_courts_before_draws_are_activated_individually() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    client = read_web("lib/adminTournamentDayOpsApi.ts")

    assert 'submitCommand("activate_day", confirmationText, {})' in panel
    assert "Draws stay inactive until you activate them one at a time" in panel
    assert "draftDayDrawIds" not in panel
    assert "activationSeedDayRef" not in panel
    assert "drawChecklist" not in panel
    assert "draw_ids?: string[]" not in client


def test_legacy_draw_and_score_routes_redirect_into_console_focus() -> None:
    scoring = read_web("app/admin/tournament-live/page.tsx")
    draws = read_web("app/admin/tournaments/live-operations/draws/page.tsx")

    assert "redirect(" in scoring
    assert 'panel: "queue"' in scoring
    assert "TournamentLiveRoute" not in scoring
    assert "redirect(" in draws
    assert 'panel: "draws"' in draws
    assert "TournamentLiveRoute" not in draws


def test_check_in_and_focused_live_routes_receive_day_context() -> None:
    check_in_page = read_web("app/admin/tournaments/live-operations/check-in/page.tsx")
    check_in_panel = read_web(
        "app/admin/tournaments/live-operations/check-in/TournamentCheckInPanel.tsx"
    )
    live_route = read_web(
        "app/admin/tournaments/live-operations/TournamentLiveRoute.tsx"
    )

    assert "initialDayId={context.dayId}" in check_in_page
    assert "tournamentRouteHref" in check_in_panel
    assert "dayId" in check_in_panel
    assert "initialDayId={context.dayId}" in live_route


def test_podium_review_never_substitutes_the_draw_live_fingerprint() -> None:
    api_types = read_web("lib/adminTournamentApi.ts")
    panel = read_web("app/admin/tournament-live/TournamentLivePanel.tsx")
    review = panel[panel.index("async function reviewPodium"):panel.index("async function reconcileOperation")]

    assert "ops_state_fingerprint?: string | null" in api_types
    assert ".ops_state_fingerprint" in review
    assert "expected_state_fingerprint: opsStateFingerprint" in review
    assert "expected_state_fingerprint: snapshot.state_fingerprint" not in review
    assert "Podium review needs the current Tournament Ops fingerprint" in panel
    assert "!podiumOpsFingerprint" in panel
    assert 'payload.scope !== "draw"' in panel


def test_day_console_is_explicit_and_accessible_about_live_scope() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")

    assert "new Date().toISOString().slice(0, 10)" not in panel
    assert "days.length === 1 ? days[0] : null" in panel
    assert "occupied ? assignment?.state : court.state" in panel
    assert 'id="day-score-error"' in panel
    assert 'aria-describedby={selectedScoreError ? "day-score-error" : undefined}' in panel
    assert 'id={`${idPrefix}-game-${score.game_number}-score-a`}' in panel
    assert 'id={`${idPrefix}-game-${score.game_number}-score-b`}' in panel
    assert "aria-labelledby={`${idPrefix}-game-${score.game_number}-title`}" in panel


def test_finished_round_robin_requires_one_stale_safe_playoff_review() -> None:
    panel = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.tsx")
    client = read_web("lib/adminTournamentDayOpsApi.ts")
    state = read_web("lib/tournamentDayWorkspaceState.mjs")
    css = read_web("app/admin/tournaments/live-operations/TournamentDayWorkspacePanel.module.css")

    assert "progression_alerts" in client
    assert "round_robin_summary" in client
    assert "tiebreak_explanations" in client
    assert "playoff_review" in client
    assert "playoff_review_fingerprint" in client
    assert "newlyReadyPlayoffNotice" in panel
    assert "Round robin complete" in panel
    assert "Round robin complete — playoff review needs attention" in panel
    assert "Review playoff setup" in panel
    assert 'size="xwide"' in panel
    assert "Round-robin summary" in panel
    assert "How tied teams were ranked" in panel
    assert "tiebreakCriterionLabel" in panel
    assert "Reset to round-robin order" in panel
    assert "Scoring by playoff round" in panel
    assert "Bracket preview" in panel
    assert "Generate reviewed playoffs" in panel
    assert "playoff_configuration" in panel
    assert "seed_team_ids" in panel
    assert "round_scoring" in panel
    assert "expectedSnapshotChanged(playoffEditor.expected)" in panel
    assert "!playoffEditor.reviewFingerprint" in panel
    assert "!currentReviewFingerprint" in panel
    assert "currentReviewFingerprint !== playoffEditor.reviewFingerprint" in panel
    assert "The server has not supplied a current playoff review fingerprint" in panel
    assert "readyPlayoffReviewDraws" in state
    assert "validatePlayoffReviewConfiguration" in state
    assert ".playoffReadyBanner" in css
    assert ".playoffReadyDraw" in css
    assert ".tiebreakAudit" in css
