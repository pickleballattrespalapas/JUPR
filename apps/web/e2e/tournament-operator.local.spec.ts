import { expect, test, type Page } from "@playwright/test";

const tournamentId = "tournament-1";
const drawId = "draw-1";
const secondDrawId = "draw-2";
const dayId = "day-1";
const selectedQuery = `tournament=${tournamentId}&tournament_name=Staging+Summer+Classic&name=Staging+Summer+Classic&draw=${drawId}&day=${dayId}`;

const players = [
  { id: 1, name: "Mateo Rivera" },
  { id: 2, name: "Liam Chen" },
  { id: 3, name: "Caleb Nguyen" },
  { id: 4, name: "Diego Alvarez" }
];
const teamsForDraw = (selectedDrawId: string) => [
  { id: "team-a", draw_id: selectedDrawId, team_number: 1, player1_id: 1, player2_id: 2, updated_at: "2026-08-15T12:00:00Z" },
  { id: "team-b", draw_id: selectedDrawId, team_number: 2, player1_id: 3, player2_id: 4, updated_at: "2026-08-15T12:00:00Z" }
];

function games(selectedDrawId = drawId) {
  return Array.from({ length: 21 }, (_, index) => ({
    id: `game-${index + 1}`,
    draw_id: selectedDrawId,
    stage: "ROUND_ROBIN",
    rr_round_number: Math.floor(index / 7) + 1,
    rr_slot_number: (index % 7) + 1,
    team_a_id: "team-a",
    team_b_id: "team-b",
    score_a: index === 0 ? 11 : null,
    score_b: index === 0 ? 7 : null,
    winner_team_id: index === 0 ? "team-a" : null,
    updated_at: "2026-08-15T12:00:00Z"
  }));
}

function blocker(code: string, message: string, count?: number) {
  return { code, scope: "tournament", count, message };
}

function liveSnapshot(selectedDrawId = drawId) {
  const rows = games(selectedDrawId);
  const teams = teamsForDraw(selectedDrawId);
  const publishBlockers = [
    blocker("OPEN_GAMES", "20 tournament games still need a finalized, non-tied score.", 20),
    blocker("PODIUM_INCOMPLETE", "The podium is incomplete and has not been explicitly reviewed."),
    blocker("AWARDS_INCOMPLETE", "Required podium awards are incomplete."),
    blocker("OFFICIAL_MATCHES_INCOMPLETE", "Official Match Log publication and replay evidence are incomplete.")
  ];
  const operations = [
    { operation_key: "op-1", request_fingerprint: "a".repeat(64), client_idempotency_key: "00000000-0000-4000-8000-000000000001", action: "tournament_live", command: "save_score", status: "completed", expected_state: "state", attempt_count: 1, updated_at: "2026-08-15T12:05:00Z", audit_evidence: { actions: ["intent", "completion"], intent_present: true, completion_present: true, failure_present: false } }
  ];
  const readiness = {
    save_score: { ready: true, confirmation: "SAVE SCORE", blockers: [] },
    generate_round_robin: { ready: false, confirmation: "GENERATE GAMES", blockers: ["Games already exist."] },
    generate_playoffs: { ready: false, confirmation: "GENERATE PLAYOFFS", blockers: ["Finish 20 open round-robin games."] },
    generate_podium: { ready: false, confirmation: "GENERATE PODIUM", blockers: ["Finish every game first."] },
    award_podium: { ready: false, confirmation: "AWARD PODIUM", blockers: ["Generate and review the podium first."] },
    publish_official_matches: { ready: false, confirmation: "PUBLISH MATCHES", blockers: publishBlockers.map((row) => row.message) }
  };
  return {
    ok: true,
    mode: "tournament_live_draw_snapshot",
    scope: "draw",
    authority: "python_fastapi",
    product_boundary: "draw_scoped_tournament_runner_not_jupr_live",
    tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", updated_at: "2026-08-15T12:00:00Z" },
    draw_id: selectedDrawId,
    summary: { draws: 2, teams: 2, games: 21, podium: 0, completed_games: 1 },
    draws: [
      { id: drawId, tournament_id: tournamentId, registration_day_id: dayId, name: "Manual Acceptance Draw", status: "DRAFT", updated_at: "2026-08-15T12:00:00Z" },
      { id: secondDrawId, tournament_id: tournamentId, registration_day_id: dayId, name: "Open Division Draw", status: "DRAFT", updated_at: "2026-08-15T12:00:00Z" }
    ],
    registration_days: checkInDays,
    teams,
    games: rows,
    podium: [],
    players,
    state_fingerprint: "b".repeat(64),
    ops_state_fingerprint: "c".repeat(64),
    runtime: { enabled: true, status: "staging_write_ready", authority: "python_fastapi", product_boundary: "draw_scoped_tournament_runner_not_jupr_live", club_id: "tres_palapas", environment: "local_test_harness", staging_only: true, writes_enabled: true, service_role_ready: true, operation_store_ready: true, audit_store_ready: true, write_flag: { name: "LOCAL", enabled: true }, streamlit_fallback_url: "#", warnings: [] },
    progression: { phase: "live", open_games: 20, completed_games: 1, published_games: 0, expected_awards: 6, verified_awards: 0 },
    readiness,
    active_operation: null,
    operations,
    lifecycle: {
      contract: "jupr:tournament-lifecycle:v1",
      authority: "python_fastapi",
      scope: "tournament",
      tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", updated_at: "2026-08-15T12:00:00Z" },
      phase: "live_in_progress",
      counts: { draws: 2, teams: 4, games: 42, finalized_games: 2, open_games: 40, tied_games: 0, podium_entries: 0, expected_awards: 12, verified_awards: 0, unexpected_awards: 0, published_games: 0, unpublished_games: 42, duplicate_publications: 0, active_operations: 0, uncertain_operations: 0 },
      draws: [
        { draw_id: drawId, name: "Manual Acceptance Draw", status: "DRAFT", protected: false, counts: { games: 21, finalized_games: 1, open_games: 20, published_games: 0, duplicate_publications: 0 }, standings: [], podium: [], states: { live_operations: "in_progress", official_publish: "blocked" }, operations, review_evidence: null, readiness: { official_publish: { ready: false, blockers: publishBlockers }, archive: { ready: false, blockers: publishBlockers } } },
        { draw_id: secondDrawId, name: "Open Division Draw", status: "DRAFT", protected: false, counts: { games: 21, finalized_games: 1, open_games: 20, published_games: 0, duplicate_publications: 0 }, standings: [], podium: [], states: { live_operations: "in_progress", official_publish: "blocked" }, operations, review_evidence: null, readiness: { official_publish: { ready: false, blockers: publishBlockers }, archive: { ready: false, blockers: publishBlockers } } }
      ],
      domain_readiness: {
        official_publish: { ready: false, blockers: publishBlockers },
        completion: { ready: false, blockers: publishBlockers },
        archive: { ready: false, blockers: [...publishBlockers, blocker("ARCHIVE_OFFICIAL_LINKS", "All tournament games require exactly one official Match Log link before archive.")] }
      },
      runtime_capability: { writes_enabled: true, official_publish_enabled: true },
      evidence: { operations }
    }
  };
}

const detail = {
  ok: true,
  tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", registration_status: "closed", start_date: "2026-09-01", end_date: "2026-09-02", updated_at: "2026-08-15T12:00:00Z" },
  settings: { location_name: "Tres Palapas", timezone: "America/Chicago" },
  days: [{ id: "day-1" }, { id: "day-2" }],
  event_options: Array.from({ length: 7 }, (_, index) => ({ id: `event-${index + 1}` })),
  registrations: [],
  selections: [],
  summary: { registrations: 32, selections: 32, by_registration_status: { confirmed: 32 }, by_payment_status: { offline: 32 } }
};

const checkInDays = [
  { id: "day-1", label: "Tuesday — Gender Doubles", event_date: "2026-09-01", sort_order: 0 },
  { id: "day-2", label: "Wednesday — Mixed & Open Doubles", event_date: "2026-09-02", sort_order: 1 }
];

function checkInRegistrant(options: {
  registrationId: string;
  dayId: string;
  playerId: number;
  name: string;
  status: "EXPECTED" | "CHECKED_IN" | "ABSENT";
  eventLabel: string;
}) {
  return {
    registration_id: options.registrationId,
    registration_day_id: options.dayId,
    registration_status: "CONFIRMED",
    registration_updated_at: "2026-08-15T12:00:00Z",
    attendance_status: options.status,
    original_registrant: { player_id: options.playerId, name: options.name },
    attendee: { player_id: options.playerId, name: options.name, is_approved_substitute: false },
    substitution: {
      allowed: true,
      event_policy_allows: true,
      blocker: { code: "NONE", status: "COMPLETE", title: "Available", detail: "" }
    },
    check_in: {
      registration_day_id: options.dayId,
      attendance_status: options.status,
      checked_in: options.status === "CHECKED_IN",
      notes: null,
      updated_at: "2026-08-15T12:00:00Z",
      updated_by: "operator@example.invalid",
      identity_current: true,
      requires_reconfirmation: false
    },
    waiver: { verified: options.status === "CHECKED_IN", subject: "attending_player", subject_name: options.name },
    payment: { status: "PAID", source: "offline_payment_tracking", ready: true },
    events: [{
      selection_id: `selection-${options.registrationId}`,
      event_option_id: `event-${options.dayId}`,
      event_label: options.eventLabel,
      team_state: "NOT_REQUIRED",
      partner_name: null,
      entered_partner_name: null,
      blockers: []
    }],
    blockers: []
  };
}

function checkInSnapshot(dayId: string) {
  const selectedDay = checkInDays.find((day) => day.id === dayId) || checkInDays[0];
  const dayOne = selectedDay.id === "day-1";
  const registrants = dayOne
    ? [
        checkInRegistrant({ registrationId: "registration-1", dayId: selectedDay.id, playerId: 1, name: "Mateo Rivera", status: "CHECKED_IN", eventLabel: "Men's Doubles" }),
        checkInRegistrant({ registrationId: "registration-8", dayId: selectedDay.id, playerId: 2, name: "Jordan Lee", status: "EXPECTED", eventLabel: "Mixed Doubles" })
      ]
    : [
        checkInRegistrant({ registrationId: "registration-4", dayId: selectedDay.id, playerId: 4, name: "Diego Alvarez", status: "ABSENT", eventLabel: "Open Doubles" })
      ];
  const unresolvedParticipants = dayOne
    ? [{ kind: "NEEDS_PARTNER", registration_id: "registration-8", registration_name: "Jordan Lee", selection_id: "selection-registration-8", event_label: "Mixed Doubles", entered_partner_name: "", title: "Partner unresolved", detail: "Jordan Lee still needs a confirmed partner." }]
    : [];
  return {
    ok: true,
    mode: "tournament_registration_check_in",
    authority: "python_fastapi_supabase",
    tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", start_date: "2026-09-01", end_date: "2026-09-02" },
    day_scope: { selected_day_id: selectedDay.id, selected_day: selectedDay, available_days: checkInDays },
    summary: {
      expected: registrants.length,
      checked_in: registrants.filter((row) => row.attendance_status === "CHECKED_IN").length,
      not_checked_in: registrants.filter((row) => row.attendance_status === "EXPECTED").length,
      absent: registrants.filter((row) => row.attendance_status === "ABSENT").length,
      unresolved: unresolvedParticipants.length
    },
    registrants,
    player_options: players,
    inactive_registrants: [],
    unresolved_participants: unresolvedParticipants,
    registration_follow_up: [],
    readiness: {
      schedule: { status: "COMPLETE", timezone: "America/Chicago", active_day_count: 1, blockers: [], days: [selectedDay] },
      draws: { status: "COMPLETE", active_division_count: 1, draw_count: 1, blockers: [] },
      staffing: { status: "NEEDS_REVIEW", source: "no_authoritative_staffing_record", blockers: [{ code: "STAFFING_REVIEW_REQUIRED", status: "NEEDS_REVIEW", title: "Staffing needs review", detail: `Confirm staffing for ${selectedDay.label}.` }] }
    },
    completed_items: [],
    blockers: [],
    runtime: { writes_enabled: true }
  };
}

function dayGame(options: {
  id: string;
  drawId: string;
  drawName: string;
  round: string;
  slot: string;
  teamA: string[];
  teamB: string[];
  courtId?: string;
}) {
  return {
    id: options.id,
    draw_id: options.drawId,
    draw_name: options.drawName,
    state: "READY",
    stage: "ROUND_ROBIN",
    round_label: options.round,
    slot_label: options.slot,
    team_a: { team_id: `${options.id}-team-a`, name: options.teamA.join(" / "), participant_names: options.teamA },
    team_b: { team_id: `${options.id}-team-b`, name: options.teamB.join(" / "), participant_names: options.teamB },
    court_id: options.courtId ?? null,
    score_a: null as number | null,
    score_b: null as number | null,
    winner_name: null as string | null,
    result_type: "PLAYED",
    result_note: null as string | null,
    updated_at: "2026-08-17T09:00:00Z",
    version: `${options.id}-v1`,
    queue_entry_version: options.courtId ? "5" : "1",
    blockers: [],
    correction_readiness: dayReadiness(false, "CORRECT COMPLETED SCORE", "Only released completed results are correctable.")
  };
}

function dayReadiness(ready: boolean, confirmation: string, message = "") {
  return {
    ready,
    confirmation,
    blockers: ready || !message ? [] : [{ code: "NOT_READY", message }]
  };
}

function dayWorkspaceSnapshot() {
  const dayGames = [
    dayGame({ id: "day-game-a", drawId, drawName: "Manual Acceptance Draw", round: "Round 1", slot: "Match 1", teamA: ["Mateo Rivera", "Liam Chen"], teamB: ["Caleb Nguyen", "Diego Alvarez"], courtId: "day-court-1" }),
    dayGame({ id: "day-game-b", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 1", slot: "Match 2", teamA: ["Avery Patel", "Jordan Lee"], teamB: ["Morgan Diaz", "Riley Smith"] }),
    dayGame({ id: "day-game-c", drawId, drawName: "Manual Acceptance Draw", round: "Round 2", slot: "Match 1", teamA: ["Nora Williams", "Sofia Kim"], teamB: ["Emma Davis", "Mia Johnson"] }),
    dayGame({ id: "day-game-held", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 2", slot: "Match 2", teamA: ["Taylor Reed", "Casey Brooks"], teamB: ["Jamie Flores", "Skyler Moore"] }),
    dayGame({ id: "day-game-blocked", drawId, drawName: "Manual Acceptance Draw", round: "Round 3", slot: "Match 1", teamA: ["Quinn Parker", "Alexis Bell"], teamB: ["Cameron Price", "Robin Ward"] }),
    {
      ...dayGame({ id: "day-game-completed", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 3", slot: "Match 2", teamA: ["Avery Patel", "Jordan Lee"], teamB: ["Morgan Diaz", "Riley Smith"] }),
      state: "COMPLETED",
      score_a: 11,
      score_b: 7,
      winner_name: "Avery Patel / Jordan Lee",
      correction_readiness: dayReadiness(true, "CORRECT COMPLETED SCORE")
    }
  ];
  return {
    ok: true,
    mode: "tournament_day_live",
    scope: { club_id: "tres_palapas", tournament_id: tournamentId, registration_day_id: dayId },
    tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE" },
    day_scope: {
      selected_day_id: dayId,
      selected_day: checkInDays[0],
      available_days: checkInDays
    },
    day_run: {
      id: "day-run-1",
      registration_day_id: dayId,
      state: "ACTIVE",
      version: "7",
      updated_at: "2026-08-17T09:00:00Z"
    },
    state_fingerprint: "d".repeat(64),
    queue_version: "11",
    generated_at: "2026-08-17T09:05:00Z",
    summary: {
      courts: 10,
      available_courts: 9,
      active_draws: 2,
      eligible_games: 2,
      held_games: 1,
      completed_games: 4
    },
    draws: [
      {
        id: drawId,
        name: "Manual Acceptance Draw",
        state: "ACTIVE",
        activation_state: "ACTIVE",
        version: "3",
        stage: "Round robin",
        total_games: 8,
        finalized_games: 4,
        queued_games: 1,
        active_games: 1,
        held_games: 0,
        readiness: {
          activate: dayReadiness(false, "ACTIVATE DRAW", "This draw is already active."),
          pause: dayReadiness(true, "PAUSE DRAW"),
          resume: dayReadiness(false, "RESUME DRAW", "Pause this draw before resuming it."),
          generate_playoffs: { ...dayReadiness(false, "GENERATE PLAYOFFS", "Finish four open round-robin matchups."), allowed_advance_counts: [4], default_advance_count: null },
          podium: dayReadiness(false, "OPEN PODIUM", "Generate and finish playoffs first.")
        }
      },
      {
        id: secondDrawId,
        name: "Open Division Draw",
        state: "ACTIVE",
        activation_state: "ACTIVE",
        version: "4",
        stage: "Round robin complete",
        total_games: 6,
        finalized_games: 6,
        queued_games: 1,
        active_games: 0,
        held_games: 1,
        readiness: {
          activate: dayReadiness(false, "ACTIVATE DRAW", "This draw is already active."),
          pause: dayReadiness(true, "PAUSE DRAW"),
          resume: dayReadiness(false, "RESUME DRAW", "Pause this draw before resuming it."),
          generate_playoffs: { ...dayReadiness(true, "GENERATE PLAYOFFS"), allowed_advance_counts: [4, 5, 6], default_advance_count: null },
          podium: dayReadiness(false, "OPEN PODIUM", "Generate playoffs first.")
        }
      }
    ],
    courts: Array.from({ length: 10 }, (_, index) => ({
      id: `day-court-${index + 1}`,
      label: `Court ${index + 1}`,
      position: index + 1,
      state: index === 0 ? "ON_COURT" : "AVAILABLE",
      version: `court-${index + 1}-v2`,
      current_assignment: index === 0 ? {
        id: "assignment-a",
        game_id: "day-game-a",
        state: "ON_COURT",
        version: "assignment-v5",
        assigned_at: "2026-08-17T09:01:00Z",
        started_at: "2026-08-17T09:02:00Z" as string | null
      } : null
    })),
    games: dayGames,
    eligible_queue: [
      { game_id: "day-game-b", draw_id: secondDrawId, position: 1, priority: 50, state: "WAITING", version: "queue-b-v3", eligible_since: "2026-08-17T08:58:00Z", reason: "First eligible across active draws", blockers: [] },
      { game_id: "day-game-c", draw_id: drawId, position: 2, priority: 40, state: "WAITING", version: "queue-c-v2", eligible_since: "2026-08-17T08:59:00Z", reason: "Second eligible across active draws", blockers: [] }
    ],
    held_games: [
      { game_id: "day-game-held", draw_id: secondDrawId, state: "HELD", reason: "Operator hold awaiting participant arrival", note: null, held_at: "2026-08-17T09:03:00Z", version: "held-v1", blockers: [] }
    ],
    blocked_games: [
      { game_id: "day-game-blocked", draw_id: drawId, state: "BLOCKED", reason: "Participant already assigned", note: null, held_at: null, version: "blocked-v1", blockers: [{ code: "PLAYER_ALREADY_CLAIMED", message: "One participant is already assigned to another court." }] }
    ],
    operations: [] as Array<{
      operation_key: string;
      client_idempotency_key: string;
      action: string;
      status: string;
      entity_label: string;
      updated_at: string;
    }>,
    readiness: {
      activate_day: dayReadiness(false, "ACTIVATE DAY", "This day is already active."),
      auto_fill_courts: dayReadiness(true, "AUTO FILL COURTS"),
      close_day: dayReadiness(false, "CLOSE TOURNAMENT DAY", "Finish the day before close."),
      correct_completed_score: dayReadiness(true, "CORRECT COMPLETED SCORE")
    },
    runtime: { writes_enabled: true, warnings: [] },
    warnings: []
  };
}

async function installMockApi(page: Page) {
  let currentDaySnapshot = dayWorkspaceSnapshot();
  const checkInUpdates = new Map<string, {
    attendanceStatus: "EXPECTED" | "CHECKED_IN" | "ABSENT";
    waiverVerified: boolean;
    notes: string | null;
    updatedAt: string;
  }>();
  const currentCheckInSnapshot = (selectedDayId: string) => {
    const snapshot = structuredClone(checkInSnapshot(selectedDayId));
    snapshot.registrants.forEach((registrant) => {
      const update = checkInUpdates.get(registrant.registration_id);
      if (!update) return;
      registrant.attendance_status = update.attendanceStatus;
      registrant.check_in.attendance_status = update.attendanceStatus;
      registrant.check_in.checked_in = update.attendanceStatus === "CHECKED_IN";
      (registrant.check_in as unknown as { notes: string | null }).notes = update.notes;
      registrant.check_in.updated_at = update.updatedAt;
      registrant.waiver.verified = update.waiverVerified;
    });
    snapshot.summary.checked_in = snapshot.registrants.filter(
      (row) => row.attendance_status === "CHECKED_IN"
    ).length;
    snapshot.summary.not_checked_in = snapshot.registrants.filter(
      (row) => row.attendance_status === "EXPECTED"
    ).length;
    snapshot.summary.absent = snapshot.registrants.filter(
      (row) => row.attendance_status === "ABSENT"
    ).length;
    return snapshot;
  };
  await page.addInitScript(() => {
    window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
      access_token: "local-operator-token",
      token_type: "bearer",
      expires_at: Date.now() + 3_600_000,
      capabilities: { authorized: true, user: { email: "operator@example.invalid" }, assignments: [{ club_id: "tres_palapas", role: "admin", permissions: ["*"] }] },
      user: { email: "operator@example.invalid" }
    }));
  });
  await page.route("http://127.0.0.1:3999/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/admin/auth/capabilities") {
      await route.fulfill({ json: { authorized: true, user: { email: "operator@example.invalid" }, assignments: [{ club_id: "tres_palapas", role: "admin", permissions: ["*"] }] } });
      return;
    }
    if (url.pathname.endsWith("/tournaments/admin/ops/tournaments")) {
      await route.fulfill({ json: { ok: true, tournaments: [detail.tournament], count: 1 } });
      return;
    }
    if (url.pathname.endsWith(`/tournaments/admin/tournaments/${tournamentId}`) && request.method() === "GET") {
      await route.fulfill({ json: detail });
      return;
    }
    const dayWorkspaceBase = `/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/days/${dayId}`;
    if (url.pathname === `${dayWorkspaceBase}/snapshot` && request.method() === "GET") {
      await route.fulfill({ json: currentDaySnapshot });
      return;
    }
    if (url.pathname === `${dayWorkspaceBase}/commands` && request.method() === "POST") {
      const command = request.postDataJSON() as {
        action: string;
        client_idempotency_key: string;
        confirmation_text: string;
        payload: {
          draw_id?: string;
          advance_count?: number;
          game_id?: string;
          court_id?: string;
          score_a?: number;
          score_b?: number;
          result_type?: "FORFEIT" | "NO_SHOW" | "RETIREMENT";
          winner_team_id?: string;
          result_note?: string;
        };
      };
      const refreshed = structuredClone(currentDaySnapshot);
      if (command.action === "assign_next_court" || command.action === "assign_game_to_court") {
        const entry = refreshed.eligible_queue.find((row) => row.game_id === command.payload.game_id);
        const targetCourt = command.action === "assign_game_to_court"
          ? refreshed.courts.find((court) => court.id === command.payload.court_id)
          : refreshed.courts.find((court) => court.state === "AVAILABLE" && !court.current_assignment);
        const game = refreshed.games.find((row) => row.id === command.payload.game_id);
        if (entry && targetCourt && game) {
          targetCourt.state = "ON_COURT";
          targetCourt.current_assignment = {
            id: `assignment-${game.id}`,
            game_id: game.id,
            state: "ON_COURT",
            version: `${entry.version}-assigned`,
            assigned_at: "2026-08-17T09:06:00Z",
            started_at: "2026-08-17T09:06:00Z"
          };
          game.court_id = targetCourt.id;
          game.queue_entry_version = `${entry.version}-assigned`;
          refreshed.eligible_queue = refreshed.eligible_queue.filter((row) => row.game_id !== game.id);
          refreshed.summary.available_courts -= 1;
          refreshed.summary.eligible_games -= 1;
        }
      }
      if (command.action === "requeue_game" || command.action === "move_game_to_court") {
        const sourceCourt = refreshed.courts.find((court) => court.current_assignment?.game_id === command.payload.game_id);
        const game = refreshed.games.find((row) => row.id === command.payload.game_id);
        if (sourceCourt && game) {
          const assignment = sourceCourt.current_assignment;
          sourceCourt.current_assignment = null;
          sourceCourt.state = "AVAILABLE";
          if (command.action === "move_game_to_court") {
            const targetCourt = refreshed.courts.find((court) => court.id === command.payload.court_id);
            if (targetCourt && assignment) {
              targetCourt.current_assignment = { ...assignment, version: `${assignment.version}-moved`, assigned_at: "2026-08-17T09:07:00Z" };
              targetCourt.state = "ON_COURT";
              game.court_id = targetCourt.id;
              game.queue_entry_version = `${assignment.version}-moved`;
            }
          } else {
            game.court_id = null;
            game.queue_entry_version = `${assignment?.version || "queue"}-requeued`;
            refreshed.eligible_queue.push({
              game_id: game.id,
              draw_id: game.draw_id,
              position: refreshed.eligible_queue.length + 1,
              priority: 50,
              state: "WAITING",
              version: game.queue_entry_version,
              eligible_since: "2026-08-17T08:58:00Z",
              reason: "Returned to existing queue priority",
              blockers: []
            });
            refreshed.summary.available_courts += 1;
            refreshed.summary.eligible_games += 1;
          }
        }
      }
      if (command.action === "score_and_release") {
        const completedGame = refreshed.games.find((game) => game.id === command.payload.game_id);
        if (completedGame) {
          completedGame.score_a = command.payload.score_a ?? null;
          completedGame.score_b = command.payload.score_b ?? null;
          completedGame.winner_name = Number(command.payload.score_a) > Number(command.payload.score_b)
            ? completedGame.team_a.name
            : completedGame.team_b.name;
          completedGame.state = "COMPLETED";
          completedGame.version = `${completedGame.id}-v2`;
        }
        refreshed.courts[0].current_assignment = null;
        refreshed.courts[0].state = "AVAILABLE";
        refreshed.summary.available_courts += 1;
        refreshed.summary.completed_games += 1;
      }
      if (command.action === "record_non_played_result") {
        const completedGame = refreshed.games.find((game) => game.id === command.payload.game_id);
        if (completedGame) {
          completedGame.result_type = command.payload.result_type ?? "NO_SHOW";
          completedGame.result_note = command.payload.result_note ?? null;
          completedGame.winner_name = command.payload.winner_team_id === completedGame.team_a.team_id
            ? completedGame.team_a.name
            : completedGame.team_b.name;
          completedGame.state = "COMPLETED";
          completedGame.version = `${completedGame.id}-v2`;
        }
        refreshed.courts[0].current_assignment = null;
        refreshed.courts[0].state = "AVAILABLE";
        refreshed.summary.available_courts += 1;
        refreshed.summary.completed_games += 1;
      }
      if (command.action === "correct_completed_score") {
        const correctedGame = refreshed.games.find((game) => game.id === command.payload.game_id);
        if (correctedGame) {
          correctedGame.score_a = command.payload.score_a ?? null;
          correctedGame.score_b = command.payload.score_b ?? null;
          correctedGame.winner_name = Number(command.payload.score_a) > Number(command.payload.score_b)
            ? correctedGame.team_a.name
            : correctedGame.team_b.name;
          correctedGame.version = `${correctedGame.id}-v2`;
        }
      }
      refreshed.queue_version = String(Number(currentDaySnapshot.queue_version) + 1);
      refreshed.state_fingerprint = Number(refreshed.queue_version).toString(16).slice(-1).repeat(64);
      refreshed.generated_at = "2026-08-17T09:06:00Z";
      refreshed.operations = [{
        operation_key: "day-operation-1",
        client_idempotency_key: command.client_idempotency_key,
        action: command.action,
        status: "completed",
        entity_label: "Tournament day",
        updated_at: refreshed.generated_at
      }];
      currentDaySnapshot = refreshed;
      await route.fulfill({
        json: {
          command: {
            action: command.action,
            confirmation_text: command.confirmation_text,
            idempotent_replay: false
          },
          operation: {
            operation_key: "day-operation-1",
            client_idempotency_key: command.client_idempotency_key,
            action: command.action,
            status: "completed",
            entity_label: "Tournament day",
            updated_at: refreshed.generated_at
          },
          snapshot: refreshed
        }
      });
      return;
    }
    if (url.pathname.startsWith(`${dayWorkspaceBase}/operations/`) && url.pathname.endsWith("/reconcile")) {
      await route.fulfill({
        json: {
          command: { action: "auto_fill_courts", confirmation_text: "RECONCILE DAY OPERATIONS", idempotent_replay: true },
          operation: { operation_key: "day-operation-1", client_idempotency_key: "reconciled", action: "auto_fill_courts", status: "completed" },
          snapshot: currentDaySnapshot
        }
      });
      return;
    }
    if (url.pathname.includes(`/tournament-live/tournaments/${tournamentId}/snapshot`)) {
      await route.fulfill({ json: liveSnapshot(url.searchParams.get("draw_id") || drawId) });
      return;
    }
    if (url.pathname.endsWith(`/tournament-live/tournaments/${tournamentId}/check-in`) && request.method() === "GET") {
      await route.fulfill({ json: currentCheckInSnapshot(url.searchParams.get("day_id") || "day-1") });
      return;
    }
    if (url.pathname.endsWith(`/tournament-live/tournaments/${tournamentId}/check-in/bulk`) && request.method() === "POST") {
      const dayId = url.searchParams.get("day_id") || "day-1";
      const input = request.postDataJSON() as {
        operation_key: string;
        updates: Array<{
          registration_id: string;
          attendance_status?: "EXPECTED" | "CHECKED_IN" | "ABSENT";
          waiver_verified?: boolean;
          notes?: string | null;
        }>;
      };
      const before = currentCheckInSnapshot(dayId);
      const checkIns = input.updates.map((update, index) => {
        const registrant = before.registrants.find(
          (row) => row.registration_id === update.registration_id
        );
        if (!registrant) throw new Error(`Unknown mocked registration ${update.registration_id}`);
        const updatedAt = `2026-08-15T12:02:${String(index).padStart(2, "0")}Z`;
        const attendanceStatus = update.attendance_status || registrant.attendance_status;
        const waiverVerified = update.waiver_verified ?? registrant.waiver.verified;
        const notes = Object.prototype.hasOwnProperty.call(update, "notes")
          ? update.notes ?? null
          : registrant.check_in.notes;
        checkInUpdates.set(update.registration_id, {
          attendanceStatus,
          waiverVerified,
          notes,
          updatedAt
        });
        return {
          registration_id: update.registration_id,
          registration_day_id: dayId,
          attendance_status: attendanceStatus,
          checked_in: attendanceStatus === "CHECKED_IN",
          waiver_verified: waiverVerified,
          approved_substitute_player_id: null,
          notes,
          updated_by: "operator@example.invalid",
          updated_at: updatedAt
        };
      });
      await route.fulfill({
        json: {
          ok: true,
          mode: "tournament_registration_check_in_bulk_update",
          operation_key: input.operation_key,
          updated_count: checkIns.length,
          check_ins: checkIns,
          idempotent_replay: false,
          message: `${checkIns.length} selected player${checkIns.length === 1 ? "" : "s"} updated.`
        }
      });
      return;
    }
    const checkInUpdateMatch = url.pathname.match(
      new RegExp(`/tournament-live/tournaments/${tournamentId}/check-in/([^/]+)$`)
    );
    if (checkInUpdateMatch && request.method() === "PUT") {
      const registrationId = decodeURIComponent(checkInUpdateMatch[1]);
      const input = request.postDataJSON() as {
        attendance_status: "EXPECTED" | "CHECKED_IN" | "ABSENT";
        waiver_verified: boolean;
        notes: string | null;
      };
      const updatedAt = `2026-08-15T12:01:${String(checkInUpdates.size).padStart(2, "0")}Z`;
      checkInUpdates.set(registrationId, {
        attendanceStatus: input.attendance_status,
        waiverVerified: input.waiver_verified,
        notes: input.notes,
        updatedAt
      });
      await route.fulfill({
        json: {
          ok: true,
          mode: "tournament_registration_check_in_update",
          check_in: {
            registration_id: registrationId,
            registration_day_id: url.searchParams.get("day_id") || "day-1",
            attendance_status: input.attendance_status,
            checked_in: input.attendance_status === "CHECKED_IN",
            waiver_verified: input.waiver_verified,
            approved_substitute_player_id: null,
            notes: input.notes,
            updated_by: "operator@example.invalid",
            updated_at: updatedAt
          },
          attendee_identity_changed: false,
          attendance_reset: false,
          idempotent_replay: false,
          message: "Check-in saved for the reviewed attendee."
        }
      });
      return;
    }
    if (url.pathname.endsWith(`/tournament-live/tournaments/${tournamentId}/draws/${drawId}/commands`)) {
      await route.fulfill({ json: { ok: true, operation_key: "saved-score-op", idempotent_replay: false } });
      return;
    }
    await route.fulfill({ status: 404, json: { detail: `Unmocked local operator API: ${request.method()} ${url.pathname}` } });
  });
}

test.beforeEach(async ({ page }) => {
  await installMockApi(page);
});

test("Home shows authoritative 1 of 21 truth and preserves selected context", async ({ page }) => {
  await page.goto(`/admin/tournaments/tournament?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Continue scoring" })).toBeVisible();
  await expect(page.getByText("1 of 21 games scored; 20 open.")).toBeVisible();
  await expect(page.getByText("Publish blockers")).toBeVisible();
  const scoringHref = await page.getByRole("link", { name: "Continue scoring" }).getAttribute("href");
  expect(scoringHref).toContain(`tournament=${tournamentId}`);
  expect(scoringHref).toContain(`draw=${drawId}`);
});

test("legacy draw route opens a clean tabbed day workspace", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/draws?${selectedQuery}`);
  await expect(page).toHaveURL(/\/admin\/tournaments\/live-operations\?/);
  await expect(page).toHaveURL(/panel=draws/);
  await expect(page).toHaveURL(new RegExp(`day=${dayId}`));
  await expect(page.getByRole("heading", { name: "Staging Summer Classic day workspace" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Tournament day scope" }).getByLabel("Tournament day")).toHaveValue(dayId);
  await expect(page.getByRole("tabpanel", { name: "Draws & progression" })).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Court board" })).toHaveCount(0);
  await expect(page.getByRole("tabpanel", { name: "Eligible queue" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Pause draw" })).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Generate playoffs" })).toHaveCount(2);

  await page.getByRole("tab", { name: "Court board" }).click();
  await expect(page).toHaveURL(/panel=board/);
  const courtBoard = page.getByRole("tabpanel", { name: "Court board" });
  const boardQueue = courtBoard.getByRole("region", { name: "Ready games from active draws" });
  await expect(boardQueue).toContainText("2 ready");
  await expect(boardQueue.locator("ol > li")).toHaveCount(2);
  await expect(boardQueue).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await expect(boardQueue).toContainText("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson");
  await expect(courtBoard.getByRole("heading", { name: /^Court \d+$/ })).toHaveCount(10);
  await expect(courtBoard.getByText("Mateo Rivera / Liam Chen vs Caleb Nguyen / Diego Alvarez")).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Draws & progression" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Pause draw" })).toHaveCount(0);

  await page.getByRole("tab", { name: "Eligible queue" }).click();
  await expect(page).toHaveURL(/panel=queue/);
  const queuePanel = page.getByRole("tabpanel", { name: "Eligible queue" });
  const queueRows = queuePanel.getByRole("region", { name: "Eligible match queue" }).locator("ol > li");
  await expect(queueRows).toHaveCount(2);
  expect(await queueRows.allTextContents()).toEqual([
    expect.stringContaining("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith"),
    expect.stringContaining("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson")
  ]);
  await expect(queueRows.nth(0)).toContainText("#1");
  await expect(queueRows.nth(1)).toContainText("#2");
  await expect(page.getByRole("heading", { name: "Held and blocked matches" })).toBeVisible();
  await expect(page.getByText("Operator hold awaiting participant arrival")).toBeVisible();
  await expect(page.getByText("One participant is already assigned to another court.")).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Court board" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Generate playoffs" })).toHaveCount(0);

  await page.reload();
  await expect(page.getByRole("region", { name: "Tournament day scope" }).getByLabel("Tournament day")).toHaveValue(dayId);
  await expect(page.getByRole("tabpanel", { name: "Eligible queue" })).toBeVisible();
});

test("playoff generation requires an explicit server-reviewed advance count", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=draws`);
  const drawCard = page.getByRole("heading", { name: "Open Division Draw", level: 3 }).locator("xpath=ancestor::article[1]");
  const advanceCount = drawCard.getByLabel("Advancing teams");
  await expect(advanceCount).toHaveValue("");
  await expect(drawCard.getByRole("button", { name: "Generate playoffs" })).toBeDisabled();
  await advanceCount.selectOption("5");
  await expect(drawCard.getByRole("button", { name: "Generate playoffs" })).toBeEnabled();
  await drawCard.getByRole("button", { name: "Generate playoffs" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: "Generate playoffs for Open Division Draw?" })).toBeVisible();
  await dialog.getByRole("button", { name: "Yes, generate playoffs" }).click();
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "generate_playoffs",
    confirmation_text: "GENERATE PLAYOFFS",
    expected: {
      day_run_version: "7",
      state_fingerprint: "d".repeat(64),
      queue_version: "11",
      draw_version: "4"
    },
    payload: { draw_id: secondDrawId, advance_count: 5 }
  });
});

test("legacy score route retains day context and focuses the unified queue", async ({ page }) => {
  await page.goto(`/admin/tournament-live?${selectedQuery}`);
  await expect(page).toHaveURL(/\/admin\/tournaments\/live-operations\?/);
  await expect(page).toHaveURL(/panel=queue/);
  await expect(page).toHaveURL(new RegExp(`day=${dayId}`));
  await expect(page.getByRole("heading", { name: "Unified eligible queue" })).toBeVisible();
});

test("Court board queue can use the next court, a chosen court, move, and requeue", async ({ page }) => {
  const commands: Array<{ action: string }> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as { action: string });
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=board`);
  const queue = page.getByRole("region", { name: "Ready games from active draws" });
  const gameBRow = queue.locator("ol > li").filter({ hasText: "Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith" });
  await gameBRow.getByRole("button", { name: /Send Avery Patel.*to next open court/ }).click();
  await expect(page.getByRole("status")).toContainText("Matchup assigned to the next authoritative open court.");

  let court2 = page.getByRole("heading", { name: "Court 2", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(court2).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await court2.getByRole("button", { name: /Move or remove Avery Patel/ }).click();
  let assignmentDialog = page.getByRole("dialog", { name: /Move or remove · Court 2/ });
  await assignmentDialog.getByLabel("Move to open court").selectOption("day-court-3");
  await assignmentDialog.getByRole("button", { name: "Move to Court 3" }).click();

  const court3 = page.getByRole("heading", { name: "Court 3", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(court3).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await court3.getByRole("button", { name: /Move or remove Avery Patel/ }).click();
  assignmentDialog = page.getByRole("dialog", { name: /Move or remove · Court 3/ });
  await assignmentDialog.getByRole("button", { name: "Return game to queue" }).click();
  await expect(court3).toContainText("Available for a queued matchup.");

  await expect(queue.locator("ol > li").filter({ hasText: "Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith" })).toBeVisible();
  const gameCRow = queue.locator("ol > li").filter({ hasText: "Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson" });
  await gameCRow.getByRole("button", { name: /Choose court for Nora Williams/ }).click();
  const chooseDialog = page.getByRole("dialog", { name: "Choose a court" });
  await chooseDialog.getByLabel("Open court").selectOption("day-court-4");
  await chooseDialog.getByRole("button", { name: "Assign to Court 4" }).click();

  const court4 = page.getByRole("heading", { name: "Court 4", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(court4).toContainText("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson");
  await expect.poll(() => commands.map((command) => command.action)).toEqual([
    "assign_next_court",
    "move_game_to_court",
    "requeue_game",
    "assign_game_to_court"
  ]);
});

test("Preflight check-in changes day without retaining old rows or losing context", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/check-in?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Staging Summer Classic preflight and check-in" })).toBeVisible();
  const summary = page.getByRole("region", { name: "Check-in summary" });
  await expect(summary.getByText("Expected today")).toBeVisible();
  await expect(summary.getByText("Checked in")).toBeVisible();
  await expect(summary.getByText("Not checked in")).toBeVisible();
  await expect(summary.getByText("Absent")).toBeVisible();
  await expect(summary.getByText("Unresolved")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Player check-in" })).toBeVisible();
  await expect(page.getByText("Partner unresolved")).toBeVisible();
  await expect(page.getByRole("row", { name: /Mateo Rivera/ })).toBeVisible();
  await expect(page.getByRole("row", { name: /Diego Alvarez/ })).toHaveCount(0);

  await page.getByLabel("Tournament day").selectOption("day-2");
  await expect(page).toHaveURL(/day=day-2/);
  await expect(page).toHaveURL(new RegExp(`tournament=${tournamentId}`));
  await expect(page).toHaveURL(new RegExp(`draw=${drawId}`));
  await expect(page.getByRole("row", { name: /Diego Alvarez/ })).toBeVisible();
  await expect(page.getByRole("row", { name: /Mateo Rivera/ })).toHaveCount(0);
  await expect(page.getByText("Partner unresolved")).toHaveCount(0);
  await expect(page.getByRole("row", { name: /Diego Alvarez.*Absent/ })).toBeVisible();

  await page.reload();
  await expect(page.getByLabel("Tournament day")).toHaveValue("day-2");
  await expect(page.getByRole("row", { name: /Diego Alvarez/ })).toBeVisible();
  const dayWorkspace = page.getByRole("link", { name: "Day workspace" });
  await expect(dayWorkspace).toHaveAttribute("href", new RegExp(`tournament=${tournamentId}.*draw=${drawId}.*day=day-2`));
});

test("Preflight applies one attendance and waiver action to selected player rows", async ({ page }) => {
  const updates: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith("/check-in/bulk?day_id=day-1")) return;
    updates.push(request.postDataJSON() as Record<string, unknown>);
  });

  await page.goto(`/admin/tournaments/live-operations/check-in?${selectedQuery}`);
  const playerTable = page.getByRole("table");
  await expect(playerTable.getByRole("row", { name: /Jordan Lee/ })).toBeVisible();
  await page.getByLabel("Select Jordan Lee").check();
  await page.getByLabel("Attendance action").selectOption("CHECKED_IN");
  await page.getByLabel("Waiver action").selectOption("VERIFY");
  await page.getByRole("button", { name: "Apply to 1 selected" }).click();

  await expect(page.getByRole("status")).toContainText("1 selected player updated.");
  await expect(playerTable.getByRole("row", { name: /Jordan Lee.*Checked in.*Verified/ })).toBeVisible();
  await expect.poll(() => updates.length).toBe(1);
  expect(updates[0]).toMatchObject({
    updates: [{
      registration_id: "registration-8",
      expected_updated_at: "2026-08-15T12:00:00Z",
      attendance_status: "CHECKED_IN",
      waiver_verified: true
    }]
  });
  expect(String(updates[0].operation_key)).toMatch(/^[0-9a-f-]{36}$/);
});

test("Preflight clears hidden selections before applying filtered bulk actions", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/check-in?${selectedQuery}`);
  await page.getByLabel("Select Jordan Lee").check();
  await expect(page.getByText("1 selected", { exact: true })).toBeVisible();

  await page.getByLabel("View").selectOption("checked_in");
  await expect(page.getByText("0 selected", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Select Jordan Lee")).toHaveCount(0);

  await page.getByRole("button", { name: "Select all shown" }).click();
  await expect(page.getByLabel("Select Mateo Rivera")).toBeChecked();
  await expect(page.getByText("1 selected", { exact: true })).toBeVisible();

  await page.getByLabel("Search players").fill("nobody matches");
  await expect(page.getByText("0 selected", { exact: true })).toBeVisible();
  await expect(page.getByText("No scheduled players match these filters.")).toBeVisible();
});

test("Score entry saves in two clicks with a live preview and the exact day fence", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}`);
  await page.getByRole("button", { name: /Enter score for Mateo Rivera \/ Liam Chen vs Caleb Nguyen \/ Diego Alvarez on Court 1/ }).click();
  const scoreDialog = page.getByRole("dialog", { name: /Enter result · Court 1/ });
  await expect(scoreDialog).toBeVisible();
  await expect(scoreDialog.getByRole("button", { name: "Played score" })).toHaveAttribute("aria-pressed", "true");
  await expect(scoreDialog.getByRole("button", { name: "Non-play result" })).toBeVisible();
  await expect(scoreDialog.getByLabel("Mateo Rivera / Liam Chen score")).toBeFocused();
  await scoreDialog.getByLabel("Mateo Rivera / Liam Chen score").fill("9");
  await scoreDialog.getByLabel("Caleb Nguyen / Diego Alvarez score").fill("9");
  await expect(scoreDialog.getByText("Tournament games cannot be saved with a tied score.")).toBeVisible();
  await expect(scoreDialog.getByRole("button", { name: "Save score & release Court 1" })).toBeDisabled();
  await expect(scoreDialog.getByRole("button", { name: "Review score" })).toHaveCount(0);
  await scoreDialog.getByLabel("Mateo Rivera / Liam Chen score").fill("11");
  await scoreDialog.getByLabel("Caleb Nguyen / Diego Alvarez score").fill("7");
  await expect(scoreDialog.getByText("Winner:")).toBeVisible();
  await expect(scoreDialog.getByText("Mateo Rivera / Liam Chen", { exact: true }).last()).toBeVisible();
  await expect(scoreDialog.getByText(/remains in the queue until an operator/)).toBeVisible();
  const saveScore = scoreDialog.getByRole("button", { name: "Save 11–7 & release Court 1" });
  await expect(saveScore).toBeEnabled();

  await saveScore.click();
  await expect(scoreDialog).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("Score saved and court released. The next matchup remains queued until an operator assigns it.");
  await expect(page.getByRole("dialog", { name: "Tournament-day operation complete" })).toHaveCount(0);
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "score_and_release",
    confirmation_text: "SAVE SCORE AND RELEASE COURT",
    expected: {
      day_run_version: "7",
      state_fingerprint: "d".repeat(64),
      queue_version: "11",
      draw_version: "3",
      game_version: "day-game-a-v1",
      court_version: "court-1-v2"
    },
    payload: { game_id: "day-game-a", score_a: 11, score_b: 7 }
  });
  const releasedCourt = page.getByRole("heading", { name: "Court 1", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(releasedCourt).toContainText("Available for a queued matchup.");
});

test("The score dialog records a non-play result without leaving the court board", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}`);
  const board = page.getByRole("tabpanel", { name: "Court board" });
  await board.getByRole("button", { name: /Enter score for Mateo Rivera \/ Liam Chen vs Caleb Nguyen \/ Diego Alvarez on Court 1/ }).click();
  let resultDialog = page.getByRole("dialog", { name: /Enter result · Court 1/ });
  await resultDialog.getByRole("button", { name: "Non-play result" }).click();
  resultDialog = page.getByRole("dialog", { name: /Enter result · Court 1/ });

  await expect(resultDialog.getByRole("button", { name: "Non-play result" })).toHaveAttribute("aria-pressed", "true");
  await expect(board).toBeVisible();
  await expect(page).not.toHaveURL(/panel=queue/);
  await resultDialog.getByLabel("Winning team").selectOption("day-game-a-team-b");
  await resultDialog.getByLabel("Operator note").fill("Mateo and Liam did not arrive; the desk verified the no-show.");
  await expect(resultDialog.getByText("Winner:")).toContainText("Caleb Nguyen / Diego Alvarez");
  const saveOutcome = resultDialog.getByRole("button", { name: "Record no show & release Court 1" });
  await expect(saveOutcome).toBeEnabled();
  await saveOutcome.click();

  await expect(resultDialog).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("Non-played outcome recorded and any court and participant claims released. The next matchup remains queued until assigned.");
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "record_non_played_result",
    confirmation_text: "RECORD NON-PLAYED RESULT",
    expected: {
      day_run_version: "7",
      state_fingerprint: "d".repeat(64),
      queue_version: "11",
      draw_version: "3",
      game_version: "day-game-a-v1",
      court_version: "court-1-v2"
    },
    payload: {
      game_id: "day-game-a",
      result_type: "NO_SHOW",
      winner_team_id: "day-game-a-team-b",
      result_note: "Mateo and Liam did not arrive; the desk verified the no-show."
    }
  });
  const releasedCourt = page.getByRole("heading", { name: "Court 1", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(releasedCourt).toContainText("Available for a queued matchup.");
});

test("Corrections & recovery submits an exact versioned day correction with before/after evidence", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations/corrections?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Corrections & recovery" })).toBeVisible();
  await page.getByRole("button", { name: "Correct completed score for Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith" }).click();
  await expect(page.getByRole("heading", { name: "Before correction" })).toBeVisible();
  await expect(page.getByText("11–7")).toBeVisible();
  await page.getByLabel("Morgan Diaz / Riley Smith score").fill("8");
  await page.getByRole("button", { name: "Review correction" }).click();
  await expect(page.getByText(/Before:/)).toContainText("11–7");
  await expect(page.getByText(/After:/)).toContainText("11–8");
  await page.getByRole("button", { name: "Confirm correction" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: "Confirm this exact completed-score correction?" })).toBeVisible();
  await dialog.getByRole("button", { name: "Confirm & save correction" }).click();
  await expect(dialog.getByRole("heading", { name: "Tournament-day operation complete" })).toBeVisible();
  await expect(dialog.getByText("Completed score corrected. The authoritative day result and all reviewed versions were refreshed.")).toBeVisible();
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "correct_completed_score",
    confirmation_text: "CORRECT COMPLETED SCORE",
    expected: {
      day_run_version: "7",
      state_fingerprint: "d".repeat(64),
      queue_version: "11",
      draw_version: "4",
      game_version: "day-game-completed-v1"
    },
    payload: { game_id: "day-game-completed", score_a: 11, score_b: 8 }
  });
  await expect(page.getByRole("heading", { name: "Recent day operations and recovery evidence" })).toBeVisible();
  await expect(page.getByText("Technical operation evidence").first()).toBeVisible();
});

test("Publish and completion remain blocked even when runtime writes are available", async ({ page }) => {
  await page.goto(`/admin/tournaments/ops/publish?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Tournament readiness" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Runtime capability" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Publish official matches" })).toBeDisabled();
  await page.goto(`/admin/tournaments/publish/closeout?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Complete tournament" })).toBeVisible();
  await expect(page.getByText("20 tournament games still need a finalized, non-tied score.")).toBeVisible();
  await expect(page.getByText("Payments, extras, and fulfillment")).toBeVisible();
});

for (const width of [1024, 1280, 1440]) {
  test(`operator routes have no page-level overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const path of ["/admin/tournaments/live-operations", "/admin/tournaments/live-operations/check-in", "/admin/tournament-live", "/admin/tournaments/live-operations/corrections", "/admin/tournaments/ops/publish"]) {
      await page.goto(`${path}?${selectedQuery}`);
      await expect(page.locator("body")).toContainText("Staging Summer Classic");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), path).toBe(true);
      expect(await page.locator("[data-nextjs-dialog]").count(), `${path} error overlay`).toBe(0);
    }
  });
}
