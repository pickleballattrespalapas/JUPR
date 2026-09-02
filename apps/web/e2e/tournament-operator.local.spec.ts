import { expect, test, type Page } from "@playwright/test";
import type { AdminTournamentDayWorkspaceSnapshot } from "../lib/adminTournamentDayOpsApi";

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
  stage?: "ROUND_ROBIN" | "PLAYOFF";
  playoffRound?: string;
  playoffGameCode?: string;
}) {
  return {
    id: options.id,
    draw_id: options.drawId,
    draw_name: options.drawName,
    state: "READY",
    stage: options.stage ?? "ROUND_ROBIN",
    round_label: options.playoffRound ?? options.round,
    slot_label: options.slot,
    playoff_round: options.playoffRound ?? null,
    playoff_game_code: options.playoffGameCode ?? null,
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

const playoffStandings = [
  { seed: 1, team_id: "playoff-team-a", team_name: "Avery Patel / Jordan Lee", participant_names: ["Avery Patel", "Jordan Lee"], wins: 5, losses: 0, points_for: 55, points_against: 31, differential: 24, non_play_results: 0, competition_status: "ACTIVE", retired: false },
  { seed: 2, team_id: "playoff-team-b", team_name: "Morgan Diaz / Riley Smith", participant_names: ["Morgan Diaz", "Riley Smith"], wins: 3, losses: 2, points_for: 52, points_against: 37, differential: 15, non_play_results: 0, competition_status: "ACTIVE", retired: false },
  { seed: 3, team_id: "playoff-team-c", team_name: "Taylor Reed / Casey Brooks", participant_names: ["Taylor Reed", "Casey Brooks"], wins: 3, losses: 2, points_for: 48, points_against: 42, differential: 6, non_play_results: 0, competition_status: "ACTIVE", retired: false },
  { seed: 4, team_id: "playoff-team-d", team_name: "Jamie Flores / Skyler Moore", participant_names: ["Jamie Flores", "Skyler Moore"], wins: 3, losses: 2, points_for: 43, points_against: 47, differential: -4, non_play_results: 1, competition_status: "ACTIVE", retired: false },
  { seed: 5, team_id: "playoff-team-e", team_name: "Nora Williams / Sofia Kim", participant_names: ["Nora Williams", "Sofia Kim"], wins: 1, losses: 4, points_for: 38, points_against: 52, differential: -14, non_play_results: 0, competition_status: "ACTIVE", retired: false },
  { seed: 6, team_id: "playoff-team-f", team_name: "Emma Davis / Mia Johnson", participant_names: ["Emma Davis", "Mia Johnson"], wins: 0, losses: 5, points_for: 29, points_against: 55, differential: -26, non_play_results: 0, competition_status: "ACTIVE", retired: false },
  { seed: 7, team_id: "playoff-team-retired", team_name: "Retired Test Team", participant_names: ["Retired Player", "Withdrawn Player"], wins: 0, losses: 5, points_for: 0, points_against: 55, differential: -55, non_play_results: 5, competition_status: "RETIRED", retired: true }
];

const playoffTemplates = [
  {
    code: "SINGLE_ELIMINATION_4",
    advance_count: 4,
    label: "4-team semifinals",
    description: "Seeds 1–4 play semifinals, followed by gold and bronze medal matches.",
    rounds: [
      { code: "SF", label: "Semifinals", game_codes: ["P1", "P2"] },
      { code: "BRONZE", label: "Bronze medal match", game_codes: ["P4"] },
      { code: "FINAL", label: "Gold medal final", game_codes: ["P3"] }
    ],
    games: [
      { code: "P1", label: "Semifinal 1", round: "SF", team_a_source: { seed: 1 }, team_b_source: { seed: 4 } },
      { code: "P2", label: "Semifinal 2", round: "SF", team_a_source: { seed: 2 }, team_b_source: { seed: 3 } },
      { code: "P3", label: "Gold Medal Match", round: "FINAL", team_a_source: { winnerOf: "P1" }, team_b_source: { winnerOf: "P2" } },
      { code: "P4", label: "Bronze Medal Match", round: "BRONZE", team_a_source: { loserOf: "P1" }, team_b_source: { loserOf: "P2" } }
    ],
    default_round_scoring: { SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "GAME_TO_11" }
  },
  {
    code: "SINGLE_ELIMINATION_5",
    advance_count: 5,
    label: "5-team play-in and semifinals",
    description: "Seeds 4 and 5 play in before the semifinals and medal matches.",
    rounds: [
      { code: "QF", label: "Play-in", game_codes: ["P1"] },
      { code: "SF", label: "Semifinals", game_codes: ["P2", "P3"] },
      { code: "BRONZE", label: "Bronze medal match", game_codes: ["P5"] },
      { code: "FINAL", label: "Gold medal final", game_codes: ["P4"] }
    ],
    games: [
      { code: "P1", label: "Play-in", round: "QF", team_a_source: { seed: 4 }, team_b_source: { seed: 5 } },
      { code: "P2", label: "Semifinal 1", round: "SF", team_a_source: { seed: 1 }, team_b_source: { winnerOf: "P1" } },
      { code: "P3", label: "Semifinal 2", round: "SF", team_a_source: { seed: 2 }, team_b_source: { seed: 3 } },
      { code: "P4", label: "Gold Medal Match", round: "FINAL", team_a_source: { winnerOf: "P2" }, team_b_source: { winnerOf: "P3" } },
      { code: "P5", label: "Bronze Medal Match", round: "BRONZE", team_a_source: { loserOf: "P2" }, team_b_source: { loserOf: "P3" } }
    ],
    default_round_scoring: { QF: "GAME_TO_11", SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "GAME_TO_11" }
  },
  {
    code: "SINGLE_ELIMINATION_6",
    advance_count: 6,
    label: "6-team quarterfinals and semifinals",
    description: "Seeds 1 and 2 receive byes while seeds 3–6 play quarterfinals.",
    rounds: [
      { code: "QF", label: "Quarterfinals", game_codes: ["P1", "P2"] },
      { code: "SF", label: "Semifinals", game_codes: ["P3", "P4"] },
      { code: "BRONZE", label: "Bronze medal match", game_codes: ["P6"] },
      { code: "FINAL", label: "Gold medal final", game_codes: ["P5"] }
    ],
    games: [
      { code: "P1", label: "Quarterfinal 1", round: "QF", team_a_source: { seed: 4 }, team_b_source: { seed: 5 } },
      { code: "P2", label: "Quarterfinal 2", round: "QF", team_a_source: { seed: 3 }, team_b_source: { seed: 6 } },
      { code: "P3", label: "Semifinal 1", round: "SF", team_a_source: { seed: 1 }, team_b_source: { winnerOf: "P1" } },
      { code: "P4", label: "Semifinal 2", round: "SF", team_a_source: { seed: 2 }, team_b_source: { winnerOf: "P2" } },
      { code: "P5", label: "Gold Medal Match", round: "FINAL", team_a_source: { winnerOf: "P3" }, team_b_source: { winnerOf: "P4" } },
      { code: "P6", label: "Bronze Medal Match", round: "BRONZE", team_a_source: { loserOf: "P3" }, team_b_source: { loserOf: "P4" } }
    ],
    default_round_scoring: { QF: "GAME_TO_11", SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "GAME_TO_11" }
  }
];

function dayWorkspaceSnapshot() {
  const dayGames = [
    dayGame({ id: "day-game-a", drawId, drawName: "Manual Acceptance Draw", round: "Round 1", slot: "Match 1", teamA: ["Mateo Rivera", "Liam Chen"], teamB: ["Caleb Nguyen", "Diego Alvarez"], courtId: "day-court-1" }),
    dayGame({ id: "day-game-b", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 1", slot: "P3", teamA: ["Avery Patel", "Jordan Lee"], teamB: ["Morgan Diaz", "Riley Smith"], stage: "PLAYOFF", playoffRound: "Final", playoffGameCode: "P3" }),
    dayGame({ id: "day-game-c", drawId, drawName: "Manual Acceptance Draw", round: "Round 2", slot: "P4", teamA: ["Nora Williams", "Sofia Kim"], teamB: ["Emma Davis", "Mia Johnson"], stage: "PLAYOFF", playoffRound: "Bronze", playoffGameCode: "P4" }),
    dayGame({ id: "day-game-held", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 2", slot: "Match 2", teamA: ["Taylor Reed", "Casey Brooks"], teamB: ["Jamie Flores", "Skyler Moore"] }),
    dayGame({ id: "day-game-blocked", drawId, drawName: "Manual Acceptance Draw", round: "Round 3", slot: "Match 1", teamA: ["Quinn Parker", "Alexis Bell"], teamB: ["Cameron Price", "Robin Ward"] }),
    {
      ...dayGame({ id: "day-game-bo3", drawId: secondDrawId, drawName: "Open Division Draw", round: "Final", slot: "P5", teamA: ["Nora Williams", "Sofia Kim"], teamB: ["Taylor Reed", "Casey Brooks"], courtId: "day-court-10", stage: "PLAYOFF", playoffRound: "Final", playoffGameCode: "P5" }),
      scoring: {
        format: "BEST_2_OF_3",
        target: 2,
        win_by_two: false,
        individual_game_format: "GAME_TO_11",
        individual_game_target: 11,
        individual_game_win_by_two: true,
        best_of_three_score_semantics: "individual_game_points_with_derived_series_result"
      },
      game_scores: [] as Array<{ game_number: 1 | 2 | 3; score_a: number; score_b: number }>
    },
    {
      ...dayGame({ id: "day-game-completed", drawId: secondDrawId, drawName: "Open Division Draw", round: "Round 3", slot: "Match 2", teamA: ["Avery Patel", "Jordan Lee"], teamB: ["Morgan Diaz", "Riley Smith"] }),
      state: "COMPLETED",
      score_a: 11,
      score_b: 7,
      winner_name: "Avery Patel / Jordan Lee",
      correction_readiness: dayReadiness(true, "CORRECT COMPLETED SCORE")
    },
    {
      ...dayGame({ id: "day-game-bo3-completed", drawId: secondDrawId, drawName: "Open Division Draw", round: "Semifinal", slot: "P2", teamA: ["Emma Davis", "Mia Johnson"], teamB: ["Jamie Flores", "Skyler Moore"], stage: "PLAYOFF", playoffRound: "SF", playoffGameCode: "P2" }),
      state: "COMPLETED",
      scoring: {
        format: "BEST_2_OF_3",
        target: 2,
        win_by_two: false,
        individual_game_format: "GAME_TO_11",
        individual_game_target: 11,
        individual_game_win_by_two: true,
        best_of_three_score_semantics: "individual_game_points_with_derived_series_result"
      },
      score_a: 2,
      score_b: 1,
      game_scores: [
        { game_number: 1 as const, score_a: 11, score_b: 7 },
        { game_number: 2 as const, score_a: 8, score_b: 11 },
        { game_number: 3 as const, score_a: 12, score_b: 10 }
      ],
      winner_name: "Emma Davis / Mia Johnson",
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
      available_courts: 8,
      active_draws: 2,
      eligible_games: 2,
      reserved_games: 0,
      held_games: 1,
      completed_games: 5
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
        round_robin_complete: true,
        progression_status: "READY_FOR_PLAYOFF_REVIEW",
        playoff_review_fingerprint: "1".repeat(64),
        total_games: 6,
        finalized_games: 6,
        queued_games: 1,
        active_games: 0,
        held_games: 1,
        round_robin_summary: {
          standings: playoffStandings,
          ranking_policy: {
            description: "Teams are ranked by wins. Tied teams are compared head-to-head first. Any tie that remains uses point differential, then total points scored, then original team number.",
            criteria: ["WINS", "HEAD_TO_HEAD", "POINT_DIFFERENTIAL", "POINTS_FOR", "TEAM_NUMBER"],
            retired_teams_eligible: false
          },
          tiebreak_explanations: [
            {
              title: "Three-way tie at 3–2",
              summary: "Morgan Diaz / Riley Smith, Taylor Reed / Casey Brooks, and Jamie Flores / Skyler Moore were tied on wins. Head-to-head could not separate them, so point differential set their final order.",
              steps: [
                {
                  criterion: "HEAD_TO_HEAD",
                  outcome: "UNRESOLVED",
                  detail: "Each tied team went 1–1 against the other tied teams."
                },
                {
                  criterion: "POINT_DIFFERENTIAL",
                  outcome: "RESOLVED",
                  detail: "Morgan Diaz / Riley Smith ranked highest at +15, followed by Taylor Reed / Casey Brooks at +6 and Jamie Flores / Skyler Moore at −4."
                }
              ]
            }
          ]
        },
        playoff_review: {
          eligible_team_ids: playoffStandings.filter((standing) => !standing.retired).map((standing) => standing.team_id),
          default_seed_team_ids: playoffStandings.slice(0, 4).map((standing) => standing.team_id),
          templates: playoffTemplates,
          default_template_code: "SINGLE_ELIMINATION_4",
          scoring_formats: [
            { code: "GAME_TO_11", label: "Game to 11" },
            { code: "GAME_TO_15", label: "Game to 15" },
            { code: "GAME_TO_21", label: "Game to 21" },
            { code: "BEST_2_OF_3", label: "Best 2 of 3 games" }
          ],
          default_round_scoring: { SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "GAME_TO_11" },
          default_scoring_format: "GAME_TO_11"
        },
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
      state: index === 0 || index === 9 ? "ON_COURT" : "AVAILABLE",
      version: `court-${index + 1}-v2`,
      current_assignment: index === 0 || index === 9 ? {
        id: index === 0 ? "assignment-a" : "assignment-bo3",
        game_id: index === 0 ? "day-game-a" : "day-game-bo3",
        state: "ON_COURT",
        version: index === 0 ? "assignment-v5" : "assignment-bo3-v1",
        assigned_at: "2026-08-17T09:01:00Z",
        started_at: "2026-08-17T09:02:00Z" as string | null
      } : null,
      next_assignment: null
    })),
    games: dayGames,
    eligible_queue: [
      { game_id: "day-game-b", draw_id: secondDrawId, position: 1, priority: 50, state: "WAITING", version: "queue-b-v3", eligible_since: "2026-08-17T08:58:00Z", reason: "First eligible across active draws", blockers: [] },
      { game_id: "day-game-c", draw_id: drawId, position: 2, priority: 40, state: "WAITING", version: "queue-c-v2", eligible_since: "2026-08-17T08:59:00Z", reason: "Second eligible across active draws", blockers: [] }
    ],
    reserved_queue: [],
    held_games: [
      { game_id: "day-game-held", draw_id: secondDrawId, state: "HELD", reason: "Operator hold awaiting participant arrival", note: null, held_at: "2026-08-17T09:03:00Z", version: "held-v1", blockers: [] }
    ],
    blocked_games: [
      { game_id: "day-game-blocked", draw_id: drawId, state: "BLOCKED", reason: "Participant already assigned", note: null, held_at: null, version: "blocked-v1", blockers: [{ code: "PLAYER_ALREADY_CLAIMED", message: "One participant is already assigned to another court." }] }
    ],
    progression_alerts: [{
      key: `draw:${secondDrawId}:round_robin_complete`,
      kind: "PLAYOFF_REVIEW_READY",
      draw_id: secondDrawId,
      draw_name: "Open Division Draw",
      message: "Review standings, qualifiers, bracket structure, and round scoring before generating playoffs.",
      ready: true,
      blockers: []
    }],
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
  let currentDaySnapshot = dayWorkspaceSnapshot() as unknown as AdminTournamentDayWorkspaceSnapshot;
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
          playoff_configuration?: {
            template_code: string;
            seed_team_ids: string[];
            round_scoring: Record<string, string>;
          };
          game_id?: string;
          court_id?: string;
          score_a?: number;
          score_b?: number;
          game_scores?: Array<{ game_number: 1 | 2 | 3; score_a: number; score_b: number }>;
          unusual_score_acknowledgement?: boolean;
          result_type?: "FORFEIT" | "NO_SHOW" | "RETIREMENT";
          non_playing_team_id?: string;
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
      if (command.action === "reserve_game_for_court") {
        const entry = refreshed.eligible_queue.find((row) => row.game_id === command.payload.game_id);
        const targetCourt = refreshed.courts.find((court) => court.id === command.payload.court_id);
        const game = refreshed.games.find((row) => row.id === command.payload.game_id);
        if (entry && targetCourt?.current_assignment && !targetCourt.next_assignment && game) {
          targetCourt.next_assignment = {
            id: `reservation-${game.id}`,
            game_id: game.id,
            state: "RESERVED",
            version: `${entry.version}-reserved`,
            reserved_at: "2026-08-17T09:06:00Z"
          };
          game.state = "RESERVED";
          game.reserved_court_id = targetCourt.id;
          game.queue_entry_version = `${entry.version}-reserved`;
          refreshed.eligible_queue = refreshed.eligible_queue.filter((row) => row.game_id !== game.id);
          refreshed.reserved_queue.push({
            ...entry,
            state: "RESERVED",
            version: game.queue_entry_version,
            reserved_court_id: targetCourt.id,
            reserved_at: "2026-08-17T09:06:00Z",
            reason: "NEXT_ON_COURT",
            note: `Next on ${targetCourt.label}; this matchup and its players are reserved while the current game finishes.`
          });
          refreshed.summary.eligible_games -= 1;
          refreshed.summary.reserved_games += 1;
        }
      }
      if (command.action === "requeue_game" || command.action === "move_game_to_court") {
        const waitCourt = refreshed.courts.find((court) => court.next_assignment?.game_id === command.payload.game_id);
        const sourceCourt = refreshed.courts.find((court) => court.current_assignment?.game_id === command.payload.game_id);
        const game = refreshed.games.find((row) => row.id === command.payload.game_id);
        if (command.action === "requeue_game" && waitCourt?.next_assignment && game) {
          const reservation = waitCourt.next_assignment;
          waitCourt.next_assignment = null;
          game.state = "WAITING";
          game.reserved_court_id = null;
          game.queue_entry_version = `${reservation.version}-requeued`;
          const reservedEntry = refreshed.reserved_queue.find((row) => row.game_id === game.id);
          refreshed.reserved_queue = refreshed.reserved_queue.filter((row) => row.game_id !== game.id);
          refreshed.eligible_queue.push({
            game_id: game.id,
            draw_id: game.draw_id,
            position: refreshed.eligible_queue.length + 1,
            priority: reservedEntry?.priority || 50,
            state: "WAITING",
            version: game.queue_entry_version,
            eligible_since: reservedEntry?.eligible_since || "2026-08-17T08:58:00Z",
            reason: "Returned to existing queue priority",
            blockers: []
          });
          refreshed.summary.eligible_games += 1;
          refreshed.summary.reserved_games -= 1;
        } else if (sourceCourt && game) {
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
          completedGame.game_scores = command.payload.game_scores ?? [];
          completedGame.winner_name = Number(command.payload.score_a) > Number(command.payload.score_b)
            ? completedGame.team_a.name
            : completedGame.team_b.name;
          completedGame.state = "COMPLETED";
          completedGame.version = `${completedGame.id}-v2`;
        }
        const releasedCourt = refreshed.courts.find((court) => court.current_assignment?.game_id === command.payload.game_id);
        if (!releasedCourt) throw new Error("Mocked scored game has no current court assignment.");
        const reservation = releasedCourt.next_assignment;
        if (reservation) {
          releasedCourt.current_assignment = {
            ...reservation,
            state: "ON_COURT",
            assigned_at: "2026-08-17T09:06:01Z",
            started_at: "2026-08-17T09:06:01Z"
          };
          releasedCourt.next_assignment = null;
          const promotedGame = refreshed.games.find((game) => game.id === reservation.game_id);
          if (promotedGame) {
            promotedGame.state = "ON_COURT";
            promotedGame.court_id = releasedCourt.id;
            promotedGame.reserved_court_id = null;
            promotedGame.queue_entry_version = reservation.version;
          }
          refreshed.reserved_queue = refreshed.reserved_queue.filter((row) => row.game_id !== reservation.game_id);
          refreshed.summary.reserved_games -= 1;
        } else {
          releasedCourt.current_assignment = null;
          releasedCourt.state = "AVAILABLE";
          refreshed.summary.available_courts += 1;
        }
        refreshed.summary.completed_games += 1;
      }
      if (command.action === "record_non_played_result") {
        const completedGame = refreshed.games.find((game) => game.id === command.payload.game_id);
        if (completedGame) {
          completedGame.result_type = command.payload.result_type ?? "NO_SHOW";
          completedGame.result_note = command.payload.result_note ?? null;
          completedGame.game_scores = command.payload.game_scores ?? [];
          completedGame.winner_name = command.payload.non_playing_team_id === completedGame.team_a.team_id
            ? completedGame.team_b.name
            : completedGame.team_a.name;
          completedGame.state = "COMPLETED";
          completedGame.version = `${completedGame.id}-v2`;
        }
        const releasedCourt = refreshed.courts.find((court) => court.current_assignment?.game_id === command.payload.game_id);
        if (releasedCourt) {
          releasedCourt.current_assignment = null;
          releasedCourt.state = "AVAILABLE";
          refreshed.summary.available_courts += 1;
        }
        refreshed.summary.completed_games += 1;
      }
      if (command.action === "correct_completed_score") {
        const correctedGame = refreshed.games.find((game) => game.id === command.payload.game_id);
        if (correctedGame) {
          correctedGame.score_a = command.payload.score_a ?? null;
          correctedGame.score_b = command.payload.score_b ?? null;
          correctedGame.game_scores = command.payload.game_scores ?? [];
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
  await expect(page.getByRole("button", { name: "Review playoff setup" })).toHaveCount(2);

  await page.getByRole("tab", { name: "Court board" }).click();
  await expect(page).toHaveURL(/panel=board/);
  const courtBoard = page.getByRole("tabpanel", { name: "Court board" });
  const boardQueue = courtBoard.getByRole("region", { name: "Ready and court-reserved games from active draws" });
  await expect(boardQueue).toContainText("2 ready");
  await expect(boardQueue.locator("ol > li")).toHaveCount(2);
  await expect(boardQueue).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await expect(boardQueue).toContainText("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson");
  const goldBoardMatch = boardQueue.locator('[data-medal-match="gold"]');
  const bronzeBoardMatch = boardQueue.locator('[data-medal-match="bronze"]');
  await expect(goldBoardMatch).toContainText("Gold medal match");
  await expect(bronzeBoardMatch).toContainText("Bronze medal match");
  await expect(courtBoard.getByRole("heading", { name: /^Court \d+$/ })).toHaveCount(10);
  await expect(courtBoard.getByText("Mateo Rivera / Liam Chen vs Caleb Nguyen / Diego Alvarez")).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Draws & progression" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Pause draw" })).toHaveCount(0);

  await page.getByRole("tab", { name: "Eligible queue" }).click();
  await expect(page).toHaveURL(/panel=queue/);
  const queuePanel = page.getByRole("tabpanel", { name: "Eligible queue" });
  const queueRows = queuePanel.getByRole("region", { name: "Tournament match queue" }).locator("ol > li");
  await expect(queueRows).toHaveCount(2);
  expect(await queueRows.allTextContents()).toEqual([
    expect.stringContaining("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith"),
    expect.stringContaining("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson")
  ]);
  await expect(queueRows.nth(0)).toContainText("#1");
  await expect(queueRows.nth(1)).toContainText("#2");
  await expect(queuePanel.locator('[data-medal-match="gold"]')).toContainText("Gold medal match");
  await expect(queuePanel.locator('[data-medal-match="bronze"]')).toContainText("Bronze medal match");
  await expect(page.getByRole("heading", { name: "Held and blocked matches" })).toBeVisible();
  await expect(page.getByText("Operator hold awaiting participant arrival")).toBeVisible();
  await expect(page.getByText("One participant is already assigned to another court.")).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Court board" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Review playoff setup" })).toHaveCount(0);

  await page.reload();
  await expect(page.getByRole("region", { name: "Tournament day scope" }).getByLabel("Tournament day")).toHaveValue(dayId);
  await expect(page.getByRole("tabpanel", { name: "Eligible queue" })).toBeVisible();
});

test("completed round robin opens a reviewed playoff setup with exact seeds and round scoring", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=draws`);

  await expect(page.getByRole("heading", { name: "Round robin complete" })).toBeVisible();
  await expect(page.getByText("Open Division Draw is ready for playoff review.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Review Open Division Draw" })).toBeVisible();
  await expect(page.getByLabel("1 ready for playoff review")).toBeVisible();

  const drawCard = page.getByRole("heading", { name: "Open Division Draw", level: 3 }).locator("xpath=ancestor::article[1]");
  await expect(drawCard).toContainText("Playoff review ready");
  await drawCard.getByRole("button", { name: "Review playoff setup" }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: "Review playoffs · Open Division Draw" })).toBeVisible();
  const tiebreakAudit = dialog.getByRole("region", { name: "How tied teams were ranked" });
  await expect(tiebreakAudit).toBeVisible();
  await expect(tiebreakAudit.getByRole("heading", { name: "Three-way tie at 3–2" })).toBeVisible();
  await expect(tiebreakAudit).toContainText("Head-to-head could not separate them, so point differential set their final order.");
  await expect(tiebreakAudit.getByRole("list", { name: "Tie-break steps for Three-way tie at 3–2" })).toContainText("Each tied team went 1–1 against the other tied teams.");
  await expect(tiebreakAudit).toContainText("Still tied");
  await expect(tiebreakAudit).toContainText("Resolved");
  const standings = dialog.getByRole("table", { name: "Completed round-robin standings and current playoff selections" });
  await expect(standings).toContainText("Avery Patel / Jordan Lee");
  await expect(standings).toContainText("5–0");
  await expect(standings).toContainText("Retired Test Team");
  await expect(standings).toContainText("Retired · ineligible");

  await dialog.getByLabel("Playoff structure").selectOption("SINGLE_ELIMINATION_5");
  await expect(dialog.getByLabel("Seed 5")).toHaveValue("playoff-team-e");
  await dialog.getByLabel("Seed 1").selectOption("playoff-team-f");
  await expect(dialog.getByText("Seed override:")).toBeVisible();
  await dialog.getByLabel("Bronze medal match").selectOption("GAME_TO_15");
  await dialog.getByLabel("Gold medal final").selectOption("BEST_2_OF_3");

  const bracketPreview = dialog.getByRole("heading", { name: "Bracket preview" }).locator("xpath=ancestor::section[1]");
  await expect(bracketPreview).toBeVisible();
  await expect(bracketPreview.getByRole("heading", { name: "Play-in" })).toBeVisible();
  await expect(bracketPreview.getByText("Seed 4 · Jamie Flores / Skyler Moore")).toBeVisible();
  await expect(bracketPreview.getByText("Seed 5 · Nora Williams / Sofia Kim")).toBeVisible();
  await expect(bracketPreview.getByText("Best 2 of 3 games")).toBeVisible();

  await dialog.getByRole("button", { name: "Generate reviewed playoffs" }).click();
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "generate_playoffs",
    confirmation_text: "GENERATE PLAYOFFS",
    expected: {
      day_run_version: "7",
      state_fingerprint: "d".repeat(64),
      queue_version: "11",
      draw_version: "4"
    }
  });
  expect(commands[0].payload).toEqual({
    draw_id: secondDrawId,
    advance_count: 5,
    playoff_configuration: {
      template_code: "SINGLE_ELIMINATION_5",
      seed_team_ids: ["playoff-team-f", "playoff-team-b", "playoff-team-c", "playoff-team-d", "playoff-team-e"],
      round_scoring: {
        QF: "GAME_TO_11",
        SF: "GAME_TO_11",
        BRONZE: "GAME_TO_15",
        FINAL: "BEST_2_OF_3"
      }
    }
  });
});

test("default playoff review stays open across unrelated day activity and generates in two clicks", async ({ page }) => {
  const initial = dayWorkspaceSnapshot() as unknown as AdminTournamentDayWorkspaceSnapshot;
  const refreshed = structuredClone(initial);
  refreshed.day_run.version = "8";
  refreshed.state_fingerprint = "e".repeat(64);
  refreshed.queue_version = "12";
  let snapshotReads = 0;
  await page.route(
    new RegExp(`/tournament-live/tournaments/${tournamentId}/days/${dayId}/snapshot$`),
    async (route) => {
      snapshotReads += 1;
      await route.fulfill({ json: snapshotReads === 1 ? initial : refreshed });
    }
  );
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });

  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=board`);
  await page.getByRole("button", { name: "Review Open Division Draw" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: "Review playoffs · Open Division Draw" })).toBeVisible();

  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect.poll(() => snapshotReads).toBeGreaterThan(1);
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Generate reviewed playoffs" }).click();

  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "generate_playoffs",
    expected: {
      day_run_version: "8",
      state_fingerprint: "e".repeat(64),
      queue_version: "12",
      draw_version: "4"
    },
    payload: {
      draw_id: secondDrawId,
      advance_count: 4,
      playoff_configuration: {
        template_code: "SINGLE_ELIMINATION_4",
        seed_team_ids: ["playoff-team-a", "playoff-team-b", "playoff-team-c", "playoff-team-d"],
        round_scoring: { SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "GAME_TO_11" }
      }
    }
  });
});

test("playoff review closes when standings change without a day-draw version bump", async ({ page }) => {
  const initial = dayWorkspaceSnapshot() as unknown as AdminTournamentDayWorkspaceSnapshot;
  const refreshed = structuredClone(initial);
  const refreshedDraw = refreshed.draws.find((draw) => draw.id === secondDrawId);
  if (!refreshedDraw?.round_robin_summary?.standings?.length) {
    throw new Error("Ready playoff draw fixture is missing standings.");
  }
  const [previousFirst, previousSecond] = refreshedDraw.round_robin_summary.standings;
  refreshedDraw.round_robin_summary.standings[0] = { ...previousSecond, seed: 1 };
  refreshedDraw.round_robin_summary.standings[1] = { ...previousFirst, seed: 2 };
  refreshedDraw.playoff_review_fingerprint = "2".repeat(64);
  refreshed.state_fingerprint = "c".repeat(64);
  let snapshotReads = 0;
  await page.route(
    new RegExp(`/tournament-live/tournaments/${tournamentId}/days/${dayId}/snapshot$`),
    async (route) => {
      snapshotReads += 1;
      await route.fulfill({ json: snapshotReads === 1 ? initial : refreshed });
    }
  );

  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=board`);
  await page.getByRole("button", { name: "Review Open Division Draw" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();

  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect.poll(() => snapshotReads).toBeGreaterThan(1);
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByText(
    "Playoff review closed because the round-robin results or authoritative tournament-day state changed. Reopen the finished draw and inspect the refreshed qualifiers."
  )).toBeVisible();
  expect(refreshedDraw.version).toBe("4");
});

test("legacy score route retains day context and focuses the unified queue", async ({ page }) => {
  await page.goto(`/admin/tournament-live?${selectedQuery}`);
  await expect(page).toHaveURL(/\/admin\/tournaments\/live-operations\?/);
  await expect(page).toHaveURL(/panel=queue/);
  await expect(page).toHaveURL(new RegExp(`day=${dayId}`));
  await expect(page.getByRole("heading", { name: "Unified game queue" })).toBeVisible();
});

test("Court board queue can use the next court, a chosen court, move, and requeue", async ({ page }) => {
  const commands: Array<{ action: string }> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as { action: string });
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=board`);
  const queue = page.getByRole("region", { name: "Ready and court-reserved games from active draws" });
  await expect(queue.locator("ol > li").nth(0)).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await expect(queue.locator("ol > li").nth(0)).toContainText("#1");
  await expect(queue.locator("ol > li").nth(1)).toContainText("Nora Williams / Sofia Kim vs Emma Davis / Mia Johnson");
  await expect(queue.locator("ol > li").nth(1)).toContainText("#2");
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
  await gameCRow.getByRole("button", { name: /Choose court or wait for a court for Nora Williams/ }).click();
  const chooseDialog = page.getByRole("dialog", { name: "Choose a court" });
  await chooseDialog.getByLabel("Court").selectOption("day-court-4");
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

test("Court reservation stays in the queue and announces automatic promotion", async ({ page }) => {
  const commands: Array<{ action: string }> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as { action: string });
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}&panel=board`);

  const queue = page.getByRole("region", { name: "Ready and court-reserved games from active draws" });
  const reservedRow = queue.locator("ol > li").filter({ hasText: "Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith" });
  await reservedRow.getByRole("button", { name: /Choose court or wait for a court for Avery Patel/ }).click();
  const chooseDialog = page.getByRole("dialog", { name: "Choose a court" });
  await chooseDialog.getByLabel("Court").selectOption("day-court-1");
  await chooseDialog.getByRole("button", { name: "Wait for Court 1" }).click();

  await expect(page.getByRole("status")).toContainText("reserved next for the selected occupied court");
  await expect(reservedRow).toContainText("Next on Court 1");
  await expect(reservedRow.getByRole("button", { name: "Cancel wait" })).toBeVisible();
  const court1 = page.getByRole("heading", { name: "Court 1", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(court1).toContainText("Next on this court");
  await expect(court1).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");

  await court1.getByRole("button", { name: /Enter score for Mateo Rivera/ }).click();
  const scoreDialog = page.getByRole("dialog", { name: /Enter result · Court 1/ });
  await scoreDialog.getByLabel("Mateo Rivera / Liam Chen score").fill("11");
  await scoreDialog.getByLabel("Caleb Nguyen / Diego Alvarez score").fill("7");
  await scoreDialog.getByRole("button", { name: "Save 11–7 & release Court 1" }).click();

  await expect(page.getByRole("status")).toContainText("Court 1 became available. The reserved next matchup is now on court.");
  await expect(court1).toContainText("Avery Patel / Jordan Lee vs Morgan Diaz / Riley Smith");
  await expect(court1.getByText("Next on this court")).toHaveCount(0);
  await expect(reservedRow).toHaveCount(0);
  await expect.poll(() => commands.map((command) => command.action)).toEqual([
    "reserve_game_for_court",
    "score_and_release"
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
  // The 9–9 tie remains blocked before the two-click score submission path.
  await scoreDialog.getByLabel("Mateo Rivera / Liam Chen score").fill("9");
  await scoreDialog.getByLabel("Caleb Nguyen / Diego Alvarez score").fill("9");
  await expect(scoreDialog.getByText("Tournament games cannot be saved with a tied score.")).toBeVisible();
  await expect(scoreDialog.getByRole("button", { name: "Save score & release Court 1" })).toBeDisabled();
  await expect(scoreDialog.getByRole("button", { name: "Review score" })).toHaveCount(0);
  await scoreDialog.getByLabel("Mateo Rivera / Liam Chen score").fill("11");
  await scoreDialog.getByLabel("Caleb Nguyen / Diego Alvarez score").fill("7");
  await expect(scoreDialog.getByText("Winner:")).toBeVisible();
  await expect(scoreDialog.getByText("Mateo Rivera / Liam Chen", { exact: true }).last()).toBeVisible();
  await expect(scoreDialog.getByText(/Next on this court moves onto it automatically/)).toBeVisible();
  const saveScore = scoreDialog.getByRole("button", { name: "Save 11–7 & release Court 1" });
  await expect(saveScore).toBeEnabled();

  await saveScore.click();
  await expect(scoreDialog).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("Score saved and court released. Any matchup reserved for this court was promoted automatically; all other games remain queued.");
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

test("Best-of-three score entry records every rating game and derives the series result", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}`);
  await page.getByRole("button", { name: /Enter score for Nora Williams \/ Sofia Kim vs Taylor Reed \/ Casey Brooks on Court 10/ }).click();
  const scoreDialog = page.getByRole("dialog", { name: /Enter result · Court 10/ });
  const gameOne = scoreDialog.getByRole("region", { name: "Game 1" });
  const gameTwo = scoreDialog.getByRole("region", { name: "Game 2" });
  await expect(gameOne.getByLabel("Nora Williams / Sofia Kim score")).toBeFocused();
  await expect(scoreDialog.getByRole("region", { name: "Game 3" })).toHaveCount(0);

  await gameOne.getByLabel("Nora Williams / Sofia Kim score").fill("11");
  await gameOne.getByLabel("Taylor Reed / Casey Brooks score").fill("8");
  await gameTwo.getByLabel("Nora Williams / Sofia Kim score").fill("9");
  await gameTwo.getByLabel("Taylor Reed / Casey Brooks score").fill("11");

  const gameThree = scoreDialog.getByRole("region", { name: "Game 3" });
  await expect(gameThree).toBeVisible();
  await expect(scoreDialog.getByText("The series is tied 1–1. Enter the deciding Game 3 scores.")).toBeVisible();
  await gameThree.getByLabel("Nora Williams / Sofia Kim score").fill("15");
  await expect(scoreDialog.getByText("Enter both scores for Game 3.")).toBeVisible();
  await gameThree.getByLabel("Taylor Reed / Casey Brooks score").fill("13");
  await expect(scoreDialog.getByText(/Rating games:/)).toContainText("Game 1: 11–8 · Game 2: 9–11 · Game 3: 15–13");

  const saveSeries = scoreDialog.getByRole("button", { name: "Save 2–1 series & release Court 10" });
  await expect(saveSeries).toBeEnabled();
  await saveSeries.click();
  await expect(scoreDialog).toHaveCount(0);
  await expect.poll(() => commands.length).toBe(1);
  expect((commands[0] as { payload: unknown }).payload).toEqual({
    game_id: "day-game-bo3",
    score_a: 2,
    score_b: 1,
    game_scores: [
      { game_number: 1, score_a: 11, score_b: 8 },
      { game_number: 2, score_a: 9, score_b: 11 },
      { game_number: 3, score_a: 15, score_b: 13 }
    ],
    unusual_score_acknowledgement: false
  });
  const releasedCourt = page.getByRole("heading", { name: "Court 10", exact: true }).locator("xpath=ancestor::article[1]");
  await expect(releasedCourt).toContainText("Available for a queued matchup.");
});

test("Best-of-three retirement preserves completed split games for ratings", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations?${selectedQuery}`);
  await page.getByRole("button", { name: /Enter score for Nora Williams \/ Sofia Kim vs Taylor Reed \/ Casey Brooks on Court 10/ }).click();
  let resultDialog = page.getByRole("dialog", { name: /Enter result · Court 10/ });
  await resultDialog.getByRole("button", { name: "Non-play result" }).click();
  resultDialog = page.getByRole("dialog", { name: /Enter result · Court 10/ });

  await resultDialog.getByLabel("Outcome").selectOption("RETIREMENT");
  await resultDialog.getByLabel("Team that retired").selectOption("day-game-bo3-team-b");
  const gameOne = resultDialog.getByRole("region", { name: "Game 1" });
  const gameTwo = resultDialog.getByRole("region", { name: "Game 2" });
  await gameOne.getByLabel("Nora Williams / Sofia Kim score").fill("11");
  await gameOne.getByLabel("Taylor Reed / Casey Brooks score").fill("8");
  await gameTwo.getByLabel("Nora Williams / Sofia Kim score").fill("9");
  await gameTwo.getByLabel("Taylor Reed / Casey Brooks score").fill("11");
  await expect(resultDialog.getByText(/Completed rating games:/)).toContainText(
    "Game 1: 11–8 · Game 2: 9–11"
  );

  const saveRetirement = resultDialog.getByRole("button", { name: "Record retirement & release Court 10" });
  await expect(saveRetirement).toBeEnabled();
  await saveRetirement.click();
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    action: "record_non_played_result",
    payload: {
      game_id: "day-game-bo3",
      result_type: "RETIREMENT",
      non_playing_team_id: "day-game-bo3-team-b",
      result_note: "",
      game_scores: [
        { game_number: 1, score_a: 11, score_b: 8 },
        { game_number: 2, score_a: 9, score_b: 11 }
      ],
      unusual_score_acknowledgement: false
    }
  });
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
  await resultDialog.getByLabel("Team that did not show").selectOption("day-game-a-team-a");
  await expect(resultDialog.getByText(/receives this game/)).toContainText("Caleb Nguyen / Diego Alvarez");
  const saveOutcome = resultDialog.getByRole("button", { name: "Record no show & release Court 1" });
  await expect(saveOutcome).toBeEnabled();
  await saveOutcome.click();

  await expect(resultDialog).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("Non-played outcome recorded and resources released. Any matchup reserved for this court was promoted automatically; all other games remain queued.");
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
      non_playing_team_id: "day-game-a-team-a",
      result_note: ""
    }
  });
  expect((commands[0] as { payload: unknown }).payload).toEqual({
    game_id: "day-game-a",
    result_type: "NO_SHOW",
    non_playing_team_id: "day-game-a-team-a",
    result_note: ""
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

test("Best-of-three correction preloads and submits changed individual rating games", async ({ page }) => {
  const commands: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST" || !request.url().endsWith(`/days/${dayId}/commands`)) return;
    commands.push(request.postDataJSON() as Record<string, unknown>);
  });
  await page.goto(`/admin/tournaments/live-operations/corrections?${selectedQuery}`);
  await page.getByRole("button", { name: "Correct completed score for Emma Davis / Mia Johnson vs Jamie Flores / Skyler Moore" }).click();
  await expect(page.getByText("Game 1: 11–7 · Game 2: 8–11 · Game 3: 12–10").first()).toBeVisible();

  const gameThree = page.getByRole("region", { name: "Game 3" });
  await gameThree.getByLabel("Emma Davis / Mia Johnson score").fill("14");
  await gameThree.getByLabel("Jamie Flores / Skyler Moore score").fill("12");
  await page.getByRole("button", { name: "Review correction" }).click();
  await expect(page.getByText(/Corrected rating games:/)).toContainText("Game 3: 14–12");
  await page.getByRole("button", { name: "Confirm correction" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", { name: "Confirm & save correction" }).click();

  await expect.poll(() => commands.length).toBe(1);
  expect((commands[0] as { payload: unknown }).payload).toEqual({
    game_id: "day-game-bo3-completed",
    score_a: 2,
    score_b: 1,
    game_scores: [
      { game_number: 1, score_a: 11, score_b: 7 },
      { game_number: 2, score_a: 8, score_b: 11 },
      { game_number: 3, score_a: 14, score_b: 12 }
    ],
    unusual_score_acknowledgement: false
  });
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
