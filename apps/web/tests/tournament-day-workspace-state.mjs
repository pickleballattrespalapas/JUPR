import assert from "node:assert/strict";

import {
  advanceCountSelection,
  dayActionConfirmation,
  dayRunAcceptsLiveCommands,
  dayRunHasStarted,
  newlyReadyPlayoffNotice,
  oldestReadyQueue,
  playoffTemplateRoundCodes,
  readyActiveDrawQueue,
  readyPlayoffReviewDraws,
  resetFocusForDay,
  retainedDayCommandStorageKey,
  tournamentDayCloseoutGuidance,
  tournamentDayMedalMatchKind,
  validateBestOfThreeCorrectionDraft,
  validateBestOfThreeGameScores,
  validateBestOfThreeRetirementGameScores,
  validateDayCorrectionDraft,
  validateDayScoreDraft,
  validateNonPlayedOutcomeDraft,
  validatePlayoffReviewConfiguration,
  visibleServerQueue,
  workspaceScopeKey
} from "../lib/tournamentDayWorkspaceState.mjs";

assert.equal(
  dayActionConfirmation("close_day"),
  "CLOSE TOURNAMENT DAY",
  "closing a day must retain the backend's exact destructive confirmation"
);
assert.equal(dayRunHasStarted("CLOSED"), true, "a closed day must never return to activation setup");
assert.equal(
  dayRunAcceptsLiveCommands("CLOSED"),
  false,
  "closed days must not expose court, scoring, or draw writes"
);
assert.equal(dayRunAcceptsLiveCommands("ACTIVE"), true);
assert.equal(
  dayActionConfirmation("correct_completed_score"),
  "CORRECT COMPLETED SCORE",
  "day-owned score corrections must retain the exact guarded confirmation"
);
assert.equal(dayActionConfirmation("assign_next_court"), "ASSIGN NEXT OPEN COURT");
assert.equal(dayActionConfirmation("assign_game_to_court"), "ASSIGN GAME TO COURT");
assert.equal(dayActionConfirmation("reserve_game_for_court"), "WAIT FOR SELECTED COURT");
assert.equal(dayActionConfirmation("requeue_game"), "RETURN GAME TO QUEUE");
assert.equal(dayActionConfirmation("move_game_to_court"), "MOVE GAME TO COURT");

assert.equal(advanceCountSelection([4, 5, 6], null, undefined), "");
assert.equal(
  advanceCountSelection([4, 5, 6], 5, undefined),
  "5",
  "only an explicit server-configured default may seed the visible selector"
);
assert.equal(advanceCountSelection([4], 5, undefined), "");
assert.equal(advanceCountSelection([4, 5], null, "5"), "5");
assert.equal(advanceCountSelection([4], null, "5"), "");

const playoffTemplate = {
  code: "TOP_4",
  advance_count: 4,
  rounds: [{ code: "SEMIFINAL" }, { code: "BRONZE" }, { code: "FINAL" }],
  games: []
};
const playoffReview = {
  eligible_team_ids: ["team-1", "team-2", "team-3", "team-4"],
  templates: [playoffTemplate],
  scoring_formats: [{ code: "GAME_TO_11" }, { code: "BEST_2_OF_3" }]
};
assert.deepEqual(
  playoffTemplateRoundCodes(playoffTemplate),
  ["SF", "BRONZE", "FINAL"],
  "playoff scoring uses the canonical backend round keys"
);
assert.equal(validatePlayoffReviewConfiguration(playoffReview, {
  template_code: "TOP_4",
  seed_team_ids: ["team-1", "team-2", "team-3", "team-4"],
  round_scoring: { SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "BEST_2_OF_3" }
}).ok, true);
assert.match(validatePlayoffReviewConfiguration(playoffReview, {
  template_code: "TOP_4",
  seed_team_ids: ["team-1", "team-1", "team-3", "team-4"],
  round_scoring: { SF: "GAME_TO_11", BRONZE: "GAME_TO_11", FINAL: "BEST_2_OF_3" }
}).message, /different team/);

const beforePlayoffReady = { draws: [{ id: "draw-a", name: "Open", progression_status: "ROUND_ROBIN_IN_PROGRESS" }] };
const afterPlayoffReady = {
  draws: [{ id: "draw-a", name: "Open", progression_status: "READY_FOR_PLAYOFF_REVIEW" }],
  progression_alerts: [{ draw_id: "draw-a", ready: true }]
};
assert.deepEqual(readyPlayoffReviewDraws(afterPlayoffReady).map((draw) => draw.id), ["draw-a"]);
assert.equal(
  newlyReadyPlayoffNotice(beforePlayoffReady, afterPlayoffReady),
  "Round robin complete — Open is ready for playoff review."
);

function closeoutSnapshot({
  dayState = "ACTIVE",
  closeReady = true,
  closeBlockers = [],
  drawOverrides = {},
  operations = [],
  eligibleQueue = [],
  reservedQueue = [],
  heldGames = [],
  blockedGames = [],
  courtOverrides = {},
  summaryOverrides = {}
} = {}) {
  return {
    day_run: { state: dayState },
    summary: {
      courts: 2,
      available_courts: 2,
      active_draws: 1,
      eligible_games: eligibleQueue.length,
      reserved_games: reservedQueue.length,
      held_games: heldGames.length,
      completed_games: 8,
      ...summaryOverrides
    },
    courts: [1, 2].map((position) => ({
      id: `court-${position}`,
      state: "AVAILABLE",
      current_assignment: null,
      next_assignment: null,
      ...courtOverrides
    })),
    draws: [{
      id: "draw-a",
      activation_state: "ACTIVE",
      total_games: 8,
      finalized_games: 8,
      queued_games: 0,
      active_games: 0,
      held_games: 0,
      readiness: { closeout: { ready: true, blockers: [] } },
      ...drawOverrides
    }],
    eligible_queue: eligibleQueue,
    reserved_queue: reservedQueue,
    held_games: heldGames,
    blocked_games: blockedGames,
    operations,
    readiness: { close_day: { ready: closeReady, blockers: closeBlockers } }
  };
}

const readyCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot());
assert.deepEqual(
  {
    phase: readyCloseout?.phase,
    nextStep: readyCloseout?.nextStep,
    playComplete: readyCloseout?.playComplete,
    progressionComplete: readyCloseout?.progressionComplete,
    podiumComplete: readyCloseout?.podiumComplete,
    readyToClose: readyCloseout?.readyToClose
  },
  {
    phase: "closeout",
    nextStep: "close",
    playComplete: true,
    progressionComplete: true,
    podiumComplete: true,
    readyToClose: true
  },
  "active draw labels must not hide a drained, server-ready day closeout"
);

assert.equal(tournamentDayCloseoutGuidance(closeoutSnapshot({
  eligibleQueue: [{ game_id: "game-open" }],
  summaryOverrides: { eligible_games: 1 }
})), null, "a ready queue must keep the end-of-day guide hidden");
assert.equal(tournamentDayCloseoutGuidance(closeoutSnapshot({
  courtOverrides: { state: "ON_COURT" },
  summaryOverrides: { available_courts: 0 }
})), null, "an unavailable court must keep the end-of-day guide hidden");

const blockedCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "GAMES_UNFINISHED", message: "Finish every game." }],
  blockedGames: [{ game_id: "game-blocked" }]
}));
assert.equal(blockedCloseout?.nextStep, "matches");
assert.equal(blockedCloseout?.playComplete, false);

const unfinishedDrawCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "PLAYOFFS_REQUIRED", draw_id: "draw-a" }],
  drawOverrides: {
    finalized_games: 7,
    readiness: { closeout: { ready: false, blockers: [{ code: "PLAYOFFS_REQUIRED" }] } }
  }
}));
assert.equal(
  unfinishedDrawCloseout?.nextStep,
  "matches",
  "an idle draw with an unfinalized game must be reviewed before progression"
);

const progressionCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "PLAYOFFS_REQUIRED", draw_id: "draw-a" }],
  drawOverrides: {
    round_robin_complete: true,
    progression_status: "READY_FOR_PLAYOFF_REVIEW",
    readiness: { closeout: { ready: false, blockers: [{ code: "PLAYOFFS_REQUIRED" }] } }
  }
}));
assert.equal(progressionCloseout?.nextStep, "draws");
assert.deepEqual(progressionCloseout?.progressionDrawIds, ["draw-a"]);

const podiumCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "PODIUM_REVIEW_REQUIRED", draw_id: "draw-a" }],
  drawOverrides: {
    readiness: { closeout: { ready: false, blockers: [{ code: "PODIUM_REVIEW_REQUIRED" }] } }
  }
}));
assert.equal(podiumCloseout?.nextStep, "podium");
assert.deepEqual(podiumCloseout?.podiumDrawIds, ["draw-a"]);

const recoveryCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "PODIUM_REVIEW_REQUIRED", draw_id: "draw-a" }],
  operations: [{ status: "recovery_required" }]
}));
assert.equal(recoveryCloseout?.nextStep, "recovery", "recovery must outrank draw and podium closeout");

const serverRecoveryCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "OPERATION_UNSETTLED" }]
}));
assert.equal(
  serverRecoveryCloseout?.nextStep,
  "recovery",
  "the server close-readiness recovery blocker must route to reconciliation even without visible operation history"
);

const unknownCloseout = tournamentDayCloseoutGuidance(closeoutSnapshot({
  closeReady: false,
  closeBlockers: [{ code: "FUTURE_CLOSEOUT_GATE" }]
}));
assert.equal(unknownCloseout?.nextStep, "review", "unknown blockers must fail closed to review");
assert.equal(unknownCloseout?.readyToClose, false);

assert.deepEqual(
  tournamentDayCloseoutGuidance(closeoutSnapshot({ dayState: "CLOSED" })),
  {
    phase: "closed",
    nextStep: "done",
    playComplete: true,
    progressionComplete: true,
    podiumComplete: true,
    readyToClose: false,
    blockerCodes: [],
    progressionDrawIds: [],
    podiumDrawIds: []
  }
);
assert.equal(tournamentDayCloseoutGuidance(closeoutSnapshot({ dayState: "DRAFT" })), null);

const queue = [
  { game_id: "g-a-1", draw_id: "draw-a", position: 1 },
  { game_id: "g-b-1", draw_id: "draw-b", position: 2 },
  { game_id: "g-a-2", draw_id: "draw-a", position: 3 }
];

assert.deepEqual(
  visibleServerQueue(queue, "all").map((row) => [row.game_id, row.position]),
  [["g-a-1", 1], ["g-b-1", 2], ["g-a-2", 3]],
  "the client must render the server's unified order without deriving or renumbering it"
);
assert.deepEqual(
  visibleServerQueue(queue, "draw-a").map((row) => [row.game_id, row.position]),
  [["g-a-1", 1], ["g-a-2", 3]],
  "a draw visibility filter may hide rows but must retain authoritative positions"
);

const mixedQueuePayload = [
  { game_id: "newest", position: 3, priority: 1, eligible_since: "2026-08-17T09:03:00Z" },
  { game_id: "oldest", position: 1, priority: 50, eligible_since: "2026-08-17T09:01:00Z" },
  { game_id: "middle", position: 2, priority: 25, eligible_since: "2026-08-17T09:02:00Z" }
];
assert.deepEqual(
  oldestReadyQueue(mixedQueuePayload).map((row) => row.game_id),
  ["oldest", "middle", "newest"],
  "the client must use authoritative FIFO positions instead of static bracket priority"
);
assert.deepEqual(
  mixedQueuePayload.map((row) => row.game_id),
  ["newest", "oldest", "middle"],
  "ordering the queue must not mutate the snapshot arrays"
);

const boardQueue = [
  { game_id: "ready-a", draw_id: "draw-a", position: 1, state: "WAITING", blockers: [] },
  { game_id: "paused-b", draw_id: "draw-b", position: 2, state: "WAITING", blockers: [] },
  { game_id: "blocked-a", draw_id: "draw-a", position: 3, state: "WAITING", blockers: [{ code: "PLAYER_ALREADY_CLAIMED" }] },
  { game_id: "on-court-a", draw_id: "draw-a", position: 4, state: "ON_COURT", blockers: [] }
];
assert.deepEqual(
  readyActiveDrawQueue(boardQueue, [
    { id: "draw-a", activation_state: "ACTIVE" },
    { id: "draw-b", activation_state: "PAUSED" }
  ]).map((row) => [row.game_id, row.position]),
  [["ready-a", 1]],
  "the Court board strip must include only unblocked waiting games from active draws"
);

assert.equal(
  tournamentDayMedalMatchKind({ stage: "PLAYOFF", playoff_round: "Final" }),
  "gold"
);
assert.equal(
  tournamentDayMedalMatchKind({ stage: "PLAYOFF", round_label: "Championship" }),
  "gold"
);
assert.equal(
  tournamentDayMedalMatchKind({ stage: "PLAYOFF", playoff_round: "Bronze" }),
  "bronze"
);
assert.equal(
  tournamentDayMedalMatchKind({ stage: "PLAYOFF", round_label: "Third_place" }),
  "bronze"
);
assert.equal(
  tournamentDayMedalMatchKind({ stage: "PLAYOFF", playoff_round: "Semifinal" }),
  null
);
assert.equal(
  tournamentDayMedalMatchKind({ stage: "ROUND_ROBIN", round_label: "Final" }),
  null,
  "a non-playoff round named Final must not receive medal styling"
);

assert.deepEqual(
  resetFocusForDay(
    { dayId: "day-1", drawId: "draw-a", courtId: "court-3", gameId: "g-a-1", panel: "queue" },
    "day-2"
  ),
  { dayId: "day-2", drawId: "", courtId: "", gameId: "", panel: "queue" },
  "changing day must clear stale draw, court, and game focus"
);

assert.equal(workspaceScopeKey("token", "tournament-1", "day-1"), "token\u0000tournament-1\u0000day-1");
assert.equal(
  retainedDayCommandStorageKey("club-a", "tournament-1", "day-1"),
  "jupr_tournament_day_ops_pending_v1:club-a:tournament-1:day-1"
);

assert.equal(validateDayScoreDraft("11", "7").ok, true);
assert.equal(validateDayScoreDraft("11", "7").unusual, false);
for (const [scoreA, scoreB] of [["9", "9"], ["-1", "11"], ["11.5", "7"], ["", "7"]]) {
  const result = validateDayScoreDraft(scoreA, scoreB);
  assert.equal(result.ok, false, `${scoreA}-${scoreB} must not reach confirmation`);
}

assert.equal(validateDayCorrectionDraft("7", "11", 11, 7).ok, true);
assert.deepEqual(
  validateDayCorrectionDraft("11", "7", 11, 7),
  { ok: false, message: "Enter a changed final score before review." }
);
assert.equal(validateDayCorrectionDraft("9", "9", 11, 7).ok, false);
assert.equal(
  validateDayScoreDraft("11", "7", { format: null, blocker: "missing" }).ok,
  false,
  "an explicit configuration blocker must never fall back to a legacy format"
);

const fatFinger = validateDayScoreDraft(
  "76",
  "11",
  { format: "GAME_TO_11", target: 11, win_by_two: true }
);
assert.equal(fatFinger.ok, true);
assert.equal(fatFinger.unusual, true);
assert.equal(fatFinger.acknowledgementRequired, true);
assert.equal(
  validateDayScoreDraft(
    "76",
    "11",
    { format: "GAME_TO_11", target: 11, win_by_two: true },
    true
  ).acknowledgementRequired,
  false
);
assert.equal(
  validateDayScoreDraft("1", "0", { format: "BEST_2_OF_3", target: 2 }).ok,
  false,
  "BEST_2_OF_3 must use its individual-game validator"
);
const straightGames = [
  { game_number: 1, score_a: "11", score_b: "7" },
  { game_number: 2, score_a: "15", score_b: "13" },
  { game_number: 3, score_a: "", score_b: "" }
];
assert.deepEqual(validateBestOfThreeGameScores(straightGames), {
  ok: true,
  scoreA: 2,
  scoreB: 0,
  gameScores: [
    { game_number: 1, score_a: 11, score_b: 7 },
    { game_number: 2, score_a: 15, score_b: 13 }
  ],
  unusual: false,
  reasons: [],
  acknowledgementRequired: false,
  scoringFormat: "BEST_2_OF_3"
});
const splitGames = [
  { game_number: 1, score_a: "11", score_b: "8" },
  { game_number: 2, score_a: "9", score_b: "11" },
  { game_number: 3, score_a: "12", score_b: "10" }
];
assert.deepEqual(validateBestOfThreeGameScores(splitGames), {
  ok: true,
  scoreA: 2,
  scoreB: 1,
  gameScores: [
    { game_number: 1, score_a: 11, score_b: 8 },
    { game_number: 2, score_a: 9, score_b: 11 },
    { game_number: 3, score_a: 12, score_b: 10 }
  ],
  unusual: false,
  reasons: [],
  acknowledgementRequired: false,
  scoringFormat: "BEST_2_OF_3"
});
assert.deepEqual(
  validateBestOfThreeGameScores(splitGames.slice(0, 2)),
  { ok: false, message: "Enter Game 3 because the series is tied 1–1." }
);
assert.match(
  validateBestOfThreeGameScores([
    { game_number: 1, score_a: "11", score_b: "11" },
    { game_number: 2, score_a: "11", score_b: "7" }
  ]).message,
  /^Game 1:/
);
const unusualSeries = validateBestOfThreeGameScores([
  { game_number: 1, score_a: "76", score_b: "11" },
  { game_number: 2, score_a: "11", score_b: "4" }
]);
assert.equal(unusualSeries.ok, true);
assert.equal(unusualSeries.acknowledgementRequired, true);
assert.equal(validateBestOfThreeGameScores([
  { game_number: 1, score_a: "76", score_b: "11" },
  { game_number: 2, score_a: "11", score_b: "4" }
], null, true).acknowledgementRequired, false);
assert.deepEqual(
  validateBestOfThreeGameScores([
    ...splitGames.slice(0, 2),
    { game_number: 3, score_a: "11", score_b: "" }
  ]),
  { ok: false, message: "Enter both scores for Game 3." }
);
assert.equal(validateBestOfThreeGameScores(
  [
    { game_number: 1, score_a: "15", score_b: "13" },
    { game_number: 2, score_a: "15", score_b: "8" }
  ],
  {
    individual_game_format: "GAME_TO_15",
    individual_game_target: 15,
    individual_game_win_by_two: true
  }
).ok, true);
assert.deepEqual(
  validateBestOfThreeGameScores([
    ...straightGames.slice(0, 2),
    { game_number: 3, score_a: "11", score_b: "4" }
  ]),
  { ok: false, message: "Game 3 must stay empty because the series was won in the first two games." }
);
assert.deepEqual(
  validateBestOfThreeCorrectionDraft(splitGames, [
    { game_number: 1, score_a: 11, score_b: 8 },
    { game_number: 2, score_a: 9, score_b: 11 },
    { game_number: 3, score_a: 12, score_b: 10 }
  ]),
  { ok: false, message: "Enter a changed individual game score before review." }
);
assert.equal(
  validateBestOfThreeCorrectionDraft(
    splitGames.map((game) => game.game_number === 3 ? { ...game, score_a: "14", score_b: "12" } : game),
    [
      { game_number: 1, score_a: 11, score_b: 8 },
      { game_number: 2, score_a: 9, score_b: 11 },
      { game_number: 3, score_a: 12, score_b: 10 }
    ]
  ).ok,
  true
);
const retirementBeforePlay = validateBestOfThreeRetirementGameScores([
  { game_number: 1, score_a: "", score_b: "" },
  { game_number: 2, score_a: "", score_b: "" },
  { game_number: 3, score_a: "", score_b: "" }
]);
assert.equal(retirementBeforePlay.ok, true);
assert.deepEqual(retirementBeforePlay.gameScores, []);
const retirementAfterOne = validateBestOfThreeRetirementGameScores([
  { game_number: 1, score_a: "11", score_b: "7" },
  { game_number: 2, score_a: "", score_b: "" }
]);
assert.equal(retirementAfterOne.ok, true);
assert.deepEqual(retirementAfterOne.gameScores, [
  { game_number: 1, score_a: 11, score_b: 7 }
]);
const retirementAfterSplit = validateBestOfThreeRetirementGameScores(splitGames.slice(0, 2));
assert.equal(retirementAfterSplit.ok, true);
assert.deepEqual(retirementAfterSplit.gameScores, [
  { game_number: 1, score_a: 11, score_b: 8 },
  { game_number: 2, score_a: 9, score_b: 11 }
]);
assert.match(
  validateBestOfThreeRetirementGameScores(straightGames.slice(0, 2)).message,
  /already won Games 1 and 2/
);
assert.deepEqual(
  validateBestOfThreeRetirementGameScores([
    { game_number: 1, score_a: "11", score_b: "7" },
    { game_number: 2, score_a: "8", score_b: "" }
  ]),
  { ok: false, message: "Enter both scores for completed Game 2." }
);
const unusualRetirementGame = validateBestOfThreeRetirementGameScores([
  { game_number: 1, score_a: "76", score_b: "11" }
]);
assert.equal(unusualRetirementGame.ok, true);
assert.equal(unusualRetirementGame.acknowledgementRequired, true);
assert.equal(validateBestOfThreeRetirementGameScores([
  { game_number: 1, score_a: "76", score_b: "11" }
], null, true).acknowledgementRequired, false);
assert.equal(validateNonPlayedOutcomeDraft("NO_SHOW", "team-a", "Opponent absent").ok, true);
assert.deepEqual(validateNonPlayedOutcomeDraft("RETIREMENT", "team-b", ""), {
  ok: true,
  resultType: "RETIREMENT",
  nonPlayingTeamId: "team-b",
  resultNote: ""
});
assert.equal(validateNonPlayedOutcomeDraft("", "team-a", "Opponent absent").ok, false);

console.log("tournament day workspace state contract: ok");
