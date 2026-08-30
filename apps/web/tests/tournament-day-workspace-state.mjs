import assert from "node:assert/strict";

import {
  advanceCountSelection,
  dayActionConfirmation,
  dayRunAcceptsLiveCommands,
  dayRunHasStarted,
  oldestReadyQueue,
  readyActiveDrawQueue,
  resetFocusForDay,
  retainedDayCommandStorageKey,
  tournamentDayMedalMatchKind,
  validateDayCorrectionDraft,
  validateDayScoreDraft,
  validateNonPlayedOutcomeDraft,
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
  "BEST_2_OF_3 uses games won and cannot finish 1-0"
);
assert.equal(validateNonPlayedOutcomeDraft("NO_SHOW", "team-a", "Opponent absent").ok, true);
assert.deepEqual(validateNonPlayedOutcomeDraft("RETIREMENT", "team-b", ""), {
  ok: true,
  resultType: "RETIREMENT",
  nonPlayingTeamId: "team-b",
  resultNote: ""
});
assert.equal(validateNonPlayedOutcomeDraft("", "team-a", "Opponent absent").ok, false);

console.log("tournament day workspace state contract: ok");
