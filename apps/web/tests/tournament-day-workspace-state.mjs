import assert from "node:assert/strict";

import {
  advanceCountSelection,
  dayActionConfirmation,
  dayRunAcceptsLiveCommands,
  dayRunHasStarted,
  resetFocusForDay,
  retainedDayCommandStorageKey,
  validateDayCorrectionDraft,
  validateDayScoreDraft,
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

assert.deepEqual(validateDayScoreDraft("11", "7"), { ok: true, scoreA: 11, scoreB: 7 });
for (const [scoreA, scoreB] of [["9", "9"], ["-1", "11"], ["11.5", "7"], ["", "7"]]) {
  const result = validateDayScoreDraft(scoreA, scoreB);
  assert.equal(result.ok, false, `${scoreA}-${scoreB} must not reach confirmation`);
}

assert.deepEqual(
  validateDayCorrectionDraft("7", "11", 11, 7),
  { ok: true, scoreA: 7, scoreB: 11 }
);
assert.deepEqual(
  validateDayCorrectionDraft("11", "7", 11, 7),
  { ok: false, message: "Enter a changed final score before review." }
);
assert.equal(validateDayCorrectionDraft("9", "9", 11, 7).ok, false);

console.log("tournament day workspace state contract: ok");
