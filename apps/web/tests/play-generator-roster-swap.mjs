import assert from "node:assert/strict";

import { swapRosterPositions } from "../lib/playGeneratorRoster.mjs";

const originalOrder = Object.freeze(["player-a", "player-b", "player-c", "player-d"]);

assert.deepEqual(
  swapRosterPositions(originalOrder, "player-a", "player-d"),
  ["player-d", "player-b", "player-c", "player-a"],
  "a swap must exchange only the two selected roster positions"
);
assert.deepEqual(
  originalOrder,
  ["player-a", "player-b", "player-c", "player-d"],
  "swapping must not mutate the current session snapshot"
);
assert.throws(
  () => swapRosterPositions(originalOrder, "player-a", "player-a"),
  /two different players/,
  "the same player cannot fill both sides of a swap"
);
assert.throws(
  () => swapRosterPositions(originalOrder, "player-a", "missing-player"),
  /current roster/,
  "a stale or missing player must not produce a partial reorder"
);
