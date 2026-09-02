import assert from "node:assert/strict";
import {
  drawOperationalStatus,
  isInactiveTournamentDraw
} from "../lib/tournamentDrawOperationalStatus.mjs";

const draft = { status: "DRAFT" };
const lifecycle = ({
  games,
  finalized,
  open,
  published = 0,
  live = "not_started",
  official = "blocked",
  ...extraCounts
}) => ({
  counts: {
    games,
    finalized_games: finalized,
    open_games: open,
    published_games: published,
    ...extraCounts
  },
  states: {
    live_operations: live,
    official_publish: official
  }
});

assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 0, finalized: 0, open: 0 })),
  "No games scheduled"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 0, open: 21 })),
  "Not started · 21 games"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 1, open: 20, live: "in_progress" })),
  "In progress · 1 of 21 scored"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 21, open: 0, live: "complete" })),
  "Scores complete · 21 of 21 scored"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 21, open: 0, published: 21, live: "complete", official: "complete" })),
  "Published · 21 official matches"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 21, open: 0, published: 1, live: "complete" })),
  "Publish recovery needed · 1 of 21 official"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 21, open: 0, published: 21, live: "complete", official: "complete", duplicate_publications: 1 })),
  "Publish recovery needed · 21 of 21 official"
);
assert.equal(drawOperationalStatus({ status: "ARCHIVED" }), "Archived");
assert.equal(isInactiveTournamentDraw({ status: "ARCHIVED" }), true);
assert.equal(drawOperationalStatus(draft), "Status unavailable");
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 21, finalized: 1, open: 19, live: "in_progress" })),
  "Status unavailable"
);

console.log("tournament draw operational status contract: ok");
