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
// Best-of-three matchups publish their individual games, not their series parent.
for (const [games, eligible] of [[27, 30], [25, 29]]) {
  assert.equal(
    drawOperationalStatus(draft, lifecycle({ games, finalized: games, open: 0, published: eligible, rating_publish_eligible_games: eligible, live: "complete", official: "complete" })),
    `Published · ${eligible} official matches`
  );
  assert.equal(
    drawOperationalStatus(draft, lifecycle({ games, finalized: games, open: 0, published: games, rating_publish_eligible_games: eligible, live: "complete" })),
    `Publish recovery needed · ${games} of ${eligible} official`
  );
}
// Non-played outcomes count toward draw completion but must never be rated.
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 10, finalized: 10, open: 0, published: 9, rating_publish_eligible_games: 9, live: "complete", official: "complete" })),
  "Published · 9 official matches"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 1, finalized: 1, open: 0, rating_publish_eligible_games: 0, live: "complete", official: "complete" })),
  "Complete · no rated games"
);
assert.equal(
  drawOperationalStatus(draft, lifecycle({ games: 27, finalized: 27, open: 0, published: 31, rating_publish_eligible_games: 30, live: "complete", official: "complete" })),
  "Status unavailable"
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
