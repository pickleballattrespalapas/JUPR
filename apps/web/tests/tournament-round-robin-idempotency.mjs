import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  clearRoundRobinGenerationIdempotencyKey,
  roundRobinGenerationReviewedFingerprint,
  stableRoundRobinGenerationIdempotencyKey
} from "../lib/tournamentRoundRobinIdempotency.mjs";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key)
  };
}

const UUID_ONE = "11111111-1111-4111-8111-111111111111";
const UUID_TWO = "22222222-2222-4222-8222-222222222222";
const UUID_THREE = "33333333-3333-4333-8333-333333333333";
const reviewedRequest = {
  expected_state_fingerprint: "state-one",
  expected_draw_updated_at: "2026-08-25T23:48:41.915211Z",
  expected_team_versions: [
    { id: "team-b", updated_at: "version-b" },
    { id: "team-a", updated_at: "version-a" }
  ]
};

assert.equal(
  roundRobinGenerationReviewedFingerprint(reviewedRequest),
  roundRobinGenerationReviewedFingerprint({
    ...reviewedRequest,
    expected_team_versions: [...reviewedRequest.expected_team_versions].reverse()
  }),
  "team row ordering must not rotate the operation key"
);

const storage = memoryStorage();
let generated = 0;
const uuids = [UUID_ONE, UUID_TWO, UUID_THREE];
const options = { storage, createUuid: () => uuids[generated++] };
const scope = "club:tournament:draw";

const first = stableRoundRobinGenerationIdempotencyKey(scope, reviewedRequest, options);
const uncertainRetry = stableRoundRobinGenerationIdempotencyKey(scope, reviewedRequest, options);
assert.equal(first, UUID_ONE);
assert.equal(uncertainRetry, UUID_ONE, "an unchanged reviewed request must reuse its UUID");
assert.equal(generated, 1);

const afterReload = stableRoundRobinGenerationIdempotencyKey(scope, reviewedRequest, {
  storage,
  createUuid: () => {
    throw new Error("a retained request must not generate a new UUID");
  }
});
assert.equal(afterReload, UUID_ONE, "session storage must retain the UUID across a reload");

const changedState = stableRoundRobinGenerationIdempotencyKey(scope, {
  ...reviewedRequest,
  expected_state_fingerprint: "state-two"
}, options);
assert.equal(changedState, UUID_TWO, "a changed reviewed state must receive a new UUID");

clearRoundRobinGenerationIdempotencyKey(scope, changedState, { storage });
const afterConclusiveResult = stableRoundRobinGenerationIdempotencyKey(scope, {
  ...reviewedRequest,
  expected_state_fingerprint: "state-two"
}, options);
assert.equal(afterConclusiveResult, UUID_THREE, "success or definite failure must release the completed UUID");

const panel = readFileSync(
  new URL("../app/admin/tournaments/ops/TournamentOpsPanel.tsx", import.meta.url),
  "utf8"
);
const generationFlow = panel
  .split("async function generateGames", 2)[1]
  .split("async function recoverRoundRobin", 1)[0];
assert.match(
  generationFlow,
  /body: JSON\.stringify\(\{ \.\.\.request, idempotency_key: idempotencyKey \}\)/,
  "round-robin generation must send the retained UUID"
);
assert.equal(
  (generationFlow.match(/clearRoundRobinGenerationIdempotencyKey\(operationScope, idempotencyKey\)/g) || []).length,
  4,
  "every conclusive success exit and definite failure must release the UUID"
);
assert.match(
  generationFlow,
  /if \(!\(error instanceof TournamentOpsRequestError\) \|\| error\.uncertain\) \{[\s\S]*actionUncertain\([\s\S]*\(\) => generateGames\(confirmationText\)/,
  "an uncertain result must expose an exact-request retry"
);

console.log("tournament round-robin idempotency contract: ok");
