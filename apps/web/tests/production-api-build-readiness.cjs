const assert = require("node:assert/strict");
const { operationsReady, waitForProductionApi, REQUIRED_FLAGS } = require("../scripts/wait-for-production-api.cjs");
const ready = { ok: true, environment: "production", fly_app_name: "juprleagues-api",
  supabase_project_ref: "dnoockbwfenunhcibwfn", production_business_write_policy: "enabled",
  staging_write_wave: "none", feature_flags: Object.fromEntries(REQUIRED_FLAGS.map((flag) => [flag, true])) };

(async () => {
  assert.equal(operationsReady(ready), true);
  for (const flag of REQUIRED_FLAGS) {
    assert.equal(operationsReady({ ...ready, feature_flags: { ...ready.feature_flags, [flag]: false } }), false);
  }
  assert.equal(operationsReady({ ...ready, supabase_project_ref: "sijpxjxvdtrehmqvirfi" }), false);
  assert.equal(operationsReady({ ...ready, production_business_write_policy: "disabled" }), false);
  let calls = 0, clock = 0;
  await waitForProductionApi({ environment: "preview", request: async () => { throw new Error("Preview must not contact production"); } });
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async () => ({ ok: true,
      json: async () => (++calls === 1 ? { ...ready, feature_flags: {} } : ready) }) });
  assert.equal(calls, 2);
  clock = 0;
  await assert.rejects(waitForProductionApi({ environment: "production", timeoutMs: 10000,
    now: () => clock, pause: async () => { clock += 10000; }, request: async () => { throw new Error("Unavailable"); } }), /current website remains active/);
  console.log("Production API build readiness checks passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
