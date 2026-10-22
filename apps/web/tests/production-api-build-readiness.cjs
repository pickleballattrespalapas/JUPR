const assert = require("node:assert/strict");
const { operationsReady, weeklyReportReady, leagueLiveReady, roundRobinHubReady, waitForProductionApi, REQUIRED_FLAGS } = require("../scripts/wait-for-production-api.cjs");
const ready = { ok: true, club_website_settings_available: true, interclub_seasons_available: true, environment: "production", fly_app_name: "juprleagues-api",
  supabase_project_ref: "dnoockbwfenunhcibwfn", production_business_write_policy: "enabled",
  staging_write_wave: "none", feature_flags: Object.fromEntries(REQUIRED_FLAGS.map((flag) => [flag, true])) };
const contract = { components: { schemas: { AdminWeeklyRecapGenerateRequest: { properties: {
  preserve_edits: { type: "boolean", default: true }, edits_json: { type: "object" },
} } } } };
for (const name of ["AdminLeagueLiveRoundPublishRequest", "AdminLeagueLiveRoundPlanRequest"]) {
  contract.components.schemas[name] = { properties: { unplayed_games: { anyOf: [{ type: "array" }, { type: "null" }] } } };
}
contract.paths = Object.fromEntries([
  ["round-robin/preview", "post"], ["players", "post"],
  ["player-operations/{operation_key}", "get"],
  ["player-operations/{operation_key}/reconcile", "post"],
].map(([suffix, method]) => [`/admin/clubs/{club_id}/league-manager/live/${suffix}`, { [method]: {} }]));
const preferencesPath = "/admin/clubs/{club_id}/weekly-recap/preferences";
contract.paths[preferencesPath] = { get: {}, put: {} };
contract.components.schemas.AdminWeeklyRecapPublishRequest = { properties: { edits_json: { type: "object" } } };
const hubRoutes = [
  ["get", "/public/round-robin-hubs/{hub_id}/registration"],
  ["post", "/public/round-robin-hubs/{hub_id}/checkins"],
];
for (const [method, path] of hubRoutes) contract.paths[path] = { [method]: {} };
contract.components.schemas.HubCommandRequest = { properties: { action: { enum: ["set_registration", "review_checkin", "process_checkins", "edit_match_players"] } } };
contract.components.schemas.GeneratorStartRequest = { properties: {
  registration_hub_id: { anyOf: [{ type: "string", format: "uuid" }, { type: "null" }] },
  participant_names: { type: "array" },
} };

(async () => {
  assert.equal(operationsReady(ready), true);
  assert.equal(operationsReady({ ...ready, club_website_settings_available: undefined }), false);
  assert.equal(operationsReady({ ...ready, interclub_seasons_available: undefined }), false);
  for (const flag of REQUIRED_FLAGS) {
    assert.equal(operationsReady({ ...ready, feature_flags: { ...ready.feature_flags, [flag]: false } }), false);
  }
  assert.equal(operationsReady({ ...ready, supabase_project_ref: "sijpxjxvdtrehmqvirfi" }), false);
  assert.equal(operationsReady({ ...ready, production_business_write_policy: "disabled" }), false);
  assert.equal(weeklyReportReady(contract), true);
  assert.equal(leagueLiveReady(contract), true);
  assert.equal(leagueLiveReady({}), false);
  for (const name of ["AdminLeagueLiveRoundPublishRequest", "AdminLeagueLiveRoundPlanRequest"]) {
    const missingGameEdits = structuredClone(contract);
    delete missingGameEdits.components.schemas[name].properties.unplayed_games;
    assert.equal(leagueLiveReady(missingGameEdits), false);
    missingGameEdits.components.schemas[name].properties.unplayed_games = { type: "boolean" };
    assert.equal(leagueLiveReady(missingGameEdits), false);
  }
  for (const path of Object.keys(contract.paths).filter((path) => path.includes("/league-manager/live/"))) {
    const missingRoute = structuredClone(contract);
    delete missingRoute.paths[path];
    assert.equal(leagueLiveReady(missingRoute), false);
  }
  assert.equal(weeklyReportReady({}), false);
  const oldContract = structuredClone(contract);
  delete oldContract.components.schemas.AdminWeeklyRecapGenerateRequest.properties.edits_json;
  assert.equal(weeklyReportReady(oldContract), false);
  const oldBuilderContract = structuredClone(contract);
  delete oldBuilderContract.paths[preferencesPath];
  assert.equal(weeklyReportReady(oldBuilderContract), false);
  for (const method of ["get", "put"]) {
    const missingPreferences = structuredClone(contract);
    delete missingPreferences.paths[preferencesPath][method];
    assert.equal(weeklyReportReady(missingPreferences), false);
  }
  const missingDraftPublication = structuredClone(contract);
  delete missingDraftPublication.components.schemas.AdminWeeklyRecapPublishRequest.properties.edits_json;
  assert.equal(weeklyReportReady(missingDraftPublication), false);
  assert.equal(roundRobinHubReady(contract), true);
  assert.equal(roundRobinHubReady({}), false);
  for (const [method, path] of hubRoutes) {
    const missingRoute = structuredClone(contract);
    delete missingRoute.paths[path][method];
    assert.equal(roundRobinHubReady(missingRoute), false);
  }
  for (const action of contract.components.schemas.HubCommandRequest.properties.action.enum) {
    const missingAction = structuredClone(contract);
    missingAction.components.schemas.HubCommandRequest.properties.action.enum = missingAction.components.schemas.HubCommandRequest.properties.action.enum.filter((value) => value !== action);
    assert.equal(roundRobinHubReady(missingAction), false);
  }
  const oldHubSetup = structuredClone(contract);
  delete oldHubSetup.components.schemas.GeneratorStartRequest.properties.registration_hub_id;
  assert.equal(roundRobinHubReady(oldHubSetup), false);
  const mandatoryRoster = structuredClone(contract);
  mandatoryRoster.components.schemas.GeneratorStartRequest.properties.participant_names.minItems = 2;
  assert.equal(roundRobinHubReady(mandatoryRoster), false);
  let calls = 0, clock = 0;
  await waitForProductionApi({ environment: "preview", request: async () => { throw new Error("Preview must not contact production"); } });
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? contract : (++calls === 1 ? { ...ready, feature_flags: {} } : ready) }) });
  assert.equal(calls, 2);
  clock = 0;
  let contracts = 0;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? oldContract : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits while a healthy API still lacks edit-preserving recap refresh");
  clock = 0;
  contracts = 0;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? oldBuilderContract : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for recap preferences and draft revision publishing before exposing the builder");
  clock = 0;
  contracts = 0;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? { ...contract, paths: {} } : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for League Live's independent preview and player setup routes");
  clock = 0;
  contracts = 0;
  const oldLiveContract = structuredClone(contract);
  delete oldLiveContract.components.schemas.AdminLeagueLiveRoundPublishRequest.properties.unplayed_games;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? oldLiveContract : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for Live Rounds game edits before publishing the new controls");
  clock = 0;
  contracts = 0;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? oldHubSetup : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for arrivals-driven RR Hub setup before publishing check-in and court player editing");
  clock = 0;
  await assert.rejects(waitForProductionApi({ environment: "production", timeoutMs: 10000,
    now: () => clock, pause: async () => { clock += 10000; }, request: async () => { throw new Error("Unavailable"); } }), /current website remains active/);
  console.log("Production API build readiness checks passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
