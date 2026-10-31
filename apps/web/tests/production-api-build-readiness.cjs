const assert = require("node:assert/strict");
const { operationsReady, weeklyReportReady, leagueLiveReady, roundRobinHubReady, siteReliabilityReady, startingRatingReviewReady, tournamentPartnerLookupReady, tournamentPoolsReady, tournamentCombinedDivisionsReady, waitForProductionApi, REQUIRED_FLAGS } = require("../scripts/wait-for-production-api.cjs");
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
const partnerLookupPath = "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/partner-profile-resolution";
contract.paths[partnerLookupPath] = { post: {} };
contract.paths[preferencesPath] = { get: {}, put: {} };
contract.components.schemas.AdminWeeklyRecapPublishRequest = { properties: { edits_json: { type: "object" } } };
const hubRoutes = [
  ["get", "/public/round-robin-hubs/{hub_id}/registration"],
  ["post", "/public/round-robin-hubs/{hub_id}/checkins"],
  ["patch", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submission/scores"],
];
for (const [method, path] of hubRoutes) contract.paths[path] = { [method]: {} };
contract.components.schemas.HubCommandRequest = { properties: { action: { enum: ["set_registration", "review_checkin", "process_checkins", "edit_match_players", "transfer_player", "rejoin_player", "resolve_game", "set_match_lineup", "remove_players", "create_extra_game"] } } };
contract.components.schemas.GeneratorStartRequest = { properties: {
  registration_hub_id: { anyOf: [{ type: "string", format: "uuid" }, { type: "null" }] },
  participant_names: { type: "array" },
} };
const reliabilityReads = [
  "/public/clubs/{slug}/site/shell",
  "/public/clubs/{slug}/site/logo",
  "/clubs/{club_slug}/players/{player_id}/share-metadata",
  "/clubs/{club_slug}/play-generators/sessions/{session_key}/share-metadata",
];
for (const path of reliabilityReads) contract.paths[path] = { get: {} };
for (const name of ["GeneratorRosterRequest", "PublicGeneratorRosterRequest"]) {
  contract.components.schemas[name] = { properties: { participant_ids: { type: "array", maxItems: 40 } } };
}
const ratingReviewPaths = ["preview", "confirm"].map((action) => `/admin/clubs/{club_id}/starting-rating-review/${action}`);
for (const path of ratingReviewPaths) contract.paths[path] = { post: {} };
contract.components.schemas.StartingRatingReviewPreviewRequest = { properties: {
  player_ids: { type: "array" }, requested_ratings: { type: "object" }, match_format: { enum: ["doubles", "singles"] },
} };
contract.components.schemas.StartingRatingReviewConfirmRequest = { properties: {
  match_format: { enum: ["doubles", "singles"] },
  reviews: { type: "array", items: { $ref: "#/components/schemas/StartingRatingReviewConfirmation" } },
} };
contract.components.schemas.StartingRatingReviewConfirmation = { properties: {
  player_id: { type: "integer" }, rating_jupr: { type: "number" }, expected_state_fingerprint: { type: "string" },
} };
const poolsPath = "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/pools";
contract.paths[poolsPath] = { put: { requestBody: { content: { "application/json": {
  schema: { $ref: "#/components/schemas/AdminTournamentDrawPoolsRequest" },
} } } } };
contract.components.schemas.AdminTournamentDrawPoolsRequest = { properties: {
  pool_config: { anyOf: [{ type: "object" }, { type: "null" }] },
  expected_state_fingerprint: { type: "string" }, expected_draw_updated_at: { type: "string" },
  expected_team_versions: { type: "array" }, confirmation_text: { type: "string" },
} };
contract.components.schemas.AdminTournamentDayLivePayload = { properties: {
  advance_count: { anyOf: [{ type: "integer", minimum: 2, maximum: 16 }, { type: "null" }] },
  playoff_configuration: { anyOf: [{ $ref: "#/components/schemas/AdminTournamentDayLivePlayoffConfiguration" }, { type: "null" }] },
} };
contract.components.schemas.AdminTournamentDayLivePlayoffConfiguration = { properties: {
  template_code: { type: "string", pattern: "^SINGLE_ELIMINATION_([2-9]|1[0-6])$" },
  seed_team_ids: { type: "array", minItems: 2, maxItems: 16 },
  override_reason: { anyOf: [{ type: "string", maxLength: 500 }, { type: "null" }] },
} };
const combinePath = "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/combine-divisions";
contract.paths[combinePath] = { post: { requestBody: { content: { "application/json": {
  schema: { $ref: "#/components/schemas/AdminTournamentDrawCombineRequest" },
} } } } };
contract.components.schemas.AdminTournamentDrawCombineRequest = {
  required: ["event_option_ids", "name", "expected_state_fingerprint"],
  properties: {
    event_option_ids: { type: "array", items: { type: "string" }, minItems: 2, maxItems: 16 },
    name: { type: "string", minLength: 1, maxLength: 180 },
    expected_state_fingerprint: { type: "string", minLength: 64, maxLength: 64 },
    confirmation_text: { type: "string" }, source: { type: "string" },
  },
};

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
  assert.equal(siteReliabilityReady(contract), true);
  assert.equal(siteReliabilityReady({}), false);
  for (const path of reliabilityReads) {
    const missingRead = structuredClone(contract);
    delete missingRead.paths[path].get;
    assert.equal(siteReliabilityReady(missingRead), false, `The web release requires GET ${path}`);
  }
  for (const name of ["GeneratorRosterRequest", "PublicGeneratorRosterRequest"]) {
    const oldRoster = structuredClone(contract);
    oldRoster.components.schemas[name].properties.participant_ids.maxItems = 4;
    assert.equal(siteReliabilityReady(oldRoster), false, `${name} must support the full departure batch`);
    delete oldRoster.components.schemas[name].properties.participant_ids;
    assert.equal(siteReliabilityReady(oldRoster), false);
  }
  assert.equal(startingRatingReviewReady(contract), true);
  assert.equal(startingRatingReviewReady({}), false);
  assert.equal(tournamentPartnerLookupReady(contract), true);
  assert.equal(tournamentPartnerLookupReady({}), false);
  const beforePartnerLookup = structuredClone(contract);
  delete beforePartnerLookup.paths[partnerLookupPath].post;
  assert.equal(tournamentPartnerLookupReady(beforePartnerLookup), false);
  assert.equal(tournamentPoolsReady(contract), true);
  assert.equal(tournamentPoolsReady({}), false);
  assert.equal(tournamentCombinedDivisionsReady(contract), true);
  assert.equal(tournamentCombinedDivisionsReady({}), false);
  const incompleteCombineContracts = [
    ["the combine POST route", api => { delete api.paths[combinePath].post; }],
    ["the correct combine request schema", api => { api.paths[combinePath].post.requestBody.content["application/json"].schema.$ref = "#/components/schemas/AdminTournamentDrawPoolsRequest"; }],
    ["the multi-division roster field", api => { api.components.schemas.AdminTournamentDrawCombineRequest.properties.event_option_ids.type = "string"; }],
    ["the supported division range", api => { api.components.schemas.AdminTournamentDrawCombineRequest.properties.event_option_ids.maxItems = 2; }],
    ["the reviewed-state guard", api => { api.components.schemas.AdminTournamentDrawCombineRequest.required = ["event_option_ids", "name"]; }],
    ["the complete state fingerprint", api => { api.components.schemas.AdminTournamentDrawCombineRequest.properties.expected_state_fingerprint.minLength = 0; }],
    ["explicit confirmation", api => { delete api.components.schemas.AdminTournamentDrawCombineRequest.properties.confirmation_text; }],
  ].map(([reason, change]) => {
    const previousApi = structuredClone(contract);
    change(previousApi);
    assert.equal(tournamentCombinedDivisionsReady(previousApi), false, reason);
    return [reason, previousApi];
  });
  const incompletePoolContracts = [
    ["the pool save route", (api) => { delete api.paths[poolsPath].put; }],
    ["the guarded pool request", (api) => { api.paths[poolsPath].put.requestBody.content["application/json"].schema.$ref = "#/components/schemas/OldRequest"; }],
    ["pool removal", (api) => { api.components.schemas.AdminTournamentDrawPoolsRequest.properties.pool_config = { type: "object" }; }],
    ...["pool_config", "expected_state_fingerprint", "expected_draw_updated_at", "expected_team_versions", "confirmation_text"].map((field) => [
      `pool save ${field}`, (api) => { delete api.components.schemas.AdminTournamentDrawPoolsRequest.properties[field]; },
    ]),
    ["playoff configuration", (api) => { delete api.components.schemas.AdminTournamentDayLivePayload.properties.playoff_configuration; }],
    ["two-team playoffs", (api) => { api.components.schemas.AdminTournamentDayLivePayload.properties.advance_count.anyOf[0].minimum = 4; }],
    ["sixteen-team playoffs", (api) => { api.components.schemas.AdminTournamentDayLivePayload.properties.advance_count.anyOf[0].maximum = 8; }],
    ["two-team seed lists", (api) => { api.components.schemas.AdminTournamentDayLivePlayoffConfiguration.properties.seed_team_ids.minItems = 4; }],
    ["sixteen-team seed lists", (api) => { api.components.schemas.AdminTournamentDayLivePlayoffConfiguration.properties.seed_team_ids.maxItems = 8; }],
    ["all playoff template sizes", (api) => { api.components.schemas.AdminTournamentDayLivePlayoffConfiguration.properties.template_code.pattern = "^SINGLE_ELIMINATION_(4|6|8)$"; }],
    ["valid playoff template schema", (api) => { api.components.schemas.AdminTournamentDayLivePlayoffConfiguration.properties.template_code.pattern = "["; }],
    ["manual override reasons", (api) => { delete api.components.schemas.AdminTournamentDayLivePlayoffConfiguration.properties.override_reason; }],
  ].map(([reason, mutate]) => {
    const api = structuredClone(contract);
    mutate(api);
    assert.equal(tournamentPoolsReady(api), false, `The release requires ${reason}`);
    return [reason, api];
  });
  const incompleteRatingReviews = [];
  for (const path of ratingReviewPaths) {
    const missingRoute = structuredClone(contract);
    delete missingRoute.paths[path].post;
    incompleteRatingReviews.push(missingRoute);
  }
  for (const [schema, field] of [
    ["StartingRatingReviewPreviewRequest", "player_ids"],
    ["StartingRatingReviewPreviewRequest", "requested_ratings"],
    ["StartingRatingReviewPreviewRequest", "match_format"],
    ["StartingRatingReviewConfirmRequest", "reviews"],
    ["StartingRatingReviewConfirmRequest", "match_format"],
    ["StartingRatingReviewConfirmation", "player_id"],
    ["StartingRatingReviewConfirmation", "rating_jupr"],
    ["StartingRatingReviewConfirmation", "expected_state_fingerprint"],
  ]) {
    const missingField = structuredClone(contract);
    delete missingField.components.schemas[schema].properties[field];
    incompleteRatingReviews.push(missingField);
  }
  for (const previousApi of incompleteRatingReviews) assert.equal(startingRatingReviewReady(previousApi), false);
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
  contracts = 0;
  const oldHubOperations = structuredClone(contract);
  oldHubOperations.components.schemas.HubCommandRequest.properties.action.enum = ["set_registration", "review_checkin", "process_checkins", "edit_match_players"];
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? oldHubOperations : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for transfer, rejoin, and not-played support before exposing the new Hub controls");
  clock = 0;
  contracts = 0;
  const beforePlannedHub = structuredClone(contract);
  beforePlannedHub.components.schemas.HubCommandRequest.properties.action.enum = beforePlannedHub.components.schemas.HubCommandRequest.properties.action.enum.filter(action => !["set_match_lineup", "remove_players", "create_extra_game"].includes(action));
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? beforePlannedHub : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for exact lineups, batch departures, and exhibition games before exposing the planned Hub controls");
  clock = 0;
  contracts = 0;
  const beforeSubmissionCorrections = structuredClone(contract);
  delete beforeSubmissionCorrections.paths["/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submission/scores"];
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? beforeSubmissionCorrections : contract) : ready }) });
  assert.equal(contracts, 2, "The website waits for submission score correction support before exposing the new editor");
  for (const previousApi of [
    (() => { const api = structuredClone(contract); delete api.paths[reliabilityReads[0]]; return api; })(),
    (() => { const api = structuredClone(contract); api.components.schemas.PublicGeneratorRosterRequest.properties.participant_ids.maxItems = 4; return api; })(),
  ]) {
    clock = 0;
    contracts = 0;
    await waitForProductionApi({ environment: "production", now: () => clock,
      pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
        json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? previousApi : contract) : ready }) });
    assert.equal(contracts, 2, "The website waits for compact shell reads and full roster departure batches");
  }
  for (const previousApi of incompleteRatingReviews) {
    clock = 0;
    contracts = 0;
    await waitForProductionApi({ environment: "production", now: () => clock,
      pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
        json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? previousApi : contract) : ready }) });
    assert.equal(contracts, 2, "The website waits for starting-rating preview, confirmation, and concurrency request fields");
  }
  clock = 0;
  contracts = 0;
  await waitForProductionApi({ environment: "production", now: () => clock,
    pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? beforePartnerLookup : contract) : ready }) });
  assert.equal(contracts, 2, "The updated partner editor waits until its lookup endpoint is deployed");
  for (const [reason, previousApi] of incompletePoolContracts) {
    clock = 0;
    contracts = 0;
    await waitForProductionApi({ environment: "production", now: () => clock,
      pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
        json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? previousApi : contract) : ready }) });
    assert.equal(contracts, 2, `The pool interface waits for ${reason} before release`);
  }
  for (const [reason, previousApi] of incompleteCombineContracts) {
    clock = 0;
    contracts = 0;
    await waitForProductionApi({ environment: "production", now: () => clock,
      pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
        json: async () => url.endsWith('/openapi.json') ? (++contracts === 1 ? previousApi : contract) : ready }) });
    assert.equal(contracts, 2, `The combined-division interface waits for ${reason} before release`);
  }
  clock = 0;
  await assert.rejects(waitForProductionApi({ environment: "production", timeoutMs: 10000,
    now: () => clock, pause: async () => { clock += 10000; }, request: async (url) => ({ ok: true,
      json: async () => url.endsWith('/openapi.json') ? incompleteCombineContracts[0][1] : ready }) }), /current website remains active/);
  clock = 0;
  await assert.rejects(waitForProductionApi({ environment: "production", timeoutMs: 10000,
    now: () => clock, pause: async () => { clock += 10000; }, request: async () => { throw new Error("Unavailable"); } }), /current website remains active/);
  console.log("Production API build readiness checks passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
