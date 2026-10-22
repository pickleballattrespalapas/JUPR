// Keep the current website serving while the separately deployed API catches up.
const REQUIRED_FLAGS = [
  "JUPR_ENABLE_TEAM_LEAGUES",
  "JUPR_ENABLE_NEXT_ADMIN_SHELL",
  "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
  "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
  "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
  "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
  "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
  "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
];

function operationsReady(health) {
  return health?.ok === true
    && health.club_website_settings_available === true
    && health.interclub_seasons_available === true
    && health.environment === "production"
    && health.fly_app_name === "juprleagues-api"
    && health.supabase_project_ref === "dnoockbwfenunhcibwfn"
    && health.production_business_write_policy === "enabled"
    && health.staging_write_wave === "none"
    && REQUIRED_FLAGS.every((flag) => health.feature_flags?.[flag] === true);
}

function weeklyReportReady(openapi) {
  const fields = openapi?.components?.schemas?.AdminWeeklyRecapGenerateRequest?.properties;
  const preferences = openapi?.paths?.["/admin/clubs/{club_id}/weekly-recap/preferences"];
  return fields?.preserve_edits?.type === "boolean"
    && fields.preserve_edits.default === true
    && Object.prototype.hasOwnProperty.call(fields, "edits_json")
    && Boolean(preferences?.get && preferences?.put)
    && Boolean(openapi?.components?.schemas?.AdminWeeklyRecapPublishRequest?.properties?.edits_json);
}

function leagueLiveReady(openapi) {
  const prefix = "/admin/clubs/{club_id}/league-manager/live";
  const gameEditsReady = ["AdminLeagueLiveRoundPublishRequest", "AdminLeagueLiveRoundPlanRequest"].every((name) => {
    const field = openapi?.components?.schemas?.[name]?.properties?.unplayed_games;
    return field?.type === "array" || field?.anyOf?.some((variant) => variant.type === "array") === true;
  });
  return gameEditsReady && [
    ["post", `${prefix}/round-robin/preview`],
    ["post", `${prefix}/players`],
    ["get", `${prefix}/player-operations/{operation_key}`],
    ["post", `${prefix}/player-operations/{operation_key}/reconcile`],
  ].every(([method, path]) => Boolean(openapi?.paths?.[path]?.[method]));
}

function roundRobinHubReady(openapi) {
  const actions = openapi?.components?.schemas?.HubCommandRequest?.properties?.action?.enum;
  const setup = openapi?.components?.schemas?.GeneratorStartRequest?.properties;
  return Array.isArray(actions)
    && ["set_registration", "review_checkin", "process_checkins", "edit_match_players"].every((action) => actions.includes(action))
    && Boolean(setup?.registration_hub_id)
    && setup?.participant_names?.type === "array"
    && !setup.participant_names.minItems
    && [
      ["get", "/public/round-robin-hubs/{hub_id}/registration"],
      ["post", "/public/round-robin-hubs/{hub_id}/checkins"],
    ].every(([method, path]) => Boolean(openapi?.paths?.[path]?.[method]));
}

async function waitForProductionApi({
  environment = process.env.VERCEL_ENV,
  request = fetch,
  pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  now = Date.now,
  timeoutMs = 15 * 60 * 1000,
} = {}) {
  if (environment !== "production") return;
  const deadline = now() + timeoutMs;
  console.log("Waiting for the production operations, weekly report, League Live, and RR Hub APIs before building the website.");
  while (now() < deadline) {
    try {
      const response = await request("https://api.juprleagues.com/health", {
        cache: "no-store", signal: AbortSignal.timeout(10000),
      });
      if (response.ok && operationsReady(await response.json())) {
        const contract = await request("https://api.juprleagues.com/openapi.json", {
          cache: "no-store", signal: AbortSignal.timeout(10000),
        });
        if (contract.ok) {
          const openapi = await contract.json();
          if (weeklyReportReady(openapi) && leagueLiveReady(openapi) && roundRobinHubReady(openapi)) {
            console.log("Production operations, weekly report, League Live, and RR Hub APIs are ready.");
            return;
          }
        }
      }
    } catch {
      // A deployment restart or timeout must leave the current website serving.
    }
    await pause(10000);
  }
  throw new Error("Production operations API is not ready. The current website remains active; retry after API verification.");
}

module.exports = { operationsReady, weeklyReportReady, leagueLiveReady, roundRobinHubReady, waitForProductionApi, REQUIRED_FLAGS };
if (require.main === module) {
  waitForProductionApi().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
