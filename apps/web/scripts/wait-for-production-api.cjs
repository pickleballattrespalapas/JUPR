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
    && ["set_registration", "review_checkin", "process_checkins", "edit_match_players",
      "transfer_player", "rejoin_player", "resolve_game",
      "set_match_lineup", "remove_players", "create_extra_game"].every((action) => actions.includes(action))
    && Boolean(setup?.registration_hub_id)
    && setup?.participant_names?.type === "array"
    && !setup.participant_names.minItems
    && [
      ["get", "/public/round-robin-hubs/{hub_id}/registration"],
      ["post", "/public/round-robin-hubs/{hub_id}/checkins"],
      ["patch", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submission/scores"],
    ].every(([method, path]) => Boolean(openapi?.paths?.[path]?.[method]));
}

function siteReliabilityReady(openapi) {
  const readsReady = [
    "/public/clubs/{slug}/site/shell",
    "/public/clubs/{slug}/site/logo",
    "/clubs/{club_slug}/players/{player_id}/share-metadata",
    "/clubs/{club_slug}/play-generators/sessions/{session_key}/share-metadata",
  ].every((path) => Boolean(openapi?.paths?.[path]?.get));
  const batchDeparturesReady = ["GeneratorRosterRequest", "PublicGeneratorRosterRequest"].every((name) => {
    const field = openapi?.components?.schemas?.[name]?.properties?.participant_ids;
    return field?.type === "array" && field.maxItems === 40;
  });
  return readsReady && batchDeparturesReady;
}

function startingRatingReviewReady(openapi) {
  const prefix = "/admin/clubs/{club_id}/starting-rating-review";
  const schemas = openapi?.components?.schemas;
  const preview = schemas?.StartingRatingReviewPreviewRequest?.properties;
  const confirm = schemas?.StartingRatingReviewConfirmRequest?.properties;
  const review = schemas?.StartingRatingReviewConfirmation?.properties;
  return ["preview", "confirm"].every((action) => Boolean(openapi?.paths?.[`${prefix}/${action}`]?.post))
    && preview?.player_ids?.type === "array"
    && preview?.requested_ratings?.type === "object"
    && [preview, confirm].every((fields) => ["doubles", "singles"].every((format) => fields?.match_format?.enum?.includes(format)))
    && confirm?.reviews?.type === "array"
    && confirm.reviews.items?.$ref === "#/components/schemas/StartingRatingReviewConfirmation"
    && review?.player_id?.type === "integer"
    && review?.rating_jupr?.type === "number"
    && review?.expected_state_fingerprint?.type === "string";
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
  console.log("Waiting for the production operations, weekly report, League Live, RR Hub, site reliability, and starting-rating review APIs before building the website.");
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
          if (weeklyReportReady(openapi) && leagueLiveReady(openapi) && roundRobinHubReady(openapi) && siteReliabilityReady(openapi) && startingRatingReviewReady(openapi)) {
            console.log("Production operations, weekly report, League Live, RR Hub, site reliability, and starting-rating review APIs are ready.");
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

module.exports = { operationsReady, weeklyReportReady, leagueLiveReady, roundRobinHubReady, siteReliabilityReady, startingRatingReviewReady, waitForProductionApi, REQUIRED_FLAGS };
if (require.main === module) {
  waitForProductionApi().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
