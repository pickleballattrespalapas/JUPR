// Keep the current website serving while the separately deployed API catches up.
const REQUIRED_FLAGS = [
  "JUPR_ENABLE_TEAM_LEAGUES",
  "JUPR_ENABLE_NEXT_ADMIN_SHELL",
  "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
  "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
  "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
];

function operationsReady(health) {
  return health?.ok === true
    && health.club_website_settings_available === true
    && health.environment === "production"
    && health.fly_app_name === "juprleagues-api"
    && health.supabase_project_ref === "dnoockbwfenunhcibwfn"
    && health.production_business_write_policy === "enabled"
    && health.staging_write_wave === "none"
    && REQUIRED_FLAGS.every((flag) => health.feature_flags?.[flag] === true);
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
  console.log("Waiting for the production club settings API before building the website.");
  while (now() < deadline) {
    try {
      const response = await request("https://api.juprleagues.com/health", {
        cache: "no-store", signal: AbortSignal.timeout(10000),
      });
      if (response.ok && operationsReady(await response.json())) {
        console.log("Production operations API is ready.");
        return;
      }
    } catch {
      // A deployment restart or timeout must leave the current website serving.
    }
    await pause(10000);
  }
  throw new Error("Production operations API is not ready. The current website remains active; retry after API verification.");
}

module.exports = { operationsReady, waitForProductionApi, REQUIRED_FLAGS };
if (require.main === module) {
  waitForProductionApi().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
