import {
  expect,
  test,
  type APIResponse,
  type BrowserContext,
  type Page
} from "@playwright/test";

const clubSlug = String(process.env.JUPR_SMOKE_CLUB_SLUG || "tres-palapas").trim();
const expectAuthIsolation = /^(1|true|yes|on)$/i.test(
  String(process.env.JUPR_EXPECT_PREVIEW_AUTH_ISOLATION || "")
);
const expectedApiOrigin = String(
  process.env.JUPR_EXPECTED_STAGING_API_ORIGIN || "https://juprleagues-api-staging.fly.dev"
).trim().replace(/\/$/, "");
const expectedAuthOrigin = String(process.env.JUPR_EXPECTED_STAGING_AUTH_ORIGIN || "")
  .trim()
  .replace(/\/$/, "");
const expectedStagingWebOrigin =
  "https://jupr-git-staging-pickleballattrespalapas1.vercel.app";
const remoteBaseUrl = String(process.env.STAGING_WEB_BASE_URL || "").trim().replace(/\/$/, "");
const bypassSecret = String(process.env.VERCEL_AUTOMATION_BYPASS_SECRET || "").trim();
const vercelBypassOrigin = (() => {
  if (!remoteBaseUrl || !bypassSecret) return "";
  if (remoteBaseUrl !== expectedStagingWebOrigin) {
    throw new Error("Refusing to send Vercel bypass credentials to a non-staging web origin.");
  }
  return expectedStagingWebOrigin;
})();

type Surface = {
  name: string;
  path: string;
  expected: RegExp;
};

const publicSurfaces: Surface[] = [
  { name: "home", path: "/", expected: /Pickleball Club Sandwich/i },
  { name: "site map", path: "/site-map", expected: /site map/i },
  { name: "ratings explainer", path: "/how-ratings-work", expected: /ratings/i },
  { name: "FAQ", path: "/faq", expected: /frequently asked|FAQ/i },
  { name: "support", path: "/support", expected: /support/i },
  { name: "club home", path: `/clubs/${clubSlug}`, expected: /Tres Palapas|club/i },
  { name: "leaderboards", path: `/clubs/${clubSlug}/leaderboards`, expected: /leaderboard/i },
  { name: "league results", path: `/clubs/${clubSlug}/league-results`, expected: /league results/i },
  { name: "players", path: `/clubs/${clubSlug}/players`, expected: /players/i },
  { name: "matches", path: `/clubs/${clubSlug}/matches`, expected: /matches/i },
  { name: "match explorer", path: `/clubs/${clubSlug}/match-explorer`, expected: /match explorer/i },
  { name: "badge codex", path: `/clubs/${clubSlug}/badge-codex`, expected: /badge/i },
  { name: "weekly recap", path: `/clubs/${clubSlug}/weekly-recap`, expected: /weekly recap/i },
  { name: "challenge ladder", path: `/clubs/${clubSlug}/challenge-ladder`, expected: /challenge ladder/i },
  { name: "tournament registration", path: `/clubs/${clubSlug}/tournament-registration`, expected: /registration/i },
  { name: "tournament roster", path: `/clubs/${clubSlug}/tournament-roster`, expected: /roster/i },
  { name: "tournament partner board", path: `/clubs/${clubSlug}/tournament-partner-board`, expected: /partner/i },
  { name: "live events", path: `/clubs/${clubSlug}/live`, expected: /live/i }
];

const adminSurfaces: Surface[] = [
  { name: "operations cockpit", path: "/admin", expected: /operations cockpit/i },
  { name: "admin login", path: "/admin/login", expected: /admin login/i },
  { name: "match log", path: "/admin/match-log", expected: /match log/i },
  { name: "replay history", path: "/admin/replay-history", expected: /replay history/i },
  { name: "match uploader", path: "/admin/match-uploader", expected: /match uploader/i },
  { name: "player editor", path: "/admin/players", expected: /player editor/i },
  { name: "player updates", path: "/admin/player-updates", expected: /player update/i },
  { name: "verified updates", path: "/admin/player-updates/verified-requests", expected: /verified update/i },
  { name: "support requests", path: "/admin/support-requests", expected: /request queue/i },
  { name: "league manager", path: "/admin/league-manager", expected: /league manager/i },
  { name: "league live", path: "/admin/league-manager/live", expected: /league live/i },
  { name: "tournament setup", path: "/admin/tournament-setup", expected: /tournament setup/i },
  { name: "tournament admin", path: "/admin/tournaments", expected: /tournament registration management/i },
  { name: "tournament operations", path: "/admin/tournaments/ops", expected: /tournament operations/i },
  { name: "tournament live", path: "/admin/tournament-live", expected: /tournament live/i },
  { name: "weekly recap admin", path: "/admin/weekly-recap", expected: /weekly recap admin/i },
  { name: "badge diagnostics", path: "/admin/badges", expected: /badge debug/i },
  { name: "moneyball", path: "/admin/moneyball", expected: /moneyball/i },
  { name: "JUPR live admin", path: "/admin/jupr-live", expected: /JUPR live admin/i },
  { name: "challenge ladder admin", path: "/admin/challenge-ladder", expected: /challenge ladder admin/i },
  { name: "match canonical audit", path: "/admin/match-canonical-audit", expected: /match canonical audit/i },
  { name: "admin tools", path: "/admin/tools", expected: /admin tools/i }
];

async function installVercelBypassCookie(context: BrowserContext): Promise<void> {
  const bootstrapUrl = `${vercelBypassOrigin}/api/environment`;
  let bootstrap: APIResponse;
  try {
    bootstrap = await context.request.get(bootstrapUrl, {
      headers: {
        "x-vercel-protection-bypass": bypassSecret,
        "x-vercel-set-bypass-cookie": "true"
      },
      maxRedirects: 0,
      failOnStatusCode: false
    });
  } catch {
    throw new Error("Unable to establish the Vercel automation bypass cookie.");
  }

  try {
    expect(
      bootstrap.status(),
      "Vercel bypass-cookie bootstrap did not redirect"
    ).toBeGreaterThanOrEqual(300);
    expect(
      bootstrap.status(),
      "Vercel bypass-cookie bootstrap did not redirect"
    ).toBeLessThan(400);
    expect(
      bootstrap.headersArray().some(({ name }) => name.toLowerCase() === "set-cookie"),
      "Vercel bypass-cookie bootstrap did not issue a cookie"
    ).toBeTruthy();
  } finally {
    await bootstrap.dispose().catch(() => {});
  }

  let verification: APIResponse;
  try {
    verification = await context.request.get(bootstrapUrl, {
      maxRedirects: 0,
      failOnStatusCode: false
    });
  } catch {
    throw new Error("Unable to verify the Vercel automation bypass cookie.");
  }

  try {
    expect(verification.status(), "Vercel bypass cookie was not accepted").toBe(200);
    expect(verification.headers()["content-type"] || "").toContain("application/json");
  } finally {
    await verification.dispose().catch(() => {});
  }
}

test.beforeEach(async ({ context }) => {
  if (!vercelBypassOrigin || !bypassSecret) return;
  await installVercelBypassCookie(context);
});

async function expectHealthySurface(page: Page, surface: Surface): Promise<void> {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto(surface.path, { waitUntil: "domcontentloaded" });
  expect(response, `${surface.name} did not return a document response`).not.toBeNull();
  expect(response?.status(), `${surface.name} returned an error status`).toBeLessThan(400);
  await expect(page.locator("body")).toContainText(surface.expected);
  await expect(page.locator("body")).not.toHaveText("");
  await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
  expect(pageErrors, `${surface.name} raised browser page errors`).toEqual([]);
  expect(consoleErrors, `${surface.name} emitted console errors`).toEqual([]);
}

test("preview environment is isolated from production", async ({ page }) => {
  const response = await page.goto("/api/environment", { waitUntil: "domcontentloaded" });
  expect(response).not.toBeNull();
  if (!response) return;
  expect(response.ok()).toBeTruthy();
  const environment = await response.json();

  expect(environment).toMatchObject({
    environment: "staging",
    vercel_environment: "preview",
    api_origin: expectedApiOrigin,
    score_entry_visible: true,
    preview_isolation_configured: true,
    preview_isolation_active: true
  });
  expect(environment.api_origin).not.toBe("https://api.juprleagues.com");

  if (expectAuthIsolation) {
    expect(
      expectedAuthOrigin,
      "JUPR_EXPECTED_STAGING_AUTH_ORIGIN must identify the isolated staging Supabase project"
    ).not.toBe("");
    expect(environment.preview_auth_isolation_configured).toBe(true);
    expect(environment.preview_auth_isolation_active).toBe(true);
    expect(environment.auth_origin).toBe(expectedAuthOrigin);
  }
});

for (const surface of publicSurfaces) {
  test(`public surface: ${surface.name}`, async ({ page }) => {
    await expectHealthySurface(page, surface);
  });
}

for (const surface of adminSurfaces) {
  test(`admin shell: ${surface.name}`, async ({ page }) => {
    await expectHealthySurface(page, surface);
  });
}
