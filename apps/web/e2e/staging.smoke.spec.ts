import { expect, test } from "@playwright/test";
import {
  bootstrapStagingContext,
  clubSlug,
  expectAuthIsolation,
  expectHealthySurface,
  expectedApiOrigin,
  expectedAuthOrigin,
  type Surface
} from "./support/staging";

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

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

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
