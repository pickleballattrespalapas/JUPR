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

const clubId = String(process.env.JUPR_SMOKE_CLUB_ID || "tres_palapas").trim();

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
  { name: "admin login", path: "/admin/login", expected: /admin login/i },
  { name: "match log", path: "/admin/match-log", expected: /match log/i },
  { name: "match log edit", path: "/admin/match-log/edit", expected: /edit a match/i },
  { name: "match log bulk", path: "/admin/match-log/bulk", expected: /bulk edit matches/i },
  { name: "match log duplicates", path: "/admin/match-log/duplicates", expected: /resolve duplicates/i },
  { name: "match log exclude", path: "/admin/match-log/exclude", expected: /exclude rated matches/i },
  { name: "match log social", path: "/admin/match-log/social", expected: /social match tools/i },
  { name: "match log replay", path: "/admin/match-log/replay", expected: /replay ratings/i },
  { name: "replay history", path: "/admin/replay-history", expected: /replay history/i },
  { name: "match uploader", path: "/admin/match-uploader", expected: /match uploader/i },
  { name: "player editor", path: "/admin/players", expected: /player editor/i },
  { name: "player updates", path: "/admin/player-updates", expected: /player update/i },
  { name: "verified updates", path: "/admin/player-updates/verified-requests", expected: /verified update/i },
  { name: "support requests", path: "/admin/support-requests", expected: /request queue/i },
  { name: "league manager", path: "/admin/league-manager", expected: /league manager/i },
  { name: "league printout", path: "/admin/league-manager/print", expected: /league night printout/i },
  { name: "top players printable", path: "/admin/top-players-printable", expected: /top active players/i },
  { name: "league awards", path: "/admin/league-manager/awards", expected: /league awards/i },
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

type ReadOnlyFeatureArea = {
  name: string;
  surfaces: Surface[];
  apiChecks: Array<{
    name: string;
    path: string;
    expectedStatuses: number[];
    disabledStatus?: number;
    disabledExpected?: RegExp;
  }>;
};

const readOnlyFeatureAreas: ReadOnlyFeatureArea[] = [
  {
    name: "tournament commerce",
    surfaces: [
      {
        name: "tournament commerce admin",
        path: "/admin/tournaments/commerce",
        expected: /extras, bundles, and fulfillment/i
      }
    ],
    apiChecks: [
      {
        name: "tournament commerce status",
        path: `/admin/clubs/${clubId}/tournaments/commerce/status`,
        expectedStatuses: [200, 401],
        disabledStatus: 200,
        disabledExpected: /"available":false/
      }
    ]
  },
  {
    name: "combined-rating and four-player team tournaments",
    surfaces: [
      {
        name: "team tournament admin",
        path: "/admin/tournaments/team-competition",
        expected: /combined ratings and four-player teams/i
      },
      {
        name: "published team tournament results",
        path: `/clubs/${clubSlug}/tournament-team-results`,
        expected: /follow four-player team standings/i
      }
    ],
    apiChecks: [
      {
        name: "team tournament admin status",
        path: `/admin/clubs/${clubId}/tournaments/team-competition/status`,
        expectedStatuses: [401, 403],
        disabledStatus: 403,
        disabledExpected: /disabled/i
      },
      {
        name: "published team tournament results",
        path: `/clubs/${clubSlug}/tournament-team-results`,
        expectedStatuses: [200, 404],
        disabledStatus: 404,
        disabledExpected: /feature not found/i
      }
    ]
  },
  {
    name: "team leagues and awards",
    surfaces: [
      {
        name: "team league admin",
        path: "/admin/league-manager/teams",
        expected: /team leagues/i
      },
      {
        name: "league awards admin",
        path: "/admin/league-manager/awards",
        expected: /league awards/i
      },
      {
        name: "published team leagues",
        path: `/clubs/${clubSlug}/team-leagues`,
        expected: /team leagues/i
      }
    ],
    apiChecks: [
      {
        name: "team league admin list",
        path: `/admin/clubs/${clubId}/league-manager/team-leagues`,
        expectedStatuses: [401, 403],
        disabledStatus: 403,
        disabledExpected: /disabled/i
      },
      {
        name: "published team leagues",
        path: `/clubs/${clubSlug}/team-leagues`,
        expectedStatuses: [200, 403],
        disabledStatus: 403,
        disabledExpected: /disabled/i
      }
    ]
  }
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

for (const featureArea of readOnlyFeatureAreas) {
  test(`read-only feature readiness: ${featureArea.name}`, async ({ page }) => {
    test.setTimeout(120_000);

    for (const surface of featureArea.surfaces) {
      await expectHealthySurface(page, surface);
    }

    for (const apiCheck of featureArea.apiChecks) {
      const response = await page.request.get(
        `${expectedApiOrigin}${apiCheck.path}`,
        { failOnStatusCode: false }
      );
      expect(
        apiCheck.expectedStatuses,
        `${apiCheck.name} returned an unexpected status`
      ).toContain(response.status());
      expect(response.headers()["content-type"] || "").toContain(
        "application/json"
      );

      const payloadText = await response.text();
      expect(() => JSON.parse(payloadText)).not.toThrow();
      if (
        apiCheck.disabledStatus === response.status() &&
        apiCheck.disabledExpected
      ) {
        expect(payloadText).toMatch(apiCheck.disabledExpected);
      }
    }
  });
}

for (const surface of publicSurfaces) {
  test(`public surface: ${surface.name}`, async ({ page }) => {
    await expectHealthySurface(page, surface);
  });
}

test("tournament roster filters and public deep links remain navigable", async ({ page }) => {
  const response = await page.goto(`/clubs/${clubSlug}/tournament-roster`, { waitUntil: "domcontentloaded" });
  expect(response).not.toBeNull();
  expect(response?.ok()).toBeTruthy();
  await expect(page.getByRole("heading", { name: /tournament roster|open/i }).first()).toBeVisible();

  const filterForm = page.getByRole("form", { name: "Tournament roster filters" });
  const rowLink = page.getByRole("link", { name: /roster entry/i }).first();
  const unavailable = page.getByRole("alert").filter({ hasText: /temporarily unavailable|not configured/i });
  const empty = page.getByRole("heading", { name: /no roster entries yet|no matching roster entries/i });

  if (await filterForm.count()) {
    await expect(filterForm.getByLabel("Day")).toBeVisible();
    await expect(filterForm.getByLabel("Event")).toBeVisible();
    await expect(filterForm.getByLabel("Division")).toBeVisible();
    await expect(filterForm.getByLabel("Status")).toBeVisible();
    await expect(filterForm.getByRole("button", { name: "Apply filters" })).toBeVisible();
  }

  if (await rowLink.count()) {
    const href = await rowLink.getAttribute("href");
    expect(href).toBeTruthy();
    const target = new URL(String(href), page.url());
    expect(target.hash).toMatch(/^#entry-/);
    await page.goto(target.toString(), { waitUntil: "domcontentloaded" });
    await expect(page.locator(target.hash)).toBeVisible();
  } else {
    await expect(empty.or(unavailable).or(page.getByText(/No tournament roster is currently published|No published tournament roster was found/i)).first()).toBeVisible();
  }

  await expect(page.getByText("Players looking for partners", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "View on partner board" })).toHaveCount(0);
});

test("badge codex: authoritative buckets, filters, anchors, and trophy room", async ({ page }) => {
  const response = await page.goto(`/clubs/${clubSlug}/badge-codex?bucket=all`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.getByRole("heading", { name: /badge codex/i })).toBeVisible();
  await expect(page.locator("[data-badge-bucket]")).toHaveCount(4);
  await expect(page.getByText(/Complete definitions/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent trophy room" })).toBeVisible();

  const firstBadge = page.locator("[data-badge-id]").first();
  await expect(firstBadge).toBeVisible();
  const badgeId = await firstBadge.getAttribute("data-badge-id");
  expect(badgeId).toBeTruthy();
  const directHref = await firstBadge.getByRole("link", { name: "Link directly to this badge" }).getAttribute("href");
  expect(directHref).toContain(`badge=${badgeId}`);
  expect(directHref).toContain(`#badge-${badgeId}`);
  await page.goto(String(directHref), { waitUntil: "domcontentloaded" });
  await expect(page.locator(`[data-badge-id="${badgeId}"]`)).toBeVisible();
});

test("challenge ladder: Python eligibility, deep links, rulebook, and status legend", async ({ page }) => {
  let response = await page.goto(`/clubs/${clubSlug}/challenge-ladder?section=rules`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.locator('[data-rulebook-authority="python"]')).toBeVisible();
  await expect(page.getByRole("heading", { name: "Swing Partner Swap format" })).toBeVisible();
  await expect(page.getByText(/exact tie favors the defender/i)).toBeVisible();
  await expect(page.locator("[data-ladder-status]")).toHaveCount(8);

  response = await page.goto(`/clubs/${clubSlug}/challenge-ladder?tier=PREM#tier-PREM`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.locator("#tier-PREM")).toBeVisible();
  response = await page.goto(`/clubs/${clubSlug}/challenge-ladder`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  const playerLink = page.locator('a[href*="player="][href*="#ladder-player-"]').first();
  await expect(playerLink).toBeVisible();
  await playerLink.click();
  await expect(page.locator('[data-python-eligibility="python"]')).toBeVisible();
  await expect(page.getByText(/Python ladder policy/i)).toBeVisible();

  response = await page.goto(`/clubs/${clubSlug}/challenge-ladder?section=challenges&status=Recently%20Completed`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  const completed = page.locator("#ladder-challenge-990001");
  await expect(completed).toBeVisible();
  await expect(completed).toContainText("Completed");
  await expect(completed).toContainText("2026-07-05 19:00 UTC");
  await expect(completed).toContainText("At challenge: #2");
  await expect(completed).toContainText("Current: #2");
  await expect(completed).toContainText("Current JUPR: 3.053");
  const challengerCard = completed.getByText("Challenger", { exact: true }).locator("..");
  await expect(challengerCard.getByRole("link", { name: "Devon Dink" })).toBeVisible();
  const winnerRow = completed.locator("p").filter({ hasText: /^Winner:/ });
  await expect(winnerRow.getByRole("link", { name: "Devon Dink" })).toBeVisible();
  const playedResult = completed.getByRole("region", { name: "Played challenge result" });
  const challengerTeam = playedResult.locator("p").filter({ hasText: "Challenger team:" });
  await expect(challengerTeam.getByRole("link", { name: "Devon Dink" })).toBeVisible();
  await expect(completed.locator('[data-result-details="available"][data-result-completeness="partial"]')).toBeVisible();
  await expect(completed).toContainText("Verified legacy Match A: 22–17");
  await expect(completed).toContainText("Games: 11–8, 11–9");
  await expect(completed.getByRole("link", { name: "Blake Baseline" })).toBeVisible();
  await expect(completed).toContainText("1 of 2 verified legacy match records");
  await expect(completed).toContainText(/partner assignment that conflicts/i);
});

for (const surface of adminSurfaces) {
  test(`admin shell: ${surface.name}`, async ({ page }) => {
    await expectHealthySurface(page, surface);
  });
}

test("admin operations cockpit is hidden until staff authentication", async ({ page }) => {
  const response = await page.goto("/admin", { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /open admin login/i })).toBeVisible();
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Workflow flags", { exact: true })).toHaveCount(0);
  await expect(page.getByText("API service role", { exact: true })).toHaveCount(0);

  const apiResponse = await page.request.get(
    `${expectedApiOrigin}/admin/operations/status`,
    { failOnStatusCode: false }
  );
  expect(apiResponse.status()).toBe(401);
});
