import { expect, test, type Page } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();
const tournamentId = "93000000-0000-4000-8000-000000000001";
const drawId = "da089489-8746-4c3f-96d2-7bb65fd6cc0e";
const selectedQuery = `tournament=${tournamentId}&tournament_name=Staging+Summer+Classic&name=Staging+Summer+Classic&draw=${drawId}`;
const pageDiagnostics = new WeakMap<Page, { consoleErrors: string[]; pageErrors: string[] }>();
const protectedWebBase = String(process.env.STAGING_WEB_BASE_URL || "").trim().replace(/\/$/, "");

function isProtectedPreviewPrefetchFallback(message: string): boolean {
  return Boolean(protectedWebBase)
    && message.startsWith(`Failed to fetch RSC payload for ${protectedWebBase}/`)
    && message.includes("Falling back to browser navigation. TypeError: Failed to fetch");
}

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ context, page }) => {
  await bootstrapStagingContext(context);
  expect(adminToken, "The protected staging admin session is required.").toBeTruthy();
  const diagnostics = { consoleErrors: [] as string[], pageErrors: [] as string[] };
  pageDiagnostics.set(page, diagnostics);
  page.on("console", (message) => {
    if (message.type() === "error") diagnostics.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  await context.addInitScript(
    ({ token, email }) => window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
      access_token: token,
      token_type: "bearer",
      user: { email }
    })),
    { token: adminToken, email: adminEmail }
  );
});

test.afterEach(async ({ page }) => {
  const diagnostics = pageDiagnostics.get(page);
  expect(diagnostics?.pageErrors || [], "The page raised browser errors.").toEqual([]);
  const unexpectedConsoleErrors = (diagnostics?.consoleErrors || []).filter(
    (message) => !isProtectedPreviewPrefetchFallback(message)
  );
  expect(unexpectedConsoleErrors, "The page emitted unexpected console errors.").toEqual([]);
  await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
});

test("the completed 11-7 command replays without changing tournament state", async ({ request }) => {
  const before = await request.get(
    `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/snapshot?draw_id=${drawId}`,
    { headers: { Authorization: `Bearer ${adminToken}` } }
  );
  expect(before.status()).toBe(200);
  const beforeSnapshot = await before.json();
  const beforeGame = beforeSnapshot.games.find((row: { id: string }) => row.id === "3d78b475-2f9a-49e5-b119-05c685a29fc4");
  expect([beforeGame?.score_a, beforeGame?.score_b]).toEqual([11, 7]);

  const replay = await request.post(
    `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/draws/${drawId}/commands`,
    {
      headers: { Authorization: `Bearer ${adminToken}` },
      data: {
        command: "save_score",
        expected_state_fingerprint: "a0ded5aefcaf8d101a26c7afb106f01bf5a33e05d535a2ff9d2b5864ba2418d7",
        idempotency_key: "c1143ac7-274e-40a8-a7bd-a5553c298695",
        confirmation_text: "SAVE SCORE",
        expected_draw_updated_at: "2026-08-15T19:04:59.416905+00:00",
        expected_game_updated_at: "2026-08-15T19:04:59.374604+00:00",
        game_id: "3d78b475-2f9a-49e5-b119-05c685a29fc4",
        score_a: 11,
        score_b: 7
      }
    }
  );
  expect(replay.status()).toBe(200);
  expect((await replay.json()).idempotent_replay).toBe(true);

  const after = await request.get(
    `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/snapshot?draw_id=${drawId}`,
    { headers: { Authorization: `Bearer ${adminToken}` } }
  );
  expect(after.status()).toBe(200);
  const afterSnapshot = await after.json();
  const afterGame = afterSnapshot.games.find((row: { id: string }) => row.id === "3d78b475-2f9a-49e5-b119-05c685a29fc4");
  expect([afterGame?.score_a, afterGame?.score_b]).toEqual([11, 7]);
  expect(afterSnapshot.operations.length).toBe(beforeSnapshot.operations.length);
});

test("Home reports the authoritative one-of-twenty-one tournament truth", async ({ page }) => {
  await page.goto(`/admin/tournaments/tournament?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Continue scoring" })).toBeVisible();
  await expect(page.getByText("1 of 21 games scored; 20 open.")).toBeVisible();
  await expect(page.getByText("Publish blockers")).toBeVisible();
  const scoringHref = await page.getByRole("link", { name: "Continue scoring" }).getAttribute("href");
  expect(scoringHref).toContain(`tournament=${tournamentId}`);
  expect(scoringHref).toContain(`draw=${drawId}`);
});

test("Preflight and check-in exposes attendance and blocker truth without saving", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/check-in?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: /Staging Summer Classic preflight and check-in/i })).toBeVisible();
  await expect(page.getByText("Expected").first()).toBeVisible();
  await expect(page.getByText("Checked in").first()).toBeVisible();
  await expect(page.getByText("Absent").first()).toBeVisible();
  await expect(page.getByText("Unresolved").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Player check-in" })).toBeVisible();
  await expect(page.getByText("Mateo Rivera").first()).toBeVisible();
  await expect(page.getByText(/staffing/i).first()).toBeVisible();
  const scoring = page.getByRole("link", { name: "Live scoring" });
  await expect(scoring).toHaveAttribute("href", new RegExp(`tournament=${tournamentId}.*draw=${drawId}`));
});

test("Open-game score review rejects a tie and shows a complete confirmation card", async ({ page, request }) => {
  const snapshotResponse = await request.get(
    `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/snapshot?draw_id=${drawId}`,
    { headers: { Authorization: `Bearer ${adminToken}` } }
  );
  expect(snapshotResponse.status()).toBe(200);
  const snapshot = await snapshotResponse.json();
  expect(snapshot.lifecycle?.counts?.games).toBe(21);
  expect(snapshot.lifecycle?.counts?.open_games).toBe(20);

  await page.goto(`/admin/tournament-live?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Enter score" }).first().click();
  await page.getByLabel("Team A score").fill("9");
  await page.getByLabel("Team B score").fill("9");
  await page.getByRole("button", { name: "Review score" }).click();
  await expect(page.getByText(/tied or invalid score cannot be reviewed or saved/i)).toBeVisible();
  await expect(page.getByRole("button", { name: "Confirm & save" })).toHaveCount(0);

  await page.getByLabel("Team A score").fill("11");
  await page.getByLabel("Team B score").fill("7");
  await page.getByRole("button", { name: "Review score" }).click();
  await expect(page.getByText("Proposed winner:")).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit score" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Confirm & save" })).toBeVisible();
});

test("Corrections review names both teams and shows before and after without saving", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/corrections?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Correct score" }).first().click();
  await expect(page.getByText(/Before correction:/)).toBeVisible();
  await expect(page.getByText("Mateo Rivera / Liam Chen").first()).toBeVisible();
  await expect(page.getByText("Caleb Nguyen / Diego Alvarez").first()).toBeVisible();
  await page.getByLabel("Team A score").fill("11");
  await page.getByLabel("Team B score").fill("8");
  await page.getByRole("button", { name: "Review score" }).click();
  await expect(page.getByText(/After correction:/)).toBeVisible();
  await expect(page.getByText("Proposed winner:")).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit score" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Confirm & save" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent operations and reconciliation" })).toBeVisible();
});

test("Corrections and recovery shows durable, human-readable evidence", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/corrections?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Recent operations and reconciliation" })).toBeVisible();
  await expect(page.getByText("Technical operation evidence").first()).toBeVisible();
  await expect(page.getByText(/Match Log corrections are for official published matches/)).toBeVisible();
  await expect(page.getByText("Mateo Rivera / Liam Chen").first()).toBeVisible();
  await expect(page.getByText("Caleb Nguyen / Diego Alvarez").first()).toBeVisible();
});

test("official publish remains blocked with runtime enabled", async ({ page, request }) => {
  const snapshotResponse = await request.get(
    `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${tournamentId}/snapshot?draw_id=${drawId}`,
    { headers: { Authorization: `Bearer ${adminToken}` } }
  );
  expect(snapshotResponse.status()).toBe(200);
  const snapshot = await snapshotResponse.json();
  expect(snapshot.lifecycle?.runtime_capability?.official_publish_enabled).toBe(true);
  expect(snapshot.lifecycle?.domain_readiness?.official_publish?.ready).toBe(false);
  expect(snapshot.lifecycle?.counts?.open_games).toBe(20);

  await page.goto(`/admin/tournaments/ops/publish?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Tournament readiness" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Runtime capability" })).toBeVisible();
  await expect(page.getByText(/20 game\(s\) without a finalized/i).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Publish official matches" })).toBeDisabled();

  await page.goto(`/admin/tournaments/publish/closeout?${selectedQuery}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Archive unavailable" })).toBeVisible();
  await expect(page.getByText("No archive write is available.")).toBeVisible();
  await expect(page.getByText("Payments, extras, and fulfillment")).toBeVisible();
});

for (const width of [1280, 1440]) {
  test(`focused operator routes have no page-level overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const path of [
      "/admin/tournaments/live-operations",
      "/admin/tournament-live",
      "/admin/tournaments/live-operations/corrections"
    ]) {
      await page.goto(`${path}?${selectedQuery}`, { waitUntil: "domcontentloaded" });
      await expect(page.locator("body")).toContainText("Staging Summer Classic");
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
      );
      expect(overflow, path).toBe(false);
      expect(await page.locator("[data-nextjs-dialog]").count(), `${path} error overlay`).toBe(0);
    }
  });
}
