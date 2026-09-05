import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("leaderboards default to active overall standings with parity fields and stable snapshots", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/leaderboards`, { waitUntil: "domcontentloaded" });

  await expect(page.getByTestId("leaderboard-status-active")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("leaderboard-scope-overall")).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("columnheader", { name: "Gain" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Gap" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Qualification" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Badges" })).toBeVisible();
  await expect(page.getByTestId("leaderboard-highlight-card")).toHaveCount(4);

  const rows = page.getByTestId("leaderboard-row");
  expect(await rows.count(), "staging leaderboard fixture should contain active players").toBeGreaterThan(0);
  for (let index = 0; index < await rows.count(); index += 1) {
    await expect(rows.nth(index)).toHaveAttribute("data-status", "active");
  }

  const snapshotLink = rows.first().getByRole("link", { name: "player snapshot" });
  const href = await snapshotLink.getAttribute("href");
  expect(href).toContain("player=");
  expect(href).toContain("#player-snapshot");
  await snapshotLink.click();
  await expect(page).toHaveURL(/player=/);
  await expect(page.getByTestId("leaderboard-player-snapshot")).toBeVisible();
  await expect(page.getByTestId("leaderboard-player-snapshot")).toContainText(/Rank/);
  await expect(page.getByTestId("leaderboard-player-snapshot")).toContainText(/Gain/);
});

test("leaderboard search, league tabs, and pagination preserve deterministic links", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/leaderboards?per_page=1`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("leaderboard-row")).toHaveCount(1);
  await expect(page.getByTestId("leaderboard-pagination")).toBeVisible();

  const playerName = (await page.getByTestId("leaderboard-row").first().locator("td").nth(1).locator("strong").innerText()).trim();
  await page.getByRole("textbox", { name: "Find player" }).fill(playerName);
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page).toHaveURL(/q=/);
  await expect(page.getByTestId("leaderboard-player-snapshot")).toContainText(playerName);

  const scopeTabs = page.getByTestId("leaderboard-scope-tabs").getByRole("link");
  if ((await scopeTabs.count()) > 1) {
    await scopeTabs.nth(1).click();
    await expect(page).toHaveURL(/league=/);
    await expect(page.getByTestId("leaderboard-qualification-note")).toBeVisible();
  }
});

test("leaderboard has explicit filtered-empty and API-error states", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/leaderboards?q=__jupr_no_player_7a9f31__`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("leaderboard-filter-empty-state")).toContainText("No players match");

  await page.goto("/clubs/__jupr_missing_club_7a9f31__/leaderboards", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("leaderboard-error-state")).toContainText("Leaderboards are unavailable right now");
  await expect(page.getByTestId("leaderboard-error-state")).toContainText("Please try again shortly");
});
