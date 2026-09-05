import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("player directory defaults active and keeps search plus stable links deterministic", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/players`, { waitUntil: "domcontentloaded" });

  await expect(page.getByTestId("players-status-active")).toHaveAttribute("aria-current", "page");
  const search = page.getByRole("searchbox", { name: "Find player" });
  await expect(search).toBeVisible();
  const rows = page.getByTestId("players-row");
  expect(await rows.count(), "staging should contain an active public player fixture").toBeGreaterThan(0);
  for (let index = 0; index < await rows.count(); index += 1) {
    await expect(rows.nth(index)).toHaveAttribute("data-status", "active");
  }

  const firstName = (await rows.first().locator("td").first().innerText()).trim();
  await search.fill(firstName);
  await page.getByTestId("players-search-form").getByRole("button", { name: "Search" }).click();
  await expect(page).toHaveURL(/q=/);
  await expect(rows.first(), "search should retain the selected public display name").toContainText(firstName);

  const stableRowLink = page.getByRole("link", { name: `Share ${firstName}` });
  const stableHref = await stableRowLink.getAttribute("href");
  expect(stableHref).toContain("player=");
  expect(stableHref).toContain("#player-");

  await page.getByRole("link", { name: `Open ${firstName} profile` }).click();
  await expect(page.getByTestId("player-profile")).toBeVisible();
  await expect(page.getByTestId("player-public-identity")).toContainText(/approved public name/i);
});

test("player profile keeps positions, trophies, badges, relationships, and history in player scope", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/players`, { waitUntil: "domcontentloaded" });
  const profileLink = page.getByTestId("players-row").first().locator("td").first().getByRole("link");
  const profileHref = await profileLink.getAttribute("href");
  expect(profileHref).toMatch(/\/players\/[^/?#]+$/);
  await profileLink.click();

  await expect(page.getByTestId("player-summary-cards")).toContainText("Doubles / overall");
  await expect(page.getByTestId("player-summary-cards")).toContainText("Singles");
  await expect(page.getByTestId("player-overview")).toBeVisible();

  await page.getByTestId("player-section-ratings").click();
  await expect(page).toHaveURL(/section=ratings/);
  await expect(page.getByTestId("player-ratings")).toBeVisible();
  await expect(page.getByTestId("player-format-row")).toHaveCount(2);

  await page.getByTestId("player-section-positions").click();
  await expect(page).toHaveURL(/\/players\/[^/?#]+\?section=positions/);
  await expect(page.getByTestId("player-league-positions")).toBeVisible();
  await expect(page.getByRole("link", { name: "Leaderboard snapshot" })).toHaveCount(0);

  await page.getByTestId("player-section-trophies").click();
  await expect(page.getByTestId("player-trophies")).toBeVisible();
  await expect(page.getByTestId("player-trophies")).toContainText(/major honors|tournament podium/i);

  await page.getByTestId("player-section-badges").click();
  await expect(page.getByTestId("player-badges")).toBeVisible();
  await expect(page.getByRole("link", { name: "Badge codex" })).toHaveCount(0);

  await page.getByTestId("player-section-social").click();
  await expect(page.getByTestId("player-best-partner")).toBeVisible();
  await expect(page.getByTestId("player-rival")).toBeVisible();
  await expect(page.getByTestId("player-social")).toBeVisible();

  await page.getByTestId("player-section-matches").click();
  await expect(page.getByTestId("player-match-history")).toBeVisible();
  const publicIdentity = page.getByTestId("player-public-identity");
  await expect(publicIdentity).toContainText(/player updates (are on|are available)|player update request is pending/i);
  const requestUpdates = page.getByRole("link", { name: "Request player updates" });
  if (await requestUpdates.count()) await expect(requestUpdates).toBeVisible();
  else await expect(publicIdentity).toContainText(/club staff will review this request|preferences link in a player update email/i);
  await expect(page.getByRole("link", { name: "Request an alias or privacy review" })).toBeVisible();

  const trend = page.getByTestId("player-rating-trend");
  const emptyTrend = page.getByTestId("player-rating-trend-empty");
  expect((await trend.count()) + (await emptyTrend.count())).toBe(1);

  await page.getByTestId("player-history-all").click();
  await expect(page).toHaveURL(/section=matches/);
  await expect(page).toHaveURL(/history=all/);
  await expect(page.getByTestId("player-history-all")).toHaveAttribute("aria-current", "page");
  const matchRows = page.getByTestId("player-match-row");
  for (let index = 0; index < await matchRows.count(); index += 1) {
    await expect(matchRows.nth(index).locator("td").nth(1)).toContainText(/Singles|Doubles/);
  }
});

test("player directory and profile expose explicit empty and error states without request interception", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/players?q=__jupr_no_player_8b7e42__`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("players-filter-empty-state")).toContainText("No players match");

  await page.goto(`/clubs/${clubSlug}/players/__jupr_missing_player_8b7e42__`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("player-profile-error-state")).toContainText("We couldn’t load this player profile");
});
