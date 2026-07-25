import { expect, test } from "@playwright/test";
import {
  bootstrapStagingContext,
  clubSlug,
  expectHealthySurface
} from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("league results overall view presents one authoritative season record", async ({ page }) => {
  await expectHealthySurface(page, {
    name: "league results overall",
    path: `/clubs/${clubSlug}/league-results`,
    expected: /league results/i
  });

  await expect(page.getByRole("heading", { name: /current standings/i })).toBeVisible();
  await expect(page.getByText(/official season record/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: /season highlights/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /season cumulative performance/i })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Print view", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Print / save PDF" })).toBeVisible();
});

test("league results deep link preserves its section and prints the current view", async ({ page }) => {
  const path = `/clubs/${clubSlug}/league-results?section=weekly&week=1&weekly_min_games=1`;
  await expectHealthySurface(page, {
    name: "league results parity",
    path,
    expected: /league results/i
  });

  await expect(page).toHaveURL(/section=weekly/);
  await expect(page).toHaveURL(/week=1/);
  await expect(page).toHaveURL(/weekly_min_games=1/);
  await expect(page.getByRole("heading", { name: /weekly results/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /current standings/i })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: /player summary/i })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: /season cumulative performance/i })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Print view", exact: true })).toHaveCount(0);
  await expect(page.getByText(/selected week/i)).toBeVisible();

  await page.evaluate(() => {
    window.print = () => window.sessionStorage.setItem("league-results-print", "called");
  });
  await page.getByRole("button", { name: "Print / save PDF" }).click();
  await expect
    .poll(() => page.evaluate(() => window.sessionStorage.getItem("league-results-print")))
    .toBe("called");

  await page.emulateMedia({ media: "print" });
  await expect(page.getByRole("heading", { name: /weekly results/i })).toBeVisible();
  await expect(page.getByRole("button", { name: "Print / save PDF" })).toBeHidden();
});

test("league results player deep link retains player detail coverage", async ({ page }) => {
  await expectHealthySurface(page, {
    name: "league results player",
    path: `/clubs/${clubSlug}/league-results?section=player`,
    expected: /league results/i
  });

  await expect(page.getByRole("heading", { name: /player summary/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /recent matches/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /current standings/i })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: /weekly results/i })).toHaveCount(0);
});
