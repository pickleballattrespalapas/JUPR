import { expect, test } from "@playwright/test";
import {
  bootstrapStagingContext,
  clubSlug,
  expectHealthySurface
} from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("league results deep link and print view preserve every selector", async ({ page }) => {
  const path = `/clubs/${clubSlug}/league-results?section=weekly&week=1&weekly_min_games=1&print=1`;
  await expectHealthySurface(page, {
    name: "league results parity",
    path,
    expected: /league results/i
  });

  await expect(page).toHaveURL(/section=weekly/);
  await expect(page).toHaveURL(/week=1/);
  await expect(page).toHaveURL(/weekly_min_games=1/);
  await expect(page).toHaveURL(/print=1/);
  await expect(page.getByRole("heading", { name: /current standings/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /weekly results/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /player summary/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /recent matches/i })).toBeVisible();
  await expect(page.getByText(/season totals only/i)).toBeVisible();
  await expect(page.getByText(/selected week/i)).toBeVisible();
});
