import { test, expect } from "@playwright/test";

// Run only against a local preview with JUPR_INTERACTION_TEST_HARNESS=1.
test.skip(
  Boolean(String(process.env.STAGING_WEB_BASE_URL || "").trim()),
  "The synthetic navigation harness is local-only."
);
const root = "/test-tournament-navigation";
test.beforeEach(async ({ page }) => {
  await page.goto(root);
  await expect(page.getByText("Test controls ready", { exact: true })).toBeVisible();
});
test("one click shows feedback, preserves context, and newer navigation wins", async ({ page }) => {
  await page.getByRole("link", { name: "Slow page", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: /Opening/ }).first()).toBeVisible();
  await page.getByRole("link", { name: "Fast page", exact: true }).click();
  await expect(page.getByRole("heading", { name: "fast page opened" })).toBeVisible();
  await expect(page).toHaveURL(/\/fast\?tournament=fixture&draw=one&day=two$/);
  // The original slow reply must not send us back after it finishes.
  await page.waitForTimeout(5_500);
  await expect(page.getByRole("heading", { name: "fast page opened" })).toBeVisible();
  await expect(page.getByText("Opening page…", { exact: true })).toHaveCount(0);
});

test("a failed server read retries once and recovers on the same URL", async ({ page }) => {
  await page.getByLabel("Simulate a failed request").check();
  await page.getByRole("link", { name: "Failure page", exact: true }).click();
  await expect(page.getByRole("heading", { name: "This page couldn’t load" })).toBeVisible();
  await page.getByLabel("Simulate a failed request").uncheck();
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByRole("heading", { name: "failure page opened" })).toBeVisible();
  await expect(page).toHaveURL(/\/failure\?tournament=fixture&draw=one&day=two$/);
});
