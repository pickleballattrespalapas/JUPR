import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";


test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});


test("League Live advertises durable complete-round publish or fails closed", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto("/admin/league-manager/live", { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.getByRole("heading", { name: /league live round entry/i })).toBeVisible();
  await expect(page.locator("body")).toContainText(/durable all-match|complete round|Streamlit League Manager/i);
  await expect(page.locator("body")).toContainText(/reconcil|Match Log|Replay History|guarded off/i);
  await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});
