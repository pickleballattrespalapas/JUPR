import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

test.describe("Match Log durability staging smoke", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
  });

  test("exposes durable edit and replay evidence without writing", async ({ page }) => {
    await page.goto("/admin/match-log");
    await expect(page.getByRole("heading", { name: "Match Log correction cockpit" })).toBeVisible();

    const filters = page.getByTestId("match-log-filters");
    if (await filters.count()) {
      await expect(filters.getByLabel("League")).toHaveJSProperty("tagName", "SELECT");
      await expect(filters.getByLabel("Week tag")).toHaveJSProperty("tagName", "SELECT");
      await expect(filters.getByRole("option", { name: "All leagues" })).toHaveCount(1);
      await expect(filters.getByRole("option", { name: "All weeks" })).toHaveCount(1);
      await expect(filters.getByRole("link", { name: "Clear filters" })).toHaveAttribute("href", "/admin/match-log");

      const resultsBeforeMutationTools = await page.evaluate(() => {
        const results = document.querySelector('[data-testid="match-log-results"]');
        const mutationHeading = Array.from(document.querySelectorAll("h2")).find((heading) => heading.textContent === "Duplicate scan");
        return Boolean(results && mutationHeading && (results.compareDocumentPosition(mutationHeading) & Node.DOCUMENT_POSITION_FOLLOWING));
      });
      expect(resultsBeforeMutationTools).toBe(true);
    }

    const editor = page.getByRole("heading", { name: "Apply audited Match Log changes" });
    if (await editor.count()) {
      await expect(editor).toBeVisible();
      await expect(page.getByRole("heading", { name: "Guided match editor" })).toBeVisible();
      await expect(page.getByTestId("match-log-bulk-editor")).toBeVisible();
      await expect(page.getByText("Nothing is written until the staged operation is confirmed below.")).toBeVisible();
    } else {
      await expect(page.getByRole("heading", { name: "Apply flow is disabled" })).toBeVisible();
    }
  });

  test("shows tracked replay status without starting a replay", async ({ page }) => {
    await page.goto("/admin/replay-history");
    await expect(page.getByRole("heading", { name: "Replay History" })).toBeVisible();
    await expect(page.getByText(/durable replay_jobs record/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /Run Replay History/i })).toBeDisabled();
  });
});
