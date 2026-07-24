import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();

test.describe("Match Log durability staging smoke", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
    test.skip(!adminToken, "STAGING_ADMIN_BEARER_TOKEN is required for authenticated durability evidence.");
    await context.addInitScript(
      ({ token, email }) => {
        window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
          access_token: token,
          token_type: "bearer",
          user: { email }
        }));
      },
      { token: adminToken, email: adminEmail }
    );
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
    await expect(page.getByRole("heading", { name: "Next replay is disabled" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Run replay" })).toHaveCount(0);
  });
});
