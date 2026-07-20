import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();
const draftWeekStart = String(process.env.JUPR_COMMUNICATIONS_DRAFT_WEEK_START || "").trim();

test.describe("communications parity staging evidence", () => {
  test.skip(!adminToken, "STAGING_ADMIN_BEARER_TOKEN is required for authenticated communications evidence.");

  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
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

  test("loads the non-mutating communications workspace with guarded controls", async ({ page }) => {
    await page.goto("/admin/player-updates", { waitUntil: "domcontentloaded" });

    await expect(page.getByRole("heading", { name: "Admin session and delivery guard" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Preview or queue digests" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Active subscriptions and replacement history" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Outbox queue and immutable delivery history" })).toBeVisible();
    await expect(page.getByText("Supabase service role", { exact: false })).toBeVisible();
    await expect(page.getByText("uncertain delivery", { exact: false })).toBeVisible();
    await expect(page.getByRole("button", { name: "Send selected pending" })).toBeDisabled();
  });

  test("renders and can print a full unpublished recap preview", async ({ page }) => {
    test.skip(!draftWeekStart, "JUPR_COMMUNICATIONS_DRAFT_WEEK_START must name a disposable staging draft.");
    await page.goto(`/admin/weekly-recap?week_start=${encodeURIComponent(draftWeekStart)}`, { waitUntil: "domcontentloaded" });

    const preview = page.getByTestId("admin-weekly-recap-preview");
    await expect(preview).toBeVisible();
    await expect(preview).toContainText("Unpublished draft — operator preview only");
    await expect(preview).toContainText("Spotlight Reel");
    await expect(preview).toContainText("Around the Club");
    await expect(preview).toContainText("Looking Ahead");
    await expect(page.getByRole("button", { name: "Print / save unpublished preview" })).toBeVisible();
  });
});
