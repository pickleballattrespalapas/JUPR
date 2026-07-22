import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const adminAccessToken = String(process.env.JUPR_STAGING_ADMIN_ACCESS_TOKEN || "").trim();
const adminEmail = String(process.env.JUPR_STAGING_ADMIN_EMAIL || "staging-admin").trim();

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("league export routes include browser-print evidence", async ({ page }) => {
  for (const route of ["/admin/league-manager/print", "/admin/top-players-printable"]) {
    const response = await page.goto(route, { waitUntil: "domcontentloaded" });
    expect(response?.status()).toBeLessThan(400);
    const styles = await page.locator("style").allTextContents();
    expect(styles.join("\n")).toContain("@media print");
    expect(styles.join("\n")).toContain("@page");
  }
});

test("authenticated league exports render Python-authoritative leaders", async ({ context, page }) => {
  test.skip(!adminAccessToken, "Set JUPR_STAGING_ADMIN_ACCESS_TOKEN for authenticated read-only league export evidence.");
  await context.addInitScript(
    ({ token, email }) => {
      window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: token, user: { email } }));
    },
    { token: adminAccessToken, email: adminEmail }
  );

  await page.goto("/admin/top-players-printable", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: /load previous-month top 50/i }).click();
  await expect(page.locator("body")).toContainText(/previous calendar month|No eligible active players/i);
  await expect(page.getByRole("button", { name: /print or save pdf/i })).toBeEnabled();

  await page.goto("/admin/league-manager/print", { waitUntil: "domcontentloaded" });
  const leagueSelect = page.getByLabel("League");
  await expect(leagueSelect).not.toHaveValue("");
  await expect(page.getByRole("button", { name: /refresh leagues/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /reload printout/i })).toBeEnabled();
  await expect(page.locator("body")).toContainText("Weekly leaders");
  await expect(page.locator("body")).toContainText("Season leaders (Top Performers)");
  await expect(page.locator("[data-print-surface='league-night']")).toBeVisible();
});
