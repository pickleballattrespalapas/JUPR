import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";


const runWrites = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_RUN_PUBLIC_LIVE_WRITE_E2E || ""));


test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});


test("public JUPR Live exposes the guarded creator or truthful view-only fallback", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto(`/clubs/${clubSlug}/live`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.getByRole("heading", { name: /live sessions/i })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /JUPR Live|New public sessions are paused here/i })
  ).toBeVisible();
  await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});


test("mobile Quick Round Robin survives refresh and completes without leaking its edit token", async ({ page }) => {
  test.skip(!runWrites, "Set JUPR_RUN_PUBLIC_LIVE_WRITE_E2E=1 only for the disposable staging write smoke.");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`/clubs/${clubSlug}/live`, { waitUntil: "domcontentloaded" });

  await expect(page.getByRole("heading", { name: "🔴 JUPR Live" })).toBeVisible();
  await page.getByLabel("Count").selectOption("4");
  await page.getByLabel("Event name").fill(`Parity Quick ${Date.now()}`);
  await page.getByRole("button", { name: "Create event" }).click();
  await expect(page).toHaveURL(new RegExp(`/clubs/${clubSlug}/live/[a-f0-9]{32}$`));
  expect(page.url()).not.toContain("edit=");
  await expect(page.getByText("Score entry enabled.", { exact: false })).toBeVisible();

  const teamAScores = page.getByLabel(/team A score/);
  const teamBScores = page.getByLabel(/team B score/);
  const matchCount = await teamAScores.count();
  expect(matchCount).toBeGreaterThan(0);
  for (let index = 0; index < matchCount; index += 1) {
    await teamAScores.nth(index).fill("11");
    await teamBScores.nth(index).fill("8");
  }
  await page.getByRole("button", { name: "Save scores" }).click();
  await expect(page.getByText("Scores saved.")).toBeVisible();
  await page.getByRole("button", { name: "Refresh" }).click();
  await expect(page.getByText("Session updated.")).toBeVisible();
  await expect(page.getByText(`${matchCount}/${matchCount}`, { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Export CSV" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Export JSON" })).toBeVisible();

  await page.getByRole("button", { name: "Complete session" }).click();
  await expect(page.getByText("Session completed.")).toBeVisible();
  await expect(page.getByText("Complete", { exact: true })).toBeVisible();
});
