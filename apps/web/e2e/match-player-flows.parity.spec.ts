import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("score entry is either write-ready or visibly routed to a safe fallback", async ({ page }) => {
  const response = await page.goto(`/clubs/${clubSlug}/admin/score-entry`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.locator("body")).toContainText(/score entry/i);
  const body = await page.locator("body").innerText();
  if (/fallback mode|score entry is disabled/i.test(body)) {
    await expect(page.getByRole("link", { name: /match uploader/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /streamlit fallback/i })).toBeVisible();
  } else {
    await expect(page.getByRole("heading", { name: /enter one rated match/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /save rated match/i })).toBeVisible();
  }
});

test("match uploader exposes every entry mode and recovery language", async ({ page }) => {
  const response = await page.goto("/admin/match-uploader", { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.locator("body")).toContainText(/match uploader/i);
  const body = await page.locator("body").innerText();
  if (!/match uploader is disabled/i.test(body)) {
    const entryMethod = page.getByLabel("Entry method");
    await expect(entryMethod).toBeVisible();
    if (/direct singles (submission|entry).*(disabled|unavailable)/i.test(body)) {
      await expect(entryMethod.locator("option")).toHaveText([
        "Doubles manual / batch",
        "Doubles round robin"
      ]);
    } else {
      await expect(entryMethod.locator("option")).toHaveText([
        "Singles match",
        "Doubles manual / batch",
        "Doubles round robin"
      ]);
    }
    await entryMethod.selectOption("manual");
    await expect(page.getByRole("button", { name: "Add 1 Match" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Add 5 Matches" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove All" })).toBeVisible();
    await expect(page.getByLabel("Team 1 · Player 1").first()).toHaveAttribute("list", /-options$/);

    await entryMethod.selectOption("round_robin");
    await expect(page.getByRole("heading", { name: "Doubles round-robin generator" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Add round robin" })).toBeVisible();
    await expect(page.getByLabel("Players").first()).toHaveAttribute("list", /-options$/);
  }
});

test("player editor exposes atomic merge preview and server readiness", async ({ page }) => {
  const response = await page.goto("/admin/players", { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.locator("body")).toContainText(/player editor/i);
  const body = await page.locator("body").innerText();
  if (!/player editor is disabled/i.test(body)) {
    await expect(page.getByRole("heading", { name: /merge player accounts/i })).toBeVisible();
    await expect(page.locator("body")).toContainText(/database transaction/i);
    await expect(page.getByRole("button", { name: /preview merge/i })).toBeVisible();
  }
});
