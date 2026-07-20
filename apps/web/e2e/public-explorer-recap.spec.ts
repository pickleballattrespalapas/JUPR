import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("Match Explorer hydrates a share link and reacts through Python projections", async ({ page }) => {
  const response = await page.goto(`/clubs/${clubSlug}/match-explorer`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);

  const me = page.getByLabel("I am");
  const partner = page.getByLabel("My partner");
  const opponentOne = page.getByLabel("Opponent 1");
  const opponentTwo = page.getByLabel("Opponent 2");
  const playerIds = await me.locator("option").evaluateAll((options) =>
    options.map((option) => (option as HTMLOptionElement).value).filter(Boolean)
  );
  expect(playerIds.length, "Match Explorer staging fixture needs four active public players").toBeGreaterThanOrEqual(4);

  // The initial projection proves React hydration and the debounced preview have
  // settled before Playwright changes the controlled selects.
  await expect(page.getByTestId("match-explorer-summary")).toContainText("Expected win rate");

  await me.selectOption(playerIds[0]);
  await partner.selectOption(playerIds[0]);
  await expect(page.getByTestId("match-explorer-validation")).toContainText("four different players");
  await expect(page.getByTestId("match-explorer-impact-chart")).toHaveCount(0);

  await partner.selectOption(playerIds[1]);
  await opponentOne.selectOption(playerIds[2]);
  await opponentTwo.selectOption(playerIds[3]);
  await page.getByLabel("Your points").fill("11");
  await page.getByLabel("Opponent points").fill("7");

  await expect(page.getByTestId("match-explorer-summary")).toContainText("Expected win rate");
  await expect(page.getByTestId("match-explorer-player-impact").locator("tbody tr")).toHaveCount(4);
  await expect(page.getByTestId("match-explorer-impact-chart")).toContainText("Actual 11–7");

  await page.getByRole("button", { name: "Copy share link" }).click();
  await expect(page).toHaveURL(new RegExp(`[?&]me=${encodeURIComponent(playerIds[0])}(?:&|$)`));
  await expect(page).toHaveURL(new RegExp(`[?&]partner=${encodeURIComponent(playerIds[1])}(?:&|$)`));
  await expect(page).toHaveURL(/[?&]sy=11(?:&|$)/);
  await expect(page).toHaveURL(/[?&]so=7(?:&|$)/);

  const shareUrl = page.url();
  await page.goto(shareUrl, { waitUntil: "domcontentloaded" });
  await expect(me).toHaveValue(playerIds[0]);
  await expect(partner).toHaveValue(playerIds[1]);
  await expect(opponentOne).toHaveValue(playerIds[2]);
  await expect(opponentTwo).toHaveValue(playerIds[3]);
  await expect(page.getByTestId("match-explorer-impact-chart")).toContainText("Actual 11–7");
});

test("Weekly Recap exposes published deep links, print view, and same-origin PDF", async ({ page }) => {
  const response = await page.goto(`/clubs/${clubSlug}/weekly-recap`, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page.getByTestId("weekly-recap-document"), "Staging needs at least one published recap fixture").toBeVisible();
  await expect(page.getByTestId("weekly-recap-number-cards").locator("article")).toHaveCount(6);

  const firstWeek = page.getByTestId("weekly-recap-week-link").first();
  await expect(firstWeek).toBeVisible();
  await firstWeek.click();
  await expect(page).toHaveURL(/[?&]week=\d{4}-\d{2}-\d{2}(?:&|$)/);

  await page.getByTestId("weekly-recap-section-spotlight").click();
  await expect(page).toHaveURL(/[?&]section=spotlight(?:&|$)/);
  await expect(page.getByTestId("weekly-recap-section-content-spotlight")).toBeVisible();
  await expect(page.getByTestId("weekly-recap-section-content-around")).toHaveCount(0);

  const pdfLink = page.getByTestId("weekly-recap-pdf-link");
  await expect(pdfLink).toHaveAttribute("href", /^\/api\/clubs\//);
  const pdfHref = await pdfLink.getAttribute("href");
  expect(pdfHref).not.toBeNull();
  const pdfResponse = await page.request.get(new URL(String(pdfHref), page.url()).toString());
  expect(pdfResponse.status()).toBe(200);
  expect(pdfResponse.headers()["content-type"] || "").toContain("application/pdf");
  expect((await pdfResponse.body()).subarray(0, 8).toString()).toBe("%PDF-1.4");

  await page.getByTestId("weekly-recap-print-link").click();
  await expect(page).toHaveURL(/[?&]print=1(?:&|$)/);
  await expect(page.getByTestId("weekly-recap-page")).toHaveAttribute("data-print-mode", "true");
  await expect(page.getByTestId("weekly-recap-document")).toBeVisible();
});
