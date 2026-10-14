import { expect, type Page } from "@playwright/test";

export async function openPublicControls(page: Page) {
  const panel = page.locator("[data-public-controls-panel]").first();
  await expect(panel).toBeAttached();
  if (await panel.getAttribute("open") === null) {
    await panel.locator("summary").first().click();
  }
}
