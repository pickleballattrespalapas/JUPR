import { expect, type Page } from "@playwright/test";
import { expectedApiOrigin } from "./staging";

export async function verifyInactivePlayerVisibility(page: Page) {
  // Existing isolated fixture. This flow never changes identity or activity.
  const name = "League Live E2E P1 fa43f59-32868369678-1";
  const query = encodeURIComponent(name);
  const errors: string[] = [];
  const onError = (error: Error) => errors.push(error.message);
  page.on("pageerror", onError);
  try {
    expect(expectedApiOrigin).toBe("https://juprleagues-api-staging.fly.dev");
    const response = await page.request.get(`${expectedApiOrigin}/clubs/tres-palapas/players?q=${query}&status=active`);
    expect(response.status()).toBe(200);
    const directory = await response.json();
    expect(directory.filters.status).toBe("all");
    expect(directory.players.map((player: { id: number }) => player.id)).toEqual([39]);
    expect(directory.players[0].is_active).toBe(false);

    await page.goto(`/clubs/tres-palapas/players?q=${query}&status=active`, { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("players-status-active")).toHaveCount(0);
    const row = page.getByTestId("players-row").filter({ hasText: name });
    await expect(row).toHaveCount(1);
    await expect(row).toHaveAttribute("data-status", "inactive");
    const profileLink = row.getByRole("link", { name: `Open ${name} profile` });
    await expect(profileLink).toHaveAttribute("href", "/clubs/tres-palapas/players/39");
    await profileLink.click();
    await expect(page.getByTestId("player-profile")).toBeVisible();

    await page.goto(`/clubs/tres-palapas/leaderboards?q=${query}&league_name=OVERALL`, { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("leaderboard-filter-empty-state")).toBeVisible();
    await expect(page.getByTestId("leaderboard-row")).toHaveCount(0);
    expect(errors).toEqual([]);
  } finally {
    page.removeListener("pageerror", onError);
  }
}
