import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const tournamentLinks = [
  ["Setup", "/admin/tournament-setup"],
  ["Registration editor", "/admin/tournaments"],
  ["Registration reports", "/admin/tournaments/registrations"],
  ["Bulk actions", "/admin/tournaments/bulk"],
  ["Status", "/admin/tournaments/status"],
  ["Operations", "/admin/tournaments/ops"],
  ["Live runner", "/admin/tournament-live"],
  ["Delete draft", "/admin/tournaments/delete-draft"]
] as const;

const routeCases = [
  ["/admin/tournament-setup", "Setup"],
  ["/admin/tournaments", "Registration editor"],
  ["/admin/tournaments/registrations", "Registration reports"],
  ["/admin/tournaments/bulk", "Bulk actions"],
  ["/admin/tournaments/status", "Status"],
  ["/admin/tournaments/delete-draft", "Delete draft"],
  ["/admin/tournaments/ops", "Operations"],
  ["/admin/tournaments/ops/draws", "Operations"],
  ["/admin/tournaments/ops/import", "Operations"],
  ["/admin/tournaments/ops/results", "Operations"],
  ["/admin/tournaments/ops/publish", "Operations"],
  ["/admin/tournament-live", "Live runner"]
] as const;

const operationCases = [
  ["/admin/tournaments/ops", "Overview"],
  ["/admin/tournaments/ops/draws", "Draws and scoring"],
  ["/admin/tournaments/ops/import", "Team imports"],
  ["/admin/tournaments/ops/results", "Results CSV"],
  ["/admin/tournaments/ops/publish", "Official publish"]
] as const;

test.describe("shared Tournament Admin navigation", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
  });

  test("exposes every tournament and operations workflow from every tournament page", async ({ page }) => {
    test.setTimeout(120_000);

    for (const [route] of routeCases) {
      await page.goto(route, { waitUntil: "domcontentloaded" });
      const navigation = page.getByRole("navigation", { name: "Tournament administration" });
      const workflowNavigation = page.getByRole("navigation", { name: "Tournament operations workflows" });

      await expect(navigation).toHaveCount(1);
      await expect(workflowNavigation).toHaveCount(1);
      for (const [label, href] of tournamentLinks) {
        await expect(navigation.getByRole("link", { name: label, exact: true })).toHaveAttribute("href", href);
      }
      for (const [href, label] of operationCases) {
        await expect(workflowNavigation.getByRole("link", { name: label, exact: true })).toHaveAttribute("href", href);
      }
    }
  });

  test("marks exactly one route current without treating every nested route as the editor", async ({ page }) => {
    test.setTimeout(120_000);

    for (const [route, currentLabel] of routeCases) {
      await page.goto(route, { waitUntil: "domcontentloaded" });
      const navigation = page.getByRole("navigation", { name: "Tournament administration" });
      const currentLinks = navigation.locator("[aria-current]");

      await expect(navigation).toHaveCount(1);
      await expect(currentLinks).toHaveCount(1);
      await expect(currentLinks).toHaveText(currentLabel);
      await expect(currentLinks).toHaveAttribute(
        "aria-current",
        route.startsWith("/admin/tournaments/ops/") ? "location" : "page"
      );
      if (route !== "/admin/tournaments") {
        await expect(navigation.getByRole("link", { name: "Registration editor", exact: true })).not.toHaveAttribute("aria-current", "page");
      }
    }
  });

  test("provides one independently active operations workflow navigation", async ({ page }) => {
    test.setTimeout(60_000);

    for (const [route, currentLabel] of operationCases) {
      await page.goto(route, { waitUntil: "domcontentloaded" });
      const tournamentNavigation = page.getByRole("navigation", { name: "Tournament administration" });
      const workflowNavigation = page.getByRole("navigation", { name: "Tournament operations workflows" });

      await expect(tournamentNavigation.getByRole("link", { name: "Operations", exact: true })).toHaveAttribute(
        "aria-current",
        route === "/admin/tournaments/ops" ? "page" : "location"
      );
      await expect(workflowNavigation).toHaveCount(1);
      await expect(workflowNavigation.locator('[aria-current="page"]')).toHaveCount(1);
      await expect(workflowNavigation.locator('[aria-current="page"]')).toHaveText(currentLabel);
      if (route !== "/admin/tournaments/ops") {
        await expect(workflowNavigation.getByRole("link", { name: "Overview", exact: true })).not.toHaveAttribute("aria-current", "page");
      }
    }
  });

  test("updates current state through keyboard navigation", async ({ page }) => {
    await page.goto("/admin/tournaments", { waitUntil: "domcontentloaded" });
    const statusLink = page.getByRole("navigation", { name: "Tournament administration" }).getByRole("link", { name: "Status", exact: true });

    await statusLink.focus();
    await expect(statusLink).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/admin\/tournaments\/status$/);
    await expect(page.getByRole("navigation", { name: "Tournament administration" }).getByRole("link", { name: "Status", exact: true })).toHaveAttribute("aria-current", "page");
  });

  test("keeps both navigation levels reachable on a narrow screen", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/admin/tournaments/ops/results", { waitUntil: "domcontentloaded" });

    const navigationShell = page.getByTestId("tournament-admin-navigation");
    await expect(navigationShell).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Tournament administration" }).getByRole("link")).toHaveCount(tournamentLinks.length);
    await expect(page.getByRole("navigation", { name: "Tournament operations workflows" }).getByRole("link")).toHaveCount(operationCases.length);
    expect(await navigationShell.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  });
});
