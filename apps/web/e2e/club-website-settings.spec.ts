import { expect, test, type Page } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";
import { leaderboardSettings } from "../lib/clubSite";

const session = { access_token: "ui-test-token", token_type: "bearer", user: { id: "user-1", email: "admin@example.invalid" } };
const capabilities = { authorized: true, assignments: [{ club_id: "tres_palapas", role: "club_owner", permissions: [] }] };
const fixtureDocument = {
  schema_version: 1, name: "Tres Palapas", description: "Welcome to the courts", location: "Los Barriles",
  visitor_info: "Visitors welcome", logo_url: "", accent: "#1d4ed8", visibility: "listed", display: {},
  page_visibility: {}, pages: [{ slug: "home", title: "Home", in_navigation: true, blocks: [] }],
};

async function installSettingsApi(page: Page) {
  const website = { club_id: "tres_palapas", slug: "tres-palapas", club_active: true, revision: 1,
    draft: structuredClone(fixtureDocument), published: structuredClone(fixtureDocument), published_at: "2026-09-24T00:00:00Z" };
  const leaderboard = { revision: 1, draft: leaderboardSettings(), published: leaderboardSettings(), published_at: "2026-09-24T00:00:00Z" };
  const writes: { path: string; body: Record<string, unknown> }[] = [];
  await page.route("**/admin/clubs/tres_palapas/**", async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (path.includes("/notifications")) {
      return route.fulfill({ json: { club_id: "tres_palapas", checked_at: "2026-09-24T00:00:00Z", categories: [], items: [], history_days: 30, truncated: false } });
    }
    if (!path.includes("/site") && !path.includes("/leaderboard-settings")) return route.abort();
    const state = path.includes("/leaderboard-settings") ? leaderboard : website;
    if (request.method() !== "GET") {
      const body = request.postDataJSON();
      expect(request.headers().authorization).toBe("Bearer ui-test-token");
      expect(body.revision).toBe(state.revision);
      writes.push({ path, body });
      if (request.method() === "PUT") state.draft = structuredClone(body.settings || body.document);
      else if (path.endsWith("/publish")) state.published = structuredClone(state.draft);
      else if (path.endsWith("/discard")) state.draft = structuredClone(state.published);
      state.revision++;
    }
    return route.fulfill({ json: state });
  });
  return { website, leaderboard, writes };
}

test.beforeEach(async ({ context, page, baseURL }) => {
  await bootstrapStagingContext(context);
  await context.addCookies([{ name: "jupr_admin_workspace_v1", value: encodeURIComponent(JSON.stringify({ clubId: "tres_palapas", clubSlug: "tres-palapas" })), url: baseURL!, sameSite: "Lax" }]);
  await page.route("**/admin/auth/workspaces", route => route.fulfill({ json: { workspaces: [{ club_id: "tres_palapas", club_slug: "tres-palapas", club_name: "Tres Palapas", roles: ["club_owner"] }] } }));
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, route => route.fulfill({ json: capabilities }));
  await page.addInitScript(value => localStorage.setItem("jupr_admin_session_v1", JSON.stringify(value)), session);
});

test("leaderboard save stays visible on desktop and mobile and publishes independently", async ({ page }, testInfo) => {
  const state = await installSettingsApi(page);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/admin/leaderboard-settings", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Leaderboard settings", exact: true })).toBeVisible();
  await page.getByLabel("Minimum games for performance cards", { exact: true }).fill("8");
  const save = page.getByRole("button", { name: "Save draft", exact: true });
  const publish = page.getByRole("button", { name: "Publish settings", exact: true });
  await page.getByRole("button", { name: "Add season or date range" }).scrollIntoViewIfNeeded();
  await expect(save).toBeInViewport();
  await expect(save).toBeEnabled();
  await expect(publish).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath("leaderboard-save-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Add season or date range" }).scrollIntoViewIfNeeded();
  await expect(save).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("leaderboard-save-mobile.png") });
  await save.click();
  await expect(page.getByRole("status")).toContainText("Draft saved. Publish");
  expect(state.leaderboard.published.min_games).toBe(0);
  await publish.click();
  await expect(page.getByRole("status")).toContainText("Leaderboard settings published");
  expect(state.leaderboard.published.min_games).toBe(8);
  expect(state.website.revision).toBe(1);
  expect(state.writes.map(write => write.path)).toEqual([
    "/admin/clubs/tres_palapas/leaderboard-settings", "/admin/clubs/tres_palapas/leaderboard-settings/publish",
  ]);
  await page.reload();
  await expect(page.getByLabel("Minimum games for performance cards", { exact: true })).toHaveValue("8");
  expect(errors).toEqual([]);
});

test("club website tabs save and publish without changing leaderboard choices", async ({ page }, testInfo) => {
  const state = await installSettingsApi(page);
  await page.goto("/admin/website", { waitUntil: "domcontentloaded" });
  const editor = page.getByRole("navigation", { name: "Website editor" });
  for (const name of ["Club introduction", "Pages & layout", "Page visibility", "Stats & information", "Overall leaderboard", "Preview & publish"]) {
    await expect(editor.getByRole("button", { name, exact: true })).toBeVisible();
  }
  await page.getByLabel("Public club name", { exact: true }).fill("Tres Palapas Baja Pickleball");
  await editor.getByRole("button", { name: "Page visibility", exact: true }).click();
  await page.getByLabel("Visibility for Players", { exact: true }).selectOption("private");
  await expect(page.getByText("Interclub leagues", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Draft saved");
  expect(state.website.published.name).toBe("Tres Palapas");
  await page.getByRole("button", { name: "Preview draft", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Tres Palapas Baja Pickleball", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Publish website", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Website published");
  expect(state.website.published.name).toBe("Tres Palapas Baja Pickleball");
  expect(state.leaderboard.revision).toBe(1);
  await page.screenshot({ path: testInfo.outputPath("club-website-published.png") });
  await editor.getByRole("button", { name: "Overall leaderboard", exact: true }).click();
  await expect(page.getByRole("link", { name: "Edit leaderboard settings", exact: true })).toHaveAttribute("href", "/admin/leaderboard-settings");
  await page.reload();
  await expect(page.getByLabel("Public club name", { exact: true })).toHaveValue("Tres Palapas Baja Pickleball");
});
