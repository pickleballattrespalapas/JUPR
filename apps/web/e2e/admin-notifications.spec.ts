import { expect, test, type Page } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const storedSession = { access_token: "ui-test-token", token_type: "bearer", user: { id: "user-1", email: "admin@example.com" } };
const capabilities = { authorized: true, user: { email: "admin@example.com" }, assignments: [{ club_id: "tres_palapas", role: "club_owner", permissions: ["manage_matches", "manage_players", "manage_tournaments"] }] };
type NoticeState = "new" | "flagged" | "cleared";
type Notice = { key: string; category: string; kind: string; title: string; description: string; href: string; occurred_at: string; state: NoticeState };
const approval: Notice = { key: "generator:rr-1", category: "generator_submissions", kind: "action", title: "Monday round robin", description: "Joe submitted three rated games for approval.", href: "/admin/play-generators/submissions?session=rr-1", occurred_at: "2026-09-21T15:00:00Z", state: "new" };
const registration: Notice = { key: "registration:player-1", category: "registrations", kind: "activity", title: "Casey registered", description: "Casey joined the fall tournament.", href: "/admin/tournaments?tournament=fall", occurred_at: "2026-09-21T14:00:00Z", state: "new" };

async function installNotificationApi(page: Page) {
  const state = {
    items: [structuredClone(approval), structuredClone(registration)],
    preferences: {} as Record<string, boolean>,
    writes: [] as Array<{ url: string; authorization: string; body: unknown }>,
    failWrites: false,
    unavailable: false
  };
  const feed = () => ({
    club_id: "tres_palapas", checked_at: "2026-09-21T16:00:00Z", history_days: 30, truncated: false,
    categories: [
      { key: "generator_submissions", label: "Generator submissions", description: "Completed sessions to review.", href: "/admin/play-generators/submissions", kind: "action" },
      { key: "registrations", label: "New registrations", description: "Players signing up for play.", href: "/admin/tournaments", kind: "activity" }
    ].map(category => ({ ...category, enabled: state.preferences[category.key] !== false, status: state.unavailable && category.kind === "action" ? "unavailable" : "ready", total_count: state.items.filter(item => item.category === category.key).length })),
    items: state.items.filter(item => state.preferences[item.category] !== false && !(state.unavailable && item.kind === "action"))
  });
  await page.route("**/admin/clubs/tres_palapas/notifications**", async route => {
    const request = route.request();
    if (request.method() === "PUT") {
      const body = request.postDataJSON();
      state.writes.push({ url: request.url(), authorization: request.headers().authorization || "", body });
      if (state.failWrites) {
        await route.fulfill({ status: 503, json: { detail: "Unable to save." } });
        return;
      }
      if (new URL(request.url()).pathname.endsWith("/preferences")) state.preferences = { ...state.preferences, ...body.categories };
      else if (new URL(request.url()).pathname.endsWith("/bulk-clear")) state.items.filter(item => body.keys.includes(item.key)).forEach(item => { item.state = "cleared"; });
      else {
        const key = decodeURIComponent(new URL(request.url()).pathname.split("/items/")[1]);
        const item = state.items.find(item => item.key === key);
        if (!item) throw new Error(`Unknown notice ${key}`);
        item.state = body.state;
      }
    }
    await route.fulfill({ json: feed() });
  });
  return state;
}

test.beforeEach(async ({ context, page, baseURL }) => {
  await bootstrapStagingContext(context);
  await context.addCookies([{ name: "jupr_admin_workspace_v1", value: encodeURIComponent(JSON.stringify({ clubId: "tres_palapas", clubSlug: "tres-palapas" })), url: baseURL!, sameSite: "Lax" }]);
  await page.route("**/admin/auth/workspaces", route => route.fulfill({ json: { workspaces: [{ club_id: "tres_palapas", club_slug: "tres-palapas", club_name: "Tres Palapas", roles: ["club_owner"] }] } }));
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, route => route.fulfill({ json: capabilities }));
  await page.addInitScript(session => localStorage.setItem("jupr_admin_session_v1", JSON.stringify(session)), storedSession);
});

test("individual notices flag, clear and restore without removing the source work", async ({ page }, testInfo) => {
  const state = await installNotificationApi(page);
  const pageErrors: string[] = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  const center = page.getByRole("region", { name: "Notifications", exact: true });
  await expect(center).toContainText("2 notices in your inbox");
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toHaveAttribute("href", approval.href);
  await page.screenshot({ path: testInfo.outputPath("admin-notifications-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("admin-notifications-mobile.png"), fullPage: true });
  await center.getByRole("button", { name: "Flag Casey registered", exact: true }).click();
  await expect(center.getByRole("button", { name: "Unflag Casey registered", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(center.locator("li").first()).toContainText("Casey registered");
  await center.getByRole("button", { name: "Clear Monday round robin", exact: true }).click();
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toHaveCount(0);
  await center.getByText("All review queues · Pending work stays here when you clear a notice", { exact: true }).click();
  await expect(center.getByRole("link", { name: "Generator submissions", exact: true })).toHaveAttribute("href", "/admin/play-generators/submissions");
  await center.getByRole("link", { name: /Open notification center/ }).click();
  await expect(page.getByRole("heading", { name: "Notification center", exact: true })).toBeVisible();
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toHaveCount(0);
  state.items.push({ ...approval, key: "generator:rr-2", title: "Tuesday round robin", href: "/admin/play-generators/submissions?session=rr-2" });
  await center.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(center.getByRole("link", { name: /^Tuesday round robin/ })).toBeVisible();
  await center.getByRole("button", { name: /^Cleared / }).click();
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toBeVisible();
  await center.getByRole("button", { name: "Restore Monday round robin", exact: true }).click();
  await center.getByRole("button", { name: /^Inbox / }).click();
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toBeVisible();
  await expect(center.getByRole("link", { name: /^Tuesday round robin/ })).toBeVisible();
  expect(state.writes.map(write => write.body)).toEqual([{ state: "flagged" }, { state: "cleared" }, { state: "new" }]);
  expect(state.writes.every(write => write.authorization === "Bearer ui-test-token")).toBe(true);
  expect(pageErrors).toEqual([]);
});

test("notification preferences survive reload and failed saves remain recoverable", async ({ page }) => {
  const state = await installNotificationApi(page);
  await page.goto("/admin/notifications", { waitUntil: "domcontentloaded" });
  const center = page.getByRole("region", { name: "Notifications", exact: true });
  await center.getByRole("button", { name: "Notification settings", exact: true }).click();
  await center.getByRole("checkbox", { name: /^New registrations/ }).uncheck();
  state.failWrites = true;
  await center.getByRole("button", { name: "Save preferences", exact: true }).click();
  await expect(center.getByRole("alert")).toContainText("Your change couldn’t be confirmed");
  await expect(center.getByRole("link", { name: /Casey registered/ })).toBeVisible();
  state.failWrites = false;
  await center.getByRole("button", { name: "Save preferences", exact: true }).click();
  await expect(center.getByRole("link", { name: /Casey registered/ })).toHaveCount(0);
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toBeVisible();
  await page.reload({ waitUntil: "domcontentloaded" });
  await center.getByRole("button", { name: "Notification settings", exact: true }).click();
  await expect(center.getByRole("checkbox", { name: /^New registrations/ })).not.toBeChecked();
  await center.getByRole("checkbox", { name: /^New registrations/ }).check();
  await center.getByRole("button", { name: "Save preferences", exact: true }).click();
  await expect(center.getByRole("link", { name: /Casey registered/ })).toBeVisible();
  state.failWrites = true;
  await center.getByRole("button", { name: "Clear Monday round robin", exact: true }).click();
  await expect(center.getByRole("alert")).toContainText("Your change couldn’t be confirmed");
  await expect(center.getByRole("link", { name: /^Monday round robin/ })).toBeVisible();
  state.failWrites = false;
  state.unavailable = true;
  await center.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(center.getByRole("alert")).toContainText("Their notices may be missing");
  await expect(center).not.toContainText("No new notices");
});

test("bulk selection clears the selected notices and updates sidebar counts and summaries", async ({ page }, testInfo) => {
  const state = await installNotificationApi(page);
  await page.goto("/admin/notifications", { waitUntil: "domcontentloaded" });
  const center = page.getByRole("region", { name: "Notifications", exact: true });
  const inboxLink = page.getByRole("link", { name: "Notifications", exact: true });
  const tournamentLink = page.getByRole("link", { name: "Tournament Manager", exact: true });
  await expect(inboxLink).toContainText("2");
  await expect(tournamentLink).toContainText("1");
  await tournamentLink.locator('[aria-hidden="true"]').hover();
  await expect(page.locator('[role="tooltip"]')).toBeVisible();
  await expect(page.locator('[role="tooltip"]')).toContainText("Casey registered");
  await center.getByRole("checkbox", { name: "Select Monday round robin", exact: true }).check();
  await expect(center.getByRole("checkbox", { name: "Select all shown notifications", exact: true })).toHaveJSProperty("indeterminate", true);
  await center.getByRole("combobox", { name: "Category" }).selectOption("registrations");
  await expect(center.getByRole("button", { name: "Clear selected", exact: true })).toBeDisabled();
  await center.getByRole("checkbox", { name: "Select all shown notifications", exact: true }).check();
  await center.getByRole("button", { name: "Clear selected (1)", exact: true }).click();
  expect(state.items.find(item => item.key === approval.key)?.state).toBe("new");
  await expect(inboxLink).toContainText("1");
  await expect(tournamentLink).not.toHaveAttribute("aria-describedby", /.+/);
  await center.getByRole("combobox", { name: "Category" }).selectOption("all");
  await expect(center.getByRole("link", { name: "Review Monday round robin", exact: true })).toHaveAttribute("href", approval.href);
  await center.getByRole("checkbox", { name: "Select all shown notifications", exact: true }).check();
  await page.screenshot({ path: testInfo.outputPath("notification-selection-and-badges.png"), fullPage: true });
  await center.getByRole("button", { name: "Clear selected (1)", exact: true }).click();
  await expect(inboxLink).not.toHaveAttribute("aria-describedby", /.+/);
  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(center).toContainText("No new notices in your selected categories");
  await center.getByRole("button", { name: /^Cleared / }).click();
  await expect(center.getByRole("button", { name: "Restore Monday round robin", exact: true })).toBeVisible();
  await expect(center.getByRole("button", { name: "Restore Casey registered", exact: true })).toBeVisible();
  expect(state.writes.map(write => write.body)).toEqual([{ keys: [registration.key] }, { keys: [approval.key] }]);
});
