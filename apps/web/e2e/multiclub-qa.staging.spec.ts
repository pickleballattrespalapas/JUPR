import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin, expectedAuthOrigin } from "./support/staging";

const clubs = [
  { id: "cabo-test-club", name: "Cabo Test Club" },
  { id: "la-paz-test-club", name: "La Paz Test Club" },
  { id: "la-ribera-pickelball-club", name: "La Ribera Pickelball Club" },
];

// Use the existing protected-deployment CI bootstrap. No sign-in bypass is
// shipped in the app: FastAPI verifies the real, short-lived Supabase JWT.
test.use({ trace: "off", video: "off", screenshot: "off" });
test("dedicated QA admin switches three clubs and previews website controls", async ({ page, context }) => {
  test.setTimeout(180_000);
  const token = process.env.STAGING_ADMIN_BEARER_TOKEN || "";
  const email = process.env.STAGING_ADMIN_EMAIL || "";
  const origin = process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN || "";
  expect(Boolean(token), "A workflow-issued test session is required").toBe(true);
  expect(email).toBe("pcs-staging-qa@example.invalid");
  expect(expectedApiOrigin).toBe("https://juprleagues-api-staging.fly.dev");
  expect(expectedAuthOrigin).toBe("https://sijpxjxvdtrehmqvirfi.supabase.co");
  expect(origin).toBe(process.env.STAGING_WEB_BASE_URL);
  await bootstrapStagingContext(context);
  await context.addInitScript(({ token: access_token, email, origin }) => {
    if (location.origin === origin) localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
      access_token, token_type: "bearer", user: { email },
    }));
  }, { token, email, origin });

  const workspacesResponse = page.waitForResponse(r => r.url() === `${expectedApiOrigin}/admin/auth/workspaces` && r.request().method() === "GET");
  await page.goto("/admin/select-club");
  const workspaces = await workspacesResponse;
  expect(workspaces.status()).toBe(200);
  const assignments = (await workspaces.json()).workspaces;
  expect(assignments.map((w: { club_id: string }) => w.club_id).sort()).toEqual(clubs.map(c => c.id).sort());
  await expect(page.getByRole("heading", { name: "Choose your club" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Your club workspaces" }).getByRole("button")).toHaveCount(3);

  const playerSets: Set<number>[] = [];
  for (const club of clubs) {
    await test.step(`${club.name}: switch, roster and page controls`, async () => {
      await page.getByRole("button", { name: `Open ${club.name}`, exact: true }).click();
      await expect(page).toHaveURL(`${origin}/admin`);
      const workspace = (await context.cookies()).find(c => c.name === "jupr_admin_workspace_v1");
      expect(JSON.parse(decodeURIComponent(workspace?.value || "{}"))).toEqual({ clubId: club.id, clubSlug: club.id });

      await page.getByRole("link", { name: "Players", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Player Editor", exact: true })).toBeVisible();
      const rosterResponse = page.waitForResponse(r => new URL(r.url()).pathname === `/admin/clubs/${club.id}/players/editor/players` && r.request().method() === "GET");
      await page.getByRole("button", { name: "Refresh players and identities" }).click();
      const roster = await rosterResponse;
      expect(roster.status()).toBe(200);
      const players = (await roster.json()).players;
      expect(players.length).toBeGreaterThanOrEqual(24);
      const ids = new Set<number>(players.map((p: { id: number }) => p.id));
      for (const previous of playerSets) expect([...ids].some(id => previous.has(id)), "A player ID appeared in two club rosters").toBe(false);
      playerSets.push(ids);

      const siteResponse = page.waitForResponse(r => new URL(r.url()).pathname === `/admin/clubs/${club.id}/site` && r.request().method() === "GET");
      await page.getByRole("link", { name: "Club website", exact: true }).click();
      const siteResult = await siteResponse;
      expect(siteResult.status()).toBe(200);
      const site = await siteResult.json();
      expect(site.club_id).toBe(club.id);
      await page.getByRole("button", { name: "Page visibility", exact: true }).click();
      const visibility = page.getByRole("combobox", { name: "Visibility for Players", exact: true });
      const original = await visibility.inputValue();
      await visibility.selectOption(original === "private" ? "public" : "private");
      await expect(page.getByText("Unsaved changes", { exact: true })).toBeVisible();
      await expect(page.getByRole("button", { name: "Save draft", exact: true })).toBeEnabled();
      await expect(page.getByRole("textbox", { name: "Link to Players", exact: true })).toHaveValue(`${origin}/clubs/${club.id}/players`);
      page.once("dialog", dialog => dialog.accept());
      await page.reload();
      await expect(page.getByText("Unsaved changes", { exact: true })).toHaveCount(0);
      await page.getByRole("button", { name: "Page visibility", exact: true }).click();
      await expect(page.getByRole("combobox", { name: "Visibility for Players", exact: true })).toHaveValue(original);
      await page.getByRole("button", { name: "Stats & information", exact: true }).click();
      await expect(page.getByRole("checkbox", { name: "Current ratings", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Preview draft", exact: true }).click();
      await expect(page.getByRole("button", { name: "Publish website", exact: true })).toBeVisible();
      // No saves or publication: preserve Joe's in-progress drafts and live sites.
      await page.getByRole("link", { name: "Switch club", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Choose your club" })).toBeVisible();
    });
  }

  // Actual protected API denial; hiding the extra club in the selector is insufficient.
  const denied = await context.request.get(`${expectedApiOrigin}/admin/clubs/tres_palapas/site`, {
    headers: { Authorization: `Bearer ${token}` }, maxRedirects: 0,
  });
  expect(denied.status()).toBe(403);
  await page.goto("/admin/login");
  const signedOut = page.waitForResponse(r => r.url().startsWith(`${expectedAuthOrigin}/auth/v1/logout`) && r.request().method() === "POST");
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  expect([200, 204]).toContain((await signedOut).status());
  await expect.poll(() => page.evaluate(() => localStorage.getItem("jupr_admin_session_v1") === null)).toBe(true);
});
