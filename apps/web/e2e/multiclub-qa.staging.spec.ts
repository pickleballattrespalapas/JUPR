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
test("club creation starts with club details and preserves them before account setup", async ({ page, context }) => {
  await bootstrapStagingContext(context);
  const creationRequests: string[] = [];
  page.on("request", request => {
    if (request.method() === "POST" && /\/(clubs\/create|auth\/v1\/(signup|token))/.test(request.url())) creationRequests.push(request.url());
  });
  await page.goto("/create-club");
  await expect(page.getByRole("heading", { name: "Club details", exact: true })).toBeVisible();
  await expect(page.getByLabel("Administrator email", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Password", { exact: true })).toHaveCount(0);
  await page.getByLabel("Club name", { exact: true }).fill("Baja Travelers QA");
  await expect(page.getByLabel("Club web address", { exact: true })).toHaveValue("baja-travelers-qa");
  await page.getByRole("button", { name: "Continue →", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Administrator", exact: true })).toBeVisible();
  await expect(page.getByText("Baja Travelers QA", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "New administrator", exact: true })).toBeDisabled();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Administrator", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "← Club details", exact: true }).click();
  await expect(page.getByLabel("Club name", { exact: true })).toHaveValue("Baja Travelers QA");
  await page.getByLabel("Club web address", { exact: true }).fill("baja-custom-qa");
  await page.getByLabel("Club name", { exact: true }).fill("Baja Visitors QA");
  await expect(page.getByLabel("Club web address", { exact: true })).toHaveValue("baja-custom-qa");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Continue →", exact: true }).scrollIntoViewIfNeeded();
  await expect(page.getByRole("button", { name: "Continue →", exact: true })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(creationRequests, "Draft setup must not create accounts, send emails or create a club").toEqual([]);
});

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

  await test.step("signed-in administrator confirms their account in Step 2 before review", async () => {
    await page.goto("/create-club");
    await page.getByLabel("Club name", { exact: true }).fill("QA Club Creation Preview");
    await page.getByRole("button", { name: "Continue →", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Administrator", exact: true })).toBeVisible();
    await expect(page.getByText(email, { exact: true })).toBeVisible();
    await page.reload();
    await expect(page.getByRole("heading", { name: "Administrator", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Use this account →", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Review and create", exact: true })).toBeVisible();
    await expect(page.getByText(email, { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Create club →", exact: true })).toBeEnabled();
    await expect(page.getByLabel("Password", { exact: true })).toHaveCount(0);
    await page.getByRole("button", { name: "← Administrator", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Administrator", exact: true })).toBeVisible();
    // Review only. Preserve the dedicated QA identity's three-club assignment boundary.
  });

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
      await expect(page.getByText(/^Unsaved changes\b/)).toBeVisible();
      await expect(page.getByRole("button", { name: "Save draft", exact: true })).toBeEnabled();
      await expect(page.getByRole("textbox", { name: "Link to Players", exact: true })).toHaveValue(`${origin}/clubs/${club.id}/players`);
      page.once("dialog", dialog => dialog.accept());
      await page.reload();
      await expect(page.getByText(/^Unsaved changes\b/)).toHaveCount(0);
      await page.getByRole("button", { name: "Page visibility", exact: true }).click();
      await expect(page.getByRole("combobox", { name: "Visibility for Players", exact: true })).toHaveValue(original);
      await page.getByRole("button", { name: "Stats & information", exact: true }).click();
      await expect(page.getByRole("checkbox", { name: "Current ratings", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Preview draft", exact: true }).click();
      await expect(page.getByRole("button", { name: "Publish website", exact: true })).toBeVisible();
      // No saves or publication: preserve Joe's in-progress drafts and live sites.

      const invitationsResponse = page.waitForResponse(r => new URL(r.url()).pathname === `/admin/clubs/${club.id}/interclub/registrations` && r.request().method() === "GET");
      await page.goto("/admin/interclub");
      const invitations = await invitationsResponse;
      expect(invitations.status()).toBe(200);
      const seasons = (await invitations.json()).seasons;
      expect(seasons.length, "The isolation season must be available").toBeGreaterThan(0);
      for (const season of seasons) {
        await page.goto("/admin/interclub");
        const invitationLink = page.locator(`a[href="/admin/interclub/registrations?season=${season.id}"]`);
        await expect(invitationLink).toBeVisible();
        if (season.organizer_club_id !== club.id && season.participation?.status === "invited") {
          await expect(page.getByRole("heading", { name: "Invitations to your club", exact: true })).toBeVisible();
          await expect(invitationLink).toHaveText("Review invitation");
        }
        const seasonPath = `/admin/clubs/${club.id}/interclub/registrations/${season.id}`;
        const seasonResponse = page.waitForResponse(r => new URL(r.url()).pathname === seasonPath && r.request().method() === "GET");
        await invitationLink.click();
        const response = await seasonResponse;
        expect(response.status(), `${club.name}: ${season.details.name} season request`).toBe(200);
        const details = await response.json();
        await expect(page.getByRole("heading", { name: season.details.name, exact: true })).toBeVisible();
        await expect(page.getByText("Loading season…", { exact: true })).toHaveCount(0);
        const organizer = season.organizer_club_id === club.id;
        expect(details.is_organizer).toBe(organizer);
        expect(details.meets.map((m: { club_ids: string[] }) => organizer || m.club_ids.includes(club.id)).every(Boolean)).toBe(true);
        if (!organizer) expect(details.participations.every((p: { club_id: string }) => p.club_id === club.id)).toBe(true);
        if (details.own_participation?.status === "invited") {
          const accept = page.getByRole("button", { name: "Accept invitation", exact: true });
          await expect(accept).toBeEnabled();
          await expect(accept).toBeInViewport();
          expect(await accept.evaluate(el => getComputedStyle(el).cursor)).toBe("pointer");
          expect((await accept.boundingBox())!.height).toBeGreaterThanOrEqual(44);
          await expect(page.getByRole("heading", { name: `${club.name} is invited`, exact: true })).toBeVisible();
        }
        const joined = details.own_participation?.status === "accepted";
        if (joined) {
          await expect(page.getByRole("heading", { name: `${club.name} has joined`, exact: true })).toBeVisible();
          const poolResult = page.waitForResponse(r => new URL(r.url()).pathname === `${seasonPath}/pool` && r.request().method() === "GET");
          await page.getByRole("button", { name: "Build season player pool", exact: true }).click();
          const poolResponse = await poolResult;
          expect(poolResponse.status()).toBe(200);
          const pool = await poolResponse.json();
          expect(pool.members.every((member: {club_id: string; season_id: string}) => member.club_id === club.id && member.season_id === season.id)).toBe(true);
          await expect(page.getByRole("region", { name: "Season player pool", exact: true })).toBeVisible();
          await page.getByRole("button", { name: "Close player pool", exact: true }).click();
          const prepare = page.getByRole("button", { name: "Prepare meet roster", exact: true });
          if (details.meets.some((m: { roster_open: boolean; club_ids: string[] }) => m.roster_open && m.club_ids.includes(club.id))) {
            await prepare.click();
            await expect(page.getByRole("region", { name: "Meet rosters", exact: true })).toBeFocused();
          }
        }
        if (!organizer && !joined) await expect(page.getByRole("combobox", { name: "Meet", exact: true })).toHaveCount(0);
        if (details.meets.length && (organizer || joined)) {
          const meet = page.getByRole("combobox", { name: "Meet", exact: true });
          const meetId = await meet.inputValue();
          const reloadMeet = page.getByRole("button", { name: "Reload meet", exact: true });
          await expect(reloadMeet).toBeEnabled();
          const meetResponse = page.waitForResponse(r => new URL(r.url()).pathname === `${seasonPath}/meets/${meetId}` && r.request().method() === "GET");
          await reloadMeet.click();
          const rosterResult = await meetResponse;
          expect(rosterResult.status()).toBe(200);
          const roster = await rosterResult.json();
          expect(roster.meet.season_id).toBe(season.id);
          expect(roster.teams.every((t: { club_id: string; meet_id: string }) => t.meet_id === meetId && (organizer || t.club_id === club.id))).toBe(true);
          await expect(page.getByText("Loading meet…", { exact: true })).toHaveCount(0);
          if (joined && roster.meet.club_ids.includes(club.id)) {
            const availabilityResult = page.waitForResponse(r => new URL(r.url()).pathname === `${seasonPath}/meets/${meetId}/availability` && r.request().method() === "GET");
            await page.getByRole("button", { name: "Invite players & view availability", exact: true }).click();
            const availabilityResponse = await availabilityResult;
            expect(availabilityResponse.status()).toBe(200);
            const availability = await availabilityResponse.json();
            expect(availability.meet.id).toBe(meetId);
            await expect(page.getByRole("button", { name: "Hide player availability", exact: true })).toBeVisible();
          }
        }
      }
      // Inspect invitations and rosters only; do not accept, decline or edit them.
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
