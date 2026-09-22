import { expect, test, type Page, type Response } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin, expectedAuthOrigin } from "./support/staging";

const clubs = [
  { id: "cabo-test-club", name: "Cabo Test Club" },
  { id: "la-paz-test-club", name: "La Paz Test Club" },
  { id: "la-ribera-pickelball-club", name: "La Ribera Pickelball Club" },
];

async function expectMeetStepsLocked(page: Page) {
  const workflow = page.getByRole("navigation", { name: "League workflow", exact: true });
  await expect(workflow).toBeVisible();
  for (const label of ["Meet availability", "Lineups", "Run meet", "Approve results"]) {
    await expect(workflow.getByRole("link", { name: label, exact: true })).toHaveCount(0);
    await expect(workflow.locator('[aria-disabled="true"]').filter({ has: page.getByText(label, { exact: true }) })).toHaveCount(1);
  }
}

async function expectNoMeetControls(page: Page) {
  // Include hidden elements: registration must not merely conceal mounted meet panels.
  await expect(page.getByRole("combobox", { name: "Meet", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reload meet", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Meet rosters", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Meet player availability", exact: true, includeHidden: true })).toHaveCount(0);
}

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
  test.setTimeout(300_000);
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
  const competitionWrites: string[] = [];
  const lateRequestWrites: string[] = [];
  const operationalReads: string[] = [];
  page.on("request", request => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.includes("/interclub/competition") && !["GET", "HEAD", "OPTIONS"].includes(request.method())) competitionWrites.push(request.method() + " " + pathname);
    if (pathname.endsWith("/pool/late-requests") && request.method() === "POST") lateRequestWrites.push(pathname);
    if (request.method() === "GET" && /\/interclub\/(registrations|competition)\/[^/]+\/meets\//.test(pathname)) operationalReads.push(pathname);
  });
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
        const registrationReadsStart = operationalReads.length;
        const seasonResponse = page.waitForResponse(r => new URL(r.url()).pathname === seasonPath && r.request().method() === "GET");
        await invitationLink.click();
        const response = await seasonResponse;
        expect(response.status(), `${club.name}: ${season.details.name} season request`).toBe(200);
        const details = await response.json();
        // Only a closed registration window with explicit permission opens meet planning.
        // Older responses without registration metadata must remain locked.
        const meetPlanningOpen = details.season.registration?.status === "closed" && details.season.registration?.meet_planning_open === true;
        await expect(page.getByRole("heading", { name: season.details.name, exact: true })).toBeVisible();
        await expect(page.getByRole("heading", { name: "Season registration", exact: true })).toBeVisible();
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
          await expect(page.getByRole("region", { name: "Season player pool", exact: true })).toBeVisible();
          await expect(page.getByRole("navigation", { name: "League workflow", exact: true }).getByRole("link", { name: "Player pool", exact: true })).toHaveAttribute("aria-current", "step");
          // The pool opens with the season now. Reload explicitly so this waiter
          // observes a fresh scoped read even if the initial request completed.
          const reloadPool = page.getByRole("button", { name: "Reload player pool", exact: true });
          await expect(reloadPool).toBeEnabled();
          const poolResult = page.waitForResponse(r => new URL(r.url()).pathname === `${seasonPath}/pool` && r.request().method() === "GET");
          await reloadPool.click();
          const poolResponse = await poolResult;
          expect(poolResponse.status()).toBe(200);
          const pool = await poolResponse.json();
          expect(pool.members.every((member: {club_id: string; season_id: string}) => member.club_id === club.id && member.season_id === season.id)).toBe(true);
          await expect(page.getByRole("region", { name: "Season player pool", exact: true })).toBeVisible();
          if (meetPlanningOpen && pool.can_request_late && pool.members.some((member: { player_id: string | null }) => member.player_id)) {
            const poolPanel = page.getByRole("region", { name: "Season player pool", exact: true });
            const searchPath = `${seasonPath}/pool/players`;
            const initialSearch = page.waitForResponse(r => new URL(r.url()).pathname === searchPath && new URL(r.url()).searchParams.get("q") === "" && r.request().method() === "GET");
            await poolPanel.getByRole("button", { name: "Request late player", exact: true }).click();
            const initial = await initialSearch;
            expect(initial.status()).toBe(200);
            const candidates: { id: string; name: string }[] = (await initial.json()).players;
            const members: { id: string; name: string; player_id: string | null }[] = pool.members;
            // Match identities first. A unique pool name makes the accessible
            // Manage target unambiguous even when club profiles share a name.
            const candidate = candidates.find(player => members.some(member => String(member.player_id) === String(player.id)
              && members.filter(other => other.name === member.name).length === 1));
            const form = poolPanel.getByRole("region", { name: "Request a late player", exact: true });
            if (candidate) {
              const member = members.find(row => String(row.player_id) === String(candidate.id))!;
              const query = candidate.name.slice(0, 80);
              const searched = page.waitForResponse(r => new URL(r.url()).pathname === searchPath && new URL(r.url()).searchParams.get("q") === query && r.request().method() === "GET");
              await form.getByRole("searchbox", { name: "Find a late player in this club", exact: true }).fill(query);
              const found = await searched;
              expect(found.status()).toBe(200);
              const matches: { id: string }[] = (await found.json()).players;
              expect(matches.some(player => String(player.id) === String(candidate.id))).toBe(true);
              const existing = form.getByRole("region", { name: "Existing season signups", exact: true });
              await expect(existing.getByRole("radio", { includeHidden: true })).toHaveCount(0);
              // Count by returned player IDs, so a new namesake may still have a radio.
              await expect(form.getByRole("radio", { includeHidden: true })).toHaveCount(matches.filter(player => !members.some(row => String(row.player_id) === String(player.id))).length);
              await existing.getByRole("button", { name: `View ${member.name} in player pool`, exact: true }).click();
              await expect(form).toHaveCount(0);
              await expect(poolPanel.getByRole("searchbox", { name: "Find in player pool", exact: true })).toHaveValue(member.name);
              await expect(poolPanel.getByRole("combobox", { name: "Show players", exact: true })).toHaveValue("all");
              const manage = poolPanel.locator("summary").filter({ hasText: /^Manage$/ }).and(poolPanel.getByLabel(`Manage ${member.name}`, { exact: true }));
              await expect(manage).toBeFocused();
              await expect(manage.locator("..")).toHaveAttribute("open", "");
              await poolPanel.getByRole("searchbox", { name: "Find in player pool", exact: true }).fill("");
              await poolPanel.getByRole("combobox", { name: "Show players", exact: true }).selectOption("active");
            } else {
              // No suitable existing identity on this read-only search page.
              await form.getByRole("button", { name: "Close late player request", exact: true }).click();
            }
          }
        }
        if (!organizer && !joined) await expect(page.getByRole("combobox", { name: "Meet", exact: true })).toHaveCount(0);
        if (!meetPlanningOpen) {
          if (organizer || joined) await expectMeetStepsLocked(page);
          await expectNoMeetControls(page);
          expect(operationalReads.slice(registrationReadsStart), "Registration must not fetch meet details or availability before meet planning opens").toEqual([]);
        }
        if (meetPlanningOpen && details.meets.length && (organizer || joined)) {
          const workflow = page.getByRole("navigation", { name: "League workflow", exact: true });
          const lineups = workflow.getByRole("link", { name: "Lineups", exact: true });
          await lineups.click();
          await expect(lineups).toHaveAttribute("aria-current", "step");
          await expect(page.getByRole("region", { name: "Meet rosters", exact: true })).toBeFocused();
          if (joined) await expect(page.getByRole("region", { name: "Season player pool", exact: true })).toBeHidden();
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
            const availabilityStep = workflow.getByRole("link", { name: "Meet availability", exact: true });
            await availabilityStep.click();
            const availabilityResponse = await availabilityResult;
            expect(availabilityResponse.status()).toBe(200);
            const availability = await availabilityResponse.json();
            expect(availability.meet.id).toBe(meetId);
            await expect(availabilityStep).toHaveAttribute("aria-current", "step");
            await expect(page.getByRole("region", { name: "Meet player availability", exact: true })).toBeVisible();
          }
        }
        if (organizer || joined) {
          await test.step(`${season.details.name}: read-only meet operations and print layout`, async () => {
            const competitionPath = `/admin/clubs/${club.id}/interclub/competition/${season.id}`;
            const competitionReadsStart = operationalReads.length;
            const responses = new Map<string, Response>();
            const capture = (response: Response) => {
              const pathname = new URL(response.url()).pathname;
              if (response.request().method() === "GET" && pathname.startsWith(competitionPath + "/meets/")) responses.set(pathname, response);
            };
            page.on("response", capture);
            try {
              const listResult = page.waitForResponse(response => new URL(response.url()).pathname === `/admin/clubs/${club.id}/interclub/competition` && response.request().method() === "GET");
              const workspaceResult = page.waitForResponse(response => new URL(response.url()).pathname === competitionPath && response.request().method() === "GET");
              await page.goto(`/admin/interclub/competition?season=${encodeURIComponent(season.id)}`);
              expect((await listResult).status()).toBe(200);
              const competitionResponse = await workspaceResult;
              expect(competitionResponse.status(), `${club.name}: competition workspace must load`).toBe(200);
              const competition = await competitionResponse.json();
              expect(competition.season.id).toBe(season.id);
              expect(competition.is_organizer).toBe(organizer);
              expect(competition.meets.every((meet: { season_id: string; club_ids: string[] }) => meet.season_id === season.id && (organizer || meet.club_ids.includes(club.id)))).toBe(true);
              const meetIds = new Set(competition.meets.map((meet: { id: string }) => meet.id));
              expect(competition.batches.every((batch: { season_id: string; meet_id: string }) => batch.season_id === season.id && meetIds.has(batch.meet_id))).toBe(true);
              await expect(page.getByRole("heading", { name: "Meet operations", exact: true })).toBeVisible();
              await expect(page.getByRole("combobox", { name: "Season", exact: true })).toHaveValue(season.id);
              await expect(page.getByText(/^(Load failed|Unable to load|Loading meet operations)/)).toHaveCount(0);

              const competitionPlanningOpen = competition.season.registration?.status === "closed" && competition.season.registration?.meet_planning_open === true;
              if (!competitionPlanningOpen) {
                await expect(page.getByRole("heading", { name: "Meet planning opens after registration closes", exact: true })).toBeVisible();
                await expect(page.getByRole("link", { name: "Go to season player pool", exact: true })).toHaveAttribute("href", `/admin/interclub/registrations?season=${encodeURIComponent(season.id)}&step=pool`);
                await expectMeetStepsLocked(page);
                await expectNoMeetControls(page);
                await expect(page.getByRole("heading", { name: "Standings & Club Cup", exact: true, includeHidden: true })).toHaveCount(0);
                await expect(page.getByRole("combobox", { name: "Scheduled competition format", exact: true, includeHidden: true })).toHaveCount(0);
                await expect(page.getByRole("button", { name: "Print meet packet", exact: true, includeHidden: true })).toHaveCount(0);
                await expect(page.getByRole("button", { name: "Generate pairings", exact: true, includeHidden: true })).toHaveCount(0);
                await expect(page.getByText("Schedule a championship, qualifying playoff or additional meet", { exact: true })).toHaveCount(0);
                expect(operationalReads.slice(competitionReadsStart), "Direct competition navigation must read only season context while meet planning is locked").toEqual([]);
                return;
              }
              await expect(page.getByRole("heading", { name: "Standings & Club Cup", exact: true })).toBeVisible();

              for (const scheduled of competition.meets) {
                const phase = scheduled.competition_phase || "regular";
                const meetPath = `${competitionPath}/meets/${scheduled.id}/${phase}`;
                const selector = page.getByRole("combobox", { name: "Meet", exact: true });
                if (await selector.inputValue() !== scheduled.id) await selector.selectOption(scheduled.id);
                await expect.poll(() => responses.has(meetPath), { message: "The selected meet must fetch its own scoped competition document" }).toBe(true);
                const meetResponse = responses.get(meetPath)!;
                expect(meetResponse.status(), `${club.name}: ${scheduled.id} competition detail`).toBe(200);
                const operations = await meetResponse.json();
                expect(operations.meet.id).toBe(scheduled.id);
                expect(operations.meet.season_id).toBe(season.id);
                expect(operations.is_organizer).toBe(organizer);
                expect(operations.can_manage).toBe(organizer || scheduled.host_club_id === club.id);
                expect(operations.teams.every((team: { meet_id: string; club_id: string; roster: Record<string, unknown>[] }) => team.meet_id === scheduled.id && scheduled.club_ids.includes(team.club_id) && team.roster.every(player => !Object.hasOwn(player, "player_id") && !Object.hasOwn(player, "email")))).toBe(true);
                expect(Object.keys(operations.eligible_players || {}).every(id => scheduled.club_ids.includes(id))).toBe(true);
                for (const candidates of Object.values(operations.eligible_players || {}) as Record<string, unknown>[][]) {
                  expect(candidates.every(player => !Object.hasOwn(player, "email") && !Object.hasOwn(player, "player_id"))).toBe(true);
                }
                await expect(page.getByRole("combobox", { name: "Scheduled competition format", exact: true })).toHaveValue(phase);
                await expect(page.getByText("Loading this meet…", { exact: true })).toHaveCount(0);
                await expect(page.getByText(/^(Load failed|This meet could not be loaded)/)).toHaveCount(0);
                if (operations.batch) {
                  expect(operations.batch.meet_id).toBe(scheduled.id);
                  expect(operations.batch.phase).toBe(phase);
                  expect(operations.batch.document.meet_id).toBe(scheduled.id);
                  const print = page.getByRole("button", { name: "Print meet packet", exact: true });
                  await expect(print).toBeEnabled();
                  expect(await print.evaluate(element => getComputedStyle(element).cursor)).toBe("pointer");
                  expect((await print.boundingBox())!.height).toBeGreaterThanOrEqual(44);
                  const packet = page.locator('[class*="printPortal"]');
                  await expect(packet).toHaveCount(1);
                  await expect(packet).toBeHidden();
                  await page.emulateMedia({ media: "print" });
                  try {
                    await expect(packet).toBeVisible();
                    await expect(page.getByRole("heading", { name: "Meet operations", exact: true })).toBeHidden();
                    expect(await packet.textContent()).toContain("Court assignments");
                    expect(await packet.textContent()).toContain("Verified by club A");
                    expect(await packet.evaluate(element => element.getBoundingClientRect().width <= document.documentElement.clientWidth)).toBe(true);
                  } finally {
                    await page.emulateMedia({ media: "screen" });
                  }
                } else {
                  await expect(page.getByRole("button", { name: "Print meet packet", exact: true })).toHaveCount(0);
                  await expect(page.getByRole("heading", { name: new RegExp("^Prepare .+ pairings$") })).toBeVisible();
                }
                if (!operations.can_manage) await expect(page.getByRole("button", { name: "Generate pairings", exact: true })).toHaveCount(0);
                if (!operations.is_organizer) {
                  await expect(page.getByRole("button", { name: "Review official approval", exact: true })).toHaveCount(0);
                  await expect(page.getByText("Schedule a championship, qualifying playoff or additional meet", { exact: true })).toHaveCount(0);
                }
              }
            } finally {
              page.off("response", capture);
            }
          });
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
  const deniedCompetition = await context.request.get(`${expectedApiOrigin}/admin/clubs/tres_palapas/interclub/competition`, {
    headers: { Authorization: `Bearer ${token}` }, maxRedirects: 0,
  });
  expect(deniedCompetition.status()).toBe(403);
  expect(competitionWrites, "The competition acceptance check must never generate, save, submit, approve or alter fixture data").toEqual([]);
  expect(lateRequestWrites, "Viewing an existing signup must not create another late request").toEqual([]);
  await page.goto("/admin/login");
  const signedOut = page.waitForResponse(r => r.url().startsWith(`${expectedAuthOrigin}/auth/v1/logout`) && r.request().method() === "POST");
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  expect([200, 204]).toContain((await signedOut).status());
  await expect.poll(() => page.evaluate(() => localStorage.getItem("jupr_admin_session_v1") === null)).toBe(true);
});
