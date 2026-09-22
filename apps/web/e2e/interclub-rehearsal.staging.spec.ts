import { expect, test } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

// This private fixture stays outside uploaded artifacts. The session was issued
// by staging Auth to a synthetic user restricted to this run's synthetic clubs.
test.use({ trace: "off", video: "off", screenshot: "off", timezoneId: "UTC" });
test("interclub paper packet, score entry, approval and public results", async ({ page, context, browser }) => {
  test.setTimeout(300_000);
  const state = JSON.parse(readFileSync(process.env.JUPR_INTERCLUB_REHEARSAL_STATE!, "utf8"));
  expect(state.marker).toBe("interclub-functional-rehearsal-v1");
  expect(state.sha).toBe(process.env.GITHUB_SHA);
  const origin = process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN!;
  const user = state.users[0];
  const club = state.clubs[0];
  const season = state.seasons.find((s: { label: string }) => s.label === "incidents");
  const official = state.seasons.find((s: { label: string }) => s.label === "full-season");
  const signup = state.seasons.find((s: { label: string }) => s.label === "browser-signup");
  expect(season.registration.status).toBe("closed");
  expect(official.registration.status).toBe("closed");
  expect(signup.registration.status).toBe("open");
  const reportDir = process.env.JUPR_INTERCLUB_REHEARSAL_REPORT!;
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await bootstrapStagingContext(context);
  await context.addInitScript(({ token: access_token, email, origin }) => {
    if (location.origin === origin) localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
      access_token, token_type: "bearer", user: { email },
    }));
  }, { token: user.token, email: user.email, origin });
  await context.addCookies([{ name: "jupr_admin_workspace_v1", value: encodeURIComponent(JSON.stringify({ clubId: club, clubSlug: club })),
    url: origin, secure: true, sameSite: "Lax" }]);
  const route = `/admin/interclub/competition?season=${season.id}&meet=${season.browser_meet}`;
  await page.goto(route);
  await expect(page.getByRole("heading", { name: "Meet score draft", exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Review and submit meet", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Print meet packet", exact: true })).toBeEnabled();
  await page.emulateMedia({ media: "print" });
  await page.pdf({ path: join(reportDir, "interclub-paper-packet.pdf"), format: "A4", printBackground: true });
  await page.emulateMedia({ media: "screen" });
  const labels: Record<string,string> = { women: "Women’s doubles", men: "Men’s doubles" };
  const played = new Date(Date.now()-60_000).toISOString().slice(0,16);
  for (const encounter of season.browser_batch.document.encounters) {
    for (const pairing of encounter.pairings) {
      for (let i=0;i<pairing.games.length;i++) {
        await page.getByLabel(`${labels[pairing.kind]} game ${i+1} status`, { exact: true }).selectOption("completed");
        await page.getByLabel(`${pairing.id} game ${i+1} club A score`, { exact: true }).fill("11");
        await page.getByLabel(`${pairing.id} game ${i+1} club B score`, { exact: true }).fill("9");
      }
    }
  }
  for (const input of await page.getByLabel("Actual time played (your device’s time)", { exact: true }).all()) await input.fill(played);
  await expect(page.getByRole("combobox", { name: /^Season/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Print meet packet", exact: true })).toBeDisabled();
  const apiRoot = `${expectedApiOrigin}/admin/clubs/${club}/interclub/competition/${season.id}/meets/${season.browser_meet}/regular`;
  const save = page.waitForResponse(r => r.url() === apiRoot && r.request().method() === "PUT");
  await page.getByRole("button", { name: "Save all draft scores", exact: true }).first().click();
  expect((await save).status()).toBe(200);
  await expect(page.getByText("All draft changes saved", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Review and submit meet", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Review and submit meet", exact: true }).click();
  await page.getByRole("button", { name: "Submit all official scores", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Awaiting organizer approval", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Review official approval", exact: true }).click();
  const approval = page.waitForResponse(r => r.url() === apiRoot+"/approve" && r.request().method() === "POST", { timeout: 90_000 });
  await page.getByRole("button", { name: "Approve this revision", exact: true }).click();
  const result = await approval;
  expect(result.status()).toBe(200);
  expect((await result.json()).ratings.status).toBe("completed");
  await expect(page.getByRole("heading", { name: "Official meet results", exact: true })).toBeVisible();
  await expect(page.getByText("Rating updates: completed", { exact: true })).toBeVisible();
  await page.screenshot({ path: join(reportDir,"interclub-approved-meet.png"), fullPage: true });

  const lockedReads: string[] = [];
  page.on("request", request => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && ["registrations", "competition"].some(area => path.startsWith(`/admin/clubs/${club}/interclub/${area}/${signup.id}/meets/`))) lockedReads.push(path);
  });
  const signupContext = page.waitForResponse(r => new URL(r.url()).pathname === `/admin/clubs/${club}/interclub/competition/${signup.id}` && r.request().method() === "GET");
  await page.goto(`/admin/interclub/competition?season=${signup.id}&meet=${signup.meets[0].id}`);
  const signupResponse = await signupContext;
  expect(signupResponse.status()).toBe(200);
  expect((await signupResponse.json()).season.id).toBe(signup.id);
  await expect(page.getByRole("heading", { name: "Meet planning opens after registration closes", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Standings & Club Cup", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("combobox", { name: "Meet", exact: true, includeHidden: true })).toHaveCount(0);
  await page.getByRole("link", { name: "Go to season player pool", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Season registration", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Season player pool", exact: true })).toBeVisible();
  const workflow = page.getByRole("navigation", { name: "League workflow", exact: true });
  await expect(workflow.getByRole("link", { name: "Player pool", exact: true })).toHaveAttribute("aria-current", "step");
  for (const label of ["Meet availability", "Lineups", "Run meet", "Approve results"]) {
    await expect(workflow.getByRole("link", { name: label, exact: true })).toHaveCount(0);
    await expect(workflow.locator('[aria-disabled="true"]').filter({ has: page.getByText(label, { exact: true }) })).toHaveCount(1);
  }
  await expect(page.getByRole("combobox", { name: "Meet", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reload meet", exact: true, includeHidden: true })).toHaveCount(0);
  expect(lockedReads, "An open signup season must not fetch operational meet details even from a direct URL").toEqual([]);

  const adjustments = state.seasons.find((s: { label: string }) => s.label === "season-adjustments");
  await page.goto(`/admin/interclub/registrations?season=${adjustments.id}`);
  const schedule = page.getByRole("region", { name: "Meet schedule", exact: true });
  await expect(schedule.getByRole("button", { name: "Add meet", exact: true })).toBeEnabled();
  await expect(schedule.getByRole("row")).toHaveCount(3);
  await schedule.getByRole("button", { name: /^Edit date for/ }).first().click();
  const localDate = (date: Date) => new Intl.DateTimeFormat("sv-SE", { timeZone: "America/Mazatlan", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    .format(date).replace(" ", "T");
  await schedule.getByLabel("Meet date and time", { exact: true }).fill(localDate(new Date(Date.parse(adjustments.meets[0].starts_at)+86_400_000)));
  const rescheduled = page.waitForResponse(r => r.url().endsWith(`/meets/${adjustments.meets[0].id}/schedule`) && r.request().method() === "PUT");
  await schedule.getByRole("button", { name: "Save meet date", exact: true }).click();
  expect((await rescheduled).status()).toBe(200);
  await expect(schedule.getByRole("status").filter({ hasText: "Meet schedule saved." })).toBeVisible();
  await schedule.getByRole("button", { name: "Add meet", exact: true }).click();
  await expect(schedule.getByLabel("Competition", { exact: true })).toHaveValue("regular");
  await schedule.getByLabel("Host club", { exact: true }).selectOption(club);
  for (const checkbox of await schedule.getByRole("checkbox").all()) {
    if (!(await checkbox.isChecked())) await checkbox.check();
  }
  await schedule.getByLabel("Meet date and time", { exact: true }).fill(localDate(new Date(Date.now()+11*86_400_000)));
  await schedule.getByLabel("Roster deadline", { exact: true }).fill(localDate(new Date(Date.now()+10*86_400_000)));
  const added = page.waitForResponse(r => r.url().endsWith(`/competition/${adjustments.id}/meets`) && r.request().method() === "POST");
  await schedule.getByRole("button", { name: "Add meet", exact: true }).click();
  expect((await added).status()).toBe(200);
  await expect(schedule.getByRole("row")).toHaveCount(4);
  const pool = page.getByRole("region", { name: "Season player pool", exact: true });
  await pool.getByRole("button", { name: "Request late player", exact: true }).click();
  await pool.getByLabel("Find a late player in this club", { exact: true }).fill(adjustments.browser_late_player);
  await pool.getByRole("radio", { name: new RegExp(adjustments.browser_late_player) }).check();
  await pool.getByLabel("Reason for late entry", { exact: true }).fill("Player committed verbally after the season registration closed.");
  await pool.getByRole("button", { name: "Submit late player request", exact: true }).click();
  const approvals = page.getByRole("region", { name: "Season eligibility approvals", exact: true });
  const applicant = approvals.getByRole("article").filter({ has: page.getByRole("heading", { name: adjustments.browser_late_player, exact: true }) });
  await expect(applicant.getByText(/Pending approval/)).toBeVisible();
  await applicant.getByLabel("Decision reason", { exact: true }).fill("Commissioner approves the late commitment.");
  await applicant.getByRole("button", { name: "Approve season eligibility", exact: true }).click();
  await expect(approvals.getByRole("status").filter({ hasText: `${adjustments.browser_late_player}: approved for the season pool.` })).toBeVisible();
  await page.screenshot({ path: join(reportDir, "interclub-season-adjustments.png"), fullPage: true });

  const anonymous = await browser.newContext({ baseURL: origin });
  await bootstrapStagingContext(anonymous);
  const publicPage = await anonymous.newPage();
  publicPage.on("pageerror", error => errors.push(error.message));
  await publicPage.goto(`/interclub/${official.id}`);
  await expect(publicPage.getByRole("heading",{ name: `Rehearsal ${state.run} full-season`, exact:true })).toBeVisible();
  await expect(publicPage.getByText("Club Cup", { exact: false }).first()).toBeVisible();
  await expect(publicPage.getByText("Private revised note", { exact: false })).toHaveCount(0);
  await publicPage.goto(`/interclub/signup/${official.signup[club].share_id}`);
  await expect(publicPage.getByRole("heading", { name: "Season registration has closed.", exact: true })).toBeVisible();
  await expect(publicPage.getByRole("button", { name: "Join the season player pool", exact: true, includeHidden: true })).toHaveCount(0);
  await publicPage.goto(`/interclub/signup/${signup.signup[club].share_id}`);
  await publicPage.getByLabel("Your name", { exact: true }).fill("Browser Rehearsal Player");
  await publicPage.getByRole("textbox", { name: /^Email/ }).fill(`browser-${state.run}@example.invalid`);
  await publicPage.getByRole("checkbox",{ name:"3.5", exact:true }).check();
  await publicPage.getByRole("checkbox",{ name:/My club can email me invitations/ }).check();
  await publicPage.getByRole("button",{ name:"Join the season player pool", exact:true }).click();
  await expect(publicPage.getByRole("heading",{ name:"Your interest is registered", exact:true })).toBeVisible();
  await expect(publicPage.getByRole("link",{ name:"Manage my season signup", exact:true })).toHaveCount(1);
  await anonymous.close();
  expect(errors).toEqual([]);
  writeFileSync(join(reportDir,"interclub-browser.json"),JSON.stringify({ status:"passed",candidate_sha:state.sha,
    checks:["paper_packet_pdf","six_game_ui_entry","dirty_navigation_lock","draft_reload","whole_meet_submission","organizer_approval","both_rating_streams","registration_phase_route_lock","upcoming_meet_edit","add_meet_after_registration","late_player_request_and_approval","anonymous_public_cup","closed_signup_readonly","anonymous_player_signup","no_browser_exceptions"] },null,2));
});
