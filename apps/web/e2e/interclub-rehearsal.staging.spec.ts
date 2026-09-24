import { expect, test } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { MeetSignupBoard, PrivateMeetSignup } from "../lib/interclubMeetSignup";
import type { PoolMember } from "../lib/interclubPlayerPool";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

function expectLinkedPoolMember(members: PoolMember[], name: string, rating: number, gender: string) {
  const matching = members.filter(member => member.name === name);
  expect(matching.length, "Inline creation must save exactly one linked season signup").toBe(1);
  const member = matching[0];
  expect(Number(member.player_id)).toBeGreaterThan(0);
  expect(member.rating).toBe(rating);
  expect(member.gender).toBe(gender);
  expect(member.eligible_divisions).toContain("3.5");
  expect(member.status).toBe("active");
  return member;
}

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
  for (const label of ["Meet signup", "Lineups", "Run meet", "Approve results"]) {
    await expect(workflow.getByRole("link", { name: label, exact: true })).toHaveCount(0);
    await expect(workflow.locator('[aria-disabled="true"]').filter({ has: page.getByText(label, { exact: true }) })).toHaveCount(1);
  }
  await expect(page.getByRole("combobox", { name: "Meet", exact: true, includeHidden: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reload meet", exact: true, includeHidden: true })).toHaveCount(0);
  expect(lockedReads, "An open signup season must not fetch operational meet details even from a direct URL").toEqual([]);

  const signupPoolRoot = `${expectedApiOrigin}/admin/clubs/${club}/interclub/registrations/${signup.id}/pool`;
  const adminPlayerName = `Browser admin new ${state.run}`;
  const signupPool = page.getByRole("region", { name: "Season player pool", exact: true });
  await signupPool.getByRole("button", { name: "Create new player", exact: true }).click();
  const newPlayer = signupPool.getByRole("region", { name: "Create a new club player", exact: true });
  await newPlayer.getByLabel("Player name", { exact: true }).fill(adminPlayerName);
  await newPlayer.getByRole("spinbutton", { name: /^Starting JUPR/ }).fill("3.25");
  await newPlayer.getByLabel("Gender (optional)", { exact: true }).selectOption("female");
  await expect(newPlayer.getByLabel("Email (optional)", { exact: true })).toHaveValue("");
  const adminCreated = page.waitForResponse(r => r.url() === `${signupPoolRoot}/create-player` && r.request().method() === "POST");
  await newPlayer.getByRole("button", { name: "Create player and add to pool", exact: true }).click();
  const adminCreatedResponse = await adminCreated;
  expect(adminCreatedResponse.status()).toBe(200);
  const adminCreatedResult = await adminCreatedResponse.json();
  const adminMember = expectLinkedPoolMember(adminCreatedResult.pool.members, adminPlayerName, 3.25, "female");
  expect(adminCreatedResult.member.id).toBe(adminMember.id);
  expect(String(adminCreatedResult.member.player_id)).toBe(String(adminMember.player_id));
  expect(adminMember.email).toBe("");
  const adminPlayerRow = signupPool.getByRole("row").filter({ has: page.getByRole("rowheader").filter({ hasText: adminPlayerName }) });
  await expect(adminPlayerRow).toHaveCount(1);
  await expect(adminPlayerRow.getByRole("cell", { name: "3.25", exact: true }).first()).toBeVisible();
  await expect(adminPlayerRow.getByRole("cell", { name: "Women", exact: true })).toBeVisible();

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
  await expect(schedule.getByText(/Season dates:/)).toBeVisible();
  await schedule.getByLabel("Host club", { exact: true }).selectOption(club);
  for (const checkbox of await schedule.getByRole("checkbox").all()) {
    if (!(await checkbox.isChecked())) await checkbox.check();
  }
  await schedule.getByLabel("Meet date and time", { exact: true }).fill(localDate(new Date(Date.now()+11*86_400_000)));
  await schedule.getByLabel("Roster deadline", { exact: true }).fill(localDate(new Date(Date.now()+10*86_400_000)));
  const added = page.waitForResponse(r => r.url().endsWith(`/competition/${adjustments.id}/meets`) && r.request().method() === "POST");
  await schedule.getByRole("button", { name: "Add meet", exact: true }).click();
  const addedResponse = await added;
  expect(addedResponse.status()).toBe(200);
  const addedMeet = (await addedResponse.json()).meet;
  await expect(schedule.getByRole("row")).toHaveCount(4);
  const pool = page.getByRole("region", { name: "Season player pool", exact: true });
  await pool.getByRole("button", { name: "Request late player", exact: true }).click();
  const lateRequest = pool.getByRole("region", { name: "Request a late player", exact: true });
  await lateRequest.getByRole("button", { name: "Create new player", exact: true }).click();
  await lateRequest.getByLabel("Player name", { exact: true }).fill(adjustments.browser_late_player);
  await lateRequest.getByRole("spinbutton", { name: /^Starting JUPR/ }).fill("3.35");
  await lateRequest.getByLabel("Gender (optional)", { exact: true }).selectOption("male");
  await expect(lateRequest.getByLabel("Notes (optional)", { exact: true })).toHaveValue("");
  const lateCreated = page.waitForResponse(r => r.url() === `${expectedApiOrigin}/admin/clubs/${club}/interclub/registrations/${adjustments.id}/pool/late-requests` && r.request().method() === "POST");
  await lateRequest.getByRole("button", { name: "Create player and request late entry", exact: true }).click();
  const lateCreatedResponse = await lateCreated;
  expect(lateCreatedResponse.status()).toBe(200);
  const lateCreatedResult = await lateCreatedResponse.json();
  const lateMember = expectLinkedPoolMember(lateCreatedResult.pool.members, adjustments.browser_late_player, 3.35, "male");
  expect(lateCreatedResult.member.id).toBe(lateMember.id);
  expect(String(lateCreatedResult.member.player_id)).toBe(String(lateMember.player_id));
  expect(lateMember.approval_status).toBe("pending");
  expect(lateMember.late_join).toBe(true);
  expect(lateMember.late_request_reason || "").toBe("");
  const latePlayerRow = pool.getByRole("row").filter({ has: page.getByRole("rowheader").filter({ hasText: adjustments.browser_late_player }) });
  await expect(latePlayerRow.getByRole("cell", { name: "Pending · cannot play", exact: true })).toBeVisible();
  const approvals = page.getByRole("region", { name: "Season eligibility approvals", exact: true });
  const applicant = approvals.getByRole("article").filter({ has: page.getByRole("heading", { name: adjustments.browser_late_player, exact: true }) });
  await expect(applicant.getByText(/Pending approval/)).toBeVisible();
  await applicant.getByLabel("Decision reason", { exact: true }).fill("Commissioner approves the late commitment.");
  const lateApproved = page.waitForResponse(r => r.url() === `${expectedApiOrigin}/admin/clubs/${club}/interclub/registrations/${adjustments.id}/pool/approvals` && r.request().method() === "POST");
  await applicant.getByRole("button", { name: "Approve season eligibility", exact: true }).click();
  const lateApprovedResponse = await lateApproved;
  expect(lateApprovedResponse.status()).toBe(200);
  const approvedMember = (await lateApprovedResponse.json()).member;
  expect(approvedMember.id).toBe(lateMember.id);
  expect(approvedMember.approval_status).toBe("approved");
  await expect(approvals.getByRole("status").filter({ hasText: `${adjustments.browser_late_player}: approved for the season pool.` })).toBeVisible();
  await page.screenshot({ path: join(reportDir, "interclub-season-adjustments.png"), fullPage: true });

  // Follow the same path an administrator uses after adding a meet. No
  // availability invitation is needed for verbally confirmed players.
  await schedule.getByRole("link", { name: "Choose players for this meet", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "League workflow" }).getByRole("link", { name: "Lineups", exact: true })).toHaveAttribute("aria-current", "step");
  await expect(page.getByLabel("Meet", { exact: true })).toHaveValue(addedMeet.id);
  const playerResponse = page.waitForResponse(r => r.url().includes(`/registrations/${adjustments.id}/meets/`) && new URL(r.url()).pathname.endsWith("/players"));
  await page.getByRole("button", { name: "Add a team for this meet", exact: true }).click();
  const availablePlayers = (await (await playerResponse).json()).players as { id: string; name: string; eligibility_rating: number; gender: string }[];
  const roster = page.getByRole("form", { name: "Choose meet players", exact: true });
  await roster.getByLabel("Lineup division", { exact: true }).selectOption("3.5");
  await expect(roster.getByLabel("Player eligibility filter", { exact: true })).toHaveValue("eligible");
  const selectedPlayers = ["female", "male"].flatMap(gender => availablePlayers.filter(player => player.gender === gender && player.eligibility_rating > 0 && player.eligibility_rating < 4).slice(0, 2));
  expect(selectedPlayers).toHaveLength(4);
  for (const player of selectedPlayers) await roster.getByRole("checkbox", { name: `Select ${player.name}`, exact: true }).check();
  await expect(roster.getByRole("status", { name: "Lineup selection" })).toContainText("2 of 2 women · 2 of 2 men");
  await page.screenshot({ path: join(reportDir, "interclub-guided-player-picker.png"), fullPage: true });
  await roster.getByRole("button", { name: "Review season player pool", exact: true }).click();
  await expect(pool).toBeVisible();
  await page.getByRole("navigation", { name: "League workflow" }).getByRole("link", { name: "Lineups", exact: true }).click();
  for (const player of selectedPlayers) await expect(roster.getByRole("checkbox", { name: `Select ${player.name}`, exact: true })).toBeChecked();
  const rosterSaved = page.waitForResponse(r => r.url().includes(`/registrations/${adjustments.id}/meets/`) && r.url().includes("/teams/") && r.request().method() === "PUT");
  await roster.getByRole("button", { name: "Submit four-player roster", exact: true }).click();
  const rosterResult = await rosterSaved;
  expect(rosterResult.status()).toBe(200);
  const savedTeam = (await rosterResult.json()).team;
  expect(savedTeam.status).toBe("eligible");
  expect(savedTeam.roster.map((player: { player_id: string }) => String(player.player_id)).sort()).toEqual(selectedPlayers.map(player => String(player.id)).sort());
  await expect(page.getByRole("status").filter({ hasText: "Lineup saved for this meet." })).toBeVisible();
  await page.screenshot({ path: join(reportDir, "interclub-guided-lineup.png"), fullPage: true });

  // Account-free signup and concurrent allocation into the actual lineup.
  const queueMeet = adjustments.meets[1];
  const queueRoot = `${expectedApiOrigin}/admin/clubs/${club}/interclub/registrations/${adjustments.id}/meets/${queueMeet.id}`;
  await page.goto(`/admin/interclub/registrations?season=${adjustments.id}&meet=${queueMeet.id}&step=availability`);
  const signupPanel = page.getByRole("region", { name: "Meet signup and substitute pool", exact: true });
  const openedSignup = page.waitForResponse(r => r.url() === `${queueRoot}/signup` && r.request().method() === "PUT");
  await signupPanel.getByRole("button", { name: "Open meet signup", exact: true }).click();
  const openedResponse = await openedSignup;
  expect(openedResponse.status()).toBe(200);
  const openedBoard: MeetSignupBoard = await openedResponse.json();
  const shareUrl = openedBoard.signup.url!;
  await expect(signupPanel.getByLabel("Share this meet signup link with your players", { exact: true })).toHaveValue(shareUrl);
  // Register directly from a vacancy, then review the declared gender separately.
  await signupPanel.getByRole("button", { name: "Add player to 3.5 women", exact: true }).click();
  const picker = signupPanel.getByRole("region", { name: "Choose player for 3.5 women", exact: true });
  const candidates = picker.getByRole("list", { name: "Eligible players by rating", exact: true });
  await expect(candidates).toBeVisible();
  const candidateText = await candidates.getByRole("button").allTextContents();
  const candidateRatings = candidateText.map(text => Number(text.match(/League rating ([0-9.]+)/)![1]));
  expect(candidateRatings).toEqual([...candidateRatings].sort((a, b) => b - a));
  expect(candidateRatings.every(rating => rating > 0 && rating < 4)).toBe(true);
  await candidates.getByRole("button").first().click();
  await picker.getByLabel("Gender", { exact: true }).selectOption("non_binary");
  await expect(picker.getByRole("status")).toContainText("An admin will review");
  // Signup mutations refresh the parent meet and remount this panel. Wait for the
  // new board before expanding Manage, or that refresh can close the form again.
  const addedBoardRefresh = page.waitForResponse(r => r.url() === `${queueRoot}/signup` && r.request().method() === "GET");
  const manualAdd = page.waitForResponse(r => r.url() === `${queueRoot}/signup/actions` && r.request().method() === "POST");
  await picker.getByRole("button", { name: "Submit for admin review", exact: true }).click();
  const manualResponse = await manualAdd;
  expect(manualResponse.status()).toBe(200);
  const manualBoard: MeetSignupBoard = await manualResponse.json();
  const reviewEntry = manualBoard.entries.find(entry => entry.declared_gender === "non_binary")!;
  expect(reviewEntry.placement).toBe("review");
  expect((await addedBoardRefresh).status()).toBe(200);
  await signupPanel.getByText(`Manage ${reviewEntry.name}`, { exact: true }).click();
  const reviewForm = signupPanel.getByRole("form", { name: `Review placement for ${reviewEntry.name}` });
  await expect(reviewForm).toBeVisible();
  const lineupPlace = reviewForm.getByLabel("Lineup place", { exact: true });
  await expect(lineupPlace).toBeVisible();
  await lineupPlace.selectOption("female");
  const reviewedBoardRefresh = page.waitForResponse(r => r.url() === `${queueRoot}/signup` && r.request().method() === "GET");
  const reviewed = page.waitForResponse(r => r.url() === `${queueRoot}/signup/actions` && r.request().method() === "POST");
  await reviewForm.getByRole("button", { name: "Approve placement", exact: true }).click();
  const reviewedResponse = await reviewed;
  expect(reviewedResponse.status()).toBe(200);
  const approved = ((await reviewedResponse.json()) as MeetSignupBoard).entries.find(entry => entry.id === reviewEntry.id)!;
  expect(approved.declared_gender).toBe("non_binary");
  expect(approved.reviewed_gender).toBe("female");
  expect(approved.placement).toBe("confirmed");
  expect((await reviewedBoardRefresh).status()).toBe(200);
  await expect(reviewForm).toHaveCount(0);
  await signupPanel.getByText(`Manage ${reviewEntry.name}`, { exact: true }).click();
  const removed = page.waitForResponse(r => r.url() === `${queueRoot}/signup/actions` && r.request().method() === "POST");
  await signupPanel.getByRole("button", { name: "Remove signup", exact: true }).click();
  expect((await removed).status()).toBe(200);
  const meetAnonymous = await browser.newContext({ baseURL: origin });
  await bootstrapStagingContext(meetAnonymous);
  const meetPage = await meetAnonymous.newPage();
  meetPage.on("pageerror", error => errors.push(error.message));
  await meetPage.goto(new URL(shareUrl).pathname);
  await meetPage.getByLabel("Find your name in the approved season pool", { exact: true }).fill(adjustments.browser_late_player);
  // Choosing a profile replaces the radio list with the selected profile card.
  await meetPage.getByRole("radio", { name: new RegExp(adjustments.browser_late_player) }).click();
  await meetPage.getByLabel("Gender", { exact: true }).selectOption("female");
  await expect(meetPage.getByText(/Your 3.35 league rating is below 3.5/)).toBeVisible();
  await meetPage.getByRole("checkbox", { name: /This is my profile and I want to play/ }).check();
  const shareId = new URL(shareUrl).pathname.split("/").at(-1)!;
  const joinUrl = `${expectedApiOrigin}/public/interclub-meet-signups/${shareId}`;
  const joined = meetPage.waitForResponse(r => r.url() === joinUrl && r.request().method() === "POST");
  await meetPage.getByRole("button", { name: "Sign me up for this meet", exact: true }).click();
  const joinedResponse = await joined;
  expect(joinedResponse.status()).toBe(200);
  const playUp: PrivateMeetSignup = await joinedResponse.json();
  expect(playUp.entry.priority).toBe("play_up");
  expect(playUp.entry.placement).toBe("waitlist");
  await expect(meetPage.getByRole("region", { name: "Your meet signup" })).toContainText("Substitute #1 · Playing up");

  const ratedPlayers = adjustments.players.filter((player: { club_id: string; division: string }) => player.club_id === club && player.division === "3.5");
  expect(ratedPlayers).toHaveLength(6);
  const joins = await Promise.all(ratedPlayers.map(async (player: { id: number; name: string }) => {
    const details = { player_id: player.id, name: player.name, division: "3.5", confirm_self: true, request_id: randomUUID() };
    const response = await meetAnonymous.request.post(joinUrl, { data: details });
    expect(response.status()).toBe(200);
    return { details, result: await response.json() as PrivateMeetSignup };
  }));
  const boardResponse = await meetAnonymous.request.get(joinUrl);
  expect(boardResponse.status()).toBe(200);
  const board: MeetSignupBoard = await boardResponse.json();
  expect(JSON.stringify(board)).not.toMatch(/token_nonce|manage_url|@example.invalid/);
  for (const gender of ["female", "male"]) {
    const ordered = board.entries.filter(entry => entry.gender === gender && entry.priority === "in_band");
    expect(ordered.map(entry => entry.placement)).toEqual(["confirmed", "confirmed", "waitlist"]);
    expect(ordered[2].queue_position).toBe(1);
  }
  expect(board.entries.find(entry => entry.id === playUp.entry.id)?.queue_position).toBe(2);
  const sameRetry = await meetAnonymous.request.post(joinUrl, { data: joins[0].details });
  expect((await sameRetry.json()).entry.manage_url).toBe(joins[0].result.entry.manage_url);
  const duplicate = await meetAnonymous.request.post(joinUrl, { data: { ...joins[0].details, request_id: randomUUID() } });
  expect((await duplicate.json()).duplicate).toBe(true);
  const firstWoman = board.entries.find(entry => entry.gender === "female" && entry.placement === "confirmed")!;
  const womanSignup = joins.find((join: { result: PrivateMeetSignup }) => join.result.entry.id === firstWoman.id)!.result;
  const nextWoman = board.entries.find(entry => entry.gender === "female" && entry.placement === "waitlist" && entry.priority === "in_band" && entry.queue_position === 1)!;
  expect(nextWoman.id).not.toBe(playUp.entry.id);
  const womanPage = await meetAnonymous.newPage();
  womanPage.on("pageerror", error => errors.push(error.message));
  const privateAddress = new URL(womanSignup.entry.manage_url!);
  await womanPage.goto(privateAddress.pathname + privateAddress.hash);
  await expect(womanPage.getByRole("region", { name: "Your meet signup" })).toContainText(firstWoman.name);
  expect(new URL(womanPage.url()).hash).toBe("");
  const withdrawal = womanPage.waitForResponse(r => r.url() === `${expectedApiOrigin}/public/interclub-meet-signups/withdraw` && r.request().method() === "POST");
  await womanPage.getByRole("button", { name: "Withdraw from this meet", exact: true }).click();
  const withdrawn = await withdrawal;
  expect(withdrawn.status()).toBe(200);
  const afterWithdrawal: PrivateMeetSignup = await withdrawn.json();
  expect(afterWithdrawal.entry.status).toBe("withdrawn");
  await expect(womanPage.getByRole("region", { name: "Your meet signup" })).toContainText("Withdrawn");
  expect(afterWithdrawal.entries.find(entry => entry.id === nextWoman.id)?.placement).toBe("confirmed");
  expect(afterWithdrawal.entries.find(entry => entry.id === playUp.entry.id)?.placement).toBe("waitlist");
  const lineupRead = await context.request.get(queueRoot, { headers: { Authorization: `Bearer ${user.token}` } });
  expect(lineupRead.status()).toBe(200);
  const autoTeam = (await lineupRead.json()).teams.find((team: { club_id: string; withdrawn: boolean }) => team.club_id === club && !team.withdrawn);
  expect(autoTeam.status).toBe("eligible");
  expect(autoTeam.roster.map((player: { name: string }) => player.name)).toContain(nextWoman.name);
  expect(autoTeam.roster.map((player: { name: string }) => player.name)).not.toContain(firstWoman.name);
  await meetPage.setViewportSize({ width: 390, height: 844 });
  await meetPage.getByRole("button", { name: "Refresh signup", exact: true }).click();
  await expect(meetPage.getByRole("region", { name: "Your meet signup" })).toContainText("Substitute #1 · Playing up");
  await meetPage.screenshot({ path: join(reportDir, "interclub-meet-signup-mobile.png"), fullPage: true });
  await signupPanel.getByRole("button", { name: "Refresh signups and lineups", exact: true }).click();
  await expect(signupPanel.getByRole("region", { name: "3.5 women", exact: true })).toContainText(nextWoman.name);
  await page.screenshot({ path: join(reportDir, "interclub-meet-signup-admin.png"), fullPage: true });
  await meetAnonymous.close();

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
  const publicPlayerName = `Browser public new ${state.run}`;
  await publicPage.getByLabel("Your name", { exact: true }).fill(publicPlayerName);
  const publicNewPlayer = publicPage.getByRole("group", { name: "New player profile", exact: true });
  await expect(publicNewPlayer).toBeVisible();
  await publicNewPlayer.getByLabel("Starting JUPR", { exact: true }).fill("3.15");
  await publicNewPlayer.getByLabel("Gender", { exact: true }).selectOption("female");
  await publicPage.getByRole("textbox", { name: /^Email/ }).fill(`browser-${state.run}@example.invalid`);
  await publicPage.getByRole("checkbox",{ name:"3.5", exact:true }).check();
  await publicPage.getByRole("checkbox",{ name:/My club can email me invitations/ }).check();
  const publicCreated = publicPage.waitForResponse(r => r.url() === `${expectedApiOrigin}/public/interclub-signups/${signup.signup[club].share_id}` && r.request().method() === "POST");
  await publicPage.getByRole("button",{ name:"Join the season player pool", exact:true }).click();
  const publicCreatedResponse = await publicCreated;
  expect(publicCreatedResponse.status()).toBe(200);
  expect((await publicCreatedResponse.json()).status).toBe("registered");
  await expect(publicPage.getByRole("heading",{ name:"Your interest is registered", exact:true })).toBeVisible();
  await expect(publicPage.getByRole("link",{ name:"Manage my season signup", exact:true })).toHaveCount(1);
  await anonymous.close();
  const refreshedPool = page.waitForResponse(r => r.url() === signupPoolRoot && r.request().method() === "GET");
  await page.goto(`/admin/interclub/registrations?season=${signup.id}`);
  const refreshedPoolResponse = await refreshedPool;
  expect(refreshedPoolResponse.status()).toBe(200);
  const persistedMembers = (await refreshedPoolResponse.json()).members;
  const publicMember = expectLinkedPoolMember(persistedMembers, publicPlayerName, 3.15, "female");
  expect(publicMember.email).toBe(`browser-${state.run}@example.invalid`);
  expect(String(publicMember.player_id)).not.toBe(String(adminMember.player_id));
  expect(expectLinkedPoolMember(persistedMembers, adminPlayerName, 3.25, "female").id).toBe(adminMember.id);
  const publicPlayerRow = signupPool.getByRole("row").filter({ has: page.getByRole("rowheader").filter({ hasText: publicPlayerName }) });
  await expect(publicPlayerRow).toHaveCount(1);
  await expect(publicPlayerRow.getByRole("cell", { name: "3.15", exact: true }).first()).toBeVisible();
  expect(errors).toEqual([]);
  writeFileSync(join(reportDir,"interclub-browser.json"),JSON.stringify({ status:"passed",candidate_sha:state.sha,
    checks:["shared_meet_signup","play_up_waitlist","concurrent_signup_capacity","rating_band_fifo","signup_retry_identity","withdrawal_promotes_actual_roster","mobile_signup","paper_packet_pdf","six_game_ui_entry","dirty_navigation_lock","draft_reload","whole_meet_submission","organizer_approval","both_rating_streams","registration_phase_route_lock","admin_inline_player_creation","upcoming_meet_edit","add_meet_after_registration","late_inline_player_creation_without_notes","late_player_request_and_approval","guided_meet_lineup","eligible_player_filter","gender_composition","lineup_draft_preserved_on_pool_visit","anonymous_public_cup","closed_signup_readonly","anonymous_inline_player_signup","persisted_inline_profile_ratings","no_browser_exceptions"] },null,2));
});
