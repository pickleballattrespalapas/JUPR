import { writeFileSync } from "node:fs";

import { expect, test, type Locator, type Page, type Response } from "@playwright/test";

import {
  bootstrapStagingContext,
  expectedApiOrigin,
  expectedAuthOrigin
} from "./support/staging";

const clubId = "tres_palapas";
const allowMutation = /^(1|true|yes|on)$/i.test(
  String(process.env.JUPR_FOUR_WEEK_ALLOW_MUTATION_E2E || "")
);
const mutationConfirmation = String(
  process.env.JUPR_PARITY_MUTATION_CONFIRMATION || ""
).trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "").trim();
const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const expectedWebOrigin = String(
  process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN || ""
).trim().replace(/\/$/, "");
const candidateSha = String(process.env.GITHUB_SHA || "").trim().toLowerCase();
const leagueName = String(process.env.JUPR_FOUR_WEEK_E2E_LEAGUE_NAME || "").trim();
const aspenName = String(process.env.JUPR_FOUR_WEEK_E2E_ASPEN_NAME || "").trim();
const birchName = String(process.env.JUPR_FOUR_WEEK_E2E_BIRCH_NAME || "").trim();
const cloverName = String(process.env.JUPR_FOUR_WEEK_E2E_CLOVER_NAME || "").trim();
const reportPath = String(process.env.JUPR_FOUR_WEEK_E2E_REPORT_PATH || "").trim();

const rosterPlayerIds = [
  991001, 991002, 991003, 991004, 991005, 991007, 991008, 991023
] as const;
const awardLabels = [
  "Highest Rating",
  "Most Improved",
  "Best Win Percentage",
  "Most Wins"
] as const;
const awardKeys = [
  "highest_rating",
  "most_improved",
  "best_win_pct",
  "most_wins"
] as const;
const weeks = [
  { week: 1, date: "2026-08-24", rosterAttendance: 8, attendance: 8, courtSizes: [4, 4], matchCount: 6, pasted: [] as string[], existingPlayerIds: [] as number[] },
  { week: 2, date: "2026-08-31", rosterAttendance: 7, attendance: 9, courtSizes: [4, 5], matchCount: 8, pasted: [aspenName], existingPlayerIds: [991016] },
  { week: 3, date: "2026-09-07", rosterAttendance: 8, attendance: 12, courtSizes: [4, 4, 4], matchCount: 9, pasted: [aspenName, birchName, cloverName], existingPlayerIds: [991017] },
  { week: 4, date: "2026-09-14", rosterAttendance: 8, attendance: 14, courtSizes: [4, 5, 5], matchCount: 13, pasted: [aspenName, birchName, cloverName], existingPlayerIds: [991016, 991017, 991019] }
] as const;

type LeagueSummary = {
  league_id?: string | number | null;
  league_name: string;
  league_type?: string | null;
  status: string;
  min_games?: number | null;
  schedule_config?: Record<string, unknown>;
  court_board_defaults?: Record<string, unknown>;
  rules_config?: Record<string, unknown>;
  awards_config?: Record<string, unknown>;
};
type LeagueDetail = {
  league: LeagueSummary;
  roster?: Array<{ player_id: number; player_name: string; in_league: boolean }>;
  schedule_preview?: Array<{ session: number; date: string }>;
};
type LiveSession = {
  id: string;
  league_name: string;
  week_tag: string;
  status: string;
  current_round: number;
  total_rounds: number;
};
type LiveDetail = {
  session: LiveSession;
  rounds: Array<{
    round_number: number;
    status: string;
    submitted_match_count?: number;
  }>;
  courts?: Array<{ court_number: number; format_type: string; player_names: string[] }>;
  publish_operations?: Array<{
    round_number: number;
    status: string;
    published_match_ids?: string[];
  }>;
};
type AwardsState = {
  league?: LeagueSummary;
  awards?: Array<{ category_key: string }>;
  award_count?: number;
  writes_enabled?: boolean;
  service_role_ready?: boolean;
  badge_definitions_ready?: boolean;
  badge_expected_count?: number;
  badge_verified_count?: number;
  provenance?: { included_count?: number };
  wizard?: {
    status: string;
    final_awards?: Array<{ category_key: string }>;
    mint?: { status?: string; expected_count?: number; verified_count?: number };
  };
};
type AdminCapabilities = {
  authorized: boolean;
  user?: { email?: string | null };
  assignments?: Array<{ club_id: string; role: string; permissions: string[] }>;
};
type DeploymentEnvironment = {
  environment?: string | null;
  git_commit_sha?: string | null;
  api_origin?: string | null;
  auth_origin?: string | null;
  preview_isolation_active?: boolean;
  preview_auth_isolation_active?: boolean;
};

test.describe.configure({ mode: "serial", retries: 0 });
test.skip(
  !allowMutation,
  "Set JUPR_FOUR_WEEK_ALLOW_MUTATION_E2E=1 only for this explicitly authorized disposable staging story."
);

function apiPath(pathname: string): string {
  return `${expectedApiOrigin}${pathname}`;
}

function leagueApiPath(suffix = ""): string {
  return `/admin/clubs/${clubId}/league-manager/leagues/${encodeURIComponent(leagueName)}${suffix}`;
}

function leagueRoute(pathname: string, leagueId: string): string {
  const route = new URL(pathname, expectedWebOrigin);
  route.searchParams.set("league_id", leagueId);
  route.searchParams.set("league", leagueName);
  route.searchParams.set("league_name", leagueName);
  route.searchParams.set("mode", "Individual");
  return route.toString();
}

async function apiGet<T>(page: Page, pathname: string): Promise<T> {
  const response = await page.request.get(apiPath(pathname), {
    headers: { Authorization: `Bearer ${adminToken}` }
  });
  expect(response.status(), `GET ${pathname} failed`).toBe(200);
  return await response.json() as T;
}

async function acknowledgeCompletion(page: Page): Promise<void> {
  const acknowledge = page.getByRole("button", { name: "OK", exact: true });
  await expect(acknowledge).toBeVisible({ timeout: 20_000 });
  await acknowledge.click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
}

async function confirmedAction(
  page: Page,
  options: {
    trigger: string;
    confirm: string;
    method: "PATCH" | "POST";
    pathname: string;
    expectedStatus?: number;
  }
): Promise<Response> {
  const trigger = page.getByRole("button", { name: options.trigger, exact: true });
  await expect(trigger).toBeVisible();
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const confirm = page.getByRole("button", { name: options.confirm, exact: true });
  await expect(confirm).toBeVisible();
  const responsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === expectedApiOrigin
      && url.pathname === options.pathname
      && response.request().method() === options.method;
  }, { timeout: 90_000 });
  await confirm.click();
  const response = await responsePromise;
  expect(response.status(), `${options.method} ${options.pathname} failed`).toBe(options.expectedStatus ?? 200);
  await acknowledgeCompletion(page);
  return response;
}

async function installAdminMutationFirewall(page: Page): Promise<void> {
  await page.route("**/admin/**", async (route) => {
    const request = route.request();
    const method = request.method();
    const url = new URL(request.url());
    if (["POST", "PUT", "PATCH", "DELETE"].includes(method) && url.origin !== expectedApiOrigin) {
      await route.abort("blockedbyclient");
      throw new Error(`Refusing ${method} ${url.origin}${url.pathname}: admin mutations are staging-API-only.`);
    }
    await route.continue();
  });
}

async function gotoLeaguePage(page: Page, pathname: string, leagueId: string): Promise<void> {
  const target = leagueRoute(pathname, leagueId);
  const response = await page.goto(target, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);
  await expect(page).toHaveURL(target);
}

async function chooseOption(select: Locator, value: string): Promise<void> {
  await expect(select).toBeVisible();
  await expect(select).toBeEnabled();
  await select.evaluate((element, nextValue) => {
    if (!(element instanceof HTMLSelectElement)) throw new Error("Expected a select element.");
    if (![...element.options].some((option) => option.value === nextValue)) {
      throw new Error(`Option ${nextValue} is unavailable.`);
    }
    element.value = nextValue;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
  await expect(select).toHaveValue(value);
}

async function chooseResettingOption(select: Locator, value: string): Promise<void> {
  await expect(select).toBeVisible();
  await expect(select).toBeEnabled();
  await select.evaluate((element, nextValue) => {
    if (!(element instanceof HTMLSelectElement)) throw new Error("Expected a select element.");
    if (![...element.options].some((option) => option.value === nextValue)) {
      throw new Error(`Option ${nextValue} is unavailable.`);
    }
    element.value = nextValue;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
  await expect(select).toHaveValue("");
}

async function addExistingPlayers(page: Page, playerIds: readonly number[]): Promise<string[]> {
  const selected: string[] = [];
  const select = page.getByRole("combobox", {
    name: /Add an existing club player, including a non-roster player/i
  });
  for (const playerId of playerIds) {
    const option = select.locator(`option[value="${playerId}"]`);
    await expect(option, `Existing staging guest ${playerId} is unavailable`).toHaveCount(1);
    const value = await option.getAttribute("value");
    const label = String(await option.textContent()).trim();
    expect(value).toBe(String(playerId));
    expect(label).not.toBe("");
    await chooseResettingOption(select, String(value));
    await expect(option, `Existing staging guest ${playerId} was not added`).toHaveCount(0);
    selected.push(label);
  }
  return selected;
}

async function runWeek(
  page: Page,
  leagueId: string,
  plan: typeof weeks[number]
): Promise<{ sessionId: string; dynamicPlayers: string[]; publishedMatchIds: string[] }> {
  await gotoLeaguePage(page, "/admin/league-manager/live", leagueId);
  await expect(page.getByRole("heading", { name: `${leagueName} live rounds`, exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "1. Setup", exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "League", exact: true })).toHaveValue(leagueName);
  await expect(page.getByRole("button", { name: "Continue to Players", exact: true })).toBeEnabled();
  await page.getByLabel("Week", { exact: true }).fill(`Week ${plan.week}`);
  await page.getByLabel("Round #", { exact: true }).fill("1");
  await page.getByLabel("Total rounds", { exact: true }).fill("1");
  await page.getByLabel("Date *", { exact: true }).fill(plan.date);
  await page.getByLabel("Round label", { exact: true }).fill(`Week ${plan.week} Round 1`);
  await page.getByRole("button", { name: "Continue to Players", exact: true }).click();

  await expect(page.getByRole("heading", { name: "2. Players", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Select all rostered players", exact: true }).click();
  const rosterSection = page.getByRole("heading", { name: "League roster", exact: true }).locator("..");
  const rosterAttendance = rosterSection.locator('input[type="checkbox"]');
  await expect(rosterAttendance).toHaveCount(rosterPlayerIds.length);
  if (plan.rosterAttendance < rosterPlayerIds.length) {
    await rosterAttendance.last().uncheck();
  }

  if (plan.pasted.length) {
    await page.getByPlaceholder("Alex Rivera, Casey Lee\nMorgan Chen").fill(plan.pasted.join("\n"));
    await page.getByRole("button", { name: "Resolve and add pasted players", exact: true }).click();
    const expectedMissing = plan.week === 2 ? 1 : plan.week === 3 ? 2 : 0;
    const startingJupr = page.getByLabel("Starting JUPR *", { exact: true });
    await expect(startingJupr).toHaveCount(expectedMissing);
    for (let index = 0; index < expectedMissing; index += 1) {
      await startingJupr.nth(index).fill(String(3.25 + index * 0.15));
    }
    if (expectedMissing) {
      const createResponse = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return url.origin === expectedApiOrigin
          && url.pathname === `/admin/clubs/${clubId}/match-uploader/players`
          && response.request().method() === "POST";
      });
      await page.getByRole("button", { name: "Create missing players", exact: true }).click();
      expect((await createResponse).status()).toBe(200);
      await expect(page.getByRole("heading", { name: "Create missing players", exact: true })).toHaveCount(0);
    }
  }

  const dynamicPlayers = await addExistingPlayers(page, plan.existingPlayerIds);
  const buildCourts = page.getByRole("button", {
    name: `Build courts from ${plan.attendance} attendees`,
    exact: true
  });
  await expect(buildCourts).toBeEnabled();
  await buildCourts.click();

  const courtsPanel = page.locator('article[aria-labelledby="league-live-courts-heading"]');
  await expect(courtsPanel).toBeVisible();
  const formatSelects = courtsPanel.getByRole("combobox", { name: "Format", exact: true });
  const playerLists = courtsPanel.getByLabel("Players, one per line", { exact: true });
  await expect(formatSelects).toHaveCount(plan.courtSizes.length);
  await expect(playerLists).toHaveCount(plan.courtSizes.length);
  for (let index = 0; index < plan.courtSizes.length; index += 1) {
    expect((await formatSelects.nth(index).inputValue()).toLowerCase()).toContain(`${plan.courtSizes[index]}-player`);
    expect((await playerLists.nth(index).inputValue()).split("\n").filter(Boolean)).toHaveLength(plan.courtSizes[index]);
  }
  await courtsPanel.getByRole("button", { name: "Validate courts and generate preview", exact: true }).click();
  await expect(courtsPanel.getByRole("heading", { name: `Match preview · ${plan.matchCount} slots`, exact: true })).toBeVisible();

  const createPath = `/admin/clubs/${clubId}/league-manager/live-sessions`;
  const createResponse = await confirmedAction(page, {
    trigger: "Create session and continue",
    confirm: "Yes, create and continue",
    method: "POST",
    pathname: createPath
  });
  const created = await createResponse.json() as { session: LiveSession };
  const sessionId = String(created.session?.id || "");
  expect(sessionId).toMatch(/^[0-9a-f-]{36}$/);
  expect(created.session).toMatchObject({
    league_name: leagueName,
    week_tag: `Week ${plan.week}`,
    total_rounds: 1,
    current_round: 1,
    status: "active"
  });

  await expect(page.getByRole("heading", { name: "4. Score Entry with Review", exact: true })).toBeVisible();
  const teamOneScores = page.locator('input[aria-label$="Team 1 score"]:enabled');
  const teamTwoScores = page.locator('input[aria-label$="Team 2 score"]:enabled');
  await expect(teamOneScores).toHaveCount(plan.matchCount);
  await expect(teamTwoScores).toHaveCount(plan.matchCount);
  for (let index = 0; index < plan.matchCount; index += 1) {
    const teamOneWins = index % 2 === 0;
    await teamOneScores.nth(index).fill(teamOneWins ? "11" : "7");
    await teamTwoScores.nth(index).fill(teamOneWins ? "7" : "11");
  }
  await page.getByRole("button", { name: "Review scores", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Review entered scores", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Confirm scores and continue", exact: true }).click();
  await expect(page.getByRole("heading", { name: "5. Movement", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Preview movement", exact: true }).click();
  await expect(page.getByText(/Verified operation key/i)).toBeVisible();
  await page.getByRole("button", { name: "Continue to Repeat or Finish", exact: true }).click();

  await confirmedAction(page, {
    trigger: "Publish reviewed round",
    confirm: "Yes, publish the round",
    method: "POST",
    pathname: `${createPath}/${sessionId}/rounds/1/submit`
  });
  await expect(page.getByRole("heading", { name: "Round 1 published", exact: true })).toBeVisible();
  await confirmedAction(page, {
    trigger: "Finish session",
    confirm: "Yes, complete session",
    method: "PATCH",
    pathname: `${createPath}/${sessionId}/snapshot`
  });
  await expect(page.getByRole("heading", { name: "Session complete", exact: true })).toBeVisible();

  const detail = await apiGet<LiveDetail>(page, `${createPath}/${sessionId}`);
  expect(detail.session).toMatchObject({
    id: sessionId,
    league_name: leagueName,
    week_tag: `Week ${plan.week}`,
    status: "complete",
    current_round: 1,
    total_rounds: 1
  });
  expect(detail.rounds).toHaveLength(1);
  expect(detail.rounds[0]).toMatchObject({
    round_number: 1,
    status: "submitted",
    submitted_match_count: plan.matchCount
  });
  expect(detail.publish_operations).toHaveLength(1);
  expect(detail.publish_operations?.[0].status).toBe("completed");
  const publishedMatchIds = detail.publish_operations?.[0].published_match_ids || [];
  expect(publishedMatchIds).toHaveLength(plan.matchCount);
  expect(new Set(publishedMatchIds).size).toBe(plan.matchCount);
  return { sessionId, dynamicPlayers, publishedMatchIds };
}

test("creates, plays, awards, and archives a four-week Flex ladder league", async ({ page, context }) => {
  test.setTimeout(1_200_000);
  page.setDefaultTimeout(20_000);
  page.setDefaultNavigationTimeout(45_000);

  expect(mutationConfirmation).toBe("RUN DISPOSABLE STAGING WRITES");
  expect(expectedApiOrigin).toBe("https://juprleagues-api-staging.fly.dev");
  expect(expectedAuthOrigin).toBe("https://sijpxjxvdtrehmqvirfi.supabase.co");
  expect(expectedWebOrigin).toMatch(
    /^https:\/\/[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}-pickleballattrespalapas1\.vercel\.app$/
  );
  expect(adminEmail).not.toBe("");
  expect(adminToken).not.toBe("");
  expect(candidateSha).toMatch(/^[0-9a-f]{40}$/);
  expect(leagueName).not.toBe("");
  expect(aspenName).not.toBe("");
  expect(birchName).not.toBe("");
  expect(cloverName).not.toBe("");
  expect(new Set([leagueName, aspenName, birchName, cloverName]).size).toBe(4);
  expect(reportPath).not.toBe("");

  await bootstrapStagingContext(context);
  const environmentResponse = await context.request.get(`${expectedWebOrigin}/api/environment`, {
    failOnStatusCode: false
  });
  expect(environmentResponse.status(), "Immutable Vercel deployment environment check failed").toBe(200);
  const environment = await environmentResponse.json() as DeploymentEnvironment;
  await environmentResponse.dispose();
  expect(environment).toMatchObject({
    environment: "staging",
    git_commit_sha: candidateSha,
    api_origin: expectedApiOrigin,
    auth_origin: expectedAuthOrigin,
    preview_isolation_active: true,
    preview_auth_isolation_active: true
  });

  const capabilities = await apiGet<AdminCapabilities>(
    page,
    `/admin/auth/capabilities?club_id=${encodeURIComponent(clubId)}`
  );
  expect(capabilities.authorized).toBe(true);
  expect(capabilities.assignments?.some((assignment) => assignment.club_id === clubId)).toBe(true);
  await context.addInitScript(
    ({ token, email, allowedOrigin, verifiedCapabilities }) => {
      if (window.location.origin !== allowedOrigin) return;
      window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
        access_token: token,
        token_type: "bearer",
        capabilities: verifiedCapabilities,
        user: { email: verifiedCapabilities.user?.email || email }
      }));
    },
    {
      token: adminToken,
      email: adminEmail,
      allowedOrigin: expectedWebOrigin,
      verifiedCapabilities: capabilities
    }
  );
  await installAdminMutationFirewall(page);

  const leagueListPath = `/admin/clubs/${clubId}/league-manager/leagues`;
  const before = await apiGet<{ leagues: LeagueSummary[] }>(page, leagueListPath);
  expect(before.leagues.some((league) => league.league_name === leagueName), "Disposable league name must be unique").toBe(false);

  const createPage = new URL("/admin/league-manager/create", expectedWebOrigin).toString();
  const browserCapabilitiesResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === expectedApiOrigin
      && url.pathname === "/admin/auth/capabilities"
      && response.request().method() === "GET";
  });
  const createDocument = await page.goto(createPage, { waitUntil: "domcontentloaded" });
  expect(createDocument?.status()).toBe(200);
  expect(createDocument?.url()).toBe(createPage);
  await expect(page).toHaveURL(createPage);
  await expect(page.getByRole("heading", { name: "Start league setup", exact: true })).toBeVisible();
  expect((await browserCapabilitiesResponse).status()).toBe(200);
  const leagueNameInput = page.getByLabel("League name", { exact: true });
  await expect(leagueNameInput).toBeVisible();
  await leagueNameInput.fill(leagueName);
  await expect(page.getByRole("combobox", { name: "League mode", exact: true })).toHaveValue("Individual");
  await expect(page.getByRole("combobox", { name: "Match modality", exact: true })).toHaveValue("doubles");
  await expect(page.getByRole("combobox", { name: "Season format", exact: true })).toHaveValue("ladder");
  await expect(page.getByRole("combobox", { name: /^Participation model/ })).toHaveValue("flex");
  await expect(page.getByRole("combobox", { name: "Session operation", exact: true })).toHaveValue("scheduled_rounds");
  await page.getByLabel("Minimum games", { exact: true }).fill("1");
  const createLeagueResponse = await confirmedAction(page, {
    trigger: "Create league",
    confirm: "Yes, create league",
    method: "POST",
    pathname: leagueListPath
  });
  const createdLeague = await createLeagueResponse.json() as { league: LeagueSummary };
  const leagueId = String(createdLeague.league?.league_id || "");
  expect(leagueId).not.toBe("");
  expect(createdLeague.league).toMatchObject({
    league_name: leagueName,
    league_type: "Individual",
    status: "draft"
  });

  await gotoLeaguePage(page, "/admin/league-manager/settings", leagueId);
  await expect(page.getByRole("heading", { name: "League setup wizard", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "2. Schedule", exact: true }).click();
  await page.getByLabel("Start date", { exact: true }).fill("2026-08-24");
  await page.getByLabel("Weeks (or use an end date)", { exact: true }).fill("4");
  await expect(page.getByRole("combobox", { name: "Weekday", exact: true })).toHaveValue("0");
  await page.getByRole("button", { name: "3. Courts & live play", exact: true }).click();
  await page.getByLabel("Total courts available", { exact: true }).fill("3");
  await page.getByLabel("Maximum courts this league may use", { exact: true }).fill("3");
  await expect(page.getByRole("combobox", { name: "Ladder pod size", exact: true })).toHaveValue("4");
  await page.getByRole("button", { name: "4. Match & standings", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Match structure", exact: true })).toHaveValue("one_game");
  await page.getByRole("button", { name: "5. Awards & eligibility", exact: true }).click();
  await page.getByLabel("Minimum games for awards", { exact: true }).fill("1");
  await confirmedAction(page, {
    trigger: "Save structured draft",
    confirm: "Yes, save draft",
    method: "PATCH",
    pathname: leagueApiPath()
  });
  const configuredDetail = await apiGet<LeagueDetail>(page, leagueApiPath());
  expect(configuredDetail.schedule_preview?.map((row) => row.date)).toEqual([
    "2026-08-24",
    "2026-08-31",
    "2026-09-07",
    "2026-09-14"
  ]);
  expect(Number(configuredDetail.league.schedule_config?.weeks)).toBe(4);
  expect(Number(configuredDetail.league.court_board_defaults?.total_courts)).toBe(3);
  expect(Number(configuredDetail.league.court_board_defaults?.max_used_courts)).toBe(3);
  expect(Number(configuredDetail.league.court_board_defaults?.players_per_court)).toBe(4);

  const awardsSetup = page.getByTestId("league-awards-setup");
  await expect(awardsSetup.getByRole("heading", { name: "Awards setup", exact: true })).toBeVisible();
  for (const label of awardLabels) {
    const fieldset = awardsSetup.getByRole("group", { name: label, exact: true });
    const enabled = fieldset.getByLabel("Enabled", { exact: true });
    if (!(await enabled.isChecked())) await enabled.check();
    await expect(fieldset.getByRole("combobox", { name: "Places", exact: true })).toHaveValue("1");
    await fieldset.getByLabel(/^Minimum /).fill("1");
  }
  const awardsConfigResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === expectedApiOrigin
      && url.pathname === leagueApiPath("/awards/config")
      && response.request().method() === "PUT";
  });
  await awardsSetup.getByRole("button", { name: "Save award setup", exact: true }).click();
  expect((await awardsConfigResponse).status()).toBe(200);

  await gotoLeaguePage(page, "/admin/league-manager/roster", leagueId);
  await expect(page.getByLabel("Search players", { exact: true })).toBeVisible();
  await chooseOption(page.getByRole("combobox", { name: "Show", exact: true }), "not_in_league");
  await expect(page.getByRole("combobox", { name: "Action", exact: true })).toHaveValue("activate");
  for (const playerId of rosterPlayerIds) {
    await page.getByLabel("Search players", { exact: true }).fill(String(playerId));
    const row = page.locator("tr").filter({ hasText: `#${playerId}` });
    await expect(row).toHaveCount(1);
    await row.locator('input[type="checkbox"]').check();
  }
  await page.getByLabel("Search players", { exact: true }).fill("");
  await confirmedAction(page, {
    trigger: "Add Players",
    confirm: "Yes, add players",
    method: "POST",
    pathname: leagueApiPath("/roster/batch")
  });
  const rosterDetail = await apiGet<LeagueDetail>(page, leagueApiPath());
  expect(rosterDetail.roster?.filter((row) => row.in_league).map((row) => row.player_id).sort((a, b) => a - b)).toEqual([...rosterPlayerIds].sort((a, b) => a - b));

  await gotoLeaguePage(page, "/admin/league-manager/league", leagueId);
  await confirmedAction(page, {
    trigger: "Start league",
    confirm: "Yes, start league",
    method: "POST",
    pathname: leagueApiPath("/lifecycle")
  });

  const weeklyResults = [];
  for (const plan of weeks) weeklyResults.push(await runWeek(page, leagueId, plan));
  const publishedMatchIds = weeklyResults.flatMap((result) => result.publishedMatchIds);
  expect(publishedMatchIds).toHaveLength(36);
  expect(new Set(publishedMatchIds).size).toBe(36);

  await gotoLeaguePage(page, "/admin/league-manager/awards", leagueId);
  const leagueSelect = page.getByRole("combobox", { name: "League", exact: true });
  await expect(leagueSelect.locator(`option[value="${leagueName}"]`)).toHaveCount(1);
  await expect(leagueSelect).toHaveValue(leagueName);
  await expect(page.getByText(/Saved step:\s*not started/i)).toBeVisible();
  const awardsPreflight = await apiGet<AwardsState>(page, leagueApiPath("/awards"));
  expect(awardsPreflight.writes_enabled).toBe(true);
  expect(awardsPreflight.service_role_ready).toBe(true);
  expect(awardsPreflight.badge_definitions_ready).toBe(true);
  expect(awardsPreflight.provenance?.included_count).toBe(36);
  await confirmedAction(page, {
    trigger: "Freeze and save",
    confirm: "Yes, freeze league",
    method: "POST",
    pathname: leagueApiPath("/awards/freeze")
  });
  const previewResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === expectedApiOrigin
      && url.pathname === leagueApiPath("/awards/preview")
      && response.request().method() === "POST";
  }, { timeout: 90_000 });
  const computePreview = page.getByRole("button", { name: "Compute and save preview", exact: true });
  await expect(computePreview).toBeEnabled();
  await computePreview.click();
  expect((await previewResponse).status()).toBe(200);
  const overrideResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === expectedApiOrigin
      && url.pathname === leagueApiPath("/awards/overrides")
      && response.request().method() === "POST";
  }, { timeout: 90_000 });
  const confirmWinners = page.getByRole("button", { name: "Confirm winners and reasons", exact: true });
  await expect(confirmWinners).toBeEnabled();
  await confirmWinners.click();
  expect((await overrideResponse).status()).toBe(200);
  await confirmedAction(page, {
    trigger: "Mint and verify",
    confirm: "Yes, mint and verify",
    method: "POST",
    pathname: leagueApiPath("/awards/mint")
  });
  await confirmedAction(page, {
    trigger: "Archive completed league",
    confirm: "Yes, archive league",
    method: "POST",
    pathname: leagueApiPath("/awards/archive")
  });
  await expect(page.getByText(/Archived\. This workflow is read-only/i)).toBeVisible();

  const sessionList = await apiGet<{ sessions: LiveSession[] }>(page, `/admin/clubs/${clubId}/league-manager/live-sessions?limit=200`);
  const leagueSessions = sessionList.sessions.filter((session) => session.league_name === leagueName);
  expect(leagueSessions).toHaveLength(4);
  expect(new Set(leagueSessions.map((session) => session.week_tag))).toEqual(new Set(["Week 1", "Week 2", "Week 3", "Week 4"]));
  expect(leagueSessions.every((session) => session.status === "complete")).toBe(true);
  const finalDetail = await apiGet<LeagueDetail>(page, leagueApiPath());
  const finalAwards = await apiGet<AwardsState>(page, leagueApiPath("/awards"));
  expect(finalDetail.league.status).toBe("archived");
  expect(finalDetail.schedule_preview).toHaveLength(4);
  expect(finalAwards.wizard?.status).toBe("archived");
  expect(finalAwards.badge_expected_count).toBeGreaterThan(0);
  expect(finalAwards.badge_verified_count).toBe(finalAwards.badge_expected_count);
  expect(finalAwards.wizard?.mint?.status).toBe("verified");
  expect(new Set(finalAwards.wizard?.final_awards?.map((award) => award.category_key))).toEqual(new Set(awardKeys));

  writeFileSync(reportPath, `${JSON.stringify({
    schema_version: 1,
    candidate_sha: String(process.env.GITHUB_SHA || ""),
    environment: "staging",
    production_targets_contacted: false,
    league_id: leagueId,
    league_name: leagueName,
    expected_schedule_weeks: 4,
    expected_session_count: 4,
    expected_match_count: 36,
    actual_session_count: leagueSessions.length,
    actual_match_count: publishedMatchIds.length,
    status: finalDetail.league.status,
    weeks: weeks.map((plan) => ({
      week: plan.week,
      date: plan.date,
      attendance: plan.attendance,
      court_sizes: plan.courtSizes,
      official_matches: plan.matchCount
    })),
    session_ids: weeklyResults.map((result) => result.sessionId),
    dynamic_existing_players: weeklyResults.flatMap((result) => result.dynamicPlayers),
    created_player_names: [aspenName, birchName, cloverName],
    unique_published_matches: new Set(publishedMatchIds).size,
    award_categories: awardKeys,
    awards_status: finalAwards.wizard?.status,
    badge_expected_count: finalAwards.badge_expected_count,
    badge_verified_count: finalAwards.badge_verified_count
  }, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
});
