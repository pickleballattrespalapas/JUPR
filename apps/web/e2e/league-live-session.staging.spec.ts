import { writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";

import {
  expect,
  test,
  type APIRequestContext,
  type Page
} from "@playwright/test";

import {
  bootstrapStagingContext,
  expectedApiOrigin
} from "./support/staging";

const clubId = "tres_palapas";
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "").trim();
const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const allowMutation = /^(1|true|yes|on)$/i.test(
  String(process.env.JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E || "")
);
const mutationConfirmation = String(
  process.env.JUPR_PARITY_MUTATION_CONFIRMATION || ""
).trim();
const expectedWebOrigin = String(
  process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN || ""
).trim().replace(/\/$/, "");
const expectedLeagueName = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_LEAGUE_NAME || ""
).trim();
const matchDate = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_MATCH_DATE || ""
).trim();
const reportPath = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_REPORT_PATH || ""
).trim();
const candidateSha = String(process.env.GITHUB_SHA || "").trim();
const workflowRunId = String(process.env.GITHUB_RUN_ID || "").trim();
const workflowRunAttempt = String(process.env.GITHUB_RUN_ATTEMPT || "").trim();
const fixtureSource = "staging_league_live_browser_acceptance";
const legacyAcceptanceLeagueName = "Acceptance Flex 0822A";
const legacyAcceptanceWeekTags = new Set([
  "E2E 4dce01a8-32797001253-1",
  "E2E 9b956dfc-32802308745-1"
]);
let sessionId = "";
let leagueId = "";
let createdPlayerIds: number[] = [];
let cleanupVerified = false;

type PlayerFixture = {
  id: number;
  name: string;
  rating: number;
  active: boolean;
  state_fingerprint: string;
};

type LeagueFixture = {
  league_id?: string | number | null;
  league_name: string;
  status: string;
};

type PublishOperation = {
  id: string;
  round_number: number;
  status: string;
  attempt_count: number;
  match_context_ids?: string[];
  published_match_ids?: string[];
};

type LeagueLiveDetail = {
  session: {
    id: string;
    league_name: string;
    status: string;
    current_round: number;
    total_rounds: number;
    week_tag?: string;
    roster_json?: Array<Record<string, unknown>>;
    current_court_state_json?: Array<Record<string, unknown>>;
  };
  rounds: Array<{
    round_number: number;
    status: string;
    match_date?: string | null;
    submitted_match_count?: number;
    submitted_match_ids?: string[];
    operation_key?: string;
    movement_json?: { operation_key?: string };
  }>;
  publish_operations?: PublishOperation[];
};

type CleanupResult = {
  leagueStatus: string;
  inactivePlayerCount: number;
};

type MatchLogRow = {
  id: number;
  row_version: number;
  league: string;
  week_tag: string;
  context_type: string;
  is_active: boolean;
};

type MatchExclusionOperation = {
  id: string;
  status: string;
  source?: string | null;
  result_json?: {
    excluded_count?: number;
  } | null;
};

type MatchLogPayload = {
  matches?: MatchLogRow[];
  recent_exclusion_operations?: MatchExclusionOperation[];
};

test.describe.configure({ mode: "serial", retries: 0 });
test.skip(
  !allowMutation,
  "Set JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E=1 only for an explicitly authorized staging session."
);

function apiPath(pathname: string): string {
  return `${expectedApiOrigin}${pathname}`;
}

async function requestJson<T>(
  request: APIRequestContext,
  method: "GET" | "PATCH" | "POST",
  pathname: string,
  data?: Record<string, unknown>,
  timeout = 90_000
): Promise<T> {
  const response = await request.fetch(apiPath(pathname), {
    method,
    headers: { Authorization: `Bearer ${adminToken}` },
    data,
    failOnStatusCode: false,
    timeout
  });
  const status = response.status();
  const payload = await response.json() as T;
  await response.dispose();
  expect(
    status,
    `${method} ${pathname} failed: ${JSON.stringify(payload)}`
  ).toBe(200);
  return payload;
}

async function recoverLegacyCleanup(
  request: APIRequestContext,
  operation: MatchExclusionOperation
): Promise<number> {
  const recovered = await requestJson<{
    excluded_count?: number;
    replay_status?: string;
    operation_status?: string;
  }>(
    request,
    "POST",
    `/admin/clubs/${clubId}/match-log/exclusions/${encodeURIComponent(operation.id)}/recover`,
    {
      confirmation_text: "RECOVER",
      source: `${fixtureSource}_legacy_cleanup_recovery`
    },
    240_000
  );
  expect(recovered.excluded_count).toBe(30);
  expect(recovered.operation_status).toBe("succeeded");
  expect(recovered.replay_status).toBe("succeeded");
  return Number(recovered.excluded_count || 0);
}

async function excludeLegacyAcceptanceMatches(
  request: APIRequestContext
): Promise<number> {
  const query = new URLSearchParams({
    league: legacyAcceptanceLeagueName,
    limit: "500"
  });
  const matchLog = await requestJson<MatchLogPayload>(
    request,
    "GET",
    `/admin/clubs/${clubId}/match-log?${query.toString()}`
  );
  const matches = (matchLog.matches || []).filter((row) =>
    legacyAcceptanceWeekTags.has(String(row.week_tag || ""))
  );
  const legacyOperation = (matchLog.recent_exclusion_operations || []).find(
    (operation) => operation.source === `${fixtureSource}_legacy_cleanup`
      && Number(operation.result_json?.excluded_count || 0) === 30
  );
  if (matches.length === 0) {
    if (!legacyOperation) return 0;
    if (legacyOperation.status === "succeeded") return 30;
    return recoverLegacyCleanup(request, legacyOperation);
  }

  expect(matches).toHaveLength(30);
  expect(new Set(matches.map((row) => row.id)).size).toBe(30);
  expect(matches.every((row) => row.league === legacyAcceptanceLeagueName)).toBe(true);
  expect(matches.every((row) => row.context_type === "league_live_session")).toBe(true);
  expect(matches.every((row) => row.is_active)).toBe(true);
  expect(matches.every((row) => Number(row.row_version) >= 1)).toBe(true);

  const excluded = await requestJson<{
    excluded_count?: number;
    replay_status?: string;
    operation_status?: string;
  }>(
    request,
    "POST",
    `/admin/clubs/${clubId}/match-log/exclude`,
    {
      targets: matches.map((row) => ({
        match_id: Number(row.id),
        expected_row_version: Number(row.row_version)
      })),
      confirmation_text: "DELETE",
      idempotency_key: randomUUID(),
      note: "Remove legacy League Live E2E matches from Acceptance Flex 0822A after isolating automated acceptance fixtures.",
      source: `${fixtureSource}_legacy_cleanup`
    },
    240_000
  );
  expect(excluded.excluded_count).toBe(30);
  expect(excluded.operation_status).toBe("succeeded");
  expect(excluded.replay_status).toBe("succeeded");

  const verified = await requestJson<MatchLogPayload>(
    request,
    "GET",
    `/admin/clubs/${clubId}/match-log?${query.toString()}`
  );
  expect(
    (verified.matches || []).filter((row) =>
      legacyAcceptanceWeekTags.has(String(row.week_tag || ""))
    )
  ).toHaveLength(0);
  return Number(excluded.excluded_count || 0);
}

async function cleanupRequest(
  request: APIRequestContext,
  method: "GET" | "PATCH" | "POST",
  pathname: string,
  data?: Record<string, unknown>
): Promise<{ status: number; payload: Record<string, unknown> }> {
  try {
    const response = await request.fetch(apiPath(pathname), {
      method,
      headers: { Authorization: `Bearer ${adminToken}` },
      data,
      failOnStatusCode: false
    });
    const status = response.status();
    let payload: Record<string, unknown> = {};
    try {
      payload = await response.json() as Record<string, unknown>;
    } catch {
      payload = {};
    }
    await response.dispose();
    return { status, payload };
  } catch {
    return { status: 0, payload: {} };
  }
}

function leaguePath(suffix = ""): string {
  return `/admin/clubs/${clubId}/league-manager/leagues/${encodeURIComponent(expectedLeagueName)}${suffix}`;
}

async function createDisposableFixtures(page: Page): Promise<{
  roster: Array<Record<string, unknown>>;
  courts: Array<Record<string, unknown>>;
}> {
  const runSuffix = `${candidateSha.slice(0, 7)}-${workflowRunId}-${workflowRunAttempt}`;
  const players: PlayerFixture[] = [];
  for (let index = 1; index <= 4; index += 1) {
    const created = await requestJson<{ player?: PlayerFixture }>(
      page.request,
      "POST",
      `/admin/clubs/${clubId}/players/editor/players`,
      {
        name: `League Live E2E P${index} ${runSuffix}`,
        starting_jupr: 3 + index / 10,
        idempotency_key: `league-live-e2e-player:${candidateSha}:${workflowRunId}:${workflowRunAttempt}:${index}`,
        source: fixtureSource
      }
    );
    const player = created.player;
    expect(player?.id).toBeGreaterThan(0);
    expect(player?.name).toContain("League Live E2E");
    expect(player?.active).toBe(true);
    players.push(player as PlayerFixture);
    createdPlayerIds.push(Number(player?.id));
  }
  expect(new Set(createdPlayerIds).size).toBe(4);

  const createdLeague = await requestJson<{ league?: LeagueFixture }>(
    page.request,
    "POST",
    `/admin/clubs/${clubId}/league-manager/leagues`,
    {
      league_name: expectedLeagueName,
      league_type: "Individual",
      match_format: "doubles",
      league_format: "ladder",
      session_mode: "scheduled_rounds",
      participation_mode: "flex",
      description: "Disposable League Live browser acceptance fixture.",
      min_games: 1,
      k_factor: 32,
      confirmation_text: "CREATE LEAGUE",
      source: fixtureSource
    }
  );
  leagueId = String(createdLeague.league?.league_id || "");
  expect(leagueId).not.toBe("");
  expect(createdLeague.league).toMatchObject({
    league_name: expectedLeagueName,
    status: "draft"
  });

  await requestJson(
    page.request,
    "POST",
    leaguePath("/roster/batch"),
    {
      action: "activate",
      player_ids: createdPlayerIds,
      starting_rating: null,
      idempotency_key: `league-live-e2e-roster:${candidateSha}:${workflowRunId}:${workflowRunAttempt}`,
      confirmation_text: "SAVE LEAGUE ROSTER BATCH",
      source: fixtureSource
    }
  );
  const started = await requestJson<{ league?: LeagueFixture }>(
    page.request,
    "POST",
    leaguePath("/lifecycle"),
    {
      action: "start",
      confirmation_text: "START LEAGUE",
      source: fixtureSource
    }
  );
  expect(started.league?.status).toBe("active");

  const roster = players.map((player, index) => ({
    player_id: player.id,
    player_name: player.name,
    rating: player.rating,
    status: "active",
    court_number: 1,
    slot: index + 1
  }));
  return {
    roster,
    courts: [{
      round_number: 1,
      court_number: 1,
      format_type: "4-Player",
      player_names: players.map((player) => player.name),
      players_json: roster
    }]
  };
}

async function cleanupDisposableFixtures(
  request: APIRequestContext
): Promise<CleanupResult> {
  let leagueStatus = "";
  if (expectedLeagueName) {
    let detail = await cleanupRequest(request, "GET", leaguePath());
    if (detail.status === 200) {
      const league = detail.payload.league as LeagueFixture | undefined;
      leagueStatus = String(league?.status || "").toLowerCase();
      if (["active", "paused"].includes(leagueStatus)) {
        await cleanupRequest(request, "POST", leaguePath("/lifecycle"), {
          action: "end",
          confirmation_text: "END LEAGUE",
          source: `${fixtureSource}_cleanup`
        });
        detail = await cleanupRequest(request, "GET", leaguePath());
        leagueStatus = String((detail.payload.league as LeagueFixture | undefined)?.status || "").toLowerCase();
      }
      if (leagueStatus === "ended") {
        await cleanupRequest(request, "POST", leaguePath("/lifecycle"), {
          action: "archive",
          confirmation_text: "ARCHIVE LEAGUE",
          source: `${fixtureSource}_cleanup`
        });
        detail = await cleanupRequest(request, "GET", leaguePath());
        leagueStatus = String((detail.payload.league as LeagueFixture | undefined)?.status || "").toLowerCase();
      }
    }
  }

  let inactivePlayerCount = 0;
  for (const playerId of createdPlayerIds) {
    const playerPath = `/admin/clubs/${clubId}/players/editor/players/${playerId}`;
    let detail = await cleanupRequest(request, "GET", playerPath);
    if (detail.status !== 200) continue;
    let player = detail.payload.player as PlayerFixture | undefined;
    if (player?.active) {
      await cleanupRequest(request, "PATCH", playerPath, {
        active: false,
        expected_state_fingerprint: player.state_fingerprint,
        idempotency_key: `league-live-e2e-deactivate:${candidateSha}:${workflowRunId}:${workflowRunAttempt}:${playerId}`,
        source: `${fixtureSource}_cleanup`
      });
      detail = await cleanupRequest(request, "GET", playerPath);
      player = detail.payload.player as PlayerFixture | undefined;
    }
    if (player && !player.active) inactivePlayerCount += 1;
  }
  return { leagueStatus, inactivePlayerCount };
}

test.afterEach(async ({ request }) => {
  if (!allowMutation || cleanupVerified || (!leagueId && createdPlayerIds.length === 0)) return;
  await cleanupDisposableFixtures(request);
});

async function fetchDetail(page: Page, targetSessionId = sessionId): Promise<LeagueLiveDetail> {
  const response = await page.request.get(
    `${expectedApiOrigin}/admin/clubs/${clubId}/league-manager/live-sessions/${targetSessionId}`,
    { headers: { Authorization: `Bearer ${adminToken}` } }
  );
  expect(response.status(), "League Live detail read failed").toBe(200);
  return await response.json() as LeagueLiveDetail;
}

async function runConfirmedAction(
  page: Page,
  options: {
    trigger: string;
    confirm: string;
    method: "PATCH" | "POST";
    pathname: string;
  }
): Promise<void> {
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
  });
  await confirm.click();
  const response = await responsePromise;
  expect(response.status(), `${options.method} ${options.pathname} failed`).toBe(200);

  const acknowledge = page.getByRole("button", { name: "OK", exact: true });
  await expect(acknowledge).toBeVisible({ timeout: 20_000 });
  await acknowledge.click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
}

async function prepareFirstRound(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: "1. Setup", exact: true })).toBeVisible();
  await expect(page.getByLabel("Round #", { exact: true })).toHaveValue("1");
  await page.getByLabel("Date *", { exact: true }).fill(matchDate);
  await page.getByRole("button", { name: "Continue to Players", exact: true }).click();

  await expect(page.getByRole("heading", { name: "2. Players", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Continue with saved courts", exact: true }).click();

  await expect(page.getByRole("heading", { name: "3. Courts and Preview", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Validate courts and generate preview", exact: true }).click();
  await expect(page.getByRole("heading", { name: /Match preview · 3 slots/i })).toBeVisible();
  await runConfirmedAction(page, {
    trigger: "Save preview and continue",
    confirm: "Yes, save and continue",
    method: "PATCH",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/snapshot`
  });
}

async function publishRound(page: Page, roundNumber: number): Promise<void> {
  await expect(page.getByRole("heading", { name: "4. Score Entry with Review", exact: true })).toBeVisible();
  const teamOneScores = page.locator('input[aria-label$="Team 1 score"]:enabled');
  const teamTwoScores = page.locator('input[aria-label$="Team 2 score"]:enabled');
  await expect(teamOneScores).toHaveCount(3);
  await expect(teamTwoScores).toHaveCount(3);
  const scores = [["11", "7"], ["6", "11"], ["11", "8"]] as const;
  for (let index = 0; index < scores.length; index += 1) {
    await teamOneScores.nth(index).fill(scores[index][0]);
    await teamTwoScores.nth(index).fill(scores[index][1]);
  }
  if (roundNumber === 1) {
    await teamTwoScores.nth(0).fill("76");
    await expect(page.getByText(/Unusual score — verify before publish/i)).toBeVisible();
    await page.getByRole("button", { name: "Review scores", exact: true }).click();
    await expect(page.getByText(/Unusual score requires review/i)).toBeVisible();
    const publishScores = page.getByRole("button", { name: "Publish reviewed scores", exact: true });
    await expect(publishScores).toBeDisabled();
    await page.getByRole("checkbox", { name: /I verified/i }).check();
    await expect(publishScores).toBeEnabled();
    await page.getByRole("button", { name: "Edit scores", exact: true }).click();
    await teamTwoScores.nth(0).fill("7");
    await expect(page.getByText(/Unusual score — verify before publish/i)).toHaveCount(0);
  }
  const reviewScores = page.getByRole("button", { name: "Review scores", exact: true });
  await expect(reviewScores).toBeEnabled();
  await reviewScores.click();
  await expect(page.getByRole("heading", { name: "Review entered scores", exact: true })).toBeVisible();
  await runConfirmedAction(page, {
    trigger: "Publish reviewed scores",
    confirm: "Yes, publish scores",
    method: "POST",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/rounds/${roundNumber}/submit`
  });

  const scoresOnly = await fetchDetail(page);
  expect(scoresOnly.session.current_round).toBe(roundNumber);
  const publishedRound = scoresOnly.rounds.find((round) => round.round_number === roundNumber);
  expect(publishedRound?.status).toBe("submitted");
  expect(publishedRound?.operation_key || publishedRound?.movement_json?.operation_key || "").toBe("");

  await expect(page.getByRole("heading", { name: "5. Movement", exact: true })).toBeVisible();
  await expect(page.getByText(/The round scores are official/i)).toBeVisible();
  await expect(page.getByText(/Verified operation key/i)).toBeVisible();
  await runConfirmedAction(page, {
    trigger: "Apply movement and continue",
    confirm: "Yes, apply movement",
    method: "POST",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/rounds/${roundNumber}/movement`
  });
  await expect(page.getByRole("heading", { name: "6. Repeat or Finish", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: `Round ${roundNumber} complete`, exact: true })).toBeVisible();
}

test("creates and completes a disposable five-round League Live session", async ({ page, context }) => {
  test.setTimeout(360_000);

  expect(mutationConfirmation).toBe("RUN DISPOSABLE STAGING WRITES");
  expect(expectedApiOrigin).toBe("https://juprleagues-api-staging.fly.dev");
  expect(expectedWebOrigin).toMatch(
    /^https:\/\/[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}-pickleballattrespalapas1\.vercel\.app$/
  );
  expect(adminEmail).not.toBe("");
  expect(adminToken).not.toBe("");
  expect(expectedLeagueName).toMatch(/^League Live E2E /);
  expect(matchDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  expect(reportPath).not.toBe("");
  expect(candidateSha).toMatch(/^[0-9a-f]{40}$/);
  expect(workflowRunId).toMatch(/^[1-9]\d*$/);
  expect(workflowRunAttempt).toMatch(/^[1-9]\d*$/);

  const legacyExcludedMatches = await excludeLegacyAcceptanceMatches(page.request);
  const fixture = await createDisposableFixtures(page);
  expect(fixture.roster).toHaveLength(4);
  expect(fixture.courts).toHaveLength(1);

  const createResponse = await page.request.post(
    `${expectedApiOrigin}/admin/clubs/${clubId}/league-manager/live-sessions`,
    {
      headers: { Authorization: `Bearer ${adminToken}` },
      data: {
        league_name: expectedLeagueName,
        week_tag: "Week 1",
        total_rounds: 5,
        current_round: 1,
        roster: fixture.roster,
        courts: fixture.courts,
        bench_player_ids: [],
        notes: "Disposable staging League Live browser acceptance session.",
        idempotency_key: `league-live-e2e:${candidateSha}:${workflowRunId}:${workflowRunAttempt}`,
        confirmation_text: "CREATE LIVE SESSION",
        source: "staging_league_live_browser_acceptance",
      }
    }
  );
  expect(createResponse.status(), "Disposable League Live session creation failed").toBe(200);
  const created = await createResponse.json() as { session?: { id?: string } };
  sessionId = String(created.session?.id || "");
  expect(sessionId).toMatch(/^[0-9a-f-]{36}$/);

  const before = await fetchDetail(page);
  expect(before.session).toMatchObject({
    id: sessionId,
    league_name: expectedLeagueName,
    status: "active",
    current_round: 1,
    total_rounds: 5
  });
  expect(before.rounds).toEqual([]);
  expect(before.publish_operations).toEqual([]);

  await bootstrapStagingContext(context);
  await context.addInitScript(
    ({ token, email, allowedOrigin }) => {
      if (window.location.origin !== allowedOrigin) return;
      window.localStorage.setItem(
        "jupr_admin_session_v1",
        JSON.stringify({
          access_token: token,
          token_type: "bearer",
          user: { email }
        })
      );
    },
    { token: adminToken, email: adminEmail, allowedOrigin: expectedWebOrigin }
  );

  const liveRoute = new URL("/admin/league-manager/live", expectedWebOrigin);
  liveRoute.searchParams.set("league_id", leagueId);
  liveRoute.searchParams.set("league", expectedLeagueName);
  liveRoute.searchParams.set("league_name", expectedLeagueName);
  liveRoute.searchParams.set("mode", "Individual");
  const documentResponse = await page.goto(liveRoute.toString(), {
    waitUntil: "domcontentloaded"
  });
  expect(documentResponse?.status()).toBeLessThan(400);
  await expect(page).toHaveURL(liveRoute.toString());
  await expect(page.getByRole("heading", {
    name: `${expectedLeagueName} live rounds`,
    exact: true
  })).toBeVisible();

  const setupPanel = page.locator('article[aria-labelledby="league-live-setup-heading"]');
  await expect(setupPanel).toBeVisible();
  const sessionSelect = setupPanel.getByRole("combobox", {
    name: "Unfinished sessions for this league",
    exact: true
  });
  await expect(sessionSelect).toBeEnabled({ timeout: 30_000 });
  await expect(sessionSelect.locator(`option[value="${sessionId}"]`)).toHaveCount(1);
  await sessionSelect.selectOption(sessionId);
  await prepareFirstRound(page);
  await publishRound(page, 1);

  for (let roundNumber = 2; roundNumber <= 5; roundNumber += 1) {
    const previewResponsePromise = page.waitForResponse((response) => {
      const url = new URL(response.url());
      return url.origin === expectedApiOrigin
        && url.pathname === `/admin/clubs/${clubId}/match-uploader/round-robin/preview`
        && response.request().method() === "POST";
    });
    await page.getByRole("button", { name: "Start next round", exact: true }).click();
    const previewResponse = await previewResponsePromise;
    expect(previewResponse.status(), `Round ${roundNumber} automatic preview failed`).toBe(200);
    await expect(page.getByRole("heading", { name: "4. Score Entry with Review", exact: true })).toBeVisible();
    await expect(page.getByText(new RegExp(`Round ${roundNumber} is ready with the approved movement`))).toBeVisible();
    await publishRound(page, roundNumber);
  }

  await runConfirmedAction(page, {
    trigger: "Finish session",
    confirm: "Yes, complete session",
    method: "PATCH",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/snapshot`
  });
  await expect(page.getByRole("heading", { name: "Session complete", exact: true })).toBeVisible();

  const after = await fetchDetail(page);
  expect(after.session).toMatchObject({
    id: sessionId,
    league_name: expectedLeagueName,
    status: "complete",
    current_round: 5,
    total_rounds: 5
  });
  expect(after.rounds.map((round) => [round.round_number, round.status])).toEqual([
    [1, "submitted"],
    [2, "submitted"],
    [3, "submitted"],
    [4, "submitted"],
    [5, "submitted"]
  ]);
  expect(after.rounds.every((round) => round.match_date === matchDate)).toBe(true);
  expect(after.rounds.every((round) => Boolean(round.operation_key || round.movement_json?.operation_key))).toBe(true);
  expect(after.rounds.reduce((total, round) => total + Number(round.submitted_match_count || 0), 0)).toBe(15);
  expect(after.publish_operations).toHaveLength(5);
  expect(after.publish_operations?.every((operation) => operation.status === "completed")).toBe(true);
  expect(after.publish_operations?.every((operation) => operation.published_match_ids?.length === 3)).toBe(true);
  const publishedMatchIds = (after.publish_operations || []).flatMap(
    (operation) => operation.published_match_ids || []
  );
  expect(publishedMatchIds).toHaveLength(15);
  expect(new Set(publishedMatchIds).size).toBe(15);
  const cleanup = await cleanupDisposableFixtures(page.request);
  expect(cleanup.leagueStatus).toBe("archived");
  expect(cleanup.inactivePlayerCount).toBe(4);
  cleanupVerified = true;

  writeFileSync(
    reportPath,
    `${JSON.stringify({
      schema_version: 2,
      candidate_sha: String(process.env.GITHUB_SHA || ""),
      environment: "staging",
      production_targets_contacted: false,
      session_id: sessionId,
      league_id: leagueId,
      league_name: expectedLeagueName,
      status: after.session.status,
      current_round: after.session.current_round,
      submitted_rounds: after.rounds.length,
      completed_publish_operations: after.publish_operations?.length || 0,
      unique_published_matches: new Set(publishedMatchIds).size,
      legacy_acceptance_matches_excluded: legacyExcludedMatches,
      isolated_player_ids: createdPlayerIds,
      cleanup: {
        league_status: cleanup.leagueStatus,
        inactive_player_count: cleanup.inactivePlayerCount
      }
    }, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" }
  );
});
