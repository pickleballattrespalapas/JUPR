import { writeFileSync } from "node:fs";

import { expect, test, type Page } from "@playwright/test";

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
const sessionId = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_SESSION_ID || ""
).trim();
const retainedOperationId = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_OPERATION_ID || ""
).trim();
const expectedLeagueName = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_LEAGUE_NAME || ""
).trim();
const expectedLeagueId = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_LEAGUE_ID || ""
).trim();
const expectedLeagueType = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_LEAGUE_TYPE || ""
).trim();
const matchDate = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_MATCH_DATE || ""
).trim();
const reportPath = String(
  process.env.JUPR_LEAGUE_LIVE_E2E_REPORT_PATH || ""
).trim();

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
  };
  rounds: Array<{
    round_number: number;
    status: string;
    match_date?: string | null;
    submitted_match_count?: number;
    submitted_match_ids?: string[];
  }>;
  publish_operations?: PublishOperation[];
};

test.describe.configure({ mode: "serial", retries: 0 });
test.skip(
  !allowMutation,
  "Set JUPR_LEAGUE_LIVE_ALLOW_MUTATION_E2E=1 only for an explicitly authorized staging session."
);

async function fetchDetail(page: Page): Promise<LeagueLiveDetail> {
  const response = await page.request.get(
    `${expectedApiOrigin}/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}`,
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
  const reviewScores = page.getByRole("button", { name: "Review scores", exact: true });
  await expect(reviewScores).toBeEnabled();
  await reviewScores.click();
  await expect(page.getByRole("heading", { name: "Review entered scores", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Confirm scores and continue", exact: true }).click();

  await expect(page.getByRole("heading", { name: "5. Movement", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Preview movement", exact: true }).click();
  await expect(page.getByText(/Verified operation key/i)).toBeVisible();
  await page.getByRole("button", { name: "Continue to Repeat or Finish", exact: true }).click();

  await expect(page.getByRole("heading", { name: "6. Repeat or Finish", exact: true })).toBeVisible();
  await runConfirmedAction(page, {
    trigger: "Publish reviewed round",
    confirm: "Yes, publish the round",
    method: "POST",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/rounds/${roundNumber}/submit`
  });
  await expect(page.getByRole("heading", { name: `Round ${roundNumber} published`, exact: true })).toBeVisible();
}

test("recovers retained Round 1 and completes the five-round League Live session", async ({ page, context }) => {
  test.setTimeout(360_000);

  expect(mutationConfirmation).toBe("RUN DISPOSABLE STAGING WRITES");
  expect(expectedApiOrigin).toBe("https://juprleagues-api-staging.fly.dev");
  expect(expectedWebOrigin).toMatch(
    /^https:\/\/[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}-pickleballattrespalapas1\.vercel\.app$/
  );
  expect(adminEmail).not.toBe("");
  expect(adminToken).not.toBe("");
  expect(sessionId).toMatch(/^[0-9a-f-]{36}$/);
  expect(retainedOperationId).toMatch(/^[0-9a-f-]{36}$/);
  expect(expectedLeagueId).toBe("9");
  expect(expectedLeagueName).not.toBe("");
  expect(expectedLeagueType).toBe("Individual");
  expect(matchDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  expect(reportPath).not.toBe("");

  const before = await fetchDetail(page);
  expect(before.session).toMatchObject({
    id: sessionId,
    league_name: expectedLeagueName,
    status: "active",
    current_round: 1,
    total_rounds: 5
  });
  expect(before.rounds).toEqual([]);
  expect(before.publish_operations).toHaveLength(1);
  expect(before.publish_operations?.[0]).toMatchObject({
    id: retainedOperationId,
    round_number: 1,
    status: "retryable",
    published_match_ids: []
  });
  expect(before.publish_operations?.[0].match_context_ids).toHaveLength(3);

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
  liveRoute.searchParams.set("league_id", expectedLeagueId);
  liveRoute.searchParams.set("league", expectedLeagueName);
  liveRoute.searchParams.set("league_name", expectedLeagueName);
  liveRoute.searchParams.set("mode", expectedLeagueType);
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
  const sessionSelect = setupPanel
    .locator("label", { hasText: /^Existing sessions/ })
    .locator("select");
  await expect(sessionSelect).toBeEnabled({ timeout: 30_000 });
  await expect(sessionSelect.locator(`option[value="${sessionId}"]`)).toHaveCount(1);
  await sessionSelect.selectOption(sessionId);
  await expect(page.getByRole("button", { name: "Retry R1", exact: true })).toBeEnabled({ timeout: 30_000 });

  await runConfirmedAction(page, {
    trigger: "Retry R1",
    confirm: "Yes, retry original publish",
    method: "POST",
    pathname: `/admin/clubs/${clubId}/league-manager/live-sessions/${sessionId}/rounds/1/retry`
  });
  await expect(page.getByRole("heading", { name: "Round 1 published", exact: true })).toBeVisible();

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
  expect(after.rounds.reduce((total, round) => total + Number(round.submitted_match_count || 0), 0)).toBe(15);
  expect(after.publish_operations).toHaveLength(5);
  expect(after.publish_operations?.every((operation) => operation.status === "completed")).toBe(true);
  expect(after.publish_operations?.every((operation) => operation.published_match_ids?.length === 3)).toBe(true);
  const publishedMatchIds = (after.publish_operations || []).flatMap(
    (operation) => operation.published_match_ids || []
  );
  expect(publishedMatchIds).toHaveLength(15);
  expect(new Set(publishedMatchIds).size).toBe(15);

  writeFileSync(
    reportPath,
    `${JSON.stringify({
      schema_version: 1,
      candidate_sha: String(process.env.GITHUB_SHA || ""),
      environment: "staging",
      production_targets_contacted: false,
      session_id: sessionId,
      league_id: expectedLeagueId,
      league_name: expectedLeagueName,
      status: after.session.status,
      current_round: after.session.current_round,
      submitted_rounds: after.rounds.length,
      completed_publish_operations: after.publish_operations?.length || 0,
      unique_published_matches: new Set(publishedMatchIds).size
    }, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" }
  );
});
