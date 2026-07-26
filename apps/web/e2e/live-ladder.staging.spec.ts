import { expect, test, type APIRequestContext, type APIResponse } from "@playwright/test";
import { bootstrapStagingContext, clubSlug, expectedApiOrigin } from "./support/staging";

const token = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const runMutations = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_RUN_LIVE_LADDER_MUTATION_E2E || ""));
const clubId = String(process.env.JUPR_LIVE_LADDER_E2E_CLUB_ID || "tres_palapas").trim();
const runId = String(process.env.JUPR_LIVE_LADDER_E2E_RUN_ID || "manual-fixture").replace(/[^A-Za-z0-9_-]/g, "-");

function assertStagingApi(): void {
  const host = new URL(expectedApiOrigin).hostname.toLowerCase();
  expect(host).toContain("staging");
  expect(host).not.toBe("api.pickleballclubsandwich.com");
}

function ratingTuplesFromCanonicalMatch(match: Record<string, any>) {
  const roundJupr = (value: unknown): number | null => {
    const numeric = Number(value);
    return value == null || !Number.isFinite(numeric)
      ? null
      : Math.round((numeric / 400) * 1_000_000) / 1_000_000;
  };
  return [
    ...(match.rating_snapshot?.team_1 || []),
    ...(match.rating_snapshot?.team_2 || [])
  ]
    .map((rating: Record<string, any>) => {
      const before = roundJupr(rating.start_rating);
      const after = roundJupr(rating.end_rating);
      return {
        player_id: Number(rating.player_id),
        before_jupr: before,
        after_jupr: after,
        delta_jupr:
          before == null || after == null
            ? null
            : Math.round((after - before) * 1_000_000) / 1_000_000
      };
    })
    .sort((left, right) => left.player_id - right.player_id);
}

function ratingTuplesFromLadderMatch(match: Record<string, any>) {
  return (match.rating_changes || [])
    .map((rating: Record<string, any>) => ({
      player_id: Number(rating.player_id),
      before_jupr: rating.before_jupr ?? null,
      after_jupr: rating.after_jupr ?? null,
      delta_jupr: rating.delta_jupr ?? null
    }))
    .sort(
      (
        left: { player_id: number },
        right: { player_id: number }
      ) => left.player_id - right.player_id
    );
}

async function apiJson(
  request: APIRequestContext,
  method: "get" | "post" | "patch",
  path: string,
  data?: Record<string, unknown>
): Promise<{ response: APIResponse; body: Record<string, any> }> {
  const response = await request[method](`${expectedApiOrigin}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
    data,
    failOnStatusCode: false
  });
  const body = (await response.json().catch(() => ({}))) as Record<string, any>;
  expect(response.ok(), `${method.toUpperCase()} ${path}: ${JSON.stringify(body)}`).toBeTruthy();
  return { response, body };
}

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("live-ladder admin surfaces render without issuing a write", async ({ page }) => {
  for (const [path, heading, evidence] of [
    ["/admin/challenge-ladder", "Challenge Ladder Admin", /Match Log|Streamlit|guarded/i],
    ["/admin/moneyball", "Moneyball", /Python|Streamlit|guarded/i],
    ["/admin/jupr-live", "JUPR Live Admin", /Tournament Live|Streamlit|guarded/i]
  ] as const) {
    const response = await page.goto(path, { waitUntil: "domcontentloaded" });
    expect(response?.status()).toBeLessThan(400);
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    await expect(page.locator("body")).toContainText(evidence);
    await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
  }
});

test.describe("explicit disposable staging lifecycle", () => {
  test.skip(!runMutations || !token, "Set the mutation opt-in, staging bearer token, and disposable fixture variables from the order-24 runbook.");

  test("Moneyball publish is replayable and reconcilable", async ({ request }) => {
    assertStagingApi();
    const playerIds = String(process.env.JUPR_MONEYBALL_E2E_PLAYER_IDS || "")
      .split(",").map((value) => Number(value.trim())).filter(Number.isInteger);
    test.skip(playerIds.length !== 8 || new Set(playerIds).size !== 8, "JUPR_MONEYBALL_E2E_PLAYER_IDS must contain eight unique disposable staging players.");

    const previewRequest = { player_ids: playerIds, rating_context: "OVERALL", win_rate: 5, point_rate: 2 };
    const { body: preview } = await apiJson(request, "post", `/admin/clubs/${clubId}/moneyball/preview`, previewRequest);
    const first = preview.matches[0];
    const scores = [{ row_id: first.row_id, score_t1: 11, score_t2: 7 }];
    const { body: settlement } = await apiJson(request, "post", `/admin/clubs/${clubId}/moneyball/settlement`, { ...previewRequest, scores });
    const publish = {
      ...previewRequest,
      scores,
      league_name: "Moneyball E2E",
      week_tag: `Moneyball E2E ${runId}`,
      match_type: "Moneyball RR E2E",
      settlement_fingerprint: settlement.settlement_fingerprint,
      expected_version: settlement.settlement_fingerprint,
      idempotency_key: `moneyball-e2e-${runId}`,
      confirmation_text: "SAVE MONEYBALL"
    };
    const { body: submitted } = await apiJson(request, "post", `/admin/clubs/${clubId}/moneyball/submit`, publish);
    const { body: replayed } = await apiJson(request, "post", `/admin/clubs/${clubId}/moneyball/submit`, publish);
    expect(replayed.idempotent_replay).toBe(true);
    expect(replayed.operation_key).toBe(submitted.operation_key);
    const { body: reconciled } = await apiJson(
      request,
      "post",
      `/admin/clubs/${clubId}/moneyball/operations/${submitted.operation_key}/reconcile`,
      { confirmation_text: "RECONCILE MONEYBALL" }
    );
    expect(reconciled.idempotent_replay).toBe(true);
    expect(reconciled.recovery.match_context_ids.length).toBeGreaterThan(0);
  });

  test("Challenge Ladder result recovers and projects the exact public result", async ({
    request,
    page
  }) => {
    assertStagingApi();
    const challengeId = Number(process.env.JUPR_LADDER_E2E_CHALLENGE_ID || 0);
    const partners = String(process.env.JUPR_LADDER_E2E_PARTNER_IDS || "")
      .split(",").map((value) => Number(value.trim())).filter(Number.isInteger);
    test.skip(!challengeId || partners.length !== 2 || new Set(partners).size !== 2, "Provide one accepted disposable challenge and two distinct club swing-partner IDs.");

    const { body: dashboard } = await apiJson(request, "get", `/admin/clubs/${clubId}/challenge-ladder/dashboard`);
    const result = {
      partner_a_challenger_id: partners[0], partner_a_defender_id: partners[1],
      partner_b_challenger_id: partners[1], partner_b_defender_id: partners[0],
      match_a_games: [[11, 7], [11, 8]], match_b_games: [[11, 6], [11, 9]],
      match_date: `2026-07-19T12:00:00Z`, winner_override: "computed", publish_official_matches: true
    };
    const { body: preview } = await apiJson(request, "post", `/admin/clubs/${clubId}/challenge-ladder/challenges/${challengeId}/result/preview`, result);
    const publish = {
      ...result,
      preview_fingerprint: preview.preview_fingerprint,
      expected_version: dashboard.state_version,
      idempotency_key: `ladder-result-e2e-${runId}`,
      confirmation_text: "PUBLISH LADDER RESULT"
    };
    const { body: submitted } = await apiJson(request, "post", `/admin/clubs/${clubId}/challenge-ladder/challenges/${challengeId}/result`, publish);
    const { body: replayed } = await apiJson(request, "post", `/admin/clubs/${clubId}/challenge-ladder/challenges/${challengeId}/result`, publish);
    expect(replayed.idempotent_replay).toBe(true);
    expect(replayed.operation_key).toBe(submitted.operation_key);
    const { body: reconciled } = await apiJson(
      request,
      "post",
      `/admin/clubs/${clubId}/challenge-ladder/operations/${submitted.operation_key}/reconcile`,
      { confirmation_text: "RECONCILE LADDER OPERATION" }
    );
    expect(reconciled.idempotent_replay).toBe(true);

    const matchIds = submitted.public_result_json?.match_ids;
    expect(matchIds?.a).toBeTruthy();
    expect(matchIds?.b).toBeTruthy();
    expect(matchIds.a).not.toBe(matchIds.b);
    const { body: publicLadder } = await apiJson(
      request,
      "get",
      `/clubs/${clubSlug}/challenge-ladder`
    );
    const publicChallenge = (publicLadder.challenge_sections || [])
      .flatMap((section: Record<string, any>) => section.challenges || [])
      .find((challenge: Record<string, any>) => String(challenge.id) === String(challengeId));
    expect(publicChallenge?.bucket).toBe("Recently Completed");
    const details = publicChallenge?.result_details;
    expect(details?.completeness).toBe("full");
    expect(details?.rank_change).toBeTruthy();
    const matchA = details.matches.find(
      (match: Record<string, any>) => match.slot === "a"
    );
    const matchB = details.matches.find(
      (match: Record<string, any>) => match.slot === "b"
    );
    expect(matchA).toMatchObject({
      match_id: matchIds.a,
      score_challenger_team: 22,
      score_defender_team: 15,
      challenger_partner: { player_id: partners[0] },
      defender_partner: { player_id: partners[1] }
    });
    expect(matchB).toMatchObject({
      match_id: matchIds.b,
      score_challenger_team: 22,
      score_defender_team: 15,
      challenger_partner: { player_id: partners[1] },
      defender_partner: { player_id: partners[0] }
    });
    const canonicalMatches = [];
    for (const matchId of [matchIds.a, matchIds.b]) {
      const { body } = await apiJson(
        request,
        "get",
        `/clubs/${clubSlug}/matches/${matchId}`
      );
      canonicalMatches.push(body.match);
    }
    expect(ratingTuplesFromLadderMatch(matchA)).toEqual(
      ratingTuplesFromCanonicalMatch(canonicalMatches[0])
    );
    expect(ratingTuplesFromLadderMatch(matchB)).toEqual(
      ratingTuplesFromCanonicalMatch(canonicalMatches[1])
    );
    expect(JSON.stringify(details)).not.toMatch(
      /operation_key|idempotency_key|context_id|atomic_core|actor_email|refresh_token|access_token/i
    );

    const cardAnchor = `ladder-challenge-${encodeURIComponent(String(challengeId))}`;
    const response = await page.goto(
      `/clubs/${clubSlug}/challenge-ladder?section=challenges&challenge=${encodeURIComponent(
        String(challengeId)
      )}#${cardAnchor}`,
      { waitUntil: "domcontentloaded" }
    );
    expect(response?.status()).toBeLessThan(400);
    const card = page.locator(`#${cardAnchor}`);
    await expect(card).toBeVisible();
    await expect(card.locator('[data-result-details="available"]')).toHaveAttribute(
      "data-result-completeness",
      "full"
    );
    await expect(card).toContainText("Position change:");
    await expect(
      card.locator(`a[href="/clubs/${clubSlug}/matches/${matchIds.a}"]`)
    ).toBeVisible();
    await expect(
      card.locator(`a[href="/clubs/${clubSlug}/matches/${matchIds.b}"]`)
    ).toBeVisible();
    await expect(card).toContainText("JUPR before → after");
  });

  test("one-off JUPR Live scores project publicly and official publish replays once", async ({ request }) => {
    assertStagingApi();
    const playerIds = String(process.env.JUPR_LIVE_E2E_PLAYER_IDS || "")
      .split(",").map((value) => Number(value.trim())).filter(Number.isInteger);
    const names = String(process.env.JUPR_LIVE_E2E_PLAYER_NAMES || "")
      .split(",").map((value) => value.trim()).filter(Boolean);
    test.skip(playerIds.length !== 4 || names.length !== 4, "Provide four linked disposable JUPR Live players and names.");

    const create = {
      title: `JUPR Live E2E ${runId}`, event_type: "round_robin", participant_names: names,
      player_ids: playerIds, total_rounds: 3, court_sizes: [], expected_version: "new",
      idempotency_key: `jupr-live-create-e2e-${runId}`, confirmation_text: "CREATE LIVE SESSION"
    };
    const { body: created } = await apiJson(request, "post", `/admin/clubs/${clubId}/jupr-live/sessions`, create);
    const session = created.session;
    const match = session.state.page_state.event.rounds[0].matches[0];
    const score = {
      scores: [{ match_id: match.id, score_a: 11, score_b: 7 }],
      expected_version: session.version,
      idempotency_key: `jupr-live-score-e2e-${runId}`,
      confirmation_text: "SAVE LIVE SCORES"
    };
    const { body: scored } = await apiJson(request, "patch", `/admin/clubs/${clubId}/jupr-live/sessions/${session.session_key}/scores`, score);
    const { body: publicState } = await apiJson(request, "get", `/clubs/${clubSlug}/live-sessions/${session.session_key}`);
    expect(publicState.session.rounds[0].matches[0]).toMatchObject({ score_a: 11, score_b: 7 });

    const publish = {
      match_date: "2026-07-19T12:00:00Z",
      expected_version: scored.session.version,
      idempotency_key: `jupr-live-publish-e2e-${runId}`,
      confirmation_text: "PUBLISH LIVE MATCHES"
    };
    const { body: submitted } = await apiJson(request, "post", `/admin/clubs/${clubId}/jupr-live/sessions/${session.session_key}/publish`, publish);
    const { body: replayed } = await apiJson(request, "post", `/admin/clubs/${clubId}/jupr-live/sessions/${session.session_key}/publish`, publish);
    expect(replayed.idempotent_replay).toBe(true);
    expect(replayed.operation_key).toBe(submitted.operation_key);
    const { body: reconciled } = await apiJson(
      request,
      "post",
      `/admin/clubs/${clubId}/jupr-live/operations/${submitted.operation_key}/reconcile`,
      { confirmation_text: "RECONCILE LIVE OPERATION" }
    );
    expect(reconciled.idempotent_replay).toBe(true);
  });
});
