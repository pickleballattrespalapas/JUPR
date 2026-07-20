import { randomUUID } from "node:crypto";
import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();
const tournamentId = String(process.env.JUPR_TOURNAMENT_LIVE_TOURNAMENT_ID || "").trim();
const drawId = String(process.env.JUPR_TOURNAMENT_LIVE_DRAW_ID || "").trim();
const gameId = String(process.env.JUPR_TOURNAMENT_LIVE_GAME_ID || "").trim();
const originalScoreA = Number(process.env.JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_A || "");
const originalScoreB = Number(process.env.JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_B || "");
const exerciseScoreA = Number(process.env.JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_A || "");
const exerciseScoreB = Number(process.env.JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_B || "");
const allowMutationEvidence = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E || ""));

const headers = () => ({ Authorization: `Bearer ${adminToken}` });
const snapshotUrl = () => `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/snapshot?draw_id=${encodeURIComponent(drawId)}`;
const commandUrl = () => `${expectedApiOrigin}/admin/clubs/tres_palapas/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/commands`;

test.describe("order-28 Tournament Live staging evidence", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
    if (adminToken) {
      await context.addInitScript(
        ({ token, email }) => window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: token, token_type: "bearer", user: { email } })),
        { token: adminToken, email: adminEmail }
      );
    }
  });

  test("runner is explicitly separate from one-off JUPR Live and keeps fallback visible", async ({ page }) => {
    await page.goto("/admin/tournament-live", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Tournament Live runner" })).toBeVisible();
    await expect(page.getByText(/not JUPR Live|separate from the one-off JUPR Live product/i).first()).toBeVisible();
    await expect(page.getByRole("link", { name: /Streamlit Tournament Live fallback/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /JUPR Live Admin/i })).toBeVisible();
  });

  test("authenticated snapshot is draw-scoped and Python authoritative", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !drawId, "Admin bearer token and disposable tournament/draw fixtures are required.");
    const response = await request.get(snapshotUrl(), { headers: headers() });
    expect(response.status()).toBe(200);
    const snapshot = await response.json();
    expect(snapshot.scope).toBe("draw");
    expect(snapshot.draw_id).toBe(drawId);
    expect(snapshot.authority).toBe("python_fastapi");
    expect(snapshot.product_boundary).toBe("draw_scoped_tournament_runner_not_jupr_live");
    expect(snapshot.state_fingerprint).toMatch(/^[0-9a-f]{64}$/);
    expect(snapshot.readiness.save_score.confirmation).toBe("SAVE SCORE");
    expect(snapshot.operations).toEqual(expect.any(Array));
  });

  test("stale score request returns 409 before an operation is created", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !drawId || !gameId, "Admin bearer token and disposable score fixture are required.");
    const snapshotResponse = await request.get(snapshotUrl(), { headers: headers() });
    expect(snapshotResponse.status()).toBe(200);
    const snapshot = await snapshotResponse.json();
    const draw = snapshot.draws.find((row: { id: string }) => row.id === drawId);
    const game = snapshot.games.find((row: { id: string }) => row.id === gameId);
    expect(draw?.updated_at).toBeTruthy();
    expect(game?.updated_at).toBeTruthy();
    const response = await request.post(commandUrl(), {
      headers: headers(),
      data: {
        command: "save_score",
        expected_state_fingerprint: "0".repeat(64),
        idempotency_key: randomUUID(),
        confirmation_text: "SAVE SCORE",
        expected_draw_updated_at: draw.updated_at,
        expected_game_updated_at: game.updated_at,
        game_id: gameId,
        score_a: 11,
        score_b: 7
      }
    });
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toMatch(/changed after it was loaded|reviewed state/i);
  });

  test("mobile operator view opens the selected draw without horizontal page overflow", async ({ page }) => {
    test.skip(!adminToken || !tournamentId || !drawId, "Admin bearer token and disposable tournament/draw fixtures are required.");
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/admin/tournament-live", { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "Load tournaments" }).click();
    await page.getByLabel("Tournament").selectOption(tournamentId);
    await page.getByRole("button", { name: "Load prepared draws" }).click();
    await page.getByLabel("Draw").selectOption(drawId);
    await page.getByRole("button", { name: "Open authoritative board" }).click();
    await expect(page.getByText(/Reviewed draw version/i)).toBeVisible();
    await expect(page.getByRole("heading", { name: "2. Enter live scores" })).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
    expect(overflow).toBe(false);
  });

  test("opt-in disposable score command replays once and restores its original score", async ({ request }) => {
    const scoresReady = [originalScoreA, originalScoreB, exerciseScoreA, exerciseScoreB].every(Number.isInteger)
      && originalScoreA !== originalScoreB
      && exerciseScoreA !== exerciseScoreB
      && (originalScoreA !== exerciseScoreA || originalScoreB !== exerciseScoreB);
    test.skip(
      !allowMutationEvidence || !adminToken || !tournamentId || !drawId || !gameId || !scoresReady,
      "Explicit mutation gate, disposable pre-playoff game, and distinct original/exercise scores are required."
    );
    let mutationAttempted = false;
    let exerciseKey = "";

    async function loadSnapshot() {
      const response = await request.get(snapshotUrl(), { headers: headers() });
      expect(response.status()).toBe(200);
      return response.json();
    }

    async function saveScore(snapshot: any, state: string, scoreA: number, scoreB: number, idempotencyKey: string) {
      const draw = snapshot.draws.find((row: { id: string }) => row.id === drawId);
      const game = snapshot.games.find((row: { id: string }) => row.id === gameId);
      expect(draw?.updated_at).toBeTruthy();
      expect(game?.updated_at).toBeTruthy();
      return request.post(commandUrl(), {
        headers: headers(),
        data: {
          command: "save_score",
          expected_state_fingerprint: state,
          idempotency_key: idempotencyKey,
          confirmation_text: "SAVE SCORE",
          expected_draw_updated_at: draw.updated_at,
          expected_game_updated_at: game.updated_at,
          game_id: gameId,
          score_a: scoreA,
          score_b: scoreB
        }
      });
    }

    try {
      const before = await loadSnapshot();
      const game = before.games.find((row: { id: string }) => row.id === gameId);
      expect(game?.score_a).toBe(originalScoreA);
      expect(game?.score_b).toBe(originalScoreB);
      const key = randomUUID();
      exerciseKey = key;
      mutationAttempted = true;
      const first = await saveScore(before, before.state_fingerprint, exerciseScoreA, exerciseScoreB, key);
      const replay = await saveScore(before, before.state_fingerprint, exerciseScoreA, exerciseScoreB, key);
      expect(first.status()).toBe(200);
      expect(replay.status()).toBe(200);
      const firstPayload = await first.json();
      const replayPayload = await replay.json();
      expect(firstPayload.idempotent_replay).toBe(false);
      expect(replayPayload.idempotent_replay).toBe(true);
      expect(replayPayload.operation_key).toBe(firstPayload.operation_key);
    } finally {
      if (mutationAttempted) {
        let changed = await loadSnapshot();
        const activeOperation = changed.operations.find(
          (operation: { client_idempotency_key: string; status: string }) =>
            operation.client_idempotency_key === exerciseKey
            && ["intent", "mutated", "recovery_required"].includes(operation.status)
        );
        if (activeOperation) {
          const reconcile = await request.post(
            `${commandUrl().replace(/\/commands$/, "")}/operations/${encodeURIComponent(activeOperation.operation_key)}/reconcile`,
            { headers: headers(), data: { confirmation_text: "RECONCILE TOURNAMENT LIVE" } }
          );
          expect(reconcile.status(), "Disposable score reconciliation failed; stop the staging exercise and inspect the draw lock.").toBe(200);
          changed = await loadSnapshot();
        }
        const changedGame = changed.games.find((row: { id: string }) => row.id === gameId);
        const remainsOriginal = changedGame?.score_a === originalScoreA && changedGame?.score_b === originalScoreB;
        const needsRestore = changedGame?.score_a === exerciseScoreA && changedGame?.score_b === exerciseScoreB;
        expect(
          remainsOriginal || needsRestore,
          "Disposable score has an unexpected value; stop the staging exercise and reconcile the draw."
        ).toBeTruthy();
        if (needsRestore) {
          async function settleRestore(baseSnapshot: any, restoreKey: string) {
            let responseStatus: number | null = null;
            try {
              const response = await saveScore(
                baseSnapshot,
                baseSnapshot.state_fingerprint,
                originalScoreA,
                originalScoreB,
                restoreKey
              );
              responseStatus = response.status();
            } catch {
              // Transport loss is ambiguous. Retain this UUID until durable
              // readback/reconciliation proves completed or not_applied.
            }
            let state = await loadSnapshot();
            let disposition = "";
            const operation = state.operations.find(
              (row: { client_idempotency_key: string; status: string }) =>
                row.client_idempotency_key === restoreKey
                && ["intent", "mutated", "recovery_required"].includes(row.status)
            );
            if (operation) {
              const reconcile = await request.post(
                `${commandUrl().replace(/\/commands$/, "")}/operations/${encodeURIComponent(operation.operation_key)}/reconcile`,
                { headers: headers(), data: { confirmation_text: "RECONCILE TOURNAMENT LIVE" } }
              );
              expect(reconcile.status(), "Disposable score restore reconciliation failed; inspect the draw lock.").toBe(200);
              disposition = String((await reconcile.json()).recovery_disposition || "");
              state = await loadSnapshot();
            }
            return { state, disposition, responseStatus };
          }

          const restoreKey = randomUUID();
          let restoreOutcome = await settleRestore(changed, restoreKey);
          let restoreGame = restoreOutcome.state.games.find((row: { id: string }) => row.id === gameId);
          if (restoreGame?.score_a === exerciseScoreA && restoreGame?.score_b === exerciseScoreB) {
            if (restoreOutcome.disposition === "not_applied" || (restoreOutcome.responseStatus != null && restoreOutcome.responseStatus !== 200)) {
              // Only definitive no-write evidence closes the old UUID. Refresh
              // all versions and use one bounded new cleanup identity.
              restoreOutcome = await settleRestore(restoreOutcome.state, randomUUID());
            } else {
              // No definitive outcome exists, so retry the exact retained request.
              restoreOutcome = await settleRestore(changed, restoreKey);
            }
            restoreGame = restoreOutcome.state.games.find((row: { id: string }) => row.id === gameId);
          }
          expect(restoreGame?.score_a).toBe(originalScoreA);
          expect(restoreGame?.score_b).toBe(originalScoreB);
        }
        const restored = await loadSnapshot();
        const game = restored.games.find((row: { id: string }) => row.id === gameId);
        expect(game?.score_a).toBe(originalScoreA);
        expect(game?.score_b).toBe(originalScoreB);
      }
    }
  });
});
