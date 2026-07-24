import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();
const tournamentId = String(process.env.JUPR_TOURNAMENT_OPS_TOURNAMENT_ID || "").trim();
const drawId = String(process.env.JUPR_TOURNAMENT_OPS_DRAW_ID || "").trim();
const gameId = String(process.env.JUPR_TOURNAMENT_OPS_GAME_ID || "").trim();
const allowMutationEvidence = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_TOURNAMENT_OPS_ALLOW_MUTATION_E2E || ""));

const authHeaders = () => ({ Authorization: `Bearer ${adminToken}` });

test.describe("order-27 Tournament Operations staging evidence", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
    if (adminToken) {
      await context.addInitScript(
        ({ token, email }) => window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: token, token_type: "bearer", user: { email } })),
        { token: adminToken, email: adminEmail }
      );
    }
  });

  test("route-specific operations surfaces remain independently addressable", async ({ page }) => {
    const routes = [
      ["/admin/tournaments/ops/draws", "Draws, scoring, playoffs, and podiums"],
      ["/admin/tournaments/ops/import", "Registration and bulk team imports"],
      ["/admin/tournaments/ops/results", "Review and import DUPR results"],
      ["/admin/tournaments/ops/publish", "Publish official tournament matches"],
    ] as const;
    for (const [route, heading] of routes) {
      await page.goto(route, { waitUntil: "domcontentloaded" });
      await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
      await expect(page.getByRole("link", { name: "Operations cockpit" })).toBeVisible();
    }
  });

  test("DUPR preview is authenticated and writes zero rows", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !drawId, "Admin token and disposable tournament/draw fixtures are required.");
    const response = await request.post(
      `${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/results-import/preview`,
      {
        headers: authHeaders(),
        data: {
          raw_text: "playerA1,playerB1,teamAGame1,teamBGame1\nOrder27 Alpha,Order27 Beta,11,7",
          import_mode: "REPLACE",
        },
      }
    );
    expect(response.ok()).toBeTruthy();
    const payload = await response.json();
    expect(payload.dry_run).toBe(true);
    expect(payload.write_count).toBe(0);
    expect(payload.review_fingerprint).toMatch(/^[a-f0-9]{64}$/);
  });

  test("read-only ops snapshot resolves the exact staging draw", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !drawId, "Admin token and exact staging tournament/draw fixtures are required.");
    const response = await request.get(
      `${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops?draw_id=${encodeURIComponent(drawId)}`,
      { headers: authHeaders() }
    );
    expect(response.ok()).toBeTruthy();
    const payload = await response.json();
    expect(payload.mode).toBe("tournament_ops_snapshot");
    expect(payload.tournament?.id).toBe(tournamentId);
    expect(payload.draw_id).toBe(drawId);
    expect(payload.draws).toEqual([expect.objectContaining({ id: drawId, tournament_id: tournamentId })]);
    expect(payload.state_ready).toBe(true);
  });

  test("DUPR preview is blocked while write wave is none", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !drawId, "Admin token and exact staging tournament/draw fixtures are required.");
    const response = await request.post(
      `${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/results-import/preview`,
      {
        headers: authHeaders(),
        data: {
          raw_text: "playerA1,playerB1,teamAGame1,teamBGame1\nTest A,Test B,11,7",
          import_mode: "REPLACE",
        },
      }
    );
    expect(response.status()).toBe(403);
    expect((await response.json()).detail).toMatch(/outside the selected staging write wave/i);
  });

  test("stale score CAS is refused before mutation", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !gameId, "Admin token and disposable tournament/game fixtures are required.");
    const snapshotResponse = await request.get(
      `${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops`,
      { headers: authHeaders() }
    );
    expect(snapshotResponse.ok()).toBeTruthy();
    const snapshot = await snapshotResponse.json();
    const response = await request.patch(
      `${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/games/${encodeURIComponent(gameId)}/score`,
      {
        headers: authHeaders(),
        data: {
          score_a: 11,
          score_b: 7,
          expected_state_fingerprint: snapshot.state_fingerprint,
          expected_game_updated_at: "order-27-deliberately-stale",
          confirmation_text: "SAVE SCORE",
          source: "playwright_tournament_ops_stale_score",
        },
      }
    );
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toMatch(/changed|stale|reload/i);
  });

  test("mutating fixture suite is explicitly opt-in", async () => {
    test.skip(!allowMutationEvidence, "Set JUPR_TOURNAMENT_OPS_ALLOW_MUTATION_E2E=1 only for the disposable staging acceptance book.");
    expect(adminToken && tournamentId && drawId).toBeTruthy();
  });
});
