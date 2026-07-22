import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "staging-admin@example.invalid").trim();
const tournamentId = String(process.env.JUPR_TOURNAMENT_ADMIN_TOURNAMENT_ID || "").trim();
const importedSelectionId = String(process.env.JUPR_TOURNAMENT_ADMIN_IMPORTED_SELECTION_ID || "").trim();
const allowMutationEvidence = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_TOURNAMENT_ADMIN_ALLOW_MUTATION_E2E || ""));

test.describe("order-26 Tournament Admin staging evidence", () => {
  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
    if (adminToken) {
      await context.addInitScript(
        ({ token, email }) => window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: token, token_type: "bearer", user: { email } })),
        { token: adminToken, email: adminEmail }
      );
    }
  });

  test("dedicated setup, management, handoff, and recovery surfaces are explicit", async ({ page }) => {
    await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Tournament Setup Manager" })).toBeVisible();
    await expect(page.getByText(/settings, registration days, event\/division options, builder drafts, publish-impact review/i)).toBeVisible();
    await expect(page.getByRole("link", { name: /Streamlit Tournament Setup fallback/i })).toBeVisible();

    await page.goto("/admin/tournaments", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: /Tournament registration management/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /bulk registration actions/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /status actions/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /Streamlit tournament recovery fallback/i })).toBeVisible();

    await page.goto("/admin/tournaments/registrations", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: /Registration management and reporting/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /Guarded operations import/i })).toBeVisible();
    await expect(page.getByText(/cannot create registrations, update entries, or send email|never writes draw teams/i)).toBeVisible();
  });

  test("authenticated setup impact review is a no-write dry run", async ({ page }) => {
    test.skip(!adminToken || !tournamentId, "Admin bearer token and disposable tournament fixture are required.");
    await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "Load list" }).click();
    const selector = page.getByLabel("Tournament");
    await selector.selectOption(tournamentId);
    await expect(page.getByRole("button", { name: "Reload setup" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "4. Publish setup" })).toBeVisible();
    await page.getByRole("button", { name: "Review publish impact (dry run)" }).click();
    await expect(page.getByRole("status")).toContainText("No rows were written");
    await expect(page.getByText(/0 writes/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Publish setup" })).toBeDisabled();
  });

  test("stale tournament mutation is refused before any write", async ({ request }) => {
    test.skip(!adminToken || !tournamentId, "Admin bearer token and disposable tournament fixture are required.");
    const detail = await request.get(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, { headers: { Authorization: `Bearer ${adminToken}` } });
    expect(detail.ok()).toBeTruthy();
    const current = await detail.json();
    const response = await request.patch(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, {
      headers: { Authorization: `Bearer ${adminToken}` },
      data: { name: current.tournament.name, expected_updated_at: "order-26-deliberately-stale", confirmation_text: "SAVE TOURNAMENT", source: "playwright_tournament_admin_stale" }
    });
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toMatch(/changed after it was loaded|reviewed state/i);
  });

  test("disposable same-value edit replays one deterministic operation", async ({ request }) => {
    test.skip(!allowMutationEvidence || !adminToken || !tournamentId, "Explicit mutation evidence gate and disposable tournament fixture are required.");
    const detail = await request.get(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, { headers: { Authorization: `Bearer ${adminToken}` } });
    const current = await detail.json();
    const body = { name: current.tournament.name, start_date: current.tournament.start_date, end_date: current.tournament.end_date, expected_updated_at: current.tournament.updated_at, confirmation_text: "SAVE TOURNAMENT", source: "playwright_tournament_admin_idempotency" };
    const first = await request.patch(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, { headers: { Authorization: `Bearer ${adminToken}` }, data: body });
    const replay = await request.patch(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, { headers: { Authorization: `Bearer ${adminToken}` }, data: body });
    expect(first.ok()).toBeTruthy(); expect(replay.ok()).toBeTruthy();
    const firstPayload = await first.json(); const replayPayload = await replay.json();
    expect(firstPayload.idempotent_replay).toBe(false);
    expect(replayPayload.idempotent_replay).toBe(true);
    expect(replayPayload.operation_key).toBe(firstPayload.operation_key);
  });

  test("imported selection refuses Registration Admin bypass", async ({ request }) => {
    test.skip(!adminToken || !tournamentId || !importedSelectionId, "An imported disposable selection fixture is required.");
    const detail = await request.get(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`, { headers: { Authorization: `Bearer ${adminToken}` } });
    const payload = await detail.json();
    const selection = payload.selections.find((row: { id: string }) => row.id === importedSelectionId);
    expect(selection).toBeTruthy();
    const response = await request.patch(`${expectedApiOrigin}/admin/clubs/tres_palapas/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/selections/${encodeURIComponent(importedSelectionId)}`, {
      headers: { Authorization: `Bearer ${adminToken}` },
      data: { event_option_id: selection.event_option_id, partner_mode: selection.partner_mode, partner_note: selection.partner_note, expected_updated_at: selection.updated_at, confirmation_text: "SAVE SELECTION", source: "playwright_imported_draw_refusal" }
    });
    expect(response.status()).toBe(400);
    expect((await response.json()).detail).toMatch(/already imported into a draw|Tournament Ops/i);
  });
});
