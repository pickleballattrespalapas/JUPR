import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const runFocusedAwardsUi = /^(1|true|yes|on)$/i.test(String(process.env.JUPR_RUN_LEAGUE_AWARDS_UI_E2E || ""));

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("League Awards recovers and advances every persisted step without a browser-side database write", async ({ page }) => {
  test.skip(!runFocusedAwardsUi, "Set JUPR_RUN_LEAGUE_AWARDS_UI_E2E=1 for the focused, API-mocked browser workflow.");

  const awards = [
    {
      category_key: "highest_rating",
      category_label: "Highest Rating",
      player_id: 1,
      player_name: "Alex",
      metric_display: "4.000",
      rank: 1,
      min_games: 2
    }
  ];
  const eligiblePlayers = [
    { player_id: 1, player_name: "Alex" },
    { player_id: 2, player_name: "Blair" }
  ];
  let status = "not_started";
  let revision = 0;
  let preview: Record<string, unknown> | null = null;
  let finalAwards: typeof awards = [];
  let mint: Record<string, unknown> = { status: "not_started", attempt_count: 0, attempts: [] };
  let leagueListReads = 0;
  let awardsStateReads = 0;

  const responsePayload = () => ({
    ok: true,
    league_name: "Awards UI Fixture",
    league: { league_name: "Awards UI Fixture", status: status === "archived" ? "archived" : status === "not_started" ? "active" : "ended", is_active: status === "not_started", min_games: 2 },
    awards: finalAwards.length ? finalAwards : preview ? awards : [],
    award_count: finalAwards.length || (preview ? awards.length : 0),
    eligible_players: eligiblePlayers,
    writes_enabled: true,
    service_role_ready: true,
    badge_definitions_ready: true,
    badge_definition_count: 4,
    badge_definition_required_count: 4,
    missing_badge_ids: [],
    badge_seed_migration: "supabase/migrations/20260720014744_seed_top_performer_badges.sql",
    badge_expected_count: status === "minted" || status === "archived" ? 1 : 0,
    badge_verified_count: status === "minted" || status === "archived" ? 1 : 0,
    wizard: {
      version: 2,
      status,
      revision,
      frozen_at: status === "not_started" ? null : "2026-07-19T12:00:00Z",
      preview,
      final_awards: finalAwards,
      override_notes: {},
      mint,
      archive: status === "archived" ? { status: "archived", archived_at: "2026-07-19T12:05:00Z" } : { status: "not_started" }
    },
    warnings: []
  });

  await page.addInitScript(() => {
    window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: "ui-test-token", user: { email: "ui-test@example.com" } }));
  });
  await page.route("**/admin/auth/capabilities**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        authorized: true,
        user: { email: "ui-test@example.com" },
        requested_club_id: "tres_palapas",
        assignments: [{
          club_id: "tres_palapas",
          role: "club_owner",
          permissions: ["manage_leagues"]
        }]
      })
    });
  });
  await page.route("**/admin/clubs/tres_palapas/league-manager/leagues**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (request.method() === "GET" && pathname.endsWith("/leagues")) {
      leagueListReads += 1;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, leagues: [{ league_name: "Awards UI Fixture", status: "active" }], count: 1 }) });
      return;
    }
    if (request.method() === "GET" && pathname.endsWith("/awards")) {
      awardsStateReads += 1;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(responsePayload()) });
      return;
    }
    if (request.method() === "POST") {
      if (pathname.endsWith("/freeze")) status = "frozen";
      if (pathname.endsWith("/preview")) {
        status = "previewed";
        preview = { awards, award_count: 1, fingerprint: "a".repeat(64), generated_at: "2026-07-19T12:01:00Z" };
      }
      if (pathname.endsWith("/overrides")) {
        status = "overrides_confirmed";
        finalAwards = awards;
      }
      if (pathname.endsWith("/mint")) {
        status = "minted";
        mint = { status: "verified", attempt_count: 1, expected_count: 1, verified_count: 1, attempts: [] };
      }
      if (pathname.endsWith("/archive")) status = "archived";
      revision += 1;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(responsePayload()) });
      return;
    }
    await route.continue();
  });

  await page.goto("/admin/league-manager/awards", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "League awards" })).toBeVisible();
  await expect(page.getByRole("option", { name: "Awards UI Fixture" })).toBeAttached();
  await expect.poll(() => leagueListReads).toBe(1);
  await page.getByLabel("League").selectOption({ label: "Awards UI Fixture" });
  await expect(page.getByText(/Saved step:\s*not started/i)).toBeVisible();
  await expect.poll(() => awardsStateReads).toBe(1);

  await page.getByRole("button", { name: "Freeze and save" }).click();
  await page.getByRole("button", { name: "Yes, freeze league" }).click();
  await page.getByRole("button", { name: "Compute and save preview" }).click();
  await page.getByRole("button", { name: "Confirm winners and reasons" }).click();
  await page.getByRole("button", { name: "Mint and verify" }).click();
  await page.getByRole("button", { name: "Yes, mint and verify" }).click();
  await expect(page.getByText(/Verified 1 of 1 expected row/i)).toBeVisible();
  await page.getByRole("button", { name: "Archive completed league" }).click();
  await page.getByRole("button", { name: "Yes, archive league" }).click();
  await expect(page.getByText(/Archived\. This workflow is read-only/i)).toBeVisible();
});
