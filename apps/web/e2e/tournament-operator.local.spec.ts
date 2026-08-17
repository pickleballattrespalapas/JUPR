import { expect, test, type Page } from "@playwright/test";

const tournamentId = "tournament-1";
const drawId = "draw-1";
const secondDrawId = "draw-2";
const selectedQuery = `tournament=${tournamentId}&tournament_name=Staging+Summer+Classic&name=Staging+Summer+Classic&draw=${drawId}`;

const players = [
  { id: 1, name: "Mateo Rivera" },
  { id: 2, name: "Liam Chen" },
  { id: 3, name: "Caleb Nguyen" },
  { id: 4, name: "Diego Alvarez" }
];
const teamsForDraw = (selectedDrawId: string) => [
  { id: "team-a", draw_id: selectedDrawId, team_number: 1, player1_id: 1, player2_id: 2, updated_at: "2026-08-15T12:00:00Z" },
  { id: "team-b", draw_id: selectedDrawId, team_number: 2, player1_id: 3, player2_id: 4, updated_at: "2026-08-15T12:00:00Z" }
];

function games(selectedDrawId = drawId) {
  return Array.from({ length: 21 }, (_, index) => ({
    id: `game-${index + 1}`,
    draw_id: selectedDrawId,
    stage: "ROUND_ROBIN",
    rr_round_number: Math.floor(index / 7) + 1,
    rr_slot_number: (index % 7) + 1,
    team_a_id: "team-a",
    team_b_id: "team-b",
    score_a: index === 0 ? 11 : null,
    score_b: index === 0 ? 7 : null,
    winner_team_id: index === 0 ? "team-a" : null,
    updated_at: "2026-08-15T12:00:00Z"
  }));
}

function blocker(code: string, message: string, count?: number) {
  return { code, scope: "tournament", count, message };
}

function liveSnapshot(selectedDrawId = drawId) {
  const rows = games(selectedDrawId);
  const teams = teamsForDraw(selectedDrawId);
  const publishBlockers = [
    blocker("OPEN_GAMES", "20 tournament games still need a finalized, non-tied score.", 20),
    blocker("PODIUM_INCOMPLETE", "The podium is incomplete and has not been explicitly reviewed."),
    blocker("AWARDS_INCOMPLETE", "Required podium awards are incomplete."),
    blocker("OFFICIAL_MATCHES_INCOMPLETE", "Official Match Log publication and replay evidence are incomplete.")
  ];
  const operations = [
    { operation_key: "op-1", request_fingerprint: "a".repeat(64), client_idempotency_key: "00000000-0000-4000-8000-000000000001", action: "tournament_live", command: "save_score", status: "completed", expected_state: "state", attempt_count: 1, updated_at: "2026-08-15T12:05:00Z", audit_evidence: { actions: ["intent", "completion"], intent_present: true, completion_present: true, failure_present: false } }
  ];
  const readiness = {
    save_score: { ready: true, confirmation: "SAVE SCORE", blockers: [] },
    generate_round_robin: { ready: false, confirmation: "GENERATE GAMES", blockers: ["Games already exist."] },
    generate_playoffs: { ready: false, confirmation: "GENERATE PLAYOFFS", blockers: ["Finish 20 open round-robin games."] },
    generate_podium: { ready: false, confirmation: "GENERATE PODIUM", blockers: ["Finish every game first."] },
    award_podium: { ready: false, confirmation: "AWARD PODIUM", blockers: ["Generate and review the podium first."] },
    publish_official_matches: { ready: false, confirmation: "PUBLISH MATCHES", blockers: publishBlockers.map((row) => row.message) }
  };
  return {
    ok: true,
    mode: "tournament_live_draw_snapshot",
    scope: "draw",
    authority: "python_fastapi",
    product_boundary: "draw_scoped_tournament_runner_not_jupr_live",
    tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", updated_at: "2026-08-15T12:00:00Z" },
    draw_id: selectedDrawId,
    summary: { draws: 2, teams: 2, games: 21, podium: 0, completed_games: 1 },
    draws: [
      { id: drawId, tournament_id: tournamentId, name: "Manual Acceptance Draw", status: "DRAFT", updated_at: "2026-08-15T12:00:00Z" },
      { id: secondDrawId, tournament_id: tournamentId, name: "Open Division Draw", status: "DRAFT", updated_at: "2026-08-15T12:00:00Z" }
    ],
    teams,
    games: rows,
    podium: [],
    players,
    state_fingerprint: "b".repeat(64),
    runtime: { enabled: true, status: "staging_write_ready", authority: "python_fastapi", product_boundary: "draw_scoped_tournament_runner_not_jupr_live", club_id: "tres_palapas", environment: "local_test_harness", staging_only: true, writes_enabled: true, service_role_ready: true, operation_store_ready: true, audit_store_ready: true, write_flag: { name: "LOCAL", enabled: true }, streamlit_fallback_url: "#", warnings: [] },
    progression: { phase: "live", open_games: 20, completed_games: 1, published_games: 0, expected_awards: 6, verified_awards: 0 },
    readiness,
    active_operation: null,
    operations,
    lifecycle: {
      contract: "jupr:tournament-lifecycle:v1",
      authority: "python_fastapi",
      scope: "tournament",
      tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", updated_at: "2026-08-15T12:00:00Z" },
      phase: "live_in_progress",
      counts: { draws: 2, teams: 4, games: 42, finalized_games: 2, open_games: 40, tied_games: 0, podium_entries: 0, expected_awards: 12, verified_awards: 0, unexpected_awards: 0, published_games: 0, unpublished_games: 42, duplicate_publications: 0, active_operations: 0, uncertain_operations: 0 },
      draws: [
        { draw_id: drawId, name: "Manual Acceptance Draw", status: "DRAFT", protected: false, counts: { games: 21, finalized_games: 1, open_games: 20, published_games: 0, duplicate_publications: 0 }, standings: [], podium: [], states: { live_operations: "in_progress", official_publish: "blocked" }, operations, review_evidence: null, readiness: { official_publish: { ready: false, blockers: publishBlockers }, archive: { ready: false, blockers: publishBlockers } } },
        { draw_id: secondDrawId, name: "Open Division Draw", status: "DRAFT", protected: false, counts: { games: 21, finalized_games: 1, open_games: 20, published_games: 0, duplicate_publications: 0 }, standings: [], podium: [], states: { live_operations: "in_progress", official_publish: "blocked" }, operations, review_evidence: null, readiness: { official_publish: { ready: false, blockers: publishBlockers }, archive: { ready: false, blockers: publishBlockers } } }
      ],
      domain_readiness: { official_publish: { ready: false, blockers: publishBlockers }, archive: { ready: false, blockers: [...publishBlockers, blocker("ARCHIVE_OFFICIAL_LINKS", "All tournament games require exactly one official Match Log link before archive.")] } },
      runtime_capability: { writes_enabled: true, official_publish_enabled: true },
      evidence: { operations }
    }
  };
}

const detail = {
  ok: true,
  tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", registration_status: "closed", start_date: "2026-09-01", end_date: "2026-09-02", updated_at: "2026-08-15T12:00:00Z" },
  settings: { location_name: "Tres Palapas", timezone: "America/Chicago" },
  days: [{ id: "day-1" }, { id: "day-2" }],
  event_options: Array.from({ length: 7 }, (_, index) => ({ id: `event-${index + 1}` })),
  registrations: [],
  selections: [],
  summary: { registrations: 32, selections: 32, by_registration_status: { confirmed: 32 }, by_payment_status: { offline: 32 } }
};

const checkInDays = [
  { id: "day-1", label: "Tuesday — Gender Doubles", event_date: "2026-09-01", sort_order: 0 },
  { id: "day-2", label: "Wednesday — Mixed & Open Doubles", event_date: "2026-09-02", sort_order: 1 }
];

function checkInRegistrant(options: {
  registrationId: string;
  dayId: string;
  playerId: number;
  name: string;
  status: "EXPECTED" | "CHECKED_IN" | "ABSENT";
  eventLabel: string;
}) {
  return {
    registration_id: options.registrationId,
    registration_day_id: options.dayId,
    registration_status: "CONFIRMED",
    registration_updated_at: "2026-08-15T12:00:00Z",
    attendance_status: options.status,
    original_registrant: { player_id: options.playerId, name: options.name },
    attendee: { player_id: options.playerId, name: options.name, is_approved_substitute: false },
    substitution: {
      allowed: true,
      event_policy_allows: true,
      blocker: { code: "NONE", status: "COMPLETE", title: "Available", detail: "" }
    },
    check_in: {
      registration_day_id: options.dayId,
      attendance_status: options.status,
      checked_in: options.status === "CHECKED_IN",
      notes: null,
      updated_at: "2026-08-15T12:00:00Z",
      updated_by: "operator@example.invalid",
      identity_current: true,
      requires_reconfirmation: false
    },
    waiver: { verified: options.status === "CHECKED_IN", subject: "attending_player", subject_name: options.name },
    payment: { status: "PAID", source: "offline_payment_tracking", ready: true },
    events: [{
      selection_id: `selection-${options.registrationId}`,
      event_option_id: `event-${options.dayId}`,
      event_label: options.eventLabel,
      team_state: "NOT_REQUIRED",
      partner_name: null,
      entered_partner_name: null,
      blockers: []
    }],
    blockers: []
  };
}

function checkInSnapshot(dayId: string) {
  const selectedDay = checkInDays.find((day) => day.id === dayId) || checkInDays[0];
  const dayOne = selectedDay.id === "day-1";
  const registrants = dayOne
    ? [
        checkInRegistrant({ registrationId: "registration-1", dayId: selectedDay.id, playerId: 1, name: "Mateo Rivera", status: "CHECKED_IN", eventLabel: "Men's Doubles" }),
        checkInRegistrant({ registrationId: "registration-8", dayId: selectedDay.id, playerId: 2, name: "Jordan Lee", status: "EXPECTED", eventLabel: "Mixed Doubles" })
      ]
    : [
        checkInRegistrant({ registrationId: "registration-4", dayId: selectedDay.id, playerId: 4, name: "Diego Alvarez", status: "ABSENT", eventLabel: "Open Doubles" })
      ];
  const unresolvedParticipants = dayOne
    ? [{ kind: "NEEDS_PARTNER", registration_id: "registration-8", registration_name: "Jordan Lee", selection_id: "selection-registration-8", event_label: "Mixed Doubles", entered_partner_name: "", title: "Partner unresolved", detail: "Jordan Lee still needs a confirmed partner." }]
    : [];
  return {
    ok: true,
    mode: "tournament_registration_check_in",
    authority: "python_fastapi_supabase",
    tournament: { id: tournamentId, name: "Staging Summer Classic", status: "LIVE", start_date: "2026-09-01", end_date: "2026-09-02" },
    day_scope: { selected_day_id: selectedDay.id, selected_day: selectedDay, available_days: checkInDays },
    summary: {
      expected: registrants.length,
      checked_in: registrants.filter((row) => row.attendance_status === "CHECKED_IN").length,
      not_checked_in: registrants.filter((row) => row.attendance_status === "EXPECTED").length,
      absent: registrants.filter((row) => row.attendance_status === "ABSENT").length,
      unresolved: unresolvedParticipants.length
    },
    registrants,
    player_options: players,
    inactive_registrants: [],
    unresolved_participants: unresolvedParticipants,
    readiness: {
      schedule: { status: "COMPLETE", timezone: "America/Chicago", active_day_count: 1, blockers: [], days: [selectedDay] },
      draws: { status: "COMPLETE", active_division_count: 1, draw_count: 1, blockers: [] },
      staffing: { status: "NEEDS_REVIEW", source: "no_authoritative_staffing_record", blockers: [{ code: "STAFFING_REVIEW_REQUIRED", status: "NEEDS_REVIEW", title: "Staffing needs review", detail: `Confirm staffing for ${selectedDay.label}.` }] }
    },
    completed_items: [],
    blockers: [],
    runtime: { writes_enabled: true }
  };
}

async function installMockApi(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem("jupr_admin_session_v1", JSON.stringify({
      access_token: "local-operator-token",
      token_type: "bearer",
      expires_at: Date.now() + 3_600_000,
      capabilities: { authorized: true, user: { email: "operator@example.invalid" }, assignments: [{ club_id: "tres_palapas", role: "admin", permissions: ["*"] }] },
      user: { email: "operator@example.invalid" }
    }));
  });
  await page.route("http://127.0.0.1:3999/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/admin/auth/capabilities") {
      await route.fulfill({ json: { authorized: true, user: { email: "operator@example.invalid" }, assignments: [{ club_id: "tres_palapas", role: "admin", permissions: ["*"] }] } });
      return;
    }
    if (url.pathname.endsWith("/tournaments/admin/ops/tournaments")) {
      await route.fulfill({ json: { ok: true, tournaments: [detail.tournament], count: 1 } });
      return;
    }
    if (url.pathname.endsWith(`/tournaments/admin/tournaments/${tournamentId}`) && request.method() === "GET") {
      await route.fulfill({ json: detail });
      return;
    }
    if (url.pathname.includes(`/tournament-live/tournaments/${tournamentId}/snapshot`)) {
      await route.fulfill({ json: liveSnapshot(url.searchParams.get("draw_id") || drawId) });
      return;
    }
    if (url.pathname.endsWith(`/tournament-live/tournaments/${tournamentId}/check-in`) && request.method() === "GET") {
      await route.fulfill({ json: checkInSnapshot(url.searchParams.get("day_id") || "day-1") });
      return;
    }
    if (url.pathname.endsWith(`/tournament-live/tournaments/${tournamentId}/draws/${drawId}/commands`)) {
      await route.fulfill({ json: { ok: true, operation_key: "saved-score-op", idempotent_replay: false } });
      return;
    }
    await route.fulfill({ status: 404, json: { detail: `Unmocked local operator API: ${request.method()} ${url.pathname}` } });
  });
}

test.beforeEach(async ({ page }) => {
  await installMockApi(page);
});

test("Home shows authoritative 1 of 21 truth and preserves selected context", async ({ page }) => {
  await page.goto(`/admin/tournaments/tournament?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Continue scoring" })).toBeVisible();
  await expect(page.getByText("1 of 21 games scored; 20 open.")).toBeVisible();
  await expect(page.getByText("Publish blockers")).toBeVisible();
  const scoringHref = await page.getByRole("link", { name: "Continue scoring" }).getAttribute("href");
  expect(scoringHref).toContain(`tournament=${tournamentId}`);
  expect(scoringHref).toContain(`draw=${drawId}`);
});

test("Draw selection stays inside the tournament workspace and survives reload", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/draws?${selectedQuery}`);
  await expect(page.getByRole("region", { name: "Tournament operating scope" })).toBeVisible();
  await expect(page.getByText("Locked to this tournament workspace")).toBeVisible();
  await expect(page.getByLabel("Tournament")).toHaveCount(0);
  await expect(page.getByText("Change or refresh selection")).toHaveCount(0);

  const drawSelector = page.getByLabel("Working draw");
  await expect(drawSelector).toHaveValue(drawId);
  await expect(drawSelector.locator(`option[value="${drawId}"]`)).toHaveText("Manual Acceptance Draw · In progress · 1 of 21 scored");
  await drawSelector.selectOption(secondDrawId);
  await expect(page).toHaveURL(new RegExp(`tournament=${tournamentId}`));
  await expect(page).toHaveURL(new RegExp(`draw=${secondDrawId}`));
  await expect(drawSelector).toHaveValue(secondDrawId);
  await expect(page.getByRole("heading", { name: "Staging Summer Classic draws and schedule" })).toBeVisible();
  await expect(page.getByText("Open Division Draw", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Live scoring" })).toHaveAttribute("href", new RegExp(`tournament=${tournamentId}.*draw=${secondDrawId}`));

  await page.getByRole("button", { name: "Refresh available draws" }).click();
  await expect(drawSelector).toHaveValue(secondDrawId);
  await page.reload();
  await expect(page.getByLabel("Working draw")).toHaveValue(secondDrawId);
});

test("Preflight check-in changes day without retaining old cards or losing context", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/check-in?${selectedQuery}&day_id=day-1`);
  await expect(page.getByRole("heading", { name: "Staging Summer Classic preflight and check-in" })).toBeVisible();
  const summary = page.getByRole("region", { name: "Check-in summary" });
  await expect(summary.getByText("Expected today")).toBeVisible();
  await expect(summary.getByText("Checked in")).toBeVisible();
  await expect(summary.getByText("Not checked in")).toBeVisible();
  await expect(summary.getByText("Absent")).toBeVisible();
  await expect(summary.getByText("Unresolved")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Player check-in" })).toBeVisible();
  await expect(page.getByText("Partner unresolved")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Mateo Rivera" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Diego Alvarez" })).toHaveCount(0);

  await page.getByLabel("Tournament day").selectOption("day-2");
  await expect(page).toHaveURL(/day_id=day-2/);
  await expect(page).toHaveURL(new RegExp(`tournament=${tournamentId}`));
  await expect(page).toHaveURL(new RegExp(`draw=${drawId}`));
  await expect(page.getByRole("heading", { name: "Diego Alvarez" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Mateo Rivera" })).toHaveCount(0);
  await expect(page.getByText("Partner unresolved")).toHaveCount(0);
  await expect(page.getByText("Absent", { exact: true }).last()).toBeVisible();

  await page.reload();
  await expect(page.getByLabel("Tournament day")).toHaveValue("day-2");
  await expect(page.getByRole("heading", { name: "Diego Alvarez" })).toBeVisible();
  const scoring = page.getByRole("link", { name: "Live scoring" });
  await expect(scoring).toHaveAttribute("href", new RegExp(`tournament=${tournamentId}.*draw=${drawId}`));
});

test("A 9–9 tie never opens confirmation; a valid score shows the full scorecard", async ({ page }) => {
  await page.goto(`/admin/tournament-live?${selectedQuery}`);
  await expect(page.getByText(/Mateo Rivera \/ Liam Chen.*Caleb Nguyen \/ Diego Alvarez/).first()).toBeVisible();
  await page.getByRole("button", { name: "Enter score" }).first().click();
  await page.getByLabel("Team A score").fill("9");
  await page.getByLabel("Team B score").fill("9");
  await page.getByRole("button", { name: "Review score" }).click();
  await expect(page.getByText(/tied or invalid score cannot be reviewed or saved/i)).toBeVisible();
  await expect(page.getByRole("button", { name: "Confirm & save" })).toHaveCount(0);
  await page.getByLabel("Team A score").fill("11");
  await page.getByLabel("Team B score").fill("7");
  await page.getByRole("button", { name: "Review score" }).click();
  await expect(page.getByText("Proposed winner:")).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit score" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Confirm & save" })).toBeVisible();
});

test("Corrections & recovery shows before/after and durable evidence", async ({ page }) => {
  await page.goto(`/admin/tournaments/live-operations/corrections?${selectedQuery}`);
  await page.getByRole("button", { name: "Correct score" }).first().click();
  await expect(page.getByText(/Before correction:/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent operations and reconciliation" })).toBeVisible();
  await expect(page.getByText("Technical operation evidence").first()).toBeVisible();
  await expect(page.getByText(/Match Log corrections are for official published matches/)).toBeVisible();
});

test("Publish and archive remain blocked even when runtime writes are available", async ({ page }) => {
  await page.goto(`/admin/tournaments/ops/publish?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Tournament readiness" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Runtime capability" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Publish official matches" })).toBeDisabled();
  await page.goto(`/admin/tournaments/publish/closeout?${selectedQuery}`);
  await expect(page.getByRole("heading", { name: "Archive unavailable" })).toBeVisible();
  await expect(page.getByText("No archive write is available.")).toBeVisible();
  await expect(page.getByText("Payments, extras, and fulfillment")).toBeVisible();
});

for (const width of [1024, 1280, 1440]) {
  test(`operator routes have no page-level overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const path of ["/admin/tournaments/live-operations", "/admin/tournaments/live-operations/check-in", "/admin/tournament-live", "/admin/tournaments/live-operations/corrections", "/admin/tournaments/ops/publish"]) {
      await page.goto(`${path}?${selectedQuery}`);
      await expect(page.locator("body")).toContainText("Staging Summer Classic");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), path).toBe(true);
      expect(await page.locator("[data-nextjs-dialog]").count(), `${path} error overlay`).toBe(0);
    }
  });
}
