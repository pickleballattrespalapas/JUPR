import { expect, test, type Page, type Route } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const listPath = "/admin/clubs/tres_palapas/tournaments/setup/tournaments";

type SetupDraft = {
  days: Array<Record<string, unknown>>;
  event_families: Array<Record<string, unknown>>;
  event_options: Array<Record<string, unknown>>;
};

function initialDraft(): SetupDraft {
  return {
    days: [
      {
        id: "day-1",
        label: "Friday",
        event_date: "2026-11-20",
        enabled: true,
        sort_order: 1
      }
    ],
    event_families: [
      {
        id: "family-1",
        event_family: "Mixed Doubles",
        participant_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        default_format: "ROUND_ROBIN_PLUS_PLAYOFF",
        default_scoring: "GAME_TO_15",
        default_capacity_teams: 16,
        default_price_usd: 0,
        default_waitlist: true,
        default_partner_board: true,
        default_status: "open"
      }
    ],
    event_options: [
      {
        id: "event-1",
        registration_day_id: "day-1",
        event_family_label: "Mixed Doubles",
        division_name: "Mixed Doubles Open",
        event_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        skill_label: "Open",
        age_mode: "ALL_AGES",
        age_label: "All Ages",
        capacity_teams: 16,
        price_usd: 0,
        waitlist_enabled: true,
        partner_board_enabled: true,
        status: "open",
        enabled: true,
        sort_order: 1,
        unknown_backend_field: { retained: true }
      }
    ]
  };
}

function detailPayload(draft: SetupDraft, id = "tour-1", name = "Setup smoke") {
  return {
    ok: true,
    tournament: { id, name, status: "draft" },
    settings: {
      registration_slug: "setup-smoke",
      registration_status: "draft",
      waitlist_enabled: true,
      partner_board_enabled: true
    },
    days: draft.days,
    event_options: draft.event_options,
    builder_draft: {
      days: draft.days,
      event_families: draft.event_families,
      divisions: draft.event_options
    },
    publish_impact: { summary: { days: draft.days.length, divisions: draft.event_options.length } },
    registration_count: 0,
    state_fingerprint: "a".repeat(64),
    templates: []
  };
}

async function seedAdminSession(page: Page) {
  await page.route("**/admin/auth/capabilities**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        authorized: true,
        user: { email: "setup-ui@example.invalid" },
        requested_club_id: "tres_palapas",
        assignments: [{
          club_id: "tres_palapas",
          role: "club_owner",
          permissions: ["manage_tournaments"]
        }]
      })
    });
  });
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify({
        access_token: "setup-ui-token",
        token_type: "bearer",
        user: { email: "setup-ui@example.invalid" }
      })
    );
  });
}

async function fulfillList(route: Route, rows = [{ id: "tour-1", name: "Setup smoke", status: "draft" }]) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ ok: true, tournaments: rows, count: rows.length })
  });
}

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("guided Tournament Setup preserves payloads, validates rows, and works at a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await seedAdminSession(page);

  let draft = initialDraft();
  let listReads = 0;
  let detailReads = 0;
  let draftWrites = 0;
  let savedPayload: Record<string, unknown> | null = null;

  await page.route("**/admin/clubs/tres_palapas/tournaments/setup/tournaments**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === listPath) {
      listReads += 1;
      await fulfillList(route);
      return;
    }
    if (request.method() === "GET" && url.pathname === `${listPath}/tour-1`) {
      detailReads += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detailPayload(draft))
      });
      return;
    }
    if (request.method() === "PUT" && url.pathname === `${listPath}/tour-1/draft`) {
      draftWrites += 1;
      savedPayload = request.postDataJSON() as Record<string, unknown>;
      draft = {
        days: savedPayload.days as SetupDraft["days"],
        event_families: savedPayload.event_families as SetupDraft["event_families"],
        event_options: savedPayload.event_options as SetupDraft["event_options"]
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true, mode: "tournament_setup_draft_save", builder_draft: draft })
      });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "unexpected setup UI request" }) });
  });

  await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Tournament Setup is disabled")).toHaveCount(0);

  await expect(page.getByRole("option", { name: /Setup smoke/ })).toBeAttached();
  await expect(page.getByRole("heading", { name: "Draft summary" })).toBeVisible();
  await expect.poll(() => listReads).toBe(1);
  await expect.poll(() => detailReads).toBe(1);
  await expect(page.locator("details", { hasText: "Advanced JSON import/export" })).not.toHaveAttribute("open", "");

  const divisionName = page.getByLabel("Division name").first();
  await divisionName.fill("Mixed 3.5");
  await page.getByRole("button", { name: "Refresh list" }).click();
  await expect.poll(() => listReads).toBe(2);
  expect(detailReads).toBe(1);
  await expect(divisionName).toHaveValue("Mixed 3.5");
  await expect(page.getByRole("status")).toContainText("Unsaved setup edits were preserved.");
  await page.getByRole("button", { name: "Add day" }).click();
  const dayLabels = page.getByLabel("Day label");
  await expect(dayLabels).toHaveCount(2);
  await dayLabels.nth(1).fill("Friday");
  await expect(page.getByText("Day labels must be unique.").first()).toBeVisible();
  await dayLabels.nth(1).fill("Saturday");
  await expect(page.getByText("Day labels must be unique.")).toHaveCount(0);
  await expect(divisionName).toHaveValue("Mixed 3.5");

  const secondDay = page.getByRole("group", { name: /Day 2: Saturday/ });
  await secondDay.getByRole("button", { name: "Remove day" }).click();
  await page.getByRole("button", { name: "Yes, remove day" }).click();
  await expect(dayLabels).toHaveCount(1);
  await expect(divisionName).toHaveValue("Mixed 3.5");

  await page.getByRole("button", { name: "Save draft" }).click();
  await page.getByRole("button", { name: "Yes, save draft" }).click();
  await expect.poll(() => draftWrites).toBe(1);
  await expect.poll(() => detailReads).toBe(2);
  await expect(page.getByLabel("Division name").first()).toHaveValue("Mixed 3.5");

  const capturedPayload = savedPayload as unknown as Record<string, unknown>;
  const savedEvents = capturedPayload.event_options as Array<Record<string, unknown>>;
  expect(savedEvents).toHaveLength(1);
  expect(savedEvents[0]).toMatchObject({
    id: "event-1",
    division_name: "Mixed 3.5",
    unknown_backend_field: { retained: true }
  });
  expect(capturedPayload.confirmation_text).toBe("SAVE SETUP DRAFT");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  );
  expect(overflow).toBe(false);
});

test("Tournament Setup waits for an authenticated session", async ({ page }) => {
  let listReads = 0;
  await page.route("**/admin/clubs/tres_palapas/tournaments/setup/tournaments**", async (route) => {
    listReads += 1;
    await fulfillList(route);
  });

  await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
  await expect(page.getByText(/not signed in|Checking session/i).first()).toBeVisible();
  await page.waitForTimeout(500);
  expect(listReads).toBe(0);
});

test("Tournament Setup ignores a deferred authenticated response after logout", async ({ page }) => {
  await seedAdminSession(page);
  let listReads = 0;
  let releaseList!: () => void;
  const waitForRelease = new Promise<void>((resolve) => {
    releaseList = resolve;
  });

  await page.route("**/admin/clubs/tres_palapas/tournaments/setup/tournaments**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === listPath) {
      listReads += 1;
      await waitForRelease;
      await fulfillList(route, [{ id: "late-tour", name: "Late tournament", status: "draft" }]);
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });

  await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
  await expect.poll(() => listReads).toBe(1);
  await page.evaluate(() => {
    window.localStorage.removeItem("jupr_admin_session_v1");
    window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
  });
  await expect(page.getByText("not signed in")).toBeVisible();
  releaseList();
  await page.waitForTimeout(100);
  await expect(page.getByRole("option", { name: /Late tournament/ })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Draft summary" })).toHaveCount(0);
});

test("Tournament Setup exposes empty and failed list states with a working retry", async ({ page }) => {
  await seedAdminSession(page);
  const draft = initialDraft();
  let listReads = 0;
  let detailReads = 0;

  await page.route("**/admin/clubs/tres_palapas/tournaments/setup/tournaments**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === listPath) {
      listReads += 1;
      if (listReads === 1) {
        await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Temporary setup list failure." }) });
      } else if (listReads === 2) {
        await fulfillList(route, []);
      } else {
        await fulfillList(route);
      }
      return;
    }
    if (request.method() === "GET" && url.pathname === `${listPath}/tour-1`) {
      detailReads += 1;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(detailPayload(draft)) });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });

  await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("status")).toContainText("Temporary setup list failure.");
  await page.getByRole("button", { name: "Refresh list" }).click();
  await expect(page.getByRole("status")).toContainText("No tournaments are available for setup.");
  await page.getByRole("button", { name: "Refresh list" }).click();
  await expect(page.getByRole("option", { name: /Setup smoke/ })).toBeAttached();
  await expect(page.getByRole("heading", { name: "Draft summary" })).toBeVisible();
  expect(listReads).toBe(3);
  expect(detailReads).toBe(1);
});

test("Tournament Setup clears the prior record while a new selection loads", async ({ page }) => {
  await seedAdminSession(page);
  const firstDraft = initialDraft();
  const secondDraft = initialDraft();
  secondDraft.event_options[0] = {
    ...secondDraft.event_options[0],
    id: "event-2",
    division_name: "Second tournament division"
  };
  let releaseSecond!: () => void;
  const waitForSecond = new Promise<void>((resolve) => {
    releaseSecond = resolve;
  });

  await page.route("**/admin/clubs/tres_palapas/tournaments/setup/tournaments**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === listPath) {
      await fulfillList(route, [
        { id: "tour-1", name: "First setup", status: "draft" },
        { id: "tour-2", name: "Second setup", status: "draft" }
      ]);
      return;
    }
    if (request.method() === "GET" && url.pathname === `${listPath}/tour-1`) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detailPayload(firstDraft, "tour-1", "First setup"))
      });
      return;
    }
    if (request.method() === "GET" && url.pathname === `${listPath}/tour-2`) {
      await waitForSecond;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detailPayload(secondDraft, "tour-2", "Second setup"))
      });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });

  await page.goto("/admin/tournament-setup", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Loaded setup: First setup")).toBeVisible();
  await page.getByLabel("Tournament").selectOption("tour-2");
  await expect(page.getByRole("heading", { name: "Draft summary" })).toHaveCount(0);
  await expect(page.getByText("Loaded setup: First setup")).toHaveCount(0);
  releaseSecond();
  await expect(page.getByText("Loaded setup: Second setup")).toBeVisible();
  await expect(page.getByLabel("Division name").first()).toHaveValue("Second tournament division");
});
