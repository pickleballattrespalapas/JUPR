import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const storedSession = {
  access_token: "ui-test-token",
  token_type: "bearer",
  user: { id: "user-1", email: "admin@example.com" }
};

const capabilities = {
  authorized: true,
  user: { email: "admin@example.com" },
  requested_club_id: "tres_palapas",
  assignments: [
    {
      club_id: "tres_palapas",
      role: "club_owner",
      permissions: ["manage_matches"]
    }
  ]
};

const dashboard = {
  club_id: "tres_palapas",
  checked_at: "2026-09-21T16:00:00Z",
  queues: [
    {
      key: "generator_submissions",
      label: "Generator submissions",
      description: "Completed sessions waiting for approval.",
      href: "/admin/play-generators/submissions",
      count: 3,
      status: "ready"
    }
  ]
};

test.beforeEach(async ({ context, page, baseURL }) => {
  await bootstrapStagingContext(context);
  await context.addCookies([{
    name: "jupr_admin_workspace_v1",
    value: encodeURIComponent(JSON.stringify({ clubId: "tres_palapas", clubSlug: "tres-palapas" })),
    url: baseURL!,
    sameSite: "Lax"
  }]);
  await page.route("**/admin/auth/workspaces", route => route.fulfill({ json: {
    workspaces: [{ club_id: "tres_palapas", club_slug: "tres-palapas", club_name: "Tres Palapas", roles: ["club_owner"] }]
  } }));
});

test("anonymous admin page never requests or renders dashboard counts", async ({
  page
}) => {
  let dashboardCalls = 0;
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    dashboardCalls += 1;
    await route.fulfill({ json: dashboard });
  });

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  expect(dashboardCalls).toBe(0);
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);
});

test("authorized admin loads dashboard once with bearer and club scope", async ({
  page
}) => {
  const dashboardRequests: Array<{ authorization: string; url: string }> = [];
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    dashboardRequests.push({
      authorization: route.request().headers().authorization || "",
      url: route.request().url()
    });
    await route.fulfill({ json: dashboard });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: /admin home/i })
  ).toBeVisible();
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toBeVisible();
  await expect.poll(() => dashboardRequests.length).toBe(1);
  expect(dashboardRequests[0].authorization).toBe("Bearer ui-test-token");
  expect(dashboardRequests[0].url).toContain("/admin/clubs/tres_palapas/dashboard");
});

test("home notifications link to approvals and refresh honestly on desktop and mobile", async ({
  page
}, testInfo) => {
  let count: number | null = 3;
  let queueStatus = "ready";
  const pageErrors: string[] = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, route => route.fulfill({ json: capabilities }));
  await page.route("**/admin/clubs/*/dashboard", route => route.fulfill({ json: {
    ...dashboard, queues: [{ ...dashboard.queues[0], count, status: queueStatus }]
  } }));
  await page.addInitScript(session => {
    localStorage.setItem("jupr_admin_session_v1", JSON.stringify(session));
  }, storedSession);
  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  const notifications = page.getByRole("region", { name: "Needs attention" });
  await expect(notifications.getByRole("link", { name: /3 Generator submissions/ })).toHaveAttribute("href", "/admin/play-generators/submissions");
  await expect(notifications).toContainText("3 items need attention");
  await page.screenshot({ path: testInfo.outputPath("admin-home-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(notifications.getByRole("link", { name: /3 Generator submissions/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("admin-home-mobile.png"), fullPage: true });
  count = 0;
  await notifications.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(notifications).toContainText("No pending work in the checked queues");
  await expect(notifications.getByRole("link", { name: /3 Generator submissions/ })).toHaveCount(0);
  count = null;
  queueStatus = "unavailable";
  await notifications.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(notifications).toContainText("Some notifications couldn’t be checked");
  await expect(notifications).not.toContainText("No pending work");
  await expect(notifications.getByRole("link", { name: "Generator submissions", exact: true })).toHaveAttribute("href", "/admin/play-generators/submissions");
  expect(pageErrors).toEqual([]);
});

test("token change clears old counts before authorization denial", async ({
  page
}) => {
  let dashboardCalls = 0;
  const denialGate: { release?: () => void } = {};
  const denialReady = new Promise<void>((resolve) => {
    denialGate.release = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    dashboardCalls += 1;
    if (dashboardCalls === 1) {
      await route.fulfill({ json: dashboard });
      return;
    }
    await denialReady;
    await route.fulfill({
      status: 403,
      contentType: "application/json",
      body: JSON.stringify({ detail: "admin access denied" })
    });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toBeVisible();
  await page.evaluate((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify({ ...session, access_token: "rotated-ui-test-token" })
    );
    window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
  }, storedSession);
  await expect.poll(() => dashboardCalls).toBe(2);
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);

  denialGate.release?.();
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))
    )
    .toBeNull();
});

test("logout during a delayed capability check cannot restore the old session", async ({
  page
}) => {
  let capabilityCalls = 0;
  let dashboardCalls = 0;
  let releaseCapability: (() => void) | undefined;
  const capabilityGate = new Promise<void>((resolve) => {
    releaseCapability = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    capabilityCalls += 1;
    await capabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    dashboardCalls += 1;
    await route.fulfill({ json: dashboard });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  await expect.poll(() => capabilityCalls).toBe(1);
  await page.evaluate(() => {
    window.localStorage.removeItem("jupr_admin_session_v1");
    window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
  });
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);

  releaseCapability?.();
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))
    )
    .toBeNull();
  expect(dashboardCalls).toBe(0);
});

test("token rotation supersedes a delayed capability check without stale dashboard counts", async ({
  page
}) => {
  const capabilityTokens: string[] = [];
  const dashboardTokens: string[] = [];
  let releaseFirstCapability: (() => void) | undefined;
  const firstCapabilityGate = new Promise<void>((resolve) => {
    releaseFirstCapability = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    capabilityTokens.push(route.request().headers().authorization || "");
    if (capabilityTokens.length === 1) await firstCapabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    dashboardTokens.push(route.request().headers().authorization || "");
    await route.fulfill({ json: dashboard });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  await expect.poll(() => capabilityTokens.length).toBe(1);
  await page.evaluate((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify({ ...session, access_token: "rotated-ui-test-token" })
    );
    window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
  }, storedSession);
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);

  releaseFirstCapability?.();
  await expect.poll(() => capabilityTokens.length).toBe(2);
  await expect(
    page.getByRole("heading", { name: /admin home/i })
  ).toBeVisible();
  await expect.poll(() => dashboardTokens).toEqual([
    "Bearer rotated-ui-test-token"
  ]);
  expect(capabilityTokens).toEqual([
    "Bearer ui-test-token",
    "Bearer rotated-ui-test-token"
  ]);
  await expect
    .poll(() =>
      page.evaluate(() => {
        const raw = localStorage.getItem("jupr_admin_session_v1");
        return raw ? JSON.parse(raw).access_token : null;
      })
    )
    .toBe("rotated-ui-test-token");
});

test("focus revalidates authorization and hides a revoked dashboard", async ({
  page
}) => {
  let denyCapabilities = false;
  let capabilityCalls = 0;
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    capabilityCalls += 1;
    if (denyCapabilities) {
      await route.fulfill({
        status: 403,
        contentType: "application/json",
        body: JSON.stringify({ detail: "admin access denied" })
      });
      return;
    }
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/dashboard", async (route) => {
    await route.fulfill({ json: dashboard });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toBeVisible();

  denyCapabilities = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));

  await expect.poll(() => capabilityCalls).toBeGreaterThanOrEqual(2);
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))
    )
    .toBeNull();
});

test("commerce evidence OK restores an explicit visible ring to the exact Inspect trigger", async ({
  page
}) => {
  const inspectedOperationIds: string[] = [];
  const commerceRoot =
    "/admin/clubs/tres_palapas/tournaments/commerce";

  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route(
    "**/admin/clubs/tres_palapas/tournaments/commerce/**",
    async (route) => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === `${commerceRoot}/status`) {
        await route.fulfill({
          json: {
            ok: true,
            runtime: { environment: "staging", admin_write_ready: false }
          }
        });
        return;
      }
      if (
        pathname ===
        `${commerceRoot}/tournaments/focus-test/operations/op-2`
      ) {
        inspectedOperationIds.push("op-2");
        await route.fulfill({
          json: {
            recovery_state: "complete",
            authoritative_mutation_complete: true,
            safe_retry: true,
            retry_mode: "same_idempotency_key"
          }
        });
        return;
      }
      if (pathname === `${commerceRoot}/tournaments/focus-test`) {
        await route.fulfill({
          json: {
            ok: true,
            tournament: { id: "focus-test", name: "Focus test" },
            catalog: {
              currency: "USD",
              catalog_revision: 1,
              catalog_fingerprint: "focus-test-catalog",
              items: [],
              variants: [],
              bundles: [],
              bundle_components: [],
              promotions: [],
              event_options: []
            },
            orders: [],
            fulfillment: [],
            operations: [
              {
                id: "op-1",
                action: "PAYMENT_UPDATE",
                status: "COMPLETED",
                created_at: "2026-08-15T03:00:00+00:00"
              },
              {
                id: "op-2",
                action: "ORDER_CANCEL",
                status: "COMPLETED",
                created_at: "2026-08-15T03:01:00+00:00"
              }
            ],
            audit: [],
            offline_payment_only: true
          }
        });
        return;
      }
      await route.abort("blockedbyclient");
    }
  );
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto(
    "/admin/tournaments/commerce?tournament=focus-test&name=Focus%20test",
    { waitUntil: "domcontentloaded" }
  );
  await page
    .getByRole("button", { name: "Recovery & audit", exact: true })
    .click();

  const inspectButtons = page.getByRole("button", {
    name: "Inspect",
    exact: true
  });
  await expect(inspectButtons).toHaveCount(2);
  const firstInspect = inspectButtons.nth(0);
  const secondInspect = inspectButtons.nth(1);
  await secondInspect.click();

  const dialog = page.getByRole("dialog", {
    name: "Authoritative evidence"
  });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("complete");
  expect(inspectedOperationIds).toEqual(["op-2"]);
  await dialog.getByRole("button", { name: "OK", exact: true }).click();

  await expect(dialog).toHaveCount(0);
  await expect(secondInspect).toBeFocused();
  await expect(firstInspect).not.toBeFocused();
  await expect
    .poll(() =>
      secondInspect.evaluate((element) => {
        const style = getComputedStyle(element);
        return {
          outlineStyle: style.outlineStyle,
          outlineWidth: style.outlineWidth,
          outlineColor: style.outlineColor,
          outlineOffset: style.outlineOffset
        };
      })
    )
    .toEqual({
      outlineStyle: "solid",
      outlineWidth: "3px",
      outlineColor: "rgb(96, 165, 250)",
      outlineOffset: "2px"
    });
});
