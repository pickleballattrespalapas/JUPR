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

const notifications = {
  club_id: "tres_palapas",
  checked_at: "2026-09-21T16:00:00Z",
  history_days: 30,
  truncated: false,
  categories: [{
    key: "generator_submissions", label: "Generator submissions", description: "Review completed sessions.",
    href: "/admin/play-generators/submissions", kind: "action", enabled: true, status: "ready", total_count: 1
  }],
  items: [{
    key: "generator:session-1", category: "generator_submissions", kind: "action", state: "new",
    title: "Monday round robin awaiting approval", description: "Completed sessions waiting for approval.",
    href: "/admin/play-generators/submissions", occurred_at: "2026-09-21T15:00:00Z"
  }]
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

test("anonymous admin page never requests or renders notifications", async ({
  page
}) => {
  let notificationCalls = 0;
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    notificationCalls += 1;
    await route.fulfill({ json: notifications });
  });

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  expect(notificationCalls).toBe(0);
  await expect(page.getByText("Completed sessions waiting for approval.", { exact: true })).toHaveCount(0);
});

test("authorized admin loads notifications once with bearer and club scope", async ({
  page
}) => {
  const notificationRequests: Array<{ authorization: string; url: string }> = [];
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    notificationRequests.push({
      authorization: route.request().headers().authorization || "",
      url: route.request().url()
    });
    await route.fulfill({ json: notifications });
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
  await expect.poll(() => notificationRequests.length).toBe(1);
  expect(notificationRequests[0].authorization).toBe("Bearer ui-test-token");
  expect(notificationRequests[0].url).toContain("/admin/clubs/tres_palapas/notifications");
});

test("token change clears old notices before authorization denial", async ({
  page
}) => {
  let notificationCalls = 0;
  const denialGate: { release?: () => void } = {};
  const denialReady = new Promise<void>((resolve) => {
    denialGate.release = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    notificationCalls += 1;
    if (notificationCalls === 1) {
      await route.fulfill({ json: notifications });
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
  await expect.poll(() => notificationCalls).toBe(2);
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
  let notificationCalls = 0;
  let releaseCapability: (() => void) | undefined;
  const capabilityGate = new Promise<void>((resolve) => {
    releaseCapability = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    capabilityCalls += 1;
    await capabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    notificationCalls += 1;
    await route.fulfill({ json: notifications });
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
  expect(notificationCalls).toBe(0);
});

test("token rotation supersedes a delayed capability check without stale notifications", async ({
  page
}) => {
  const capabilityTokens: string[] = [];
  const notificationTokens: string[] = [];
  let releaseFirstCapability: (() => void) | undefined;
  const firstCapabilityGate = new Promise<void>((resolve) => {
    releaseFirstCapability = resolve;
  });
  await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, async (route) => {
    capabilityTokens.push(route.request().headers().authorization || "");
    if (capabilityTokens.length === 1) await firstCapabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    notificationTokens.push(route.request().headers().authorization || "");
    await route.fulfill({ json: notifications });
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
  await expect.poll(() => notificationTokens).toEqual([
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

test("focus revalidates authorization and hides revoked notifications", async ({
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
  await page.route("**/admin/clubs/*/notifications", async (route) => {
    await route.fulfill({ json: notifications });
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
