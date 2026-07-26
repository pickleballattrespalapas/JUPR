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

const operationsStatus = {
  service: "jupr-api",
  environment: "staging",
  mode: "guarded",
  write_pilot_enabled: false,
  streamlit_fallback_url: "https://fallback.example.test",
  strict_audit_required: true,
  service_role_configured: true,
  jwt_verification_configured: true,
  jwt_verification_mode: "jwks",
  enabled_workflows: [],
  recommended_sequence: [],
  pilot_gates: ["Keep writes off."],
  permanent_guardrails: ["Require authorization."],
  workflows: []
};

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("anonymous admin page never requests or renders operations status", async ({
  page
}) => {
  let statusCalls = 0;
  await page.route("**/admin/operations/status?**", async (route) => {
    statusCalls += 1;
    await route.fulfill({ json: operationsStatus });
  });

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  expect(statusCalls).toBe(0);
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);
});

test("authorized admin loads protected status exactly once with bearer and scope", async ({
  page
}) => {
  const statusRequests: Array<{ authorization: string; url: string }> = [];
  await page.route("**/admin/auth/capabilities?**", async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/operations/status?**", async (route) => {
    statusRequests.push({
      authorization: route.request().headers().authorization || "",
      url: route.request().url()
    });
    await route.fulfill({ json: operationsStatus });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: /operations cockpit/i })
  ).toBeVisible();
  await expect(page.getByText("Environment", { exact: true })).toBeVisible();
  await expect.poll(() => statusRequests.length).toBe(1);
  expect(statusRequests[0].authorization).toBe("Bearer ui-test-token");
  expect(statusRequests[0].url).toContain("club_id=tres_palapas");
});

test("token change clears old posture before authorization denial", async ({
  page
}) => {
  let statusCalls = 0;
  const denialGate: { release?: () => void } = {};
  const denialReady = new Promise<void>((resolve) => {
    denialGate.release = resolve;
  });
  await page.route("**/admin/auth/capabilities?**", async (route) => {
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/operations/status?**", async (route) => {
    statusCalls += 1;
    if (statusCalls === 1) {
      await route.fulfill({ json: operationsStatus });
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

  await expect(page.getByText("Environment", { exact: true })).toBeVisible();
  await page.evaluate((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify({ ...session, access_token: "rotated-ui-test-token" })
    );
    window.dispatchEvent(new CustomEvent("jupr-admin-session-change"));
  }, storedSession);
  await expect.poll(() => statusCalls).toBe(2);
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);

  denialGate.release?.();
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);
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
  let statusCalls = 0;
  let releaseCapability: (() => void) | undefined;
  const capabilityGate = new Promise<void>((resolve) => {
    releaseCapability = resolve;
  });
  await page.route("**/admin/auth/capabilities?**", async (route) => {
    capabilityCalls += 1;
    await capabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/operations/status?**", async (route) => {
    statusCalls += 1;
    await route.fulfill({ json: operationsStatus });
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
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);

  releaseCapability?.();
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))
    )
    .toBeNull();
  expect(statusCalls).toBe(0);
});

test("token rotation supersedes a delayed capability check without stale status", async ({
  page
}) => {
  const capabilityTokens: string[] = [];
  const statusTokens: string[] = [];
  let releaseFirstCapability: (() => void) | undefined;
  const firstCapabilityGate = new Promise<void>((resolve) => {
    releaseFirstCapability = resolve;
  });
  await page.route("**/admin/auth/capabilities?**", async (route) => {
    capabilityTokens.push(route.request().headers().authorization || "");
    if (capabilityTokens.length === 1) await firstCapabilityGate;
    await route.fulfill({ json: capabilities });
  });
  await page.route("**/admin/operations/status?**", async (route) => {
    statusTokens.push(route.request().headers().authorization || "");
    await route.fulfill({ json: operationsStatus });
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
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);

  releaseFirstCapability?.();
  await expect.poll(() => capabilityTokens.length).toBe(2);
  await expect(
    page.getByRole("heading", { name: /operations cockpit/i })
  ).toBeVisible();
  await expect.poll(() => statusTokens).toEqual([
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

test("focus revalidates authorization and hides a revoked cockpit", async ({
  page
}) => {
  let denyCapabilities = false;
  let capabilityCalls = 0;
  await page.route("**/admin/auth/capabilities?**", async (route) => {
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
  await page.route("**/admin/operations/status?**", async (route) => {
    await route.fulfill({ json: operationsStatus });
  });
  await page.addInitScript((session) => {
    window.localStorage.setItem(
      "jupr_admin_session_v1",
      JSON.stringify(session)
    );
  }, storedSession);

  await page.goto("/admin", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Environment", { exact: true })).toBeVisible();

  denyCapabilities = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));

  await expect.poll(() => capabilityCalls).toBeGreaterThanOrEqual(2);
  await expect(
    page.getByRole("heading", { name: /admin sign-in required/i })
  ).toBeVisible();
  await expect(page.getByText("Environment", { exact: true })).toHaveCount(0);
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))
    )
    .toBeNull();
});
