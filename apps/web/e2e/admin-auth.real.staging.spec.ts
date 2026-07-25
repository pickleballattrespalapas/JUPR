import { expect, test } from "@playwright/test";
import {
  bootstrapStagingContext,
  expectedApiOrigin,
  expectedAuthOrigin
} from "./support/staging";

const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "").trim();
const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const expectedRole = String(process.env.JUPR_REAL_AUTH_EXPECTED_ROLE || "club_owner").trim();
const expectedWebOrigin = String(
  process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN ||
    process.env.STAGING_WEB_BASE_URL ||
    ""
)
  .trim()
  .replace(/\/$/, "");

test("workflow-minted real staging admin session is capability checked and signs out", async ({ page, context }) => {
  expect(adminEmail, "STAGING_ADMIN_EMAIL is required").not.toBe("");
  expect(adminToken, "STAGING_ADMIN_BEARER_TOKEN is required").not.toBe("");
  expect(expectedAuthOrigin, "JUPR_EXPECTED_STAGING_AUTH_ORIGIN is required").not.toBe("");
  expect(expectedWebOrigin, "Attested staging web origin is required").not.toBe("");

  await bootstrapStagingContext(context);

  await context.addInitScript(
    ({ token, email, allowedOrigin }) => {
      if (window.location.origin !== allowedOrigin) return;
      window.localStorage.setItem(
        "jupr_admin_session_v1",
        JSON.stringify({
          access_token: token,
          token_type: "bearer",
          user: { email }
        })
      );
    },
    { token: adminToken, email: adminEmail, allowedOrigin: expectedWebOrigin }
  );

  const capabilitiesResponsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.origin === expectedApiOrigin &&
      url.pathname === "/admin/auth/capabilities" &&
      response.request().method() === "GET"
    );
  });
  await page.goto("/admin/login", { waitUntil: "domcontentloaded" });
  const capabilitiesResponse = await capabilitiesResponsePromise;
  expect(capabilitiesResponse.status(), "Live FastAPI capability check failed").toBe(200);
  const liveCapabilities = await capabilitiesResponse.json();
  expect(liveCapabilities?.authorized).toBe(true);
  expect(liveCapabilities?.requested_club_id).toBe("tres_palapas");
  expect(
    String(liveCapabilities?.user?.email || "").toLowerCase() === adminEmail.toLowerCase(),
    "Live FastAPI capability identity did not match the workflow-minted session"
  ).toBe(true);
  expect(liveCapabilities?.assignments).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        club_id: "tres_palapas",
        role: expectedRole
      })
    ])
  );

  const session = await page.evaluate(() => JSON.parse(localStorage.getItem("jupr_admin_session_v1") || "null"));
  expect(Boolean(session?.access_token), "Authorized browser session was not persisted").toBe(true);
  expect(
    session?.access_token === adminToken,
    "Browser session did not retain the workflow-minted bearer token"
  ).toBe(true);
  expect(
    String(session?.user?.email || "").toLowerCase() === adminEmail.toLowerCase(),
    "Persisted browser identity did not match the workflow-minted session"
  ).toBe(true);
  expect(session?.capabilities?.authorized).toBe(true);
  expect(session?.capabilities?.assignments).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        club_id: "tres_palapas",
        role: expectedRole
      })
    ])
  );
  await expect(page.getByRole("heading", { name: "Signed in" })).toBeVisible();

  const logoutResponsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.origin === expectedAuthOrigin &&
      url.pathname === "/auth/v1/logout" &&
      url.searchParams.get("scope") === "local" &&
      response.request().method() === "POST"
    );
  });
  await page.getByRole("button", { name: "Sign out" }).click();
  const logoutResponse = await logoutResponsePromise;
  expect([200, 204], "Supabase local sign-out failed").toContain(logoutResponse.status());
  await expect.poll(() => page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))).toBeNull();
  await expect(page.getByRole("heading", { name: "Sign in with Supabase Auth" })).toBeVisible();
  await expect(page.getByText("Signed out.", { exact: true })).toBeVisible();
});
