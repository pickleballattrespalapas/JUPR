import { expect, test, type Page, type Route } from "@playwright/test";

const accessToken = "header.payload.signature";
const refreshToken = "refresh-token";

async function fulfillCapabilities(route: Route) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      authorized: true,
      user: { email: "admin@example.com" },
      requested_club_id: "tres_palapas",
      assignments: [{ club_id: "tres_palapas", role: "club_owner", permissions: ["enter_scores"] }]
    })
  });
}

async function installCapabilityMock(page: Page) {
  await page.route("**/admin/auth/capabilities**", fulfillCapabilities);
}

test("password login is capability checked and rejects an external next redirect", async ({ page }) => {
  await installCapabilityMock(page);
  await page.route("**/auth/v1/token?grant_type=password", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_in: 3600,
        token_type: "bearer",
        user: { email: "admin@example.com" }
      })
    });
  });

  await page.goto("/admin/login?next=https%3A%2F%2Fattacker.example%2Fsteal");
  if (await page.getByText("Admin auth is not configured").isVisible().catch(() => false)) {
    test.skip(true, "The build does not include staging public auth/API configuration.");
  }
  await page.getByLabel("Email").fill("admin@example.com");
  await page.getByLabel("Password").fill("correct horse battery staple");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();

  await expect(page).toHaveURL(/\/admin$/);
  expect(new URL(page.url()).hostname).not.toBe("attacker.example");
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("jupr_admin_session_v1") || "null"));
  expect(stored.capabilities.assignments[0]).toMatchObject({ club_id: "tres_palapas", role: "club_owner" });
});

test("wrong-club capability denial does not persist the Supabase session", async ({ page }) => {
  let localLogoutCalled = false;
  await page.route("**/admin/auth/capabilities**", async (route) => {
    await route.fulfill({ status: 403, contentType: "application/json", body: JSON.stringify({ detail: "admin access denied" }) });
  });
  await page.route("**/auth/v1/token?grant_type=password", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_token: accessToken, refresh_token: refreshToken, expires_in: 3600, token_type: "bearer" })
    });
  });
  await page.route("**/auth/v1/logout?scope=local", async (route) => {
    localLogoutCalled = true;
    await route.fulfill({ status: 204, body: "" });
  });

  await page.goto("/admin/login");
  if (await page.getByText("Admin auth is not configured").isVisible().catch(() => false)) {
    test.skip(true, "The build does not include staging public auth/API configuration.");
  }
  await page.getByLabel("Email").fill("admin@example.com");
  await page.getByLabel("Password").fill("correct horse battery staple");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();

  await expect(page.getByText("not authorized for the requested JUPR admin workspace")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))).toBeNull();
  expect(localLogoutCalled).toBe(true);
});

test("PKCE recovery request, callback, password update, and cleanup stay browser local", async ({ page }) => {
  await installCapabilityMock(page);
  let recoveryRequest: Record<string, unknown> | null = null;
  await page.route("**/auth/v1/recover", async (route) => {
    recoveryRequest = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  await page.route("**/auth/v1/token?grant_type=pkce", async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    expect(body.auth_code).toBe("recovery-code");
    expect(String(body.code_verifier || "").length).toBeGreaterThanOrEqual(43);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_in: 3600,
        token_type: "bearer",
        user: { email: "admin@example.com" }
      })
    });
  });
  await page.route("**/auth/v1/user", async (route) => {
    expect(route.request().postDataJSON()).toEqual({ password: "Stronger!Password9" });
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ user: { id: "user-1" } }) });
  });
  await page.route("**/auth/v1/logout?scope=local", async (route) => {
    await route.fulfill({ status: 204, body: "" });
  });

  await page.goto("/admin/reset-password");
  if (await page.getByText("Admin auth is not configured").isVisible().catch(() => false)) {
    test.skip(true, "The build does not include staging public auth/API configuration.");
  }
  await page.getByLabel("Email").fill("admin@example.com");
  await page.getByRole("button", { name: "Send recovery email" }).click();
  await expect(page.getByRole("status")).toContainText("eligible admin account");
  expect(recoveryRequest).toMatchObject({
    email: "admin@example.com",
    code_challenge_method: "s256"
  });
  const recoveryChallenge = (recoveryRequest as unknown as Record<string, unknown>).code_challenge;
  expect(String(recoveryChallenge || "").length).toBeGreaterThan(20);

  await page.goto("/admin/reset-password?code=recovery-code");
  await expect(page.getByRole("status")).toContainText("recovery link verified");
  await page.getByLabel("New password", { exact: true }).fill("Stronger!Password9");
  await page.getByLabel("Confirm new password").fill("Stronger!Password9");
  await page.getByRole("button", { name: "Update password" }).click();

  await expect(page.getByRole("status")).toContainText("Password updated");
  await expect(page).not.toHaveURL(/code=/);
  const residue = await page.evaluate(() => ({
    pkce: localStorage.getItem("jupr_admin_recovery_pkce_v1"),
    admin: localStorage.getItem("jupr_admin_session_v1"),
    recovery: sessionStorage.getItem("jupr_admin_recovery_session_v1")
  }));
  expect(residue).toEqual({ pkce: null, admin: null, recovery: null });
});
