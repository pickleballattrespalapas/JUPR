import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const adminEmail = String(process.env.STAGING_ADMIN_EMAIL || "").trim();
const adminPassword = String(process.env.STAGING_ADMIN_PASSWORD || "").trim();
const expectedRole = String(process.env.JUPR_REAL_AUTH_EXPECTED_ROLE || "club_owner").trim();

test("real staging login obtains a live capability-checked session and signs out", async ({ page, context }) => {
  expect(adminEmail, "STAGING_ADMIN_EMAIL is required").not.toBe("");
  expect(adminPassword, "STAGING_ADMIN_PASSWORD is required").not.toBe("");
  await bootstrapStagingContext(context);
  await page.goto("/admin/login", { waitUntil: "domcontentloaded" });
  await page.getByLabel("Email").fill(adminEmail);
  await page.getByLabel("Password").fill(adminPassword);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page).toHaveURL(/\/admin(?:\?|$)/);

  const session = await page.evaluate(() => JSON.parse(localStorage.getItem("jupr_admin_session_v1") || "null"));
  expect(session?.access_token).toBeTruthy();
  expect(session?.user?.email).toBe(adminEmail);
  expect(session?.capabilities?.authorized).toBe(true);
  expect(session?.capabilities?.assignments).toEqual(
    expect.arrayContaining([expect.objectContaining({ role: expectedRole })])
  );

  await page.goto("/admin/login", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("jupr_admin_session_v1"))).toBeNull();
});
