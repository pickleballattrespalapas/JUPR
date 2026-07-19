import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("rating rules, FAQ, and policy request links are complete", async ({ page }) => {
  await page.goto("/how-ratings-work");
  await expect(page.getByRole("heading", { name: "How club ratings work" })).toBeVisible();
  await expect(page.getByRole("heading", { name: /How ratings move/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "Read rating FAQs" })).toHaveAttribute("href", "/faq");

  await page.goto("/privacy");
  await expect(page.getByRole("heading", { name: "Public aliases and private identity" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Request profile privacy review" })).toHaveAttribute("href", "/profile-privacy");
  await expect(page.getByRole("link", { name: "Manage email preferences" })).toHaveAttribute("href", "/email-preferences");
});

test("general support form posts one durable staff-review request", async ({ page }) => {
  let posted: Record<string, unknown> | null = null;
  await page.route("**/clubs/tres-palapas/support/intake", async (route) => {
    posted = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        club: { id: "tres_palapas", slug: "tres-palapas", name: "Tres Palapas" },
        ok: true,
        accepted: true,
        deduplicated: false,
        message: "Request received. Staff will review it before any data changes are made."
      })
    });
  });

  await page.goto("/support#general-support-form");
  await page.getByLabel("Your name").fill("Parity Tester");
  await page.getByLabel("Your email").fill("parity@example.com");
  await page.getByLabel("Short subject").fill("Staging support smoke");
  await page.getByLabel("How can we help?").fill("Verify the durable support queue without changing source data.");
  await page.getByLabel(/Staff may contact me/).check();
  await page.getByRole("button", { name: "Submit support request" }).click();

  await expect(page.getByRole("status")).toContainText("Request received");
  expect(posted).toMatchObject({
    request_type: "general_support",
    requester_email: "parity@example.com",
    consent_to_contact: true,
    source: "next_general_support_form"
  });
});

