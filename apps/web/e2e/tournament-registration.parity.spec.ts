import { expect, test } from "@playwright/test";
import { bootstrapStagingContext, clubSlug } from "./support/staging";

const registrationSlug = String(process.env.JUPR_TOURNAMENT_REGISTRATION_FIXTURE_SLUG || "").trim();
const fixturePath = `/clubs/${clubSlug}/tournament-registration${registrationSlug ? `?tournament=${encodeURIComponent(registrationSlug)}` : ""}`;

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

test("tournament registration exposes explicit start and edit recovery paths", async ({ page }) => {
  const response = await page.goto(fixturePath, { waitUntil: "domcontentloaded" });
  expect(response?.status()).toBeLessThan(400);

  const chooser = page.getByTestId("registration-mode-chooser");
  await expect(chooser).toBeVisible();
  await expect(chooser.getByRole("button", { name: "Start a registration" })).toBeVisible();
  await chooser.getByRole("button", { name: "Edit my registration" }).click();
  await expect(page.getByTestId("registration-edit-mode")).toBeVisible();
  await expect(page.getByTestId("registration-edit-link-form")).toBeVisible();
  await expect(page.getByText(/We’ll send your edit link there/i)).toBeVisible();
});

test("new-registration wizard requires demographics and resolves profiles through FastAPI", async ({ page }) => {
  await page.route("**/clubs/*/tournament-registration/profile-resolution", async (route) => {
    const request = route.request();
    expect(request.method()).toBe("POST");
    const body = request.postDataJSON();
    expect(body).toMatchObject({
      first_name: "Avery",
      last_name: "Ace",
      email: "avery@example.test",
      age: 34,
      gender: "Women"
    });
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        club: { id: "fixture", slug: clubSlug, name: "Fixture Club" },
        ok: true,
        status: "ready",
        can_start_new: true,
        registration_open: true,
        masked_email: "a***y@example.test",
        profile_match_kind: "email_exact",
        profile_candidates: [
          { id: "player-10", display_name: "Avery Ace", dupr_id: "DUPR-10", doubles_skill: 4.0, singles_skill: null }
        ],
        profile_policy: { linkage: "staff_review_required", public_submission_links_player: false },
        message: "Review the suggested profile or continue without one."
      })
    });
  });

  await page.goto(fixturePath, { waitUntil: "domcontentloaded" });
  const start = page.getByRole("button", { name: "Start a registration" });
  test.skip(await start.isDisabled(), "The configured staging tournament is closed; closed-state contracts are covered by FastAPI tests.");
  await start.click();
  await expect(page.getByTestId("registration-step-contact")).toBeVisible();

  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "First name, last name, email, age, and gender are required" })
  ).toBeVisible();

  await page.getByLabel("First name").fill("Avery");
  await page.getByLabel("Last name").fill("Ace");
  await page.getByLabel("Email", { exact: true }).fill("avery@example.test");
  await page.getByLabel("Age", { exact: true }).fill("34");
  await page.getByLabel("Gender", { exact: true }).selectOption("Women");
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  await expect(page.getByTestId("registration-step-profile")).toBeVisible();
  await page.getByText(/Avery Ace · Doubles 4 · Singles not set/).click();
  await expect(page.getByLabel("Display name")).toHaveValue("Avery Ace");
  await expect(page.getByLabel("Doubles skill")).toHaveValue("4");
  await expect(page.getByLabel("Singles skill")).toHaveValue("");
  await page.getByLabel("Singles skill").fill("3.5");
  await expect(page.getByLabel("Singles skill")).toHaveValue("3.5");
  await expect(page.getByText(/Choosing a profile only fills in this form/i)).toBeVisible();
});
