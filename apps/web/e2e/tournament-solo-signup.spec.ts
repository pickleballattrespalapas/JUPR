import { expect, test } from "@playwright/test";

for (const entry of ["solo", "team"] as const) {
  test(`${entry} four-player tournament signup saves on ${entry === "solo" ? "mobile" : "desktop"}`, async ({ page }, testInfo) => {
    if (entry === "solo") await page.setViewportSize({ width: 390, height: 844 });
    const writes: Array<{ path: string; body: any }> = [];
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/clubs/fixture/tournament-registration**", async route => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path.endsWith("/confirmation")) return route.fulfill({ status: 302, headers: { location: "/tournament-signup-test?confirmation=1" }, body: "" });
      if (path.endsWith("/profile-resolution")) return route.fulfill({ json: {
        can_start_new: true, profile_match_kind: "name_exact",
        profile_candidates: [{ id: "fixture-player", display_name: "Fixture Player", doubles_skill: 4.5 }],
      } });
      expect(request.method()).toBe("POST");
      writes.push({ path, body: request.postDataJSON() });
      if (path.endsWith("/four-player-team")) return route.fulfill({ json: { ok: true, team_id: "team1" } });
      expect(path).toBe("/clubs/fixture/tournament-registration");
      return route.fulfill({ json: { ok: true, registration_id: "solo", confirmation_token: "fixture-only", email_delivery: { status: "dry_run" } } });
    });
    await page.goto("/tournament-signup-test");
    await page.getByRole("button", { name: "Start a registration", exact: true }).click();
    for (const [label, value] of [["First name", "Fixture"], ["Last name", "Player"], ["Email", "fixture@example.invalid"], ["Age", "40"]]) {
      await page.getByLabel(label, { exact: true }).fill(value);
    }
    await page.getByLabel("Gender", { exact: true }).selectOption("Men");
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    await page.getByRole("button", { name: "Continue to events", exact: true }).click();
    await page.getByRole("checkbox", { name: "Team Tournament Mixed 5.0", exact: true }).check();
    const plan = page.getByLabel("Mixed 5.0 team plan", { exact: true });
    const roster = page.getByTestId("four-player-team-roster");
    await expect(roster).toBeVisible();
    await page.getByRole("button", { name: "Review registration", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Enter a team name");
    await plan.selectOption("solo");
    await expect(roster).toHaveCount(0);
    if (entry === "team") {
      await plan.selectOption("team");
      await roster.getByLabel("Team name", { exact: true }).fill("Fixture Team");
      for (let i = 0; i < 3; i++) {
        await roster.getByLabel("Player name", { exact: true }).nth(i).fill(`Teammate ${i}`);
        await roster.getByLabel("Player email", { exact: true }).nth(i).fill(`teammate${i}@example.invalid`);
      }
      await plan.selectOption("solo");
      await plan.selectOption("team");
      await expect(roster.getByLabel("Team name", { exact: true })).toHaveValue("Fixture Team");
    }
    await page.getByRole("button", { name: "Review registration", exact: true }).click();
    await expect(page.getByText(entry === "solo" ? "Individual signup · Needs a team" : "Team: Fixture Team")).toBeVisible();
    await page.getByRole("checkbox", { name: /My information is correct/ }).check();
    await page.getByRole("button", { name: "Submit registration", exact: true }).click();
    await expect(page).toHaveURL(/tournament-signup-test\?confirmation=1/);
    expect(writes).toHaveLength(entry === "solo" ? 1 : 2);
    expect(writes[0].body.selections).toEqual([expect.objectContaining({ event_option_id: "team-event", partner_mode: "NONE" })]);
    expect(writes[0].body.selections[0].partner_email || "").toBe("");
    if (entry === "team") {
      expect(writes[1].body.team_name).toBe("Fixture Team");
      expect(writes[1].body.members).toHaveLength(4);
      expect(writes[1].body.members.filter((member: any) => member.registration_id === "solo")).toHaveLength(1);
    } else {
      await expect(page.getByTestId("team-placement-pending")).toContainText("Your individual registration is saved");
      await expect(page.getByRole("button", { name: "Save team", exact: true })).not.toBeVisible();
      await page.getByText("I have a team to add", { exact: true }).click();
      await expect(page.getByRole("button", { name: "Save team", exact: true })).toBeVisible();
      await page.getByText("I have a team to add", { exact: true }).click();
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(errors).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`${entry}-registration.png`) });
  });
}
