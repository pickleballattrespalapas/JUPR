import { expect, test, type Page } from "@playwright/test";

async function enterRegistrant(page: Page) {
  await page.getByRole("button", { name: "Start a registration", exact: true }).click();
  for (const [label, value] of [["First name", "Fixture"], ["Last name", "Player"], ["Email", "fixture@example.invalid"], ["Age", "40"]]) {
    await page.getByLabel(label, { exact: true }).fill(value);
  }
  await page.getByLabel("Gender", { exact: true }).selectOption("Men");
}

async function selectTeamEvent(page: Page) {
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByRole("button", { name: "Continue to events", exact: true }).click();
  await page.getByRole("checkbox", { name: "Team Tournament Mixed 5.0", exact: true }).check();
}

for (const entry of ["solo", "team", "partial", "partial-explicit", "partial-retry"] as const) {
  test(`${entry} four-player tournament signup saves on ${entry === "solo" ? "mobile" : "desktop"}`, async ({ page }, testInfo) => {
    if (entry === "solo") await page.setViewportSize({ width: 390, height: 844 });
    const writes: Array<{ path: string; body: any }> = [];
    const auditRequests: Array<{ path: string; body: any }> = [];
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/clubs/fixture/tournament-registration**", async route => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path.endsWith("/confirmation")) {
        expect(new URL(request.url()).searchParams.get("team_setup")).toBe(entry === "partial-retry" ? "attention" : null);
        const fixture = entry === "solo" || entry === "team" ? "1" : "partial";
        return route.fulfill({ status: 302, headers: { location: `/tournament-signup-test?confirmation=${fixture}` }, body: "" });
      }
      if (path.endsWith("/profile-resolution") || path.endsWith("/partner-profile-resolution")) return route.fulfill({ json: {
        can_start_new: true, profile_match_kind: "name_exact",
        profile_candidates: [{ id: "fixture-player", display_name: "Fixture Player", doubles_skill: 4.5 }],
      } });
      expect(request.method()).toBe("POST");
      if (path.endsWith("/attempts")) {
        const body = request.postDataJSON();
        expect(body.attempt_id).toMatch(/^[0-9a-f-]{36}$/);
        expect(body.flow).toBe("new");
        expect(body).not.toHaveProperty("phone");
        expect(body).not.toHaveProperty("notes");
        expect(body).not.toHaveProperty("confirmation_token");
        auditRequests.push({ path, body });
        return route.fulfill({ status: 202, json: { accepted: true } });
      }
      writes.push({ path, body: request.postDataJSON() });
      if (path.endsWith("/four-player-team")) {
        // A lost response after a successful commit must recover the saved team, without signing the captain up again.
        if (entry === "partial-retry") return route.abort("connectionfailed");
        return route.fulfill({ json: { ok: true, team_id: "team1" } });
      }
      expect(path).toBe("/clubs/fixture/tournament-registration");
      return route.fulfill({ json: { ok: true, registration_id: "solo", confirmation_token: "fixture-only", email_delivery: { status: "dry_run" } } });
    });
    await page.goto("/tournament-signup-test");
    await enterRegistrant(page);
    // A person can leave the form here without ever submitting a registration.
    await expect.poll(() => auditRequests.some(row => row.body.email === "fixture@example.invalid")).toBe(true);
    expect(writes).toHaveLength(0);
    await selectTeamEvent(page);
    const plan = page.getByLabel("Mixed 5.0 team plan", { exact: true });
    const roster = page.getByTestId("four-player-team-roster");
    await expect(roster).toBeVisible();
    await page.getByRole("button", { name: "Review registration", exact: true }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Enter a team name" })).toBeVisible();
    await plan.selectOption("solo");
    await expect(roster).toHaveCount(0);
    if (entry !== "solo") {
      await plan.selectOption("team");
      await roster.getByLabel("Team name", { exact: true }).fill("Fixture Team");
      const invitedCount = entry === "team" ? 3 : 2;
      for (let i = 0; i < invitedCount; i++) {
        await roster.getByLabel("Player name", { exact: true }).nth(i).fill(`Teammate ${i}`);
        await roster.getByLabel("Player email", { exact: true }).nth(i).fill(`teammate${i}@example.invalid`);
      }
      if (entry === "partial-explicit") {
        await roster.getByLabel("Women’s spot 2 player choice", { exact: true }).selectOption({ label: "Need a player" });
      }
      await plan.selectOption("solo");
      await plan.selectOption("team");
      await expect(roster.getByLabel("Team name", { exact: true })).toHaveValue("Fixture Team");
    }
    await page.getByRole("button", { name: "Review registration", exact: true }).click();
    if (entry === "solo") {
      await expect(page.getByText("Individual signup · Needs a team")).toBeVisible();
    } else {
      await expect(page.getByText("Team: Fixture Team", { exact: false })).toBeVisible();
      if (entry !== "team") await expect(page.getByText(/Needs 1 player/i).first()).toBeVisible();
    }
    await page.getByRole("checkbox", { name: /My information is correct/ }).check();
    await page.getByRole("button", { name: "Submit registration", exact: true }).click();
    await expect(page).toHaveURL(/tournament-signup-test\?confirmation=/);
    await expect(page.getByRole("heading", { name: "Your four-player team" })).toBeVisible();
    if (entry.startsWith("partial")) {
      await expect(page.getByText("Women’s spot 2: Need a player", { exact: true })).toBeVisible();
      await expect(page.getByRole("button", { name: "Save team" })).toHaveCount(0);
    }
    expect(writes).toHaveLength(entry === "solo" ? 1 : 2);
    expect(writes[0].body.attempt_id).toBe(auditRequests.find(row => row.body.email === "fixture@example.invalid")!.body.attempt_id);
    expect(writes[0].body.selections).toEqual([expect.objectContaining({ event_option_id: "team-event", partner_mode: entry === "solo" ? "NEEDS_PARTNER" : "NONE" })]);
    expect(writes[0].body.selections[0].partner_email || "").toBe("");
    if (entry !== "solo") {
      expect(writes[1].body.team_name).toBe("Fixture Team");
      expect(writes[1].body.members.map((member: {slot: string}) => member.slot).sort())
        .toEqual(entry === "team" ? ["MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2"] : ["MAN_1", "MAN_2", "WOMAN_1"]);
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

test("a returning captain invites the fourth player into the saved team", async ({ page }) => {
  await page.route("**/clubs/fixture/tournament-registration/four-player-team", async route => {
    expect(route.request().postDataJSON()).toMatchObject({
      tournament_id: "t1", event_option_id: "team-event", fill_team_id: "team1",
      captain_registration_id: "solo", members: [
        { slot: "WOMAN_2", display_name: "Final Player", email: "final@example.invalid" },
      ],
    });
    await route.fulfill({ json: { ok: true, team_id: "team1" } });
  });
  await page.route("**/clubs/fixture/tournament-registration/four-player-team/recover", async route => {
    await route.fulfill({ json: {
      ok: true, tournament: { id: "t1", name: "Team tournament" },
      captain: { registration_id: "solo", display_name: "Fixture Player", email: "fixture@example.invalid", gender: "Men", registration_status: "CONFIRMED" },
      events: [{ id: "team-event", division_name: "Mixed 5.0", event_family_label: "Team Tournament",
        setup_state: "COMPLETE", team: { id: "team1", name: "Fixture Team", status: "FORMING", is_captain: true, open_slots: [],
          members: [
            { member_id: "captain", slot: "MAN_1", display_name: "Fixture Player", status: "ACCEPTED" },
            { member_id: "teammate1", slot: "MAN_2", display_name: "Teammate One", status: "INVITED" },
            { member_id: "teammate2", slot: "WOMAN_1", display_name: "Teammate Two", status: "INVITED" },
            { member_id: "teammate3", slot: "WOMAN_2", display_name: "Final Player", status: "INVITED" },
          ],
        } }],
    } });
  });

  await page.goto("/tournament-signup-test?confirmation=partial");
  await expect(page.getByText("Women’s spot 2: Need a player", { exact: true })).toBeVisible();
  await page.getByText("Invite players to open spots", { exact: true }).click();
  const form = page.getByTestId("four-player-team-roster");
  await expect(form.getByLabel("Player name", { exact: true })).toHaveCount(1);
  await form.getByLabel("Player name", { exact: true }).fill("Final Player");
  await form.getByLabel("Player email", { exact: true }).fill("final@example.invalid");
  await page.getByRole("button", { name: "Invite players", exact: true }).click();
  await expect(page.getByText("Women’s spot 2: Final Player · Invitation pending", { exact: true })).toBeVisible();
  await expect(page.getByText("Invite players to open spots", { exact: true })).toHaveCount(0);
});

for (const action of ["ACCEPT", "DECLINE"] as const) {
  test(`invited player can ${action.toLowerCase()} a team spot`, async ({ page, context }) => {
    const token = "fixture-invitation-private-token";
    let registered = action === "DECLINE";
    const responses: string[] = [];
    await context.route("**/clubs/fixture/tournaments/t1/sponsors", route =>
      route.fulfill({ json: { tournament_id: "t1", sponsors: [] } })
    );
    await page.route("**/clubs/fixture/tournament-team-invitation/resolve", async route => {
      expect(route.request().postDataJSON()).toEqual({ token });
      await route.fulfill({ json: {
        ok: true, tournament: { id: "t1", name: "Team tournament" }, team: { name: "Fixture Team" },
        invitation: { invited_name: "Fixture Player", slot: "MAN_2", status: responses.includes("ACCEPT") ? "ACCEPTED" : "INVITED", event_label: "Mixed 5.0" },
        registration: registered ? { id: "invited-registration", display_name: "Fixture Player" } : null,
      } });
    });
    await page.route("**/clubs/fixture/tournament-team-invitation/respond", async route => {
      expect(route.request().postDataJSON()).toMatchObject({ token, action, registration_id: "invited-registration" });
      responses.push(action);
      await route.fulfill({ json: { ok: true } });
    });
    await context.route("**/clubs/fixture/tournament-registration**", async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (request.method() === "GET") {
        if (url.pathname.endsWith("/confirmation")) {
          return route.fulfill({ status: 302, headers: { location: "/tournament-signup-test?confirmation=1" }, body: "" });
        }
        expect(url.searchParams.get("tournament_id")).toBe("t1");
        expect(url.toString()).not.toContain(token);
        return route.fulfill({ status: 302, headers: { location: "/tournament-signup-test" }, body: "" });
      }
      if (url.pathname.endsWith("/attempts")) return route.fulfill({ status: 202, json: { accepted: true } });
      if (url.pathname.endsWith("/profile-resolution")) return route.fulfill({ json: {
        can_start_new: true, profile_match_kind: "name_exact",
        profile_candidates: [{ id: "fixture-player", display_name: "Fixture Player", doubles_skill: 4.5 }],
      } });
      expect(url.pathname).toBe("/clubs/fixture/tournament-registration");
      expect(request.postDataJSON().selections).toMatchObject([{ event_option_id: "team-event", partner_mode: "NEEDS_PARTNER" }]);
      registered = true;
      return route.fulfill({ json: { ok: true, registration_id: "invited-registration", confirmation_token: "fixture-only" } });
    });
    await page.goto(`/tournament-signup-test?invitation=1#token=${token}`);
    await expect(page.getByRole("heading", { name: "Fixture Team" })).toBeVisible();
    expect(new URL(page.url()).hash).toBe("");
    if (!registered) {
      await expect(page.getByRole("button", { name: "Accept invitation" })).toHaveCount(0);
      const popup = page.waitForEvent("popup");
      await page.getByRole("link", { name: /register/i }).click();
      const signup = await popup;
      await signup.waitForLoadState("networkidle");
      await enterRegistrant(signup);
      await selectTeamEvent(signup);
      await signup.getByLabel("Mixed 5.0 team plan", { exact: true }).selectOption("solo");
      await signup.getByRole("button", { name: "Review registration", exact: true }).click();
      await signup.getByRole("checkbox", { name: /My information is correct/ }).check();
      await signup.getByRole("button", { name: "Submit registration", exact: true }).click();
      await expect(signup).toHaveURL(/tournament-signup-test\?confirmation=1/);
      await signup.close();
      await page.getByRole("button", { name: /check my registration/i }).click();
    }
    await page.getByRole("button", { name: action === "ACCEPT" ? "Accept invitation" : "Decline", exact: true }).click();
    await expect(page.getByRole("status")).toContainText(action === "ACCEPT" ? "You joined the team" : "You declined");
    if (action === "ACCEPT") {
      await page.goto("about:blank");
      await page.goto(`/tournament-signup-test?invitation=1#token=${token}`);
      await expect(page.getByText("You’ve already accepted this team invitation.")).toBeVisible();
      await expect(page.getByRole("button", { name: "Accept invitation" })).toHaveCount(0);
    }
    expect(responses).toEqual([action]);
  });
}
