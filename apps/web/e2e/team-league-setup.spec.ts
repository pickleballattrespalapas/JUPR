import { expect, test } from "@playwright/test";

for (const size of [2, 4]) {
  test(`${size}-player team setup saves and reloads`, async ({ context, page, baseURL }, testInfo) => {
    const league = `Season ${size}-player teams`;
    let saved: Record<string, unknown> | null = null;
    let writes = 0;
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    if (size === 4) await page.setViewportSize({ width: 390, height: 844 });
    await context.addCookies([{ name: "jupr_admin_workspace_v1", value: encodeURIComponent(JSON.stringify({ clubId: "tres_palapas", clubSlug: "tres-palapas" })), url: baseURL!, sameSite: "Lax" }]);
    await page.addInitScript(() => localStorage.setItem("jupr_admin_session_v1", JSON.stringify({ access_token: "ui-test-token", user: { id: "fixture", email: "admin@example.invalid" } })));
    await page.route("**/admin/auth/workspaces", route => route.fulfill({ json: { workspaces: [{ club_id: "tres_palapas", club_slug: "tres-palapas", club_name: "Tres Palapas", roles: ["club_owner"] }] } }));
    await page.route(/\/admin\/auth\/capabilities(?:\?.*)?$/, route => route.fulfill({ json: { authorized: true, assignments: [{ club_id: "tres_palapas", role: "club_owner", permissions: [] }] } }));
    await page.route("**/admin/clubs/tres_palapas/**", async route => {
      const req = route.request(), path = decodeURIComponent(new URL(req.url()).pathname);
      if (path.endsWith(`/team-leagues/${league}/settings`)) {
        expect(req.method()).toBe("PUT");
        const body = req.postDataJSON();
        expect(req.headers().authorization).toBe("Bearer ui-test-token");
        expect(body.expected_settings_version).toBe(0);
        expect(body.settings.team_size).toBe(size);
        expect(body.confirmation_text).toBe("SAVE TEAM LEAGUE");
        saved = { ...body.settings, league_name: league, settings_version: 1 };
        writes++;
        return route.fulfill({ json: { committed: true } });
      }
      expect(req.method()).toBe("GET");
      if (path.endsWith("/team-leagues")) return route.fulfill({ json: { leagues: saved ? [saved] : [] } });
      if (path.endsWith(`/leagues/${league}`)) return route.fulfill({ json: {
        ok: true, league: { league_name: league, league_type: "Team", status: "draft", match_format: "doubles" },
        capabilities: { settings_mode: "full", roster_mutable: true, lifecycle_actions: ["start"], printable: true },
        schedule_preview: [], standings: [], standings_count: 0, roster: [],
      } });
      if (path.includes("/notifications")) return route.fulfill({ json: { categories: [], items: [], history_days: 30 } });
      return route.fulfill({ json: { award_catalog: [], league: {}, writes_enabled: true } });
    });
    await page.goto(`/admin/league-manager/settings?league_id=fixture-${size}&league_name=${encodeURIComponent(league)}&mode=Team`);
    const panel = page.getByTestId("team-league-setup");
    await expect(panel.getByLabel("Primary roster size")).toBeVisible();
    await panel.getByLabel("Primary roster size").selectOption(String(size));
    if (size === 4) await expect(panel).toContainText("An admin assembles these rosters");
    await panel.getByRole("button", { name: "Save team league setup", exact: true }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Yes, save setup", exact: true }).click();
    await expect(dialog.getByRole("heading", { name: "Team league setup saved" })).toBeVisible();
    await dialog.getByRole("button", { name: "OK", exact: true }).click();
    expect(writes).toBe(1);
    await page.reload();
    await expect(panel.getByLabel("Primary roster size")).toHaveValue(String(size));
    await expect(panel.getByRole("button", { name: "Save team league setup", exact: true })).toBeDisabled();
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`team-${size}-saved.png`) });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(errors).toEqual([]);
  });
}
