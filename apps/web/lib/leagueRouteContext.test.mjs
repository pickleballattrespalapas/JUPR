import assert from "node:assert/strict";
import test from "node:test";
import {
  isExactLeagueResult,
  leagueRouteHref,
  readLeagueRouteContext
} from "./leagueRouteContext.ts";

for (const [leagueId, leagueName] of [
  ["league-spring-123", "Spring League"],
  ["league-team-456", "Acceptance Team League"]
]) {
  test(`preserves distinct league identity and display name for ${leagueName}`, () => {
    const context = readLeagueRouteContext(new URLSearchParams({
      league_id: leagueId,
      league: leagueName,
      league_name: leagueName,
      mode: leagueName === "Acceptance Team League" ? "Team" : "Individual"
    }));
    const href = leagueRouteHref("/admin/league-manager/results", context);
    const target = new URL(href, "https://staging.example.invalid");

    assert.equal(context.leagueId, leagueId);
    assert.equal(target.searchParams.get("league_id"), leagueId);
    assert.equal(target.searchParams.get("league"), leagueName);
    assert.equal(isExactLeagueResult(leagueName, context.leagueName), true);
    assert.equal(isExactLeagueResult("Acceptance Singles", context.leagueName), false);
  });
}

test("upgrades legacy name-only links when generating the next route", () => {
  const context = readLeagueRouteContext(new URLSearchParams({
    league: "Spring League",
    mode: "Individual"
  }));
  const href = leagueRouteHref("/admin/league-manager/settings", context);
  const target = new URL(href, "https://staging.example.invalid");

  assert.equal(context.leagueId, "Spring League");
  assert.equal(target.searchParams.get("league_id"), "Spring League");
});
