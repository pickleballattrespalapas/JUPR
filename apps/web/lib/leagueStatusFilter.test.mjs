import assert from "node:assert/strict";
import test from "node:test";
import {
  filterLeaguesByStatus,
  leagueStatusCategory,
  leagueStatusCounts
} from "./leagueStatusFilter.ts";

const leagues = [
  { league_name: "Summer", status: "active" },
  { league_name: "Next season", status: "draft" },
  { league_name: "Legacy inactive", status: " INACTIVE " },
  { league_name: "Rain delay", status: "paused" },
  { league_name: "Spring", status: "ended" },
  { league_name: "Old spring", status: "archived" },
  { league_name: "Imported", status: "legacy" }
];

test("groups draft and legacy inactive statuses without hiding unknown rows from All", () => {
  assert.equal(leagueStatusCategory("inactive"), "draft");
  assert.equal(leagueStatusCategory("LEGACY"), "other");
  assert.deepEqual(leagueStatusCounts(leagues), {
    active: 1,
    draft: 2,
    paused: 1,
    ended: 1,
    archived: 1,
    all: 7
  });
});

test("filters each lifecycle state and preserves the complete list for All", () => {
  assert.deepEqual(filterLeaguesByStatus(leagues, "active").map((league) => league.league_name), ["Summer"]);
  assert.deepEqual(filterLeaguesByStatus(leagues, "draft").map((league) => league.league_name), ["Next season", "Legacy inactive"]);
  assert.deepEqual(filterLeaguesByStatus(leagues, "paused").map((league) => league.league_name), ["Rain delay"]);
  assert.deepEqual(filterLeaguesByStatus(leagues, "ended").map((league) => league.league_name), ["Spring"]);
  assert.deepEqual(filterLeaguesByStatus(leagues, "archived").map((league) => league.league_name), ["Old spring"]);
  assert.equal(filterLeaguesByStatus(leagues, "all").length, leagues.length);
});
