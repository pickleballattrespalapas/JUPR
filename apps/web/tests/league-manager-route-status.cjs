const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

function loadTypeScriptModule(relativePath) {
  const filename = path.resolve(__dirname, relativePath);
  const source = fs.readFileSync(filename, "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020
    },
    fileName: filename
  }).outputText;
  const loaded = { exports: {} };
  const evaluate = new Function("require", "module", "exports", "__filename", "__dirname", compiled);
  evaluate(require, loaded, loaded.exports, filename, path.dirname(filename));
  return loaded.exports;
}

const {
  isExactLeagueResult,
  isTeamLeagueType,
  leagueRouteHref,
  normalizeLeagueType,
  readLeagueRouteContext
} = loadTypeScriptModule("../lib/leagueRouteContext.ts");
const {
  filterLeaguesByStatus,
  leagueStatusCategory,
  leagueStatusCounts
} = loadTypeScriptModule("../lib/leagueStatusFilter.ts");

const teamContext = readLeagueRouteContext(new URLSearchParams({
  league_id: "league-team-456",
  league: "Acceptance Team League",
  league_name: "Acceptance Team League",
  mode: " team "
}));
const teamHref = new URL(
  leagueRouteHref("/admin/league-manager/results", teamContext),
  "https://staging.example.invalid"
);
assert.equal(teamContext.leagueId, "league-team-456");
assert.equal(teamContext.leagueName, "Acceptance Team League");
assert.equal(teamContext.leagueType, "Team");
assert.equal(teamHref.searchParams.get("league_id"), "league-team-456");
assert.equal(teamHref.searchParams.get("league"), "Acceptance Team League");
assert.equal(teamHref.searchParams.get("mode"), "Team");
assert.equal(normalizeLeagueType("TEAM"), "Team");
assert.equal(isTeamLeagueType(" team "), true);
assert.equal(isTeamLeagueType("Individual"), false);
assert.equal(isExactLeagueResult("Acceptance Team League", teamContext.leagueName), true);
assert.equal(isExactLeagueResult("Acceptance Singles League", teamContext.leagueName), false);

const legacyContext = readLeagueRouteContext(new URLSearchParams({ league: "Spring League" }));
const legacyHref = new URL(
  leagueRouteHref("/admin/league-manager/settings", legacyContext),
  "https://staging.example.invalid"
);
assert.equal(legacyContext.leagueId, "Spring League");
assert.equal(legacyHref.searchParams.get("league_id"), "Spring League");

const leagues = [
  { league_name: "Summer", status: "active" },
  { league_name: "Next season", status: "draft" },
  { league_name: "Legacy inactive", status: " INACTIVE " },
  { league_name: "Rain delay", status: "paused" },
  { league_name: "Spring", status: "ended" },
  { league_name: "Old spring", status: "archived" },
  { league_name: "Imported", status: "legacy" }
];
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
assert.deepEqual(
  filterLeaguesByStatus(leagues, "draft").map((league) => league.league_name),
  ["Next season", "Legacy inactive"]
);
assert.equal(filterLeaguesByStatus(leagues, "all").length, leagues.length);

console.log("league manager route and status behavior passed");
