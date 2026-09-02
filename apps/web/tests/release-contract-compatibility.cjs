const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

function loadTypeScriptModule(relativePath, dependencies = {}) {
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
  const localRequire = (specifier) => dependencies[specifier] || require(specifier);
  const evaluate = new Function(
    "require",
    "module",
    "exports",
    "__filename",
    "__dirname",
    compiled
  );
  evaluate(
    localRequire,
    loaded,
    loaded.exports,
    filename,
    path.dirname(filename)
  );
  return loaded.exports;
}

async function main() {
  const originalFetch = global.fetch;
  const originalApiBase = process.env.JUPR_API_BASE_URL;
  const payloads = [
    { club: { id: "club", slug: "club", name: "Club" }, leagues: [] },
    {
      club: { id: "club", slug: "club", name: "Club" },
      leagues: [],
      past_leagues: [{ name: "Past league" }],
      award_progress: {
        awards: [{ category_key: "mvp", category_label: "MVP" }],
        award_count: 1,
        races: [],
        race_count: 0
      }
    },
    {
      ok: true,
      league: {},
      teams: [],
      fixtures: [],
      standings: [],
      registration: {},
      registration_players: []
    }
  ];

  try {
    process.env.JUPR_API_BASE_URL = "https://legacy-api.example.invalid";
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => payloads.shift()
    });

    const api = loadTypeScriptModule("../lib/api.ts");
    const teamLeagueApi = loadTypeScriptModule("../lib/teamLeagueApi.ts", {
      "./api": api
    });

    const legacyLeague = await api.getClubLeagueResults("club");
    assert.deepEqual(legacyLeague.data.past_leagues, []);
    assert.deepEqual(legacyLeague.data.award_progress, {
      awards: [],
      award_count: 0
    });

    const currentLeague = await api.getClubLeagueResults("club");
    assert.deepEqual(currentLeague.data.past_leagues, [{ name: "Past league" }]);
    assert.equal(currentLeague.data.award_progress.award_count, 1);
    assert.equal(currentLeague.data.award_progress.awards[0].category_key, "mvp");

    const legacyTeamLeague = await teamLeagueApi.getPublicTeamLeague(
      "club",
      "Team league"
    );
    assert.deepEqual(legacyTeamLeague.data.award_progress, {
      awards: [],
      award_count: 0
    });
    assert.equal(payloads.length, 0);
  } finally {
    global.fetch = originalFetch;
    if (originalApiBase === undefined) delete process.env.JUPR_API_BASE_URL;
    else process.env.JUPR_API_BASE_URL = originalApiBase;
  }
}

main()
  .then(() => console.log("release contract compatibility passed"))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
