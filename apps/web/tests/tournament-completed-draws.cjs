const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const ts = require("typescript");

const pagePath = path.resolve(__dirname, "../app/clubs/[clubSlug]/tournament-results/page.tsx");
const compiled = ts.transpileModule(fs.readFileSync(pagePath, "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
}).outputText;
let draws;
const mockData = () => ({
  club: { id: "test", slug: "test-club", name: "Test club" },
  tournament: { id: "test-tournament", name: "Summer Classic", status: "ACTIVE", settings: {} },
  draws
});
const loaded = { exports: {} };
new Function("require", "module", "exports", compiled)((name) => {
  if (name.endsWith(".module.css")) return { __esModule: true, default: new Proxy({}, { get: (_, key) => key }) };
  if (name === "@/components/ClubDisplay") return { Display: ({ children }) => children };
  if (name === "@/components/PublicClubLink") return { __esModule: true, default: ({ prefetch, scroll, ...props }) => React.createElement("a", props) };
  if (name === "@/components/PublicTournamentSponsors" || name === "@/components/PublicTournamentNav") return { __esModule: true, default: () => null };
  if (name === "@/lib/tournamentResultsApi") return { getPublicTournamentResults: async () => ({ data: mockData(), error: null }) };
  return require(name);
}, loaded, loaded.exports);

function draw(key, name, state) {
  return {
    public_draw_key: key, name, state,
    event_family_label: "Doubles", division_name: name,
    scheduled_days: [], teams: [], bracket: [],
    podium: state === "COMPLETE" ? [
      { placement: 3, medal: "Bronze", team_name: `${name} bronze pair` },
      { placement: 1, medal: "Gold", team_name: `${name} gold pair` },
      { placement: 2, medal: "Silver", team_name: `${name} silver pair` },
      { placement: 3, medal: "Bronze", team_name: `${name} tied bronze pair` }
    ] : [],
    standings: [{ public_team_key: `${key}-team`, team_name: `${name} gold pair`, rank: 1, wins: 3, losses: 0 }],
    scores: [{ public_game_key: `${key}-game`, stage: "ROUND_ROBIN", round_number: 1, state: "FINAL", team_a_name: `${name} game team A`, team_b_name: `${name} game team B`, score_a: 11, score_b: 7 }]
  };
}
const finished = [draw("complete-one", "Men's 3.5+", "COMPLETE"), draw("complete-two", "Women's 3.5+", "COMPLETE")];
const active = draw("live-one", "Mixed 3.5+", "LIVE");
const upcoming = draw("upcoming-one", "Open 4.0+", "SCHEDULED");
async function render(search = {}) {
  return renderToStaticMarkup(await loaded.exports.default({
    params: { clubSlug: "test-club" },
    searchParams: { tournament_id: "test-tournament", ...search }
  }));
}
function hrefs(html) {
  return [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1].replaceAll("&amp;", "&"));
}

(async () => {
  draws = [...finished, active, upcoming];
  const live = await render();
  assert.match(live, /Current draws/);
  assert.match(live, /Mixed 3\.5\+ game team A/);
  assert.doesNotMatch(live, /completedCard|Men&#x27;s 3\.5\+ gold pair/);

  const overview = await render({ tab: "completed", view: "past" });
  assert.equal((overview.match(/class="completedCard"/g) || []).length, 2);
  assert.doesNotMatch(overview, /<table|Playoff bracket|Completed scores|Current draws|Upcoming draws/);
  assert.equal((overview.match(/tied bronze pair/g) || []).length, 2);
  assert.ok(overview.indexOf("gold pair") < overview.indexOf("silver pair"));
  assert.ok(overview.indexOf("silver pair") < overview.indexOf("bronze pair"));
  const cardHref = hrefs(overview).find((href) => href.includes("draw=complete-two"));
  assert.ok(cardHref);
  const cardParams = Object.fromEntries(new URL(cardHref, "https://test.invalid").searchParams);
  assert.equal(cardParams.tab, "completed");
  assert.equal(cardParams.view, "past");
  const detail = await render(cardParams);
  assert.match(detail, /Women&#x27;s 3\.5\+ game team A/);
  assert.match(detail, /Standings/);
  assert.doesNotMatch(detail, /Men&#x27;s 3\.5\+ game team A|Mixed 3\.5\+ game team A|completedCard/);
  assert.ok(hrefs(detail).includes("/clubs/test-club/tournament-results?tournament_id=test-tournament&tab=completed&view=past"));

  const legacyDeepLink = await render({ draw: "complete-one" });
  assert.match(legacyDeepLink, /Men&#x27;s 3\.5\+ game team A/);
  assert.doesNotMatch(legacyDeepLink, /Mixed 3\.5\+ game team A/);
  const missingDraw = await render({ tab: "completed", draw: "removed" });
  assert.equal((missingDraw.match(/class="completedCard"/g) || []).length, 2);

  draws = finished;
  const allDone = await render();
  assert.equal((allDone.match(/class="completedCard"/g) || []).length, 2);
  assert.match(await render({ tab: "live" }), /All draws are complete/);
  draws = [active];
  assert.match(await render({ tab: "completed" }), /No completed draws yet/);
  draws = [];
  assert.match(await render(), /No results yet/);
  console.log("Completed draw navigation, podium cards, tied medals, deep links, archive context, and empty states passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
