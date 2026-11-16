const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

let player = { id: 1, name: "Social Player", rating: null, rating_jupr: null, starting_rating_jupr: null,
  singles_rating: null, singles_rating_jupr: null, matches_played: 0, singles_matches_played: 0 };
const rated = { id: 2, name: "Rated Player", rating: 1612, rating_jupr: 4.03, matches_played: 1 };
let matches = [];
const api = {
  getClubPlayers: async () => ({ data: { players: [rated, player] } }),
  getClubPlayerProfile: async () => ({ data: {
    player, identity: { display_name: player.name }, verified_updates: { status: "available" },
    awards: { trophies: [] }, social: {}, league_ratings: [{ league_name: "Open", rating_jupr: null, matches_played: 0 }],
    rating_summary: {}, rating_history: [], rating_breakdowns: [], match_history: matches, recent_matches: matches,
    history: { total_matches: 0, history_limit: 50, recent_limit: 12 },
  } }),
};
const imports = {
  "@/lib/api": api,
  "@/lib/publicMatchLabels": { publicSocialEventTypeLabel: value => value },
  "@/components/ClubDisplay": { Display: ({ children }) => children },
  "@/components/PublicClubLink": { __esModule: true, default: props => React.createElement("a", props) },
  "@/components/PublicPlayerSearch": { __esModule: true, default: () => null },
};
function compile(file) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => imports[name] || require("./helpers/public-controls-module.cjs")(name) || require(name), module, module.exports);
  return module.exports.default;
}

(async () => {
  const Directory = compile("app/clubs/[clubSlug]/players/page.tsx");
  const Profile = compile("app/clubs/[clubSlug]/players/[playerId]/page.tsx");
  const props = { params: Promise.resolve({ clubSlug: "club", playerId: "1" }), searchParams: Promise.resolve({ section: "ratings" }) };
  let html = renderToStaticMarkup(await Directory(props));
  assert.match(html, /Social Player/);
  assert.match(html, /Not rated/);
  assert.match(html, /4\.030/);
  html = renderToStaticMarkup(await Profile(props));
  assert.equal((html.match(/Not rated/g) || []).length, 6, "Doubles, singles, starting, highest, lowest and league estimates all show Not rated");
  assert.match(html, /No rating history is available yet/);
  assert.doesNotMatch(html, /data-testid="player-rating-trend"/);
  player = { ...player, rating: 1564, rating_jupr: 3.91, matches_played: 1 };
  html = renderToStaticMarkup(await Profile(props));
  assert.match(html, /Doubles \/ overall<\/strong><div[^>]*>3\.910/);
  assert.match(html, /Singles<\/strong><div[^>]*>Not rated/);
  matches = [{ id: 99, rating_scope: "unrated", team_1: [], team_2: [], score_t1: 11, score_t2: 8 }];
  html = renderToStaticMarkup(await Profile({ ...props, searchParams: Promise.resolve({ section: "matches" }) }));
  assert.match(html, /data-testid="player-match-row"[\s\S]*Not rated/);
  assert.doesNotMatch(html.match(/<tr data-testid="player-match-row"[\s\S]*?<\/tr>/)[0], /→/);
  console.log("PASS provisional public ratings: directory, profile cards, summaries, league ratings and independent disciplines.");
})().catch(error => { console.error(error); process.exitCode = 1; });
