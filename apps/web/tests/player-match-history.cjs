const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const ts = require("typescript");

function compile(file, imports) {
  const code = ts.transpileModule(fs.readFileSync(path.resolve(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => imports[name] ?? (require('./helpers/public-controls-module.cjs')(name) || require(name)), module, module.exports);
  return module.exports;
}

let requested, history, matches;
let awards = { trophies: [], badge_award_count: 0 };
const page = compile("app/clubs/[clubSlug]/players/[playerId]/page.tsx", {
  "@/components/ClubDisplay": { Display: ({ children }) => children },
  "@/components/PublicClubLink": { __esModule: true, default: props => React.createElement("a", props) },
  "@/lib/badgeApi": { publicBadgeRarityLabel: value => value },
  "@/lib/publicMatchLabels": { publicSocialEventTypeLabel: value => value },
  "@/lib/api": { getClubPlayerProfile: async (club, player, filters) => {
    requested = filters;
    return { error: null, data: {
      player: { id: 159, name: "Returning Player", wins: 24, losses: 17 },
      identity: { display_name: "Returning Player" },
      verified_updates: { status: "available", can_request: true },
      league_ratings: [], awards, social: {},
      recent_matches: matches.slice(0, 12), match_history: matches, history
    }};
  }}
}).default;

function fixture(total, offset, count) {
  history = { total_matches: total, filtered_matches: total, recent_limit: 12, history_limit: 50, history_offset: offset, has_more: offset + count < total, leagues: ["Fall 2025", "Summer 2026"] };
  matches = Array.from({ length: count }, (_, i) => ({ id: total - offset - i, league: "Fall 2025", date: "2026-02-04", team_1: [{ id: 159, name: "Returning Player" }], team_2: [], score_t1: 11, score_t2: 8 }));
}
async function render(search = {}) {
  return renderToStaticMarkup(await page({ params: { clubSlug: "tres-palapas", playerId: "159" }, searchParams: { section: "matches", history: "all", ...search } }));
}
function link(html, testId) {
  const tag = html.match(new RegExp(`<a[^>]*data-testid="${testId}"[^>]*>`));
  return tag ? new URL(tag[0].match(/href="([^"]+)"/)[1].replaceAll("&amp;", "&"), "https://test.invalid") : null;
}

(async () => {
  fixture(41, 0, 41);
  let html = await render();
  assert.match(html, /Full history \(41\)/);
  assert.equal((html.match(/data-testid="player-match-row"/g) || []).length, 41);
  assert.match(html, /Matches 1–41 of 41/);
  assert.equal(link(html, "player-history-older"), null);

  fixture(105, 0, 50);
  html = await render({ league: "Fall 2025" });
  assert.deepEqual(requested, { recent: 12, history: 50, offset: 0, league: "Fall 2025" });
  let older = link(html, "player-history-older");
  assert.equal(older.searchParams.get("page"), "2");
  assert.equal(older.searchParams.get("league"), "Fall 2025");
  assert.equal(older.searchParams.get("history"), "all");
  assert.match(html, /Summer 2026/, "League choices survive when that league is not on this page");

  fixture(105, 50, 50);
  html = await render(Object.fromEntries(older.searchParams));
  assert.equal(requested.offset, 50);
  assert.match(html, /Matches 51–100 of 105/);
  assert.equal(link(html, "player-history-newer").searchParams.get("page"), null);
  older = link(html, "player-history-older");
  assert.equal(older.searchParams.get("page"), "3");

  fixture(105, 100, 5);
  html = await render(Object.fromEntries(older.searchParams));
  assert.equal(requested.offset, 100);
  assert.match(html, /Matches 101–105 of 105/);
  assert.equal(link(html, "player-history-older"), null);
  assert.equal(link(html, "player-history-newer").searchParams.get("page"), "2");
  assert.equal(link(html, "player-history-recent").searchParams.get("page"), null);

  await render({ history: "recent", page: "3" });
  assert.equal(requested.offset, 0);
  for (const invalid of ["-1", "1.5", "garbage", "Infinity", "9007199254740991"]) {
    await render({ page: invalid });
    assert.equal(requested.offset, 0);
  }
  fixture(0, 0, 0);
  html = await render();
  assert.match(html, /No matches match this view/);
  assert.doesNotMatch(html, /player-history-pagination/);

  const trophy = (badge_id, placement, context_type, title, extra = {}) => ({ badge_id, placement, context_type, title, ...extra });
  const qualifying = [
    trophy("tournament_champion", 1, "tournament", "Tournament gold"),
    trophy("tournament_runner_up", 2, "tournament_award", "Tournament silver"),
    trophy("tournament_third_place", 3, "tournament_podium", "Tournament bronze"),
    trophy("league_champion", 1, "league_end", "League champion"),
    trophy("top_performer_most_wins", 2, "league", "League award silver"),
    trophy("award-uuid", 1, "interclub", "Interclub champion", { award_key: "division_champion" }),
  ];
  const excluded = [
    trophy("mountain_climber", 27, "league", "Excluded climb 27"),
    trophy("mountain_climber", 1, "league", "Excluded climb 1"),
    trophy("top_performer_most_wins", 66, "league", "Excluded wins 66"),
    trophy("award-uuid", 1, "interclub", "Excluded participant", { award_key: "participation" }),
    trophy("unknown", 1, "league", "Excluded unknown badge"),
    trophy("tournament_champion", 1, "weekly", "Excluded weekly award"),
    ...[0, -1, 4, null, undefined, "2", true, 2.5, NaN, Infinity].map((rank, index) => trophy("podium", rank, "league", `Excluded malformed ${index}`)),
  ];
  awards = { trophies: [...qualifying, ...excluded], trophy_count: 99, badge_award_count: 0 };
  html = await render({ section: "trophies" });
  assert.equal((html.match(/data-testid="player-trophy"/g) || []).length, qualifying.length);
  assert.match(html, /Major honors<\/strong><div[^>]*>6<\/div>/, "Summary counts only visible qualifying honors, even with an old API count");
  for (const award of qualifying) assert.ok(html.includes(award.title));
  assert.doesNotMatch(html, /Excluded|Place #66|Place #27/);
  assert.equal(awards.trophies.length, qualifying.length + excluded.length, "Presentation filtering never changes stored/API awards");
  awards = { ...awards, trophies: excluded };
  html = await render({ section: "trophies" });
  assert.match(html, /No top-three tournament or end-of-league awards yet/);
  assert.match(html, /Major honors<\/strong><div[^>]*>0<\/div>/);

  const api = compile("lib/api.ts", {});
  process.env.JUPR_API_BASE_URL = "https://test.invalid";
  let apiUrl;
  global.fetch = async url => { apiUrl = new URL(url); return { ok: true, json: async () => ({}) }; };
  await api.getClubPlayerProfile("tres-palapas", "159", { recent: 12, history: 50, offset: 100, league: "Fall 2025 & Friends" });
  assert.equal(apiUrl.searchParams.get("history_offset"), "100");
  assert.equal(apiUrl.searchParams.get("history_league"), "Fall 2025 & Friends");
  console.log("PASS player history, navigation, API queries, and top-three trophy rendering with legacy award payloads.");
})().catch(error => { console.error(error); process.exitCode = 1; });
