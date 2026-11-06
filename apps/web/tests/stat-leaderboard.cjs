const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { renderToStaticMarkup } = require("react-dom/server");
const h = React.createElement;
const Link = ({ children, prefetch, ...props }) => h("a", props, children);
const calls = [];
let data, leagueData, teamData, error;
const mocks = {
  "@/components/PublicClubLink": Link,
  "@/lib/api": {
    getClubLeaderboard: async (slug, opts) => { calls.push(["leaderboard", slug, opts]); return { data, error }; },
    getClubLeagueResults: async (...args) => { calls.push(["league", ...args]); return { data: leagueData, error }; },
  },
  "@/lib/teamLeagueApi": { getPublicTeamLeague: async (...args) => { calls.push(["team", ...args]); return { data: teamData, error }; } },
};
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function("require", "module", "exports", code)(name => {
    if (Object.hasOwn(mocks, name)) return mocks[name];
    if (name.startsWith("@/") || name.startsWith(".")) {
      const base = name.startsWith("@/") ? name.slice(2) : path.join(path.dirname(file), name);
      return load(fs.existsSync(path.join(__dirname, "..", base + ".tsx")) ? base + ".tsx" : base + ".ts");
    }
    return require(name);
  }, module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const decode = s => s.replaceAll("&amp;", "&");
(async () => {
  const { statLeaderboardHref, safeStatReturnTo } = load("lib/statLeaderboardLinks.ts");
  const source = "/clubs/tres-palapas/leaderboards?season=2026&status=all&page=3#stat-card-most_wins";
  assert.equal(safeStatReturnTo("tres-palapas", source), source);
  for (const malicious of ["https://evil.test", "//evil.test", "/clubs/elsewhere/leaderboards", "/clubs/tres-palapas/../../admin", "/clubs/tres-palapas/\\evil.test", "javascript:alert(1)"]) assert.equal(safeStatReturnTo("tres-palapas", malicious), "/clubs/tres-palapas/leaderboards");
  const url = new URL(statLeaderboardHref("tres-palapas", "most_wins", { league: "Sunday Ladder", leagueView: "past", status: "all", season: "2026", search: "Joe", returnTo: source }), "https://fixture.test");
  assert.equal(url.searchParams.get("return_to"), source);
  assert.equal(url.searchParams.get("league"), "Sunday Ladder");
  const Page = load("app/clubs/[clubSlug]/leaderboards/[stat]/page.tsx").default;
  const render = async (stat, searchParams = {}) => renderToStaticMarkup(await Page({ params: Promise.resolve({ clubSlug: "tres-palapas", stat }), searchParams: Promise.resolve(searchParams) }));
  const { LEADERBOARD_CARD_LABELS } = load("lib/clubSite.ts");
  for (const [stat, title] of Object.entries(LEADERBOARD_CARD_LABELS)) {
    data = { club: { name: "Tres Palapas" }, selected_scope: "OVERALL", period: { id: "2026", name: "2026–27" }, filters: { status: "all" }, stat_ranking: { key: stat, entries: [{ rank: 73, player_id: 7, player_name: "Joe <safe>", metric_value: 42, metric_display: "+42 wins", metric_sample: 80 }], total_qualified: 103, minimum: 5, minimum_metric: "games", pagination: { total: 103, offset: 50, limit: 50, has_more: true } } };
    const html = await render(stat, { season: "2026", status: "all", page: "2", return_to: source });
    assert.ok(html.includes(title.replaceAll("&", "&amp;")), stat);
    assert.match(html, /#73/);
    assert.match(html, /Joe &lt;safe&gt;/);
    assert.match(html, /103 qualifying/);
    assert.match(html, /Minimum 5 games/);
    assert.match(html, /Page 2 of 3/);
    const request = calls.at(-1)[2];
    assert.equal(request.stat, stat); assert.equal(request.offset, 50); assert.equal(request.season, "2026");
    assert.equal(request.status, "all");
    for (const link of html.matchAll(/href="([^"]+)"[^>]*>(Previous|Next)<\/a>/g)) {
      const query = new URL(decode(link[1]), "https://fixture.test").searchParams;
      assert.equal(query.get("season"), "2026"); assert.equal(query.get("return_to"), source);
    }
  }
  data.stat_ranking.key = "most_wins";
  for (const status of ["active", "inactive", "all"]) {
    await render("most_wins", { league: "Archived Ladder", league_view: "past", status });
    assert.equal(calls.at(-1)[2].status, status, "Past league rankings preserve the card's player filter");
    assert.equal(calls.at(-1)[2].leagueView, "past");
  }
  await render("most_wins");
  assert.equal(calls.at(-1)[2].season, null, "Unspecified season uses the club's current default");
  data.stat_ranking.key = "best_partnership";
  data.stat_ranking.entries = [{ rank: 5, team_key: "1:2", player_name: "Suzanne & Esteban", team_members: [{ player_id: 1, player_name: "Suzanne" }, { player_id: 2, player_name: "Esteban" }], metric_display: "90.0% (9/10)", metric_sample: 10 }];
  data.stat_ranking.total_qualified = 30; data.stat_ranking.pagination = { total: 1, offset: 0, limit: 50, has_more: false };
  let html = await render("best_partnership", { q: "Suzanne" });
  assert.equal((html.match(/data-testid="stat-ranking-row"/g) || []).length, 1);
  assert.match(html, /#5/); assert.match(html, /players\/1/); assert.match(html, /players\/2/);
  assert.match(html, /Ranks show positions in the full qualifying field/);
  assert.doesNotMatch(html, /players\/(null|undefined)/);
  data.stat_ranking.entries = []; data.stat_ranking.pagination.total = 0;
  assert.match(await render("best_partnership", { q: "Nobody" }), /No players or teams match/);
  data.stat_ranking.total_qualified = 0;
  assert.match(await render("best_partnership"), /Not enough qualifying data yet/);
  error = "Unavailable";
  assert.match(await render("most_wins"), /role="alert"/);
  error = null;
  const before = calls.length;
  assert.match(await render("invalid"), /This statistic is not available/);
  assert.equal(calls.length, before, "Unknown stats do not query the API");

  const entries = Array.from({ length: 55 }, (_, index) => ({ category_key: "most_wins", category_label: "Most wins", player_id: index + 1, recipient_name: `Player ${index + 1}`, rank: index + 1, metric_display: `${60 - index} wins`, min_games: 5 }));
  const progress = { awards: entries.slice(0, 3), finalized: true, frozen_at: "2026-11-05T20:00:00Z", races: [{ category_key: "most_wins", category_label: "Most wins", min_games: 5, minimum_metric: "games", eligible_count: 55, entries }] };
  leagueData = { club: { name: "Tres Palapas" }, selected_league: "Ladder", selected_week: 3, weeks: [{ week_num: 3, week_label: "Week 3" }], award_progress: progress };
  html = await render("most_wins", { source: "awards", league: "Ladder", page: "2" });
  assert.equal((html.match(/data-testid="stat-ranking-row"/g) || []).length, 5);
  assert.match(html, /#51/); assert.match(html, /Final award results/);
  html = await render("most_wins", { source: "awards", league: "Ladder", q: "Player 55" });
  assert.match(html, /#55/); assert.match(html, /55 qualifying players/);
  leagueData.award_progress.unavailable_categories = ["best_partnership"];
  assert.match(await render("best_partnership", { source: "awards", league: "Ladder" }), /complete ranking was not saved/);
  teamData = { league: { league_name: "Ladder" }, award_progress: progress };
  assert.match(await render("most_wins", { source: "team-awards", league: "Ladder" }), /Player 1/);
  assert.equal(calls.at(-1)[0], "team");

  leagueData.weekly_stat_rankings = { wins: { key: "wins", title: "Weekly wins", qualification: "At least 2 games", total_qualified: 55, entries: entries.map(entry => ({ ...entry, player_name: entry.recipient_name, metric_value: 10, metric_sample: 12 })) } };
  html = await render("wins", { source: "weekly", league: "Ladder", week: "3", weekly_min_games: "4", q: "Player 55" });
  assert.match(html, /Weekly wins/); assert.match(html, /Week 3/); assert.match(html, /#55/);
  assert.equal(calls.at(-1)[5], 4);
  assert.match(html, /name="weekly_min_games" value="4"/);
  assert.match(await render("wins", { source: "weekly", league: "Ladder", week: "8" }), /selected league or week is unavailable/);

  leagueData.season_stat_rankings = { ...leagueData.weekly_stat_rankings, most_active: { ...leagueData.weekly_stat_rankings.wins, title: "Most games played", qualification: "Every league participant" } };
  html = await render("most_active", { source: "season", league: "Ladder", q: "Player 55" });
  assert.match(html, /Full season/); assert.match(html, /#55/); assert.match(html, /Every league participant/);
  assert.equal(calls.at(-1)[3], undefined, "Season ranking does not send a selected week");
  data.stat_ranking.key = "most_improved"; data.stat_ranking.entries = [{ player_id: 1, player_name: "Joe", rank: 1, metric_display: "+0.400", starting_rating_jupr: 3.5, rating_gain_jupr: 0.4, rating_jupr: 4.8 }];
  html = await render("most_improved");
  assert.match(html, /3.500 → 3.900/); assert.doesNotMatch(html, /3.500 → 4.800/);

  const { LeagueAwardRaceGrid } = load("components/LeagueAwardRace.tsx");
  const pair = { category_key: "best_partnership", category_label: "Best partnership", rank: 1, recipient_type: "team", recipient_name: "Suzanne & Esteban", team_id: "pair:1:2", team_members: [{ player_id: 1, player_name: "Suzanne" }, { player_id: 2, player_name: "Esteban" }], metric_display: "100% (6/6)" };
  const pairs = { awards: [pair], races: [{ category_key: "best_partnership", category_label: "Best partnership", entries: [pair], recipient_type: "team" }] };
  html = renderToStaticMarkup(h(LeagueAwardRaceGrid, { progress: pairs, clubSlug: "tres-palapas", leagueName: "Ladder", returnTo: "/clubs/tres-palapas/leagues/Ladder#awards" }));
  assert.match(html, /leaderboards\/best_partnership\?/); assert.match(html, /source=awards/); assert.match(html, /league=Ladder/);
  assert.match(html, /players\/1/); assert.match(html, /players\/2/);
  console.log("Stat leaderboard: all 17 stats, scoped links, safe return, preserved ranking, search, pagination, pairs, final awards, weekly and team sources passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
