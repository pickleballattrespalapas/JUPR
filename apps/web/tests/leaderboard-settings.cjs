const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const { renderToStaticMarkup } = require("react-dom/server");
const h = React.createElement;
const initial = {};
let site = { club_id: "tres_palapas", slug: "tres-palapas", club_active: true, revision: 1, draft: structuredClone(initial), published: structuredClone(initial), published_at: "2026-09-19" };
const Link = ({ children, ...props }) => h("a", props, children);
const mocks = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "@/lib/useAdminWorkspace": { useAdminWorkspace: () => ({ clubId: "tres_palapas" }) },
  "@/lib/useAdminSession": { useAdminSession: () => ({ loading: false, accessToken: "fixture", session: { user: { id: "one" }, capabilities: { assignments: [{ club_id: "tres_palapas", role: "administrator" }] } } }) },
  "@/lib/adminAuthClient": { getAdminApiBaseUrl: () => "https://api.test" },
  "@/lib/adminWorkspace": { readBrowserWorkspace: () => ({ clubId: "tres_palapas" }) },
  "@/lib/badgeApi": { publicBadgeRarityLabel: value => value },
};
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function("require", "module", "exports", code)(name => {
    if (name.endsWith(".css")) return {};
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
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
global.window = { addEventListener() {}, removeEventListener() {}, location: { origin: "https://staging.test" } };
global.crypto ??= require("node:crypto").webcrypto;
const mutations = [];
global.fetch = async (url, options = {}) => {
  if (options.method) {
    const body = JSON.parse(options.body);
    assert.equal(body.revision, site.revision);
    mutations.push({ url, body });
    if (options.method === "PUT") site.draft = structuredClone(body.settings);
    else if (url.endsWith("/publish")) site.published = structuredClone(site.draft);
    site.revision++;
  }
  return { ok: true, status: 200, json: async () => structuredClone(site) };
};

(async () => {
  const { leaderboardSettings, LEADERBOARD_CARD_LABELS, LEADERBOARD_CARD_GROUPS } = load("lib/clubSite.ts");
  assert.deepEqual(leaderboardSettings().seasons, [], "No season is created automatically for existing clubs");
  assert.deepEqual(leaderboardSettings().cards, ["highest_rating", "most_improved", "best_win_pct", "most_wins"], "New choices do not change existing clubs' selected cards");
  assert.deepEqual(leaderboardSettings().card_options, {});
  const Website = load("app/admin/leaderboard-settings/page.tsx").default;
  let tree;
  await act(async () => { tree = create(h(Website)); });
  const button = label => tree.root.findAllByType("button").find(item => text(item) === label);
  const field = label => tree.root.findByProps({ "aria-label": label });
  assert.equal(Object.keys(LEADERBOARD_CARD_LABELS).length, 17);
  for (const group of LEADERBOARD_CARD_GROUPS) assert.ok(tree.root.findAllByType("legend").some(node => text(node) === group));
  for (const label of Object.values(LEADERBOARD_CARD_LABELS)) assert.ok(field(`Show ${label} card`));
  await act(async () => field("Remove Highest rating card").props.onClick());
  await act(async () => field("Show Most games played card").props.onChange({ target: { checked: true } }));
  await act(async () => field("Move Most games played up").props.onClick());
  await act(async () => field("Move Most games played up").props.onClick());
  await act(async () => field("Show summary counts above the cards").props.onChange({ target: { checked: false } }));
  await act(async () => field("Minimum games for performance cards").props.onChange({ target: { value: "10" } }));
  await act(async () => field("Show Best close-game record card").props.onChange({ target: { checked: true } }));
  await act(async () => field("Best close-game record minimum close games").props.onChange({ target: { value: "6" } }));
  await act(async () => field("Best close-game record players to show").props.onChange({ target: { value: "3" } }));
  await act(async () => field("Remove Best close-game record card").props.onClick());
  await act(async () => field("Show Best close-game record card").props.onChange({ target: { checked: true } }));
  assert.equal(field("Best close-game record minimum close games").props.value, 6, "Hiding a card preserves its criteria");
  assert.equal(field("Best close-game record players to show").props.value, 3);
  await act(async () => field("Show Best partnership card").props.onChange({ target: { checked: true } }));
  await act(async () => field("Best partnership minimum games with that partner").props.onChange({ target: { value: "8" } }));
  await act(async () => field("Best partnership players to show").props.onChange({ target: { value: "11" } }));
  await act(async () => button("Save draft").props.onClick());
  assert.equal(mutations.length, 0, "Invalid card depth cannot be saved");
  assert.match(text(tree.toJSON()), /show between 1 and 10 players/);
  await act(async () => field("Best partnership players to show").props.onChange({ target: { value: "2" } }));
  await act(async () => field("Show Biggest upset card").props.onChange({ target: { checked: true } }));
  await act(async () => field("Biggest upset teams to show").props.onChange({ target: { value: "11" } }));
  await act(async () => button("Save draft").props.onClick());
  assert.equal(mutations.length, 0, "Invalid team depth cannot be saved");
  assert.match(text(tree.toJSON()), /Biggest upset: show between 1 and 10 teams/);
  await act(async () => field("Biggest upset teams to show").props.onChange({ target: { value: "3" } }));
  await act(async () => field("Biggest upset minimum upset wins together").props.onChange({ target: { value: "2" } }));
  assert.match(text(tree.toJSON()), /minimum counts games the two teammates played together/);
  await act(async () => field("Timezone for All time statistics").props.onChange({ target: { value: "America/Phoenix" } }));
  await act(async () => button("Add season or date range").props.onClick());
  await act(async () => button("Save draft").props.onClick());
  assert.equal(mutations.length, 0, "An unfinished season cannot be saved");
  assert.match(text(tree.toJSON()), /add a season name/);
  assert.equal(field("Season 1 timezone").props.value, "America/Mazatlan");
  await act(async () => field("Season 1 name").props.onChange({ target: { value: "2026–27" } }));
  await act(async () => field("Season 1 start date").props.onChange({ target: { value: "2026-09-15" } }));
  await act(async () => field("Season 1 end date").props.onChange({ target: { value: "2026-09-01" } }));
  await act(async () => button("Save draft").props.onClick());
  assert.equal(mutations.length, 0, "An inverted range cannot be saved");
  await act(async () => field("Season 1 end date").props.onChange({ target: { value: "" } }));
  let select = tree.root.findByType("select");
  const seasonId = select.findAllByType("option")[1].props.value;
  await act(async () => select.props.onChange({ target: { value: seasonId } }));
  await act(async () => button("Save draft").props.onClick());
  assert.deepEqual(site.draft.cards, ["most_improved", "most_matches", "best_win_pct", "most_wins", "close_game_record", "best_partnership", "biggest_upset"]);
  assert.deepEqual(site.draft.card_options, { close_game_record: { minimum: 6, depth: 3 }, best_partnership: { minimum: 8, depth: 2 }, biggest_upset: { minimum: 2, depth: 3 } });
  assert.equal(site.draft.timezone, "America/Phoenix");
  assert.equal(site.draft.default_season_id, seasonId);
  assert.equal(site.draft.seasons[0].end_date, null);
  assert.equal(site.draft.min_games, 10);
  assert.deepEqual(site.published, {}, "Saving a draft leaves public settings untouched");
  await act(async () => tree.unmount());
  await act(async () => { tree = create(h(Website)); });
  assert.equal(field("Best close-game record minimum close games").props.value, 6, "Card criteria survive a fresh load");
  assert.equal(field("Best close-game record players to show").props.value, 3);
  assert.equal(field("Best partnership minimum games with that partner").props.value, 8);
  assert.equal(field("Biggest upset teams to show").props.value, 3, "Team depth survives a fresh load");
  assert.equal(field("Biggest upset minimum upset wins together").props.value, 2);
  await act(async () => button("Publish settings").props.onClick());
  assert.deepEqual(site.published, site.draft);
  await act(async () => field("Remove 2026–27 from leaderboard").props.onClick());
  assert.equal(tree.root.findByType("select").props.value, "all", "Removing the default season restores All time");
  await act(async () => tree.unmount());

  const publicSettings = structuredClone(site.published);
  const row = { player_id: 1, player_name: "Stu", rank: 1, rating_jupr: 4.2, rating_gain_jupr: 0.05, matches_played: 12, wins: 8, losses: 4, win_pct: 66.7, is_active: true };
  const data = { club: { id: "tres_palapas", slug: "tres-palapas", name: "Tres Palapas" }, scopes: [{ name: "OVERALL", label: "Overall", min_games: 0 }, { name: "Ladder", label: "Ladder", min_games: 3 }], selected_scope: "OVERALL", scope: { name: "OVERALL", min_games: 0 }, filters: { league_view: "active", status: "active", search: "", sort: "rank" }, summary: { ranked_players: 80, active_players: 80, inactive_players: 0, leaderboard_scopes: 2, filtered_players: 80 }, leaderboard: [row], snapshot: row, leaderboard_settings: publicSettings, period: { ...publicSettings.seasons[0] }, highlights: Object.fromEntries(["highest_rating", "most_improved", "best_win_pct", "most_wins", "most_matches"].map(key => [key, [row]])), pagination: { total: 80, offset: 0, limit: 50, has_more: true } };
  const requests = [];
  mocks["@/lib/api"] = { getClubLeaderboard: async (slug, options) => { requests.push({ slug, options }); return { data: structuredClone(data), error: null }; } };
  const Page = load("app/clubs/[clubSlug]/leaderboards/page.tsx").default;
  const { ClubDisplayProvider } = load("components/ClubDisplay.tsx");
  const render = async (searchParams = {}, display = {}) => {
    const page = await Page({ params: { clubSlug: "tres-palapas" }, searchParams });
    return renderToStaticMarkup(h(ClubDisplayProvider, { display }, page));
  };
  let markup = await render({ season: seasonId });
  assert.equal(requests.at(-1).options.season, seasonId);
  const titles = [...markup.matchAll(/<h2[^>]*>([^<]+)<\/h2>/g)].map(match => match[1]);
  assert.deepEqual(titles, ["Stu", "Most improved", "Most games played", "Best win %", "Most wins", "Best close-game record", "Best partnership", "Biggest upset"]);
  assert.doesNotMatch(markup, /data-testid="leaderboard-summary"/);
  assert.match(markup, /Sep 15, 2026 onward/);
  assert.match(markup, /Ratings and ranks show current standing/);
  assert.match(markup, /at least 10 recorded games/);
  assert.match(markup, /Minimum 6 close games/);
  assert.match(markup, /Minimum 8 games with that partner/);
  assert.match(markup, /Minimum 2 upset wins together/);
  const searchForm = markup.match(/<form[^>]*data-testid="leaderboard-search-form"[\s\S]*?<\/form>/)[0];
  assert.ok(searchForm.includes(`name="season" value="${seasonId}"`), "Searching preserves the selected season");
  for (const match of markup.matchAll(/href="([^"]+)"[^>]*>(Next|Win %|See all|Share this player|view summary)<\/a>/g)) {
    assert.ok(match[1].includes(`season=${seasonId}`), `${match[2]} preserves the selected season`);
  }
  markup = await render({ season: seasonId }, { match_counts: false, records: false });
  assert.match(markup, /<h2[^>]*>Most games played/);
  assert.match(markup, /<h2[^>]*>Most wins/);
  assert.match(markup, /<h2[^>]*>Most improved/);
  assert.doesNotMatch(markup, /<th\b[^>]*>(Games|W-L)<\/th>/, "Stats controls still hide table columns");

  const savedCards = [...data.leaderboard_settings.cards];
  data.leaderboard_settings.cards = ["most_improved", "most_wins", "average_margin", "biggest_upset"];
  markup = await render({ season: seasonId }, { rating_changes: false });
  assert.deepEqual([...markup.matchAll(/<h2[^>]*>([^<]+)<\/h2>/g)].map(match => match[1]),
    ["Stu", "Most improved", "Most wins", "Best average margin", "Biggest upset"],
    "All four selected cards appear even when rating gains are hidden elsewhere");
  assert.doesNotMatch(markup, /<th\b[^>]*>(Gain|Gap)<\/th>/, "Showing Most improved does not enable Gain or Gap columns");
  data.leaderboard_settings.cards = savedCards;
  data.period = { id: null, name: "All time", start_date: null, end_date: null, timezone: "UTC" };
  markup = await render({ season: "all" });
  assert.equal(requests.at(-1).options.season, "all");
  assert.match(markup, /Showing all-time statistics/);
  assert.match(markup, /name="season" value="all"/);
  data.leaderboard_settings.cards = Object.keys(LEADERBOARD_CARD_LABELS);
  for (const key of data.leaderboard_settings.cards) {
    data.highlights[key] = [{ ...row, metric_value: 7, metric_display: `Metric ${key}`, metric_sample: 8 }];
  }
  data.highlights.best_partnership[0].metric_display = "75.0% with Alex & Sam · 8 games";
  data.highlights.point_differential = [
    { ...row, metric_value: -5, metric_display: "−5 points", metric_sample: 8 },
    { ...row, player_id: 2, player_name: "Pat", metric_value: 0, metric_display: "0 points", metric_sample: 8 },
  ];
  const teamNames = [["Caleb Nguyen", "Camila Flores"], ["Rafael Torres", "Alex Rivera"], ["Sam & Pat", "Jo Chen"]];
  data.highlights.biggest_upset = teamNames.map((names, index) => ({
    player_id: null, player_name: names.join(" & "), rank: index + 1,
    team_key: `${101 + index * 2}:${102 + index * 2}`,
    team_members: names.map((player_name, memberIndex) => ({ player_id: 101 + index * 2 + memberIndex, player_name })),
    metric_value: 0.5 - index / 10, metric_display: `+${(0.5 - index / 10).toFixed(3)} JUPR`, metric_sample: 2,
  }));
  data.highlights.average_margin = [201, 202, 203].map(player_id => ({ ...row, player_id, player_name: "Clover", metric_value: 4, metric_display: "+4.0 pts/game", metric_sample: 8 }));
  markup = await render();
  for (const key of data.leaderboard_settings.cards) assert.ok(markup.includes(LEADERBOARD_CARD_LABELS[key]), `${key} renders`);
  assert.match(markup, /75\.0% with Alex &amp; Sam · 8 games/);
  assert.match(markup, /Metric highest_rating/, "Generic formatting overrides the legacy field");
  assert.match(markup, /width:100%;height:100%;border-radius:999px;background:#dc2626/);
  assert.match(markup, /width:0%;height:100%;border-radius:999px;background:#2563eb/, "A zero metric has an empty bar");
  assert.doesNotMatch(markup, /width:(?:NaN|Infinity|-)/, "Bars use finite, nonnegative widths");
  const cardMarkup = (html, title) => [...html.matchAll(/<article\b[^>]*data-testid="leaderboard-highlight-card"[\s\S]*?<\/article>/g)]
    .map(match => match[0]).find(card => card.includes(`>${title}</h2>`));
  const upsetCard = cardMarkup(markup, "Biggest upset");
  assert.equal((upsetCard.match(/data-testid="leaderboard-team-row"/g) || []).length, 3, "Top three upset places represent three teams");
  assert.deepEqual([...upsetCard.matchAll(/data-testid="leaderboard-team-place"[^>]*>#(\d+)<\/span>/g)].map(match => Number(match[1])), [1, 2, 3]);
  assert.deepEqual([...upsetCard.matchAll(/<a href="([^"]+)"/g)].map(match => match[1]),
    [101, 102, 103, 104, 105, 106].map(id => `/clubs/tres-palapas/players/${id}`), "Each team's two names link to their own profile");
  assert.match(upsetCard, /Sam &amp; Pat/);
  assert.doesNotMatch(upsetCard, /players\/(?:null|undefined)/);
  const marginCard = cardMarkup(markup, "Best average margin");
  assert.deepEqual([...marginCard.matchAll(/<a href="([^"]+)"/g)].map(match => match[1]),
    [201, 202, 203].map(id => `/clubs/tres-palapas/players/${id}`), "Distinct players remain separate even when their names match");
  assert.doesNotMatch(marginCard, /leaderboard-team-(?:row|place)/, "Player cards keep their existing presentation");
  const displayColumns = { ratings: ["Rating"], records: ["W-L"], win_percentage: ["Win %"], match_counts: ["Games"], rating_changes: ["Gain", "Gap"] };
  for (const [displayField, hiddenColumns] of Object.entries(displayColumns)) {
    const gated = await render({}, { [displayField]: false });
    const renderedTitles = [...gated.matchAll(/<h2[^>]*>([^<]+)<\/h2>/g)].map(match => match[1]);
    for (const key of data.leaderboard_settings.cards) {
      assert.ok(renderedTitles.includes(LEADERBOARD_CARD_LABELS[key]), `Selected ${key} remains visible when ${displayField} is hidden elsewhere`);
    }
    const renderedColumns = [...gated.matchAll(/<th\b[^>]*>([^<]+)<\/th>/g)].map(match => match[1]);
    for (const column of hiddenColumns) {
      assert.ok(!renderedColumns.includes(column), `${column} table column still honors ${displayField}`);
    }
  }
  data.leaderboard_settings.cards = [];
  assert.doesNotMatch(await render(), /leaderboard-highlight-card/, "An empty card selection is respected");
  data.selected_scope = "Ladder";
  markup = await render({ league: "Ladder", season: seasonId });
  assert.match(markup, /leaderboard-summary/);
  assert.match(markup, /Highest rating/);
  assert.doesNotMatch(markup, /leaderboard-period-controls/);
  const defaultLeagueCards = ["Highest rating", "Most improved", "Best win %", "Most wins"];
  for (const [displayField, hiddenCard] of Object.entries({ ratings: "Highest rating", rating_changes: "Most improved", win_percentage: "Best win %", records: "Most wins" })) {
    const gated = await render({ league: "Ladder" }, { [displayField]: false });
    const renderedTitles = [...gated.matchAll(/<h2[^>]*>([^<]+)<\/h2>/g)].map(match => match[1]);
    for (const title of defaultLeagueCards) {
      assert.equal(renderedTitles.includes(title), title !== hiddenCard, `League card ${title} still honors ${displayField}`);
    }
  }

  process.env.JUPR_API_BASE_URL = "https://api.test";
  let requestedUrl;
  global.fetch = async url => { requestedUrl = new URL(url); return { ok: true, json: async () => ({}) }; };
  const api = load("lib/api.ts");
  await api.getClubLeaderboard("tres-palapas", { season: "all", status: "all", offset: 50 });
  assert.equal(requestedUrl.searchParams.get("season"), "all");
  assert.equal(requestedUrl.searchParams.get("offset"), "50");
  await api.getClubLeaderboard("tres-palapas");
  assert.equal(requestedUrl.searchParams.has("season"), false, "Omitted season lets the API choose the published default");
  console.log("Overall leaderboard: 17 grouped choices, persisted card depth/minimums, generic metrics, independent Overall card visibility, preserved table/league display controls, card order, season validation and draft/publish isolation passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
