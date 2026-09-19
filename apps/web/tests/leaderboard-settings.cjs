const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const { renderToStaticMarkup } = require("react-dom/server");
const h = React.createElement;
const initial = { schema_version: 1, name: "Tres Palapas", description: "", location: "", visitor_info: "", logo_url: "", accent: "#1d4ed8", visibility: "listed", display: {}, pages: [{ slug: "home", title: "Home", in_navigation: true, blocks: [] }] };
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
    if (options.method === "PUT") site.draft = structuredClone(body.document);
    else if (url.endsWith("/publish")) site.published = structuredClone(site.draft);
    site.revision++;
  }
  return { ok: true, status: 200, json: async () => structuredClone(site) };
};

(async () => {
  const { leaderboardSettings } = load("lib/clubSite.ts");
  assert.deepEqual(leaderboardSettings().seasons, [], "No season is created automatically for existing clubs");
  const Website = load("app/admin/website/page.tsx").default;
  let tree;
  await act(async () => { tree = create(h(Website)); });
  const button = label => tree.root.findAllByType("button").find(item => text(item) === label);
  const field = label => tree.root.findByProps({ "aria-label": label });
  await act(async () => button("Overall leaderboard").props.onClick());
  await act(async () => field("Remove Highest rating card").props.onClick());
  await act(async () => button("Add Most games played").props.onClick());
  await act(async () => field("Move Most games played up").props.onClick());
  await act(async () => field("Move Most games played up").props.onClick());
  await act(async () => tree.root.findByProps({ type: "checkbox" }).props.onChange({ target: { checked: false } }));
  await act(async () => tree.root.findByProps({ type: "number" }).props.onChange({ target: { value: "10" } }));
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
  assert.deepEqual(site.draft.leaderboard.cards, ["most_improved", "most_matches", "best_win_pct", "most_wins"]);
  assert.equal(site.draft.leaderboard.default_season_id, seasonId);
  assert.equal(site.draft.leaderboard.seasons[0].end_date, null);
  assert.equal(site.draft.leaderboard.min_games, 10);
  assert.equal(site.published.leaderboard, undefined, "Saving a draft leaves public settings untouched");
  await act(async () => button("Publish website").props.onClick());
  assert.deepEqual(site.published.leaderboard, site.draft.leaderboard);
  await act(async () => field("Remove 2026–27 from leaderboard").props.onClick());
  assert.equal(tree.root.findByType("select").props.value, "all", "Removing the default season restores All time");
  await act(async () => tree.unmount());

  const publicSettings = structuredClone(site.published.leaderboard);
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
  assert.deepEqual(titles, ["Stu", "Most improved", "Most games played", "Best win %", "Most wins"]);
  assert.doesNotMatch(markup, /data-testid="leaderboard-summary"/);
  assert.match(markup, /Sep 15, 2026 onward/);
  assert.match(markup, /Ratings and ranks show current standing/);
  assert.match(markup, /at least 10 recorded games/);
  const searchForm = markup.match(/<form[^>]*data-testid="leaderboard-search-form"[\s\S]*?<\/form>/)[0];
  assert.ok(searchForm.includes(`name="season" value="${seasonId}"`), "Searching preserves the selected season");
  for (const match of markup.matchAll(/href="([^"]+)"[^>]*>(Next|Win %|See all|Share this player|view summary)<\/a>/g)) {
    assert.ok(match[1].includes(`season=${seasonId}`), `${match[2]} preserves the selected season`);
  }
  markup = await render({ season: seasonId }, { match_counts: false, records: false });
  assert.doesNotMatch(markup, /<h2[^>]*>Most games played|<h2[^>]*>Most wins/);
  assert.match(markup, /<h2[^>]*>Most improved/);
  data.period = { id: null, name: "All time", start_date: null, end_date: null, timezone: "UTC" };
  markup = await render({ season: "all" });
  assert.equal(requests.at(-1).options.season, "all");
  assert.match(markup, /Showing all-time statistics/);
  assert.match(markup, /name="season" value="all"/);
  data.leaderboard_settings.cards = [];
  assert.doesNotMatch(await render(), /leaderboard-highlight-card/, "An empty card selection is respected");
  data.selected_scope = "Ladder";
  markup = await render({ league: "Ladder", season: seasonId });
  assert.match(markup, /leaderboard-summary/);
  assert.match(markup, /Highest rating/);
  assert.doesNotMatch(markup, /leaderboard-period-controls/);

  process.env.JUPR_API_BASE_URL = "https://api.test";
  let requestedUrl;
  global.fetch = async url => { requestedUrl = new URL(url); return { ok: true, json: async () => ({}) }; };
  const api = load("lib/api.ts");
  await api.getClubLeaderboard("tres-palapas", { season: "all", status: "all", offset: 50 });
  assert.equal(requestedUrl.searchParams.get("season"), "all");
  assert.equal(requestedUrl.searchParams.get("offset"), "50");
  await api.getClubLeaderboard("tres-palapas");
  assert.equal(requestedUrl.searchParams.has("season"), false, "Omitted season lets the API choose the published default");
  console.log("Overall leaderboard: card order/visibility, season validation, draft/publish isolation, default removal, period controls and link/filter persistence passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
