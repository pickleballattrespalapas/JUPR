const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(file, overrides = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const mod = { exports: {} };
  new Function("require", "module", "exports", code)(name => overrides[name] || require(name), mod, mod.exports);
  return mod.exports;
}

const savedGlobals = { fetch: global.fetch, sessionStorage: global.sessionStorage, addEventListener: global.addEventListener, removeEventListener: global.removeEventListener };
const storage = new Map();
if (!global.crypto) global.crypto = require("node:crypto").webcrypto;
global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
global.addEventListener = () => {};
global.removeEventListener = () => {};
const ConfirmAction = () => null;
const editorModule = load("app/admin/league-manager/live/LeagueCourtRosterEditor.tsx");
const CourtRosterEditor = editorModule.LeagueCourtRosterEditor;
const Panel = load("app/admin/league-manager/live/LeagueLiveRoundPanel.tsx", {
  "./LeagueCourtRosterEditor": editorModule,
  "next/dynamic": { __esModule: true, default: () => () => null },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: props => React.createElement("select", props) },
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": load("components/interaction/types.ts"),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-token", session: {} }), adminSessionLabel: () => "Signed in" },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts")
}).default;

const leagueName = "Sunday Ladder";
const base = "/admin/clubs/club-a/league-manager";
const players = ["Alex", "Casey", "Jordan", "Sam"].map((name, index) => ({ id: index + 1, name, rating: 1200, rating_jupr: 3 }));
const roster = players.map((player, index) => ({ player_id: player.id, player_name: player.name, rating: player.rating, status: "active", court_number: 1, slot: index + 1 }));
const courts = [{ round_number: 1, court_number: 1, format_type: "4-Player", player_names: players.map(player => player.name), players_json: roster }];
const detail = { ok: true, league: { league_name: leagueName, league_type: "Individual", match_format: "doubles", rules_config: { participation_mode: "set" } }, roster: roster.map(row => ({ ...row, in_league: true })), standings: [], schedule_preview: [] };
const suggestion = { ok: true, roster, active_roster: roster, bench: [], bench_player_ids: [], default_bench_player_ids: [], court_sizes: [4], courts, fingerprint: "roster-v1" };
const pairings = [[1, 2, 3, 4], [1, 3, 2, 4], [1, 4, 2, 3]];
const matches = pairings.map(([a, b, c, d], index) => ({ row_id: `court-1-${index}`, label: `Match ${index + 1}`, court: 1, t1: [players[a - 1], players[b - 1]], t2: [players[c - 1], players[d - 1]], t1_p1: a, t1_p2: b, t2_p1: c, t2_p2: d }));
const preview = { ok: true, courts: [{ court: 1, format_type: "4-Player", matches }], match_count: 3, missing_players: [] };
const liveStatus = { enabled: true, submit_enabled: true, player_create_enabled: true, movement_authority: "python_fastapi", round_robin_format_options: ["4-Player", "5-Player"], warnings: [] };
const props = { apiBase: "https://api.example.invalid", clubId: "club-a", selectedLeagueName: leagueName, leagueStatus: { enabled: true, warnings: [] }, liveDomainStatus: liveStatus, players };
const session = { id: "session-a", league_name: leagueName, week_tag: "Week 1", status: "active", total_rounds: 5, current_round: 1, roster_json: roster, updated_at: "2026-10-16T19:00:00Z" };
const newPlayer = { id: 5, name: "Taylor", rating: 1300 };
let tree, requests = [], published = false;
const rendered = () => JSON.stringify(tree.toJSON());
const text = node => node.children.map(child => typeof child === "string" ? child : text(child)).join("");
const button = label => tree.root.findAllByType("button").find(node => text(node) === label);
async function click(label) {
  const found = button(label);
  assert.ok(found, `Expected button ${label}`);
  assert.equal(found.props.disabled, false, `${label} must be enabled`);
  await act(async () => { await found.props.onClick(); });
}
async function clickAria(label) {
  const found = tree.root.findAllByType("button").find(node => node.props["aria-label"] === label);
  assert.ok(found, `Expected button with aria-label ${label}`);
  assert.equal(found.props.disabled, false, `${label} must be enabled`);
  await act(async () => { await found.props.onClick(); });
}
async function mount(overrides = {}) {
  requests = []; published = false;
  await act(async () => { tree = create(React.createElement(Panel, { ...props, ...overrides })); });
}
const response = payload => ({ ok: true, status: 200, json: async () => structuredClone(payload) });
global.fetch = async (url, options = {}) => {
  const pathname = new URL(url).pathname;
  const body = options.body ? JSON.parse(options.body) : null;
  requests.push({ path: pathname, method: options.method || "GET", body });
  assert.equal(options.headers.get("Authorization"), "Bearer test-token");
  assert.doesNotMatch(pathname, /match-uploader/, "League Live must never depend on Match Uploader routes");
  if (pathname === `${base}/leagues`) return response({ leagues: [detail.league] });
  if (pathname === `${base}/leagues/${encodeURIComponent(leagueName)}`) return response(detail);
  if (pathname === `${base}/live/roster-suggestion`) return response(suggestion);
  if (pathname === `${base}/live/round-robin/preview`) return response(preview);
  if (pathname === `${base}/live/players`) return response({ ok: true, players: [newPlayer], accepted_count: 1 });
  if (pathname === `${base}/live/player-operations/retained-key`) return response({ ok: true, status: "recovery_required", recovery_required: true });
  if (pathname === `${base}/live/player-operations/retained-key/reconcile`) return response({ ok: true, status: "completed", players: [newPlayer], accepted_count: 1 });
  if (pathname === `${base}/live-sessions`) return response(options.method === "POST" ? { ok: true, session, courts } : { ok: true, sessions: [], count: 0 });
  if (pathname === `${base}/live-sessions/session-a/snapshot`) return response({ ok: true, session, courts });
  if (pathname === `${base}/live-sessions/session-a/rounds/1/submit`) {
    published = true;
    return response({ ok: true, session, published_match_ids: [101, 102, 103] });
  }
  if (pathname === `${base}/live-sessions/session-a`) return response({ ok: true, session, courts, rounds: published ? [{ round_number: 1, status: "submitted", match_date: "2026-10-16", matches_json: requests.find(request => request.path.endsWith("/submit")).body.matches }] : [] });
  if (pathname === `${base}/live-sessions/session-a/rounds/1/plan`) return response({ ok: true, operation_key: "movement-v1", next_courts: courts, bench: [], movement: { rows: [], next_round: 2, applied: false }, next_roster: roster });
  throw new Error(`Unexpected request ${options.method || "GET"} ${url}`);
};

(async () => {
  // Exercise the actual server route without a functioning Match Uploader service.
  let uploaderReads = 0;
  const Page = load("app/admin/league-manager/live/page.tsx", {
    "../LeagueManagerLoadError": { __esModule: true, default: () => null },
    "../LeagueManagerNav": { __esModule: true, default: () => null },
    "../SelectedLeaguePanelScope": { __esModule: true, default: ({ children }) => children },
    "./LeagueLiveRoundPanel": { __esModule: true, default: Panel },
    "next/navigation": { redirect: () => { throw new Error("Unexpected redirect"); } },
    "@/lib/adminWorkspaceServer": { requireAdminWorkspace: () => ({ clubId: "club-a", clubSlug: "club-a" }) },
    "@/lib/leagueRouteContext": { readLeagueRouteContext: () => ({ leagueId: "league-a", leagueName }) },
    "@/lib/api": { getClubPlayerOptions: async () => ({ data: { players } }) },
    "@/lib/adminLeagueManagerApi": { getAdminLeagueManagerApiBaseUrl: () => props.apiBase, getAdminLeagueManagerStatus: async () => ({ data: props.leagueStatus }), getAdminLeagueLiveStatus: async () => ({ data: liveStatus }) },
    "@/lib/adminMatchUploaderApi": { getAdminMatchUploaderStatus: async () => { uploaderReads++; return { data: { enabled: false } }; } }
  }).default;
  await act(async () => { tree = create(await Page({})); });
  assert.equal(uploaderReads, 0);
  assert.match(rendered(), /1. Setup/);
  assert.equal(button("Continue to Players").props.disabled, false);
  await click("Continue to Players");
  await click("Build courts from 4 attendees");
  await click("Validate courts and generate preview");
  assert.match(rendered(), /Match preview/);
  assert.deepEqual(requests.find(request => request.path.endsWith("/round-robin/preview")).body.courts[0].player_names, players.map(player => player.name));

  // Actual editor interactions invalidate the reviewed lineup and feed the exact
  // updated order into both the regenerated preview and session creation.
  await clickAria("Move Casey up on Court 1");
  assert.doesNotMatch(rendered(), /Match preview ·/);
  assert.equal(tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Create session and continue").props.disabled, true);
  await clickAria("Select Alex on Court 1, position 2");
  await clickAria("Swap Alex with Sam on Court 1");
  const createdOrder = ["Casey", "Sam", "Jordan", "Alex"];
  await click("Validate courts and generate preview");
  assert.deepEqual(requests.filter(request => request.path.endsWith("/round-robin/preview")).at(-1).body.courts[0].player_names, createdOrder);
  let confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Create session and continue");
  assert.equal(confirm.props.disabled, false);
  await act(async () => { await confirm.props.onConfirm("CREATE LIVE SESSION"); });
  assert.deepEqual(requests.find(request => request.path === `${base}/live-sessions` && request.method === "POST").body.courts[0].player_names, createdOrder);
  assert.match(rendered(), /4. Score Entry with Review/);

  // Returning to court setup cannot silently discard any entered draft score.
  const firstScore = tree.root.findAllByType("input").find(node => /Team 1 score/.test(node.props["aria-label"] || ""));
  await act(async () => firstScore.props.onChange({ target: { value: "11" } }));
  await click("Back to Courts and Preview");
  let editor = tree.root.findByType(CourtRosterEditor);
  assert.equal(editor.props.disabled, true);
  assert.ok(editor.findAllByType("button").every(node => node.props.disabled), "Every court mutation must remain disabled while draft scores exist");
  assert.equal(button("Validate courts and generate preview").props.disabled, true);
  const protectedCourts = structuredClone(editor.props.courts);
  await act(async () => editor.props.onChange([{ ...protectedCourts[0], playerNames: "Sam\nCasey\nJordan\nAlex" }]));
  assert.deepEqual(tree.root.findByType(CourtRosterEditor).props.courts, protectedCourts, "The parent guard rejects editing even when a stale callback fires");
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Clear draft scores to edit courts");
  assert.equal(confirm.props.disabled, false);
  await act(async () => { await confirm.props.onConfirm("CLEAR DRAFT SCORES"); });
  editor = tree.root.findByType(CourtRosterEditor);
  assert.equal(editor.props.disabled, false);
  assert.doesNotMatch(rendered(), /Match preview ·/);

  // The mock returns the original saved courts, so edit that returned arrangement
  // and verify the snapshot request independently of the mocked response.
  await clickAria("Move Sam up on Court 1");
  const savedOrder = ["Alex", "Casey", "Sam", "Jordan"];
  await click("Validate courts and generate preview");
  assert.deepEqual(requests.filter(request => request.path.endsWith("/round-robin/preview")).at(-1).body.courts[0].player_names, savedOrder);
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Save preview and continue");
  assert.equal(confirm.props.disabled, false);
  await act(async () => { await confirm.props.onConfirm("SAVE SESSION"); });
  assert.deepEqual(requests.find(request => request.path.endsWith("/snapshot")).body.courts[0].player_names, savedOrder);
  assert.ok(tree.root.findAllByType("input").filter(node => /Team [12] score/.test(node.props["aria-label"] || "")).every(node => node.props.value === ""), "Regenerated matchups must start without stale scores");
  for (const input of tree.root.findAllByType("input").filter(node => /Team [12] score/.test(node.props["aria-label"] || ""))) {
    await act(async () => input.props.onChange({ target: { value: input.props["aria-label"].includes("Team 1") ? "11" : "7" } }));
  }
  await click("Review scores");
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Publish reviewed scores");
  assert.equal(confirm.props.disabled, false);
  await act(async () => { await confirm.props.onConfirm("SUBMIT LEAGUE ROUND"); });
  assert.match(rendered(), /The round scores are official/);
  const submit = requests.find(request => request.path.endsWith("/rounds/1/submit"));
  assert.equal(submit.body.expected_match_count, 3);
  assert.deepEqual(submit.body.matches.map(match => [match.score_t1, match.score_t2]), [[11, 7], [11, 7], [11, 7]]);
  assert.equal(submit.body.confirmation_text, "SUBMIT LEAGUE ROUND");
  const courtsNavigation = tree.root.findAllByType("button").find(node => /Step 3Courts and Preview/.test(text(node)));
  assert.ok(courtsNavigation);
  assert.equal(courtsNavigation.props.disabled, true, "Official rounds must not expose court editing");
  await act(async () => courtsNavigation.props.onClick());
  assert.equal(tree.root.findAllByType(CourtRosterEditor).length, 0, "Navigation guards must keep the editor closed after publication");
  await act(async () => tree.unmount());

  // Player batch creation uses the Live capability and guarded Live route.
  await mount();
  await click("Continue to Players");
  await act(async () => tree.root.findByType("textarea").props.onChange({ target: { value: "Taylor" } }));
  await click("Resolve and add pasted players");
  await act(async () => tree.root.findAllByType("input").find(node => node.props.placeholder === "Required").props.onChange({ target: { value: "3.5" } }));
  await click("Create missing players");
  // crypto.subtle digest is asynchronous outside React's microtask flush.
  for (let attempt = 0; !requests.some(request => request.path.endsWith("/live/players")) && attempt < 20; attempt++) {
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 5)); });
  }
  const playerWrite = requests.find(request => request.path.endsWith("/live/players"));
  assert.ok(playerWrite);
  assert.deepEqual(playerWrite.body.players, [{ name: "Taylor", starting_jupr: 3.5 }]);
  assert.equal(playerWrite.body.confirmation_text, "CREATE PLAYERS");
  assert.match(playerWrite.body.idempotency_key, /^league-live-players:/);
  assert.match(rendered(), /Session player order \(/);
  assert.match(rendered(), /Taylor/);
  await act(async () => tree.unmount());

  // An interrupted player batch is inspected and reconciled with its original key.
  storage.set("jupr-league-live-player-create-recovery:club-a", JSON.stringify({ version: 1, operationKey: "retained-key", status: "uncertain", message: "Connection interrupted", leagueName }));
  await mount();
  await click("Continue to Players");
  await click("Check and reconcile exact player operation");
  assert.deepEqual(requests.filter(request => request.path.includes("player-operations")).map(request => [request.method, request.path]), [
    ["GET", `${base}/live/player-operations/retained-key`], ["POST", `${base}/live/player-operations/retained-key/reconcile`]
  ]);
  assert.equal(storage.has("jupr-league-live-player-create-recovery:club-a"), false);
  await act(async () => tree.unmount());

  await mount({ liveDomainStatus: { ...liveStatus, enabled: false } });
  assert.match(rendered(), /Live league operations are disabled on this site/);
  assert.doesNotMatch(rendered(), /this build|existing League Manager scoring/);
  assert.equal(requests.length, 0);
  await act(async () => tree.unmount());

  await mount({ liveDomainStatus: { ...liveStatus, submit_enabled: false, player_create_enabled: false } });
  assert.match(rendered(), /Disabled on this site/);
  await click("Continue to Players");
  await act(async () => tree.root.findByType("textarea").props.onChange({ target: { value: "Taylor" } }));
  await click("Resolve and add pasted players");
  await act(async () => tree.root.findAllByType("input").find(node => node.props.placeholder === "Required").props.onChange({ target: { value: "3.5" } }));
  assert.equal(button("Create missing players").props.disabled, true);
  assert.equal(requests.filter(request => request.path.endsWith("/live/players")).length, 0);
  await act(async () => tree.unmount());
  console.log("League Live availability: setup, player creation/recovery, exact court edits and persistence, guarded draft-score clearing, and reviewed scoring work independently of Match Uploader; official rounds and Live capabilities remain protected.");
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => Object.assign(global, savedGlobals));
