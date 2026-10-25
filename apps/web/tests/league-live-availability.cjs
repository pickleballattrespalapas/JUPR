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
const dndMock = {
  DragDropContext: ({ children, ...props }) => React.createElement("mock-dnd-context", props, children),
  Droppable: ({ children, ...props }) => React.createElement("mock-droppable", props, children({ innerRef() {}, droppableProps: {}, placeholder: null }, { isDraggingOver: false })),
  Draggable: ({ children, ...props }) => React.createElement("mock-draggable", props, children({ innerRef() {}, draggableProps: { style: {} }, dragHandleProps: {} }, { isDragging: false }))
};
const editorModule = load("app/admin/league-manager/live/LeagueCourtRosterEditor.tsx", { "@hello-pangea/dnd": dndMock });
const CourtRosterEditor = editorModule.LeagueCourtRosterEditor;
const gameEditsModule = load("app/admin/league-manager/live/leagueLiveGameEdits.ts");
const playerSelectMock = { __esModule: true, default: props => React.createElement("select", props) };
const gameEditorModule = load("app/admin/league-manager/live/LeagueLiveGameEditor.tsx", { "./leagueLiveGameEdits": gameEditsModule, "@/components/SearchablePlayerSelect": playerSelectMock });
const attendanceEditorModule = load("app/admin/league-manager/live/LeagueLiveAttendanceEditor.tsx", { "@/components/SearchablePlayerSelect": playerSelectMock });
const scoreHelpers = load("app/admin/league-manager/live/leagueRoundScores.ts");
const scoreReviewModule = load("app/admin/league-manager/live/LeagueRoundScoreReview.tsx", { "./leagueRoundScores": scoreHelpers });
const savedScoresModule = load("app/admin/league-manager/live/LeagueSavedRoundScores.tsx", { "./leagueRoundScores": scoreHelpers, "./LeagueRoundScoreReview": scoreReviewModule });
const Panel = load("app/admin/league-manager/live/LeagueLiveRoundPanel.tsx", {
  "./LeagueRoundScoreReview": scoreReviewModule,
  "./LeagueSavedRoundScores": savedScoresModule,
  "./LeagueLiveGameEditor": gameEditorModule,
  "./LeagueLiveAttendanceEditor": attendanceEditorModule,
  "./leagueLiveGameEdits": gameEditsModule,
  "./LeagueCourtRosterEditor": editorModule,
  "next/dynamic": { __esModule: true, default: factory => String(factory).includes("LeagueCourtRosterEditor") ? CourtRosterEditor : () => null },
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
  const alexCard = tree.root.findAllByType("button").find(node => node.props["aria-label"] === "Select Alex on Court 1, position 1");
  assert.ok(alexCard);
  assert.match(text(alexCard), /Rating\s*3\.00/, "The parent supplies the session roster rating to each court card on the displayed JUPR scale");
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

  // A one-game staff substitution and unfinished game serialize exact people,
  // preserve the other schedules, clear partial scores, and require reasons.
  await mount({ players: [...players, newPlayer] });
  await click("Continue to Players");
  await click("Build courts from 4 attendees");
  await click("Validate courts and generate preview");
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Create session and continue");
  await act(async () => { await confirm.props.onConfirm("CREATE LIVE SESSION"); });
  for (const input of tree.root.findAllByType("input").filter(node => /Team [12] score/.test(node.props["aria-label"] || ""))) {
    await act(async () => input.props.onChange({ target: { value: input.props["aria-label"].includes("Team 1") ? "11" : "7" } }));
  }
  let gameEditor = tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor)[1];
  await act(async () => gameEditor.findByType("details").props.onToggle({ currentTarget: { open: true } }));
  const staffSelect = gameEditor.findAllByType("select").find(node => node.props["aria-label"] === "Court 1 Match 2 Game 1 Team 1 player 1");
  await act(async () => staffSelect.props.onValueChange("5"));
  assert.equal(button("Review scores").props.disabled, true, "Substitution without a reason cannot be published");
  let reason = tree.root.findAllByType("input").find(node => node.props["aria-label"] === "Court 1 Match 2 Game 1 Substitution reason");
  await act(async () => reason.props.onChange({ target: { value: "Staff filled in for this game" } }));
  assert.equal(tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor)[0].props.edit, undefined, "Game one keeps its original lineup");
  assert.equal(tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor)[2].props.edit, undefined, "Game three keeps its original lineup");
  const unplayed = tree.root.findAllByType("input").find(node => node.props["aria-label"] === "Court 1 Match 3 Game 1 Not played or unfinished");
  await act(async () => unplayed.props.onChange({ target: { checked: true } }));
  assert.equal(button("Review scores").props.disabled, true, "An unfinished game requires a reason");
  const lastScores = tree.root.findAllByType("input").filter(node => /Match 3 Game 1 Team [12] score/.test(node.props["aria-label"] || ""));
  assert.ok(lastScores.every(node => node.props.value === "" && node.props.disabled), "Partial scores cannot become an official result");
  reason = tree.root.findAllByType("input").find(node => node.props["aria-label"] === "Court 1 Match 3 Game 1 Unplayed reason");
  await act(async () => reason.props.onChange({ target: { value: "Stopped because of heat" } }));
  const draft = JSON.parse(storage.get("jupr-league-live-round-draft:club-a:session-a:1"));
  assert.deepEqual(draft.gameEdits["court-1-1:game:1"].playerIds, [5, 3, 2, 4]);
  assert.equal(draft.gameEdits["court-1-2:game:1"].notPlayed, true);
  await act(async () => tree.root.findAllByType("button").find(node => /Step 1Setup/.test(text(node))).props.onClick());
  await click("Retry selected session");
  assert.equal(tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor).length, 3, text(tree.root));
  assert.deepEqual(tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor)[1].props.edit.playerIds, [5, 3, 2, 4], "Reload restores the one-game substitution");
  assert.equal(tree.root.findAllByType(gameEditorModule.LeagueLiveGameEditor)[2].props.edit.notPlayed, true, "Reload restores explicit unfinished games");
  await click("Review scores");
  assert.match(rendered(), /Taylor/);
  assert.match(rendered(), /Stopped because of heat/);
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Publish reviewed scores");
  await act(async () => { await confirm.props.onConfirm("SUBMIT LEAGUE ROUND"); });
  const flexSubmit = requests.find(request => request.path.endsWith("/rounds/1/submit")).body;
  assert.equal(flexSubmit.expected_match_count, 3);
  assert.equal(flexSubmit.matches.length, 2);
  assert.equal(flexSubmit.matches[0].t1_p1, 1);
  assert.equal(flexSubmit.matches[1].t1_p1, 5);
  assert.deepEqual(flexSubmit.matches[1].scheduled_player_ids, [1, 3, 2, 4]);
  assert.equal(flexSubmit.matches[1].substitution_reason, "Staff filled in for this game");
  assert.deepEqual(flexSubmit.unplayed_games, [{ series_key: "court-1-2", game_number: 1, court: 1, reason: "Stopped because of heat" }]);
  const out = tree.root.findAllByType("input").find(node => node.props["aria-label"] === "Playing next round: Sam");
  await act(async () => out.props.onChange({ target: { checked: false } }));
  let attendance = tree.root.findByType(attendanceEditorModule.LeagueLiveAttendanceEditor);
  assert.deepEqual(attendance.props.activePlayerIds, [1, 2, 3]);
  const lateArrival = attendance.findAllByType("select").find(node => node.props["aria-label"] === "Add next-round player");
  await act(async () => lateArrival.props.onValueChange("5"));
  await click("Add player");
  await click("Load court board");
  const attendanceRequest = requests.filter(request => request.path.endsWith("/rounds/1/plan")).at(-1).body;
  assert.deepEqual(attendanceRequest.roster_change.active_player_ids, [1, 2, 3, 5]);
  assert.deepEqual(attendanceRequest.roster_change.players, [{ player_id: 5, player_name: "Taylor", rating: 1300 }]);
  assert.equal(attendanceRequest.roster_change.action, "attendance");
  assert.equal(tree.root.findByType(attendanceEditorModule.LeagueLiveAttendanceEditor).props.roster.length, 4, "Sitting out must not delete historical roster players");
  assert.ok(storage.has("jupr-league-live-attendance-draft:club-a:session-a:1"));
  await act(async () => tree.root.findAllByType("button").find(node => /Step 1Setup/.test(text(node))).props.onClick());
  await click("Retry selected session");
  assert.deepEqual(tree.root.findByType(attendanceEditorModule.LeagueLiveAttendanceEditor).props.activePlayerIds, [1, 2, 3, 5], "Reload restores pending attendance after scores are official");
  assert.match(rendered(), /Recovered next-round attendance/);
  await act(async () => tree.unmount());
  storage.clear();

  // Rehearse the actual 36 -> 32 -> 28 attendance controls, with two
  // first-round sit-outs returning for the third round.
  const dayPlayers = Array.from({ length: 36 }, (_, index) => ({ id: index + 1, name: `Player ${index + 1}` }));
  let selectedDayPlayers;
  function AttendanceHarness() {
    const [active, setActive] = React.useState(dayPlayers.map(player => player.id));
    const [sizes, setSizes] = React.useState("");
    selectedDayPlayers = active;
    return React.createElement(attendanceEditorModule.LeagueLiveAttendanceEditor, { roster: dayPlayers, availablePlayers: dayPlayers, activePlayerIds: active, courtSizes: sizes, disabled: false, onActivePlayersChange: setActive, onCourtSizesChange: setSizes });
  }
  await act(async () => { tree = create(React.createElement(AttendanceHarness)); });
  async function attend(id, checked) {
    const input = tree.root.findAllByType("input").find(node => node.props["aria-label"] === `Playing next round: Player ${id}`);
    assert.ok(input, `Returning player ${id} must remain selectable`);
    await act(async () => input.props.onChange({ target: { checked } }));
  }
  for (const id of [33, 34, 35, 36]) await attend(id, false);
  assert.equal(selectedDayPlayers.length, 32);
  assert.match(rendered(), /32/);
  for (const id of [27, 28, 29, 30, 31, 32]) await attend(id, false);
  for (const id of [33, 34]) await attend(id, true);
  assert.equal(selectedDayPlayers.length, 28);
  assert.ok(selectedDayPlayers.includes(33) && selectedDayPlayers.includes(34));
  assert.ok(!selectedDayPlayers.includes(27) && !selectedDayPlayers.includes(36));
  assert.equal(tree.root.findAllByType("input").filter(node => node.props.type === "checkbox").length, 36, "Every historical player remains available to return");
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

  await mount({ players: [...players, { id: 677, name: "Lily Lee", rating: 1460, rating_jupr: 3.65 }] });
  await click("Continue to Players");
  await click("Build courts from 4 attendees");
  await clickAria("Replace Casey");
  const replacementSelect = tree.root.findAllByType("select").find(node => node.props["aria-label"] === "Replacement for Casey");
  assert.match(text(replacementSelect), /Lily Lee.*3.65/);
  await act(async () => replacementSelect.props.onChange({ target: { value: "677" } }));
  assert.deepEqual(tree.root.findByType(CourtRosterEditor).props.courts[0].playerNames.split("\n"), ["Alex", "Lily Lee", "Jordan", "Sam"]);
  assert.deepEqual(tree.root.findByType(CourtRosterEditor).props.playerNames, ["Alex", "Lily Lee", "Jordan", "Sam"]);
  assert.ok(tree.root.findAllByType("button").find(node => node.props["aria-label"] === "Edit rating for Lily Lee"));
  await click("Validate courts and generate preview");
  confirm = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Create session and continue");
  await act(async () => { await confirm.props.onConfirm("CREATE LIVE SESSION"); });
  const replacedWrite = requests.find(request => request.path === `${base}/live-sessions` && request.method === "POST").body;
  assert.deepEqual(replacedWrite.courts[0].player_names, ["Alex", "Lily Lee", "Jordan", "Sam"]);
  assert.deepEqual(replacedWrite.roster.map(row => row.player_id), [1, 677, 3, 4]);
  assert.equal(replacedWrite.roster[1].rating, 1460);
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
  console.log("League Live availability: setup, player creation/recovery, exact court edits and persistence, guarded draft-score clearing, one-game staff substitutions, explicit unfinished games, 36-to-32-to-28 attendance with returning players, and reviewed scoring work independently of Match Uploader; official rounds and Live capabilities remain protected.");
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => Object.assign(global, savedGlobals));
