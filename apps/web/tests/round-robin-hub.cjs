const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const h = React.createElement, root = path.resolve(__dirname, ".."), cache = new Map();
global.IS_REACT_ACT_ENVIRONMENT = true;
const stored = new Map();
global.sessionStorage = { getItem: key => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
const replacedUrls = [];
function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(listener); },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    dispatchEvent(event) { for (const listener of [...(listeners.get(event.type) || [])]) listener(event); }
  };
}
let intervalId = 0;
const intervals = new Map();
global.window = { ...eventTarget(), history: { replaceState(_state, _title, url) { replacedUrls.push(url); } },
  setInterval(callback) { const id = ++intervalId; intervals.set(id, callback); return id; }, clearInterval(id) { intervals.delete(id); } };
global.document = { ...eventTarget(), visibilityState: "visible" };
const tickIntervals = () => { for (const callback of [...intervals.values()]) callback(); };
let standingsMounts = 0;
function Standings(props) {
  const [instance] = React.useState(() => ++standingsMounts);
  return h("div", { "data-standings": props.sessionKey, "data-instance": instance, "data-refresh-key": props.refreshKey, "data-embedded": props.embedded, "data-club": props.clubId },
    "Full session standings", h("button", { onClick: () => props.onNavigate({ view: "round", roundNumber: 3 }) }, "Current round"),
    h("button", { onClick: props.onSessionChanged }, "Refresh after sharing"));
}
const stubs = {
  "next/link": ({ children, ...props }) => h("a", props, children),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-staff" }) },
  "@/components/interaction": { InteractionDialog: ({ title, children, actions }) => h("section", { "data-dialog": title }, children, actions) },
  "@/app/admin/play-generators/GeneratorRoundRunner": { __esModule: true, default: props => h("div", { "data-runner": props.sessionKey, "data-round": props.roundNumber }, h("button", { onClick: () => props.onNavigate({ view: "standings" }) }, "Embedded standings")) },
  "@/app/admin/play-generators/GeneratorStandings": { __esModule: true, default: Standings }
};
function load(name, parent = root) {
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (name.endsWith(".module.css")) return new Proxy({}, { get: (_, key) => key === "__esModule" ? false : String(key) });
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts"].find(file => fs.existsSync(file) && fs.statSync(file).isFile());
  if (!file) throw new Error(`Missing ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const source = ts.transpileModule(fs.readFileSync(file, "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function("require", "module", "exports", source)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports); return module.exports;
}
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const buttons = (tree, label) => tree.root.findAllByType("button").filter(node => text(node) === label);
const clone = value => JSON.parse(JSON.stringify(value));
const people = (prefix, from) => Array.from({ length: 4 }, (_, index) => ({ id: `${prefix}${from + index}`, identity: `player:${prefix}${from + index}`, name: `${prefix.toUpperCase()} Player ${from + index}` }));
function match(session, id, start, courts, wait) { const players = people(session, start); return { session_key: session, match_id: id, round_number: 1, side_a: players.slice(0, 2), side_b: players.slice(2), label: id, wait_seconds: wait, allowed_court_ids: courts, blockers: [], status: "waiting" }; }
const a1 = match("a", "a-1", 1, ["c1"], 0), a2 = match("a", "a-2", 5, ["c1"], 720), b1 = match("b", "b-1", 1, ["c2"], 600);
let detail = {
  ok: true, hub: { id: "event-1", title: "Friday Night", status: "active", version: 1 },
  workspace: {
    courts: [{ id: "c1", label: "Stadium", dedicated_session_key: "a", open: true, current: a1, next: null }, { id: "c2", label: "Court 4", dedicated_session_key: null, open: true, current: null, next: null }],
    groups: ["a", "b"].map(key => ({ session_key: key, title: key === "a" ? "3.5 Round Robin" : "Moneyball", current_round_number: 1, round_status: "active", completed_matches: 0, total_matches: key === "a" ? 2 : 1, can_advance: false, event_format: key === "a" ? "standard" : "moneyball_expectations", scoring_mode: "scored" })),
    eligible_queue: [a2, b1], blocked_queue: [], reserved_queue: [],
    players: [{ identity: "player:b1", name: "B Player 1", paused: false, wait_seconds: 600, session_keys: ["b"], participant_ids: { b: "b1" } }]
  }
};
const requests = [], requestUrls = [], receipts = new Map();
let failNext = false, conflictNext = false, loseAddResponse = false, loseFinishResponse = false, listReads = 0;
let deferDetailAfter = -1, releaseDetail;
let availableSessions = [{ session_key: "new", title: "New RR", status: "active", rating_mode: "unrated" }];
let listedHubs = null;
global.fetch = async (url, options = {}) => {
  assert.equal(options.headers.Authorization, "Bearer test-staff");
  if (!options.body) {
    if (url.endsWith("/round-robin-hubs")) { listReads += 1; return { ok: true, json: async () => ({ ok: true, hubs: listedHubs ?? [detail.hub], sessions: clone(availableSessions) }) }; }
    if (deferDetailAfter === 0) {
      deferDetailAfter = -1;
      const snapshot = clone(detail);
      return new Promise(resolve => { releaseDetail = () => resolve({ ok: true, json: async () => snapshot }); });
    }
    if (deferDetailAfter > 0) deferDetailAfter -= 1;
    return { ok: true, json: async () => clone(detail) };
  }
  const body = JSON.parse(options.body); requests.push(body); requestUrls.push(url);
  if (receipts.has(body.idempotency_key)) {
    assert.deepEqual(body, receipts.get(body.idempotency_key), "A replay must retain the exact request");
    return { ok: true, json: async () => clone(detail) };
  }
  if (failNext) { failNext = false; throw new Error("Network interrupted"); }
  if (conflictNext) { conflictNext = false; detail.hub.version += 1; return { ok: false, status: 409, json: async () => ({ detail: "The court board changed. Refresh before continuing." }) }; }
  assert.equal(body.expected_version, detail.hub.version);
  if (body.action === "reserve_game_for_court") {
    assert.equal(body.payload.court_id, "c1"); assert.equal(body.payload.session_key, "a"); assert.equal(body.payload.match_id, "a-2");
    detail.workspace.courts[0].next = { ...a2, reserved_court_id: "c1", status: "reserved" };
    detail.workspace.reserved_queue = [detail.workspace.courts[0].next];
    detail.workspace.eligible_queue = [b1];
  } else if (body.action === "score_and_release") {
    assert.deepEqual(body.payload, { court_id: "c1", session_key: "a", match_id: "a-1", score_a: 11, score_b: 7 });
    detail.workspace.courts[0].current = detail.workspace.courts[0].next;
    detail.workspace.courts[0].next = null; detail.workspace.reserved_queue = [];
    detail.workspace.groups[0].completed_matches = 1;
  } else if (body.action === "pause_player") {
    assert.deepEqual(body.payload, { identity: "player:b1" });
    detail.workspace.players[0].paused = true;
  } else if (body.action === "resume_player") detail.workspace.players[0].paused = false;
  else if (body.action === "finish_group") {
    const group = detail.workspace.groups.find(group => group.session_key === body.payload.session_key);
    assert.ok(group && group.can_finish);
    group.finish_requested = true; group.finish_after_round = group.finish_boundary_round; group.can_finish = false;
    if (!group.finish_pending_matches) group.status = "completed";
    if (detail.workspace.groups.every(group => group.status === "completed")) detail.hub.status = "completed";
  }
  else if (body.action === "add_group") {
    const key = body.payload.session_key;
    assert.ok(!detail.workspace.groups.some(group => group.session_key === key), "Do not add a group twice");
    detail.workspace.groups.push({ session_key: key, title: "Created round robin", current_round_number: 1, round_status: "active", completed_matches: 0, total_matches: 1, can_advance: false, event_format: "standard", scoring_mode: "scored" });
    detail.workspace.eligible_queue.push(match(key, `${key}-1`, 1, ["c2"], 0));
  }
  else if (!["assign_next_court", "auto_fill_courts"].includes(body.action)) throw new Error(`Unexpected action ${body.action}`);
  detail.hub.version += 1;
  receipts.set(body.idempotency_key, clone(body));
  if (body.action === "add_group" && loseAddResponse) { loseAddResponse = false; throw new Error("Response lost after attachment"); }
  if (body.action === "finish_group" && loseFinishResponse) { loseFinishResponse = false; throw new Error("Response lost after finish request"); }
  return { ok: true, json: async () => clone(detail) };
};

(async () => {
  const Panel = load("@/app/admin/round-robin-hub/RoundRobinHubPanel").default;
  let tree;
  await act(async () => { tree = create(h(Panel, { apiBase: "http://test.local", clubId: "club", initialHubId: "event-1" })); });
  assert.ok(text(tree.toJSON()).includes("Friday Night"));
  assert.equal(buttons(tree, "Next open court")[0].props.disabled, true, "Dedicated group cannot spill into an open shared court");
  assert.equal(buttons(tree, "Next open court")[1].props.disabled, false);
  await act(async () => buttons(tree, "Choose court")[0].props.onClick());
  const chooser = tree.root.findByProps({ "data-dialog": "Choose court" });
  assert.deepEqual(chooser.findAllByType("option").map(option => option.props.value), ["", "c1"], "Dedicated group only sees its dedicated court");
  await act(async () => button(tree, "Reserve next").props.onClick());
  assert.ok(text(tree.toJSON()).includes("Next on this court"));

  await act(async () => button(tree, "Enter score").props.onClick());
  await act(async () => tree.root.findByProps({ "aria-label": "Side A score" }).props.onChange({ target: { value: "11" } }));
  await act(async () => tree.root.findByProps({ "aria-label": "Side B score" }).props.onChange({ target: { value: "7" } }));
  await act(async () => tree.root.findByProps({ id: "hub-score-form" }).props.onSubmit({ preventDefault() {} }));
  assert.ok(text(tree.toJSON()).includes("Score saved and court released."));
  await act(async () => button(tree, "Round robins").props.onClick());
  assert.ok(text(tree.toJSON()).includes("1 of 2 matches complete"));
  assert.ok(text(tree.toJSON()).includes("Standings update when this round is complete."));
  assert.equal(button(tree, "Next round"), undefined, "The hub manages the rolling queue without a manual next-round gate");
  assert.ok(tree.root.findAllByType("a").some(link => link.props.href.startsWith("/admin/round-robin-generator?hub=event-1&draft=")));
  await act(async () => buttons(tree, "Manage round robin")[0].props.onClick());
  assert.equal(tree.root.findByProps({ "data-runner": "a" }).props["data-runner"], "a");
  await act(async () => button(tree, "Embedded standings").props.onClick());
  assert.ok(text(tree.toJSON()).includes("Full session standings"));
  assert.equal(tree.root.findByProps({ id: "hub-tab-standings" }).props["aria-selected"], true);
  assert.equal(tree.root.findByProps({ "aria-label": "Standings round robin" }).props.value, "a");
  assert.equal(tree.root.findAllByProps({ "data-runner": "a" }).length, 0, "Standings replaces the embedded manager");
  await act(async () => button(tree, "Current round").props.onClick());
  assert.equal(button(tree, "Round robins").props["aria-selected"], true);
  assert.equal(tree.root.findByProps({ "data-runner": "a" }).props["data-round"], 3, "Standings round links stay in this hub and select the requested round");
  assert.equal(tree.root.findAllByProps({ "data-standings": "a" }).length, 0);
  await act(async () => button(tree, "Close round robin controls").props.onClick());

  await act(async () => button(tree, "Players").props.onClick());
  failNext = true;
  await act(async () => button(tree, "Pause player").props.onClick());
  const failed = requests[requests.length - 1];
  assert.ok(stored.has("rr-hub-pending:club"), "Uncertain action is retained across refresh");
  assert.equal(button(tree, "Pause player").props.disabled, true, "Other writes freeze until uncertain action resolves");
  await act(async () => button(tree, "Retry saved action").props.onClick());
  assert.deepEqual(requests[requests.length - 1], failed, "Retry uses the exact idempotency key and expected version");
  assert.equal(stored.has("rr-hub-pending:club"), false);
  assert.ok(button(tree, "Ready again"));
  await act(async () => button(tree, "Court board").props.onClick());
  const paused = tree.root.findByProps({ "aria-label": "Paused players" });
  assert.match(text(paused), /Moneyball.*B Player 1/);
  await act(async () => paused.findByType("button").props.onClick());
  assert.equal(requests.at(-1).action, "resume_player");
  assert.deepEqual(requests.at(-1).payload, { identity: "player:b1" });
  assert.equal(tree.root.findAllByProps({ "aria-label": "Paused players" }).length, 0, "Resuming the last paused player clears the board section");
  await act(async () => button(tree, "Players").props.onClick());
  assert.ok(button(tree, "Pause player"));

  await act(async () => button(tree, "Court board").props.onClick());
  conflictNext = true;
  await act(async () => button(tree, "Next open court").props.onClick());
  assert.equal(stored.has("rr-hub-pending:club"), false, "Definite version conflict does not leave a stale retry");
  assert.ok(text(tree.toJSON()).includes("The court board changed."));
  await act(async () => tree.unmount());

  function emptyHub(id) {
    detail = { ok: true, hub: { id, title: id, status: "active", version: 1 }, workspace: {
      courts: [{ id: "c2", label: "Court 2", dedicated_session_key: null, open: true, current: null, next: null }],
      groups: [], eligible_queue: [], blocked_queue: [], reserved_queue: [], players: []
    } };
    availableSessions = []; stored.clear(); requests.length = 0; requestUrls.length = 0; replacedUrls.length = 0;
    deferDetailAfter = -1; releaseDetail = undefined;
    listedHubs = null; document.visibilityState = "visible";
  }
  const props = (hub, session = "") => ({ apiBase: "http://test.local", clubId: "club", initialHubId: hub, initialAddSessionKey: session });
  const standingsGroup = (key, status = "active", scoring_mode = "scored") => ({ session_key: key, title: `${key} round robin`, status, scoring_mode,
    current_round_number: 1, round_status: "active", completed_matches: 0, total_matches: 2, can_advance: false, event_format: "standard" });
  for (const change of ["substitution", "team_swap", "identity_change"]) {
    emptyHub(`score-lineup-${change}`);
    detail.workspace.groups = [standingsGroup("a")];
    detail.workspace.courts[0].current = clone(a1);
    await act(async () => { tree = create(h(Panel, props(detail.hub.id))); });
    await act(async () => button(tree, "Enter score").props.onClick());
    for (const [side, value] of [["A", "11"], ["B", "7"]])
      await act(async () => tree.root.findByProps({ "aria-label": `Side ${side} score` }).props.onChange({ target: { value } }));
    const current = detail.workspace.courts[0].current;
    if (change === "substitution") current.side_a[0] = { id: "new", identity: "guest:a:new", name: "New arrival" };
    else if (change === "team_swap") [current.side_a[0], current.side_b[0]] = [current.side_b[0], current.side_a[0]];
    else current.side_a[0].identity = "player:resolved-profile";
    detail.hub.version += 1;
    await act(async () => window.dispatchEvent({ type: "focus" }));
    assert.equal(button(tree, "Save score & release court").props.disabled, true, `${change} invalidates an open score draft`);
    assert.equal(button(tree, "Reviewed — keep these scores"), undefined, "Old scores cannot be carried onto changed players");
    await act(async () => tree.root.findByProps({ id: "hub-score-form" }).props.onSubmit({ preventDefault() {} }));
    assert.equal(requests.length, 0);
    assert.match(text(tree.toJSON()), /match or players changed/);
    await act(async () => tree.unmount());
  }
  emptyHub("paused-queue-event");
  detail.workspace.groups = [standingsGroup("a")];
  detail.workspace.blocked_queue = [{ ...a2, status: "blocked", blockers: ["A player is paused."] }];
  await act(async () => { tree = create(h(Panel, props("paused-queue-event"))); });
  assert.equal(button(tree, "Fill open courts").props.disabled, false, "An existing paused queue can request replacement planning even before any match is ready");
  await act(async () => button(tree, "Fill open courts").props.onClick());
  assert.equal(requests.at(-1).action, "auto_fill_courts");
  detail.workspace.courts[0].open = false;
  await act(async () => window.dispatchEvent({ type: "focus" }));
  assert.equal(button(tree, "Fill open courts").props.disabled, true, "Closed courts cannot be filled");
  await act(async () => tree.unmount());
  emptyHub("finish-event");
  detail.workspace.groups = [
    { ...standingsGroup("already-finished", "completed"), can_finish: false },
    { ...standingsGroup("draining"), can_finish: true, finish_requested: false, finish_boundary_round: 2, finish_pending_matches: 3 },
    { ...standingsGroup("last"), can_finish: true, finish_requested: false, finish_boundary_round: 1, finish_pending_matches: 0 },
    { ...standingsGroup("locked"), can_finish: false, finish_unavailable_reason: "Results are already submitted." }
  ];
  await act(async () => { tree = create(h(Panel, props("finish-event"))); });
  await act(async () => button(tree, "Finish").props.onClick());
  const finishDialog = () => tree.root.findByProps({ "data-dialog": "Finish round robin" });
  const finishSelect = () => finishDialog().findByProps({ "aria-label": "Round robin to finish" });
  const finishSubmit = () => finishDialog().findAllByType("button").find(node => text(node) === "Finish round robin");
  assert.equal(finishSelect().props.value, "", "Choose the RR explicitly before requesting its finish");
  assert.equal(finishSubmit().props.disabled, true);
  assert.deepEqual(finishSelect().findAllByType("option").map(option => option.props.value), ["", "draining", "last", "locked"], "Finished RRs are excluded from finish choices");
  assert.equal(finishSelect().findAllByType("option").find(option => option.props.value === "locked").props.disabled, true);
  assert.ok(text(finishDialog()).includes("Results are already submitted."));
  await act(async () => finishSelect().props.onChange({ target: { value: "draining" } }));
  assert.ok(text(finishDialog()).includes("Finish after round 2."));
  assert.ok(text(finishDialog()).includes("3 matches still need to finish"));
  assert.ok(text(finishDialog()).includes("No scores are discarded."));
  loseFinishResponse = true;
  await act(async () => tree.root.findByProps({ id: "hub-finish-group-form" }).props.onSubmit({ preventDefault() {} }));
  const finishRequest = clone(requests.at(-1));
  assert.deepEqual(finishRequest.payload, { session_key: "draining" });
  assert.equal(finishSelect().props.disabled, true, "An uncertain finish cannot be redirected to another RR");
  assert.equal(finishSelect().props.value, "draining");
  assert.equal(finishSubmit().props.disabled, true);
  assert.equal(button(tree, "Keep playing").props.disabled, true);
  await act(async () => button(tree, "Retry saved action").props.onClick());
  assert.deepEqual(requests.at(-1), finishRequest, "Finish retry retains exact target, version and operation key");
  assert.equal(tree.root.findAllByProps({ "data-dialog": "Finish round robin" }).length, 0);
  assert.equal(button(tree, "Court board").props["aria-selected"], true, "Finishing keeps the operator on the board");
  assert.ok(text(tree.root.findByProps({ "aria-label": "Finishing round robins" })).includes("draining round robin · Finishing after round 2 · 3 matches left"));
  detail.workspace.groups.find(group => group.session_key === "draining").status = "completed";
  detail.workspace.groups.find(group => group.session_key === "locked").status = "completed";
  detail.hub.version += 1;
  await act(async () => button(tree, "Refresh").props.onClick());
  await act(async () => button(tree, "Finish").props.onClick());
  await act(async () => finishSelect().props.onChange({ target: { value: "last" } }));
  assert.ok(text(finishDialog()).includes("This round robin will finish now."));
  await act(async () => tree.root.findByProps({ id: "hub-finish-group-form" }).props.onSubmit({ preventDefault() {} }));
  assert.ok(text(tree.toJSON()).includes("All round robins are finished. Event closed."));
  assert.equal(button(tree, "Finish").props.disabled, true);
  await act(async () => tree.root.findByProps({ id: "hub-tab-standings" }).props.onClick());
  assert.equal(tree.root.findByProps({ "aria-label": "Standings round robin" }).findAllByType("option").length, 4, "Finished RR results remain available after the last finish closes the event");
  assert.equal(tree.root.findByType(Standings).parent.props.inert, false, "Completed events still allow viewing and sharing standings");
  await act(async () => tree.unmount());

  emptyHub("standings-event");
  const completed = standingsGroup("finished", "completed"), unscored = standingsGroup("casual", "active", "unscored"), active = standingsGroup("live");
  detail.workspace.groups = [completed, unscored, active];
  await act(async () => { tree = create(h(Panel, props("standings-event"))); });
  await act(async () => tree.root.findByProps({ id: "hub-tab-standings" }).props.onClick());
  const selectStandings = async key => act(async () => tree.root.findByProps({ "aria-label": "Standings round robin" }).props.onChange({ target: { value: key } }));
  const selectedStandings = () => tree.root.findByProps({ "aria-label": "Standings round robin" }).props.value;
  assert.equal(selectedStandings(), "live", "Default to an active scored group rather than completed or unscored groups");
  const liveView = tree.root.findByProps({ "data-standings": "live" });
  assert.equal(liveView.props["data-embedded"], true);
  assert.equal(liveView.props["data-club"], "club");
  const liveInstance = liveView.props["data-instance"];
  assert.deepEqual(tree.root.findByProps({ "aria-label": "Standings round robin" }).findAllByType("option").map(option => text(option)), ["finished round robin · Finished", "casual round robin · Unscored", "live round robin"]);
  await selectStandings("finished");
  const finishedInstance = tree.root.findByProps({ "data-standings": "finished" }).props["data-instance"];
  assert.notEqual(finishedInstance, liveInstance, "Changing groups remounts standings so sharing state cannot leak");
  detail.workspace.groups.reverse(); detail.hub.version += 1;
  await act(async () => button(tree, "Refresh after sharing").props.onClick());
  assert.equal(selectedStandings(), "finished", "Refresh and list reordering preserve an explicit choice");
  assert.equal(tree.root.findByProps({ "data-standings": "finished" }).props["data-instance"], finishedInstance);
  assert.equal(tree.root.findByProps({ "data-standings": "finished" }).props["data-refresh-key"], detail.hub.version, "Hub refresh reaches the selected embedded standings");
  await selectStandings("casual");
  assert.equal(tree.root.findByProps({ "data-standings": "casual" }).props["data-standings"], "casual", "Unscored groups delegate their explanation to the existing standings view");
  detail.workspace.groups = [completed, active]; detail.hub.version += 1;
  await act(async () => button(tree, "Refresh after sharing").props.onClick());
  assert.equal(selectedStandings(), "live", "Removing the selected group falls back to an active scored group");
  detail.workspace.groups.unshift(unscored);
  await act(async () => button(tree, "Refresh after sharing").props.onClick());
  assert.equal(selectedStandings(), "live", "Reattaching a previously selected group does not revive stale selection");
  await act(async () => button(tree, "Round robins").props.onClick());
  await act(async () => tree.root.findAllByType("article").find(article => text(article).includes("finished round robin")).findAllByType("button").find(node => text(node) === "Standings").props.onClick());
  assert.equal(selectedStandings(), "finished", "The group card opens this group's Standings tab");
  await act(async () => button(tree, "Round robins").props.onClick());
  await act(async () => buttons(tree, "Manage round robin")[0].props.onClick());
  await act(async () => tree.root.findByProps({ id: "hub-tab-standings" }).props.onClick());
  assert.equal(tree.root.findAllByProps({ "aria-label": "Manage selected round robin" }).length, 0, "Opening the tab clears any stale round controls");
  await selectStandings("live");
  const beforeHubSwitch = tree.root.findByProps({ "data-standings": "live" }).props["data-instance"];
  detail.hub.id = "other-standings-event"; detail.workspace.groups = [active];
  await act(async () => tree.update(h(Panel, props("other-standings-event"))));
  assert.equal(selectedStandings(), "live");
  assert.notEqual(tree.root.findByProps({ "data-standings": "live" }).props["data-instance"], beforeHubSwitch, "Even a matching session key remounts when the hub changes");
  detail.workspace.groups = []; detail.hub.version += 1;
  await act(async () => button(tree, "Refresh after sharing").props.onClick());
  assert.equal(tree.root.findByProps({ "aria-label": "Standings round robin" }).props.disabled, true);
  assert.equal(tree.root.findAllByType(Standings).length, 0);
  assert.ok(text(tree.toJSON()).includes("Add a round robin to this event to view its standings here."));
  await act(async () => tree.unmount());

  emptyHub("old-event");
  await act(async () => { tree = create(h(Panel, props("old-event"))); });
  await act(async () => button(tree, "Round robins").props.onClick());
  const readsBeforeReturn = listReads;
  detail.hub.id = "created-event"; detail.hub.title = "Created event";
  await act(async () => tree.update(h(Panel, props("created-event", "created-session"))));
  assert.equal(requests.length, 1, "A cached panel attaches the returned session exactly once");
  assert.ok(requestUrls[0].endsWith("/created-event/commands"), "The return URL selects the exact hub, not the previously selected hub");
  assert.deepEqual(requests[0].payload, { session_key: "created-session" });
  assert.ok(listReads > readsBeforeReturn, "Returning from creation refreshes the session list");
  assert.equal(button(tree, "Court board").props["aria-selected"], true);
  assert.ok(text(tree.toJSON()).includes("Fill open courts"));
  assert.ok(text(tree.toJSON()).includes("Round robin added. Choose Next open court or Fill open courts to start play."));
  assert.equal(replacedUrls.at(-1), "/admin/round-robin-hub?hub=created-event", "Consume the return query only after attachment succeeds");
  await act(async () => tree.update(h(Panel, props("created-event", "created-session"))));
  assert.equal(requests.length, 1, "Ordinary rerenders do not repeat attachment");
  await act(async () => tree.unmount());

  requests.length = 0;
  await act(async () => { tree = create(h(React.StrictMode, null, h(Panel, props("created-event", "created-session")))); });
  assert.equal(requests.length, 0, "An already attached return session succeeds without another write");
  assert.equal(button(tree, "Court board").props["aria-selected"], true);
  await act(async () => tree.unmount());

  emptyHub("interrupted-event"); loseAddResponse = true;
  await act(async () => { tree = create(h(React.StrictMode, null, h(Panel, props("interrupted-event", "interrupted-session")))); });
  assert.equal(requests.length, 1, "Strict Mode does not duplicate automatic attachment");
  const interruptedRequest = clone(requests[0]);
  assert.ok(stored.has("rr-hub-pending:club"));
  assert.ok(tree.root.findAllByType("a").some(link => link.props.href === "/admin/round-robin-generator/sessions/interrupted-session/rounds/1"), "A failed attachment keeps the created RR accessible");
  assert.equal(replacedUrls.length, 0, "An uncertain attachment keeps its return intent");
  await act(async () => tree.unmount());
  await act(async () => { tree = create(h(React.StrictMode, null, h(Panel, props("interrupted-event", "interrupted-session")))); });
  assert.equal(requests.length, 1, "Restore pending action before deciding whether to auto-attach");
  await act(async () => button(tree, "Retry saved action").props.onClick());
  assert.deepEqual(requests.at(-1), interruptedRequest, "Uncertain attachment replays the same key and version after remount");
  assert.equal(button(tree, "Court board").props["aria-selected"], true);
  assert.equal(stored.size, 0);
  assert.equal(replacedUrls.at(-1), "/admin/round-robin-hub?hub=interrupted-event");
  await act(async () => tree.unmount());

  emptyHub("conflicting-event"); conflictNext = true;
  await act(async () => { tree = create(h(Panel, props("conflicting-event", "conflicting-session"))); });
  assert.equal(requests.length, 1, "A conflict stops automatic attachment until an explicit retry");
  assert.equal(stored.size, 0);
  assert.equal(replacedUrls.length, 0);
  await act(async () => button(tree, "Retry adding to hub").props.onClick());
  assert.equal(requests.length, 2);
  assert.notEqual(requests[0].idempotency_key, requests[1].idempotency_key);
  assert.equal(requests[1].expected_version, 2, "An explicit retry obtains the latest hub version");
  assert.equal(button(tree, "Court board").props["aria-selected"], true);
  await act(async () => tree.unmount());

  emptyHub("slow-old-event"); deferDetailAfter = 1;
  await act(async () => { tree = create(h(Panel, props("slow-old-event", "slow-session"))); });
  assert.equal(typeof releaseDetail, "function");
  assert.equal(requests.length, 0);
  assert.equal(tree.root.findByProps({ "aria-label": "Select hub" }).props.disabled, true, "Preparing an attachment locks hub switching");
  detail.hub.id = "new-destination"; detail.hub.title = "New destination";
  await act(async () => tree.update(h(Panel, props("new-destination"))));
  await act(async () => releaseDetail());
  assert.equal(requests.length, 0, "A delayed preparation cannot attach after the return intent was removed");
  assert.equal(tree.root.findByProps({ "aria-label": "Select hub" }).props.value, "new-destination");
  assert.equal(replacedUrls.length, 0, "A stale response cannot navigate back to the old event");
  assert.equal(button(tree, "Retry adding to hub"), undefined);
  await act(async () => tree.unmount());

  emptyHub("old-failed-event"); conflictNext = true;
  await act(async () => { tree = create(h(Panel, props("old-failed-event", "old-session"))); });
  assert.ok(button(tree, "Retry adding to hub"));
  detail.hub.id = "local-selection";
  await act(async () => tree.root.findByProps({ "aria-label": "Select hub" }).props.onChange({ target: { value: "local-selection" } }));
  assert.equal(button(tree, "Retry adding to hub"), undefined, "Local event selection abandons the unsent return intent");
  assert.equal(requests.length, 1);
  await act(async () => tree.unmount());

  emptyHub("pending-event"); loseAddResponse = true;
  await act(async () => { tree = create(h(Panel, props("pending-event", "first-pending-session"))); });
  const firstPending = clone(requests[0]);
  await act(async () => tree.update(h(Panel, props("pending-event", "second-return-session"))));
  assert.equal(requests.length, 1, "A new return intent waits for the earlier uncertain command");
  assert.deepEqual(JSON.parse(stored.get("rr-hub-pending:club")).body, firstPending, "Changing return intent preserves the sent pending command");
  await act(async () => button(tree, "Retry saved action").props.onClick());
  assert.equal(requests.length, 3);
  assert.deepEqual(requests[1], firstPending, "Confirm the first exact request before sending the next one");
  assert.deepEqual(requests[2].payload, { session_key: "second-return-session" });
  assert.equal(detail.workspace.groups.length, 2);
  assert.equal(button(tree, "Court board").props["aria-selected"], true);
  await act(async () => tree.unmount());

  emptyHub("manual-event");
  availableSessions = [{ session_key: "existing", title: "Existing RR", status: "active" }];
  await act(async () => { tree = create(h(Panel, props("manual-event"))); });
  await act(async () => button(tree, "Round robins").props.onClick());
  await act(async () => tree.root.findAllByType("select").find(select => select.props.value === "").props.onChange({ target: { value: "existing" } }));
  await act(async () => button(tree, "Add to hub").props.onClick());
  assert.deepEqual(requests[0].payload, { session_key: "existing" }, "Existing-session manual attachment remains available");
  await act(async () => tree.unmount());

  emptyHub("external-sessions-event");
  const available = (session_key, title) => ({ session_key, title, status: "active", rating_mode: "unrated" });
  availableSessions = [available("existing", "Existing RR")];
  await act(async () => { tree = create(h(Panel, props("external-sessions-event"))); });
  availableSessions.push(available("opened-elsewhere", "3.0 Round Robin"));
  await act(async () => button(tree, "Round robins").props.onClick());
  const existingSelect = () => tree.root.findAllByType("select").find(select => select.findAllByType("option").some(option => text(option) === "Choose a round robin"));
  const availableKeys = () => existingSelect().findAllByType("option").map(option => option.props.value);
  assert.ok(availableKeys().includes("opened-elsewhere"), "Opening Round robins discovers sessions started elsewhere after this page loaded");
  await act(async () => existingSelect().props.onChange({ target: { value: "opened-elsewhere" } }));
  availableSessions.push(available("returned-elsewhere", "3.5/4.0 Round Robin"));
  await act(async () => window.dispatchEvent({ type: "focus" }));
  assert.ok(availableKeys().includes("returned-elsewhere"), "Returning focus refreshes the existing-RR selector without a manual page refresh");
  assert.equal(existingSelect().props.value, "opened-elsewhere", "Discovering another RR preserves the organizer's current selection");
  assert.equal(button(tree, "Add to hub").props.disabled, false);
  availableSessions.push(available("polled-session", "Next RR"));
  await act(async () => tickIntervals());
  assert.ok(availableKeys().includes("polled-session"), "Visible polling refreshes available RRs as well as the court board");
  const beforeHidden = listReads;
  document.visibilityState = "hidden";
  availableSessions.push(available("hidden-session", "Later RR"));
  await act(async () => { window.dispatchEvent({ type: "focus" }); document.dispatchEvent({ type: "visibilitychange" }); tickIntervals(); });
  assert.equal(listReads, beforeHidden, "Hidden tabs do not request the RR list on focus, visibility changes or polling");
  assert.ok(!availableKeys().includes("hidden-session"));
  document.visibilityState = "visible";
  await act(async () => document.dispatchEvent({ type: "visibilitychange" }));
  assert.ok(availableKeys().includes("hidden-session"), "Becoming visible discovers RRs created while the hub was hidden");
  assert.equal(existingSelect().props.value, "opened-elsewhere");
  assert.equal(requests.length, 0, "Refreshing choices never attaches an RR or changes court operations");
  await act(async () => tree.unmount());
  assert.equal(intervals.size, 0, "Unmounting stops the periodic refresh");

  emptyHub("not-created-yet"); listedHubs = [];
  await act(async () => { tree = create(h(Panel, props(""))); });
  assert.ok(tree.root.findByProps({ "aria-label": "Create event" }));
  const beforeInitialFocus = listReads;
  availableSessions.push(available("first-session", "First RR"));
  await act(async () => window.dispatchEvent({ type: "focus" }));
  assert.ok(listReads > beforeInitialFocus, "Available RRs refresh even before a hub event has been created or selected");
  assert.equal(requests.length, 0, "Discovering RRs before setup does not create an event automatically");
  await act(async () => tree.unmount());
  console.log("round-robin hub: court controls, cached creation return, exact event selection, Strict Mode, durable retry, conflicts, manual attachment, and live RR selector refresh passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
