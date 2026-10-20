const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const h = React.createElement, root = path.resolve(__dirname, ".."), cache = new Map();
global.IS_REACT_ACT_ENVIRONMENT = true;
const stored = new Map();
global.sessionStorage = { getItem: key => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
global.window = { history: { replaceState() {} }, setInterval: () => 1, clearInterval() {}, addEventListener() {}, removeEventListener() {} };
global.document = { visibilityState: "visible", addEventListener() {}, removeEventListener() {} };
const stubs = {
  "next/link": ({ children, ...props }) => h("a", props, children),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-staff" }) },
  "@/components/interaction": { InteractionDialog: ({ title, children, actions }) => h("section", { "data-dialog": title }, children, actions) },
  "@/app/admin/play-generators/GeneratorRoundRunner": { __esModule: true, default: props => h("div", { "data-runner": props.sessionKey }, h("button", { onClick: () => props.onNavigate({ view: "standings" }) }, "Embedded standings")) },
  "@/app/admin/play-generators/GeneratorStandings": { __esModule: true, default: props => h("div", { "data-standings": props.sessionKey }, "Full session standings") }
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
const requests = [];
let failNext = false, conflictNext = false;
global.fetch = async (url, options = {}) => {
  assert.equal(options.headers.Authorization, "Bearer test-staff");
  if (!options.body) {
    if (url.endsWith("/round-robin-hubs")) return { ok: true, json: async () => ({ ok: true, hubs: [detail.hub], sessions: [{ session_key: "new", title: "New RR", status: "active", rating_mode: "unrated" }] }) };
    return { ok: true, json: async () => clone(detail) };
  }
  const body = JSON.parse(options.body); requests.push(body);
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
  else if (body.action !== "assign_next_court") throw new Error(`Unexpected action ${body.action}`);
  detail.hub.version += 1;
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
  assert.equal(buttons(tree, "Next round")[0].props.disabled, true, "Partially completed round cannot advance");
  assert.ok(tree.root.findAllByType("a").some(link => link.props.href === "/admin/round-robin-generator?hub=event-1"));
  await act(async () => buttons(tree, "Manage round robin")[0].props.onClick());
  assert.equal(tree.root.findByProps({ "data-runner": "a" }).props["data-runner"], "a");
  await act(async () => button(tree, "Embedded standings").props.onClick());
  assert.ok(text(tree.toJSON()).includes("Full session standings"));
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
  await act(async () => button(tree, "Ready again").props.onClick());
  assert.ok(button(tree, "Pause player"));

  await act(async () => button(tree, "Court board").props.onClick());
  conflictNext = true;
  await act(async () => button(tree, "Next open court").props.onClick());
  assert.equal(stored.has("rr-hub-pending:club"), false, "Definite version conflict does not leave a stale retry");
  assert.ok(text(tree.toJSON()).includes("The court board changed."));
  await act(async () => tree.unmount());
  console.log("round-robin hub: exclusive courts, reservation, partial-round scoring, embedded controls, pause recovery, and version conflicts passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
