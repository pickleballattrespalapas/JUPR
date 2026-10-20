const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const h = React.createElement, root = path.resolve(__dirname, ".."), cache = new Map();
global.IS_REACT_ACT_ENVIRONMENT = true;
const stored = new Map();
global.sessionStorage = { getItem: key => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
const replacedUrls = [];
global.window = { history: { replaceState(_state, _title, url) { replacedUrls.push(url); } }, setInterval: () => 1, clearInterval() {}, addEventListener() {}, removeEventListener() {} };
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
const requests = [], requestUrls = [], receipts = new Map();
let failNext = false, conflictNext = false, loseAddResponse = false, listReads = 0;
let deferDetailAfter = -1, releaseDetail;
let availableSessions = [{ session_key: "new", title: "New RR", status: "active", rating_mode: "unrated" }];
global.fetch = async (url, options = {}) => {
  assert.equal(options.headers.Authorization, "Bearer test-staff");
  if (!options.body) {
    if (url.endsWith("/round-robin-hubs")) { listReads += 1; return { ok: true, json: async () => ({ ok: true, hubs: [detail.hub], sessions: clone(availableSessions) }) }; }
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
  else if (body.action === "add_group") {
    const key = body.payload.session_key;
    assert.ok(!detail.workspace.groups.some(group => group.session_key === key), "Do not add a group twice");
    detail.workspace.groups.push({ session_key: key, title: "Created round robin", current_round_number: 1, round_status: "active", completed_matches: 0, total_matches: 1, can_advance: false, event_format: "standard", scoring_mode: "scored" });
    detail.workspace.eligible_queue.push(match(key, `${key}-1`, 1, ["c2"], 0));
  }
  else if (body.action !== "assign_next_court") throw new Error(`Unexpected action ${body.action}`);
  detail.hub.version += 1;
  receipts.set(body.idempotency_key, clone(body));
  if (body.action === "add_group" && loseAddResponse) { loseAddResponse = false; throw new Error("Response lost after attachment"); }
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

  function emptyHub(id) {
    detail = { ok: true, hub: { id, title: id, status: "active", version: 1 }, workspace: {
      courts: [{ id: "c2", label: "Court 2", dedicated_session_key: null, open: true, current: null, next: null }],
      groups: [], eligible_queue: [], blocked_queue: [], reserved_queue: [], players: []
    } };
    availableSessions = []; stored.clear(); requests.length = 0; requestUrls.length = 0; replacedUrls.length = 0;
    deferDetailAfter = -1; releaseDetail = undefined;
  }
  const props = (hub, session = "") => ({ apiBase: "http://test.local", clubId: "club", initialHubId: hub, initialAddSessionKey: session });
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
  console.log("round-robin hub: court controls, cached creation return, exact event selection, Strict Mode, durable retry, conflicts, and manual attachment passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
