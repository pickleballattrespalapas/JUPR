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

const participant = (session, profile = null, name = "1") => ({
  name, display_name: `${name} · Numbers ${session.toUpperCase()}`, session_key: session,
  group_title: `Numbers ${session.toUpperCase()}`, participant_id: `${session}-1`,
  identity: profile == null ? `participant:${session}:1` : `profile:${profile}`, profile_id: profile
});
const guestA = participant("a"), guestB = participant("b");
const review = (participants = [guestA, guestB], extra = {}) => ({
  id: "name:1", name: "1", status: "pending", can_merge: true, assigned: false,
  participants: clone(participants), ...extra
});
const readyMatch = {
  session_key: "c", match_id: "c-1", round_number: 1,
  side_a: [{ id: "c1", identity: "c1", name: "Unaffected A" }],
  side_b: [{ id: "c2", identity: "c2", name: "Unaffected B" }],
  label: "Unaffected match", wait_seconds: 300, allowed_court_ids: ["c1"], blockers: [], status: "waiting"
};
const blockedMatch = {
  ...readyMatch, session_key: "a", match_id: "a-1",
  side_a: [{ id: guestA.participant_id, ...guestA }],
  side_b: [{ id: "a2", identity: "a2", name: "Partner" }],
  blockers: ["Matching player name needs identity review: 1"], status: "blocked"
};
let detail, failAfterCommit = false, conflictNext = false;
const requests = [], receipts = new Map();
function setup(reviews = [review()]) {
  stored.clear(); requests.length = 0; receipts.clear(); failAfterCommit = false; conflictNext = false;
  const occurrences = reviews.flatMap(item => item.participants);
  detail = { ok: true, hub: { id: "identity-event", title: "Numbered groups", status: "active", version: 1 }, workspace: {
    courts: [{ id: "c1", label: "Court 1", dedicated_session_key: null, open: true, current: null, next: null }],
    groups: [...new Set(["a", "b", "c", ...occurrences.map(item => item.session_key)])].map(session => ({
      session_key: session, title: `Numbers ${session.toUpperCase()}`, current_round_number: 1,
      round_status: "active", completed_matches: 0, total_matches: 2, can_advance: false,
      event_format: "standard", scoring_mode: "scored"
    })),
    eligible_queue: [clone(readyMatch)], blocked_queue: [clone(blockedMatch)], reserved_queue: [],
    players: occurrences.map(item => ({ identity: item.identity, name: item.name, display_name: item.display_name,
      paused: false, wait_seconds: 300, session_keys: [item.session_key], participant_ids: [item.participant_id] })),
    identity_review: clone(reviews), identity_pending_count: reviews.filter(item => item.status === "pending").length
  } };
}
global.fetch = async (url, options = {}) => {
  assert.equal(options.headers.Authorization, "Bearer test-staff");
  if (!options.body) return { ok: true, json: async () => url.endsWith("/round-robin-hubs") ? { ok: true, hubs: [clone(detail.hub)], sessions: [] } : clone(detail) };
  const body = JSON.parse(options.body); requests.push(body);
  if (receipts.has(body.idempotency_key)) {
    assert.deepEqual(body, receipts.get(body.idempotency_key), "Identity replay retains the complete exact request");
    return { ok: true, json: async () => clone(detail) };
  }
  if (conflictNext) {
    conflictNext = false; detail.hub.version += 1;
    return { ok: false, status: 409, json: async () => ({ detail: "The roster changed. Review matching names again." }) };
  }
  assert.equal(body.expected_version, detail.hub.version);
  if (body.action === "resolve_identities") {
    const selected = body.payload.participants || [];
    for (const item of detail.workspace.identity_review) {
      if (body.payload.resolution === "all_different" && item.status === "pending") item.status = "different";
      else if (selected.length && item.participants.some(person => selected.some(ref => ref.session_key === person.session_key && ref.participant_id === person.participant_id))) item.status = body.payload.resolution;
    }
    detail.workspace.identity_pending_count = detail.workspace.identity_review.filter(item => item.status === "pending").length;
  } else if (body.action === "pause_player") {
    detail.workspace.players.find(player => player.identity === body.payload.identity).paused = true;
  } else throw new Error(`Unexpected action ${body.action}`);
  detail.hub.version += 1; receipts.set(body.idempotency_key, clone(body));
  if (failAfterCommit) { failAfterCommit = false; throw new Error("Response lost after identity decision saved"); }
  return { ok: true, json: async () => clone(detail) };
};

(async () => {
  const Panel = load("@/app/admin/round-robin-hub/RoundRobinHubPanel").default;
  const api = load("@/lib/adminRoundRobinHubApi");
  const props = { apiBase: "http://test.local", clubId: "club", initialHubId: "identity-event" };
  let tree;
  async function mount() { await act(async () => { tree = create(h(Panel, props)); }); }
  async function click(label) { await act(async () => button(tree, label).props.onClick()); }
  const reviewCard = () => tree.root.findByProps({ "data-review-id": "name:1" });
  const choice = label => reviewCard().findAllByType("button").find(node => text(node) === label);

  setup(); await mount();
  assert.ok(button(tree, "Review matching names"), "Legacy/loaded pending identities are discoverable immediately");
  assert.equal(button(tree, "Next open court").props.disabled, false, "Unrelated ready matches remain dispatchable during review");
  await click("Match queue");
  assert.ok(text(tree.toJSON()).includes("Matching player name needs identity review: 1"));
  assert.ok(text(tree.toJSON()).includes("1 · Numbers A vs Partner"), "Match labels distinguish separate guest entries");
  await click("Review matching names");
  assert.ok(tree.root.findByProps({ "aria-label": "Matching player names" }));
  assert.equal(choice("Same person").props.disabled, false);
  assert.equal(choice("Different people").props.disabled, false);
  failAfterCommit = true;
  await click("These are all different people");
  const lostRequest = clone(requests.at(-1));
  assert.deepEqual(lostRequest.payload, { resolution: "all_different" }, "Batch review uses one explicit command without stale participant enumeration");
  assert.equal(lostRequest.action, "resolve_identities");
  assert.ok(stored.has("rr-hub-pending:club"));
  assert.equal(choice("Same person").props.disabled, true, "Uncertain save freezes competing decisions");
  await act(async () => tree.unmount());
  await mount();
  await click("Retry saved action");
  assert.deepEqual(requests.at(-1), lostRequest, "Refresh/retry preserves identity command key, version and payload");
  assert.equal(stored.size, 0);
  await click("Players");
  assert.equal(button(tree, "These are all different people"), undefined, "Saved decisions do not require repeat review");
  await click("Review player identities");
  assert.ok(text(reviewCard()).includes("Different people"), "Saved decisions remain accessible for correction");
  const playerA = tree.root.findByProps({ "data-player-identity": guestA.identity });
  const playerB = tree.root.findByProps({ "data-player-identity": guestB.identity });
  assert.equal(playerA.props["aria-label"], "1 · Numbers A");
  assert.equal(playerB.props["aria-label"], "1 · Numbers B");
  await act(async () => playerA.findByType("button").props.onClick());
  assert.deepEqual(requests.at(-1).payload, { identity: guestA.identity });
  assert.equal(text(tree.root.findByProps({ "data-player-identity": guestB.identity }).findByType("button")), "Pause player", "The other guest retains independent pause controls");
  await act(async () => choice("Same person").props.onClick());
  assert.deepEqual(requests.at(-1).payload, { resolution: "same", participants: [{ session_key: "a", participant_id: "a-1" }, { session_key: "b", participant_id: "b-1" }] });
  assert.equal(requests.some(request => /score|round/.test(request.action)), false, "Identity correction never sends round or score mutations");
  await act(async () => tree.unmount());

  setup([review([guestA, guestB], { assigned: true })]); await mount(); await click("Review matching names");
  assert.equal(choice("Same person").props.disabled, true, "Do not merge while affected players are assigned");
  assert.equal(choice("Different people").props.disabled, false, "Splitting remains available during play");
  assert.equal(button(tree, "These are all different people").props.disabled, false);
  assert.ok(text(tree.toJSON()).includes("Finish or release affected assignments before marking these as the same person."));
  await act(async () => tree.unmount());

  const linkedA = participant("a", 101, "Mike"), linkedB = participant("b", 102, "Mike"), guestC = participant("c", null, "Mike");
  setup([review([linkedA, linkedB, guestC], { can_merge: false })]); await mount(); await click("Review matching names");
  assert.equal(choice("Same person").props.disabled, true, "Different authoritative profiles cannot be merged");
  const linkedBSelection = reviewCard().findAllByType("input").find(node => node.props["aria-label"].includes("Numbers B"));
  await act(async () => linkedBSelection.props.onChange({ target: { checked: false } }));
  assert.equal(choice("Same person").props.disabled, false, "A guest can match one linked profile without merging the other profile");
  await act(async () => choice("Same person").props.onClick());
  assert.deepEqual(requests.at(-1).payload.participants, [{ session_key: "a", participant_id: "a-1" }, { session_key: "c", participant_id: "c-1" }]);
  await act(async () => tree.unmount());

  setup([review([participant("a", 101), participant("b", 101)], { status: "same" })]);
  // Real snapshots collapse same-profile rows; keep one here to avoid duplicate React keys.
  detail.workspace.players = detail.workspace.players.slice(0, 1);
  await mount(); await click("Players"); await click("Review player identities");
  assert.equal(choice("Different people").props.disabled, true, "A shared authoritative profile cannot be split");
  assert.ok(text(tree.toJSON()).includes("cannot be split here"));
  await act(async () => tree.unmount());

  setup([review([participant("a", 101), participant("b", 101), participant("c")], { status: "same" })]);
  detail.workspace.players = detail.workspace.players.filter((player, index) => index !== 1);
  await mount(); await click("Players"); await click("Review player identities");
  assert.equal(choice("Different people").props.disabled, false, "Select the complete merged component to separate a guest from repeated authoritative profiles");
  assert.ok(text(tree.toJSON()).includes("same club profile stay together"));
  await act(async () => choice("Different people").props.onClick());
  assert.deepEqual(requests.at(-1).payload.participants, ["a", "b", "c"].map(session => ({ session_key: session, participant_id: `${session}-1` })));
  await act(async () => tree.unmount());

  setup(); await mount(); await click("Review matching names"); conflictNext = true;
  await act(async () => choice("Different people").props.onClick());
  assert.equal(stored.size, 0, "Definite conflict clears the old command receipt");
  assert.equal(requests.length, 1, "Roster conflict is never silently replayed against a newer roster");
  assert.ok(text(tree.toJSON()).includes("The roster changed. Review matching names again."));
  await act(async () => choice("Different people").props.onClick());
  assert.equal(requests.at(-1).expected_version, 2);
  assert.notEqual(requests[0].idempotency_key, requests[1].idempotency_key);
  await act(async () => tree.unmount());

  assert.equal(api.hubMatchLabel(blockedMatch), "1 · Numbers A vs Partner");
  assert.equal(api.hubPlayerLabel({ name: "Legacy name" }), "Legacy name", "Older snapshots still render safely");
  console.log("Round robin hub identity review checks passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
