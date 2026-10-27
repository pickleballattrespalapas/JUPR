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
const Panel = load("@/app/admin/round-robin-hub/RoundRobinHubPanel").default;
const member = (id, name = id, extra = {}) => ({participant_id: id, name, identity: `guest:${id}`, profile_id: null, paused: false, wait_seconds: 60, on_court: false, reserved: false, active: true, active_from_round: 1, can_remove: true, membership_status: "active", ...extra});
function game(id, session = "a", extra = {}) {
  const people = [1, 2, 3, 4].map(n => ({id: `${id}-${n}`, name: `${id} Player ${n}`, identity: `guest:${id}-${n}`}));
  return {session_key: session, match_id: id, round_number: 3, side_a: people.slice(0, 2), side_b: people.slice(2), label: id, wait_seconds: 60, allowed_court_ids: ["c1", "c2"], blockers: [], status: "waiting", result_status: "pending", can_edit_players: true, ...extra};
}
function fixture() {
  const current = game("playing", "a", {status: "on_court", court_id: "c1"});
  const waiting = game("waiting");
  return {ok: true, hub: {id: "event-1", title: "Tonight", status: "active", version: 1}, registration: {open: true, pending_count: 0, automatic_pending_count: 0, checkins: []}, workspace: {
    courts: [{id: "c1", label: "Court 6", dedicated_session_key: null, open: true, current, next: null}, {id: "c2", label: "Court 7", dedicated_session_key: null, open: true, current: null, next: null}],
    groups: ["a", "b"].map(key => ({session_key: key, title: key === "a" ? "3.5/4.0" : "2.5/3.0", status: "active", event_format: "standard", scoring_mode: "scored", current_round_number: 3, total_rounds: 3, round_status: "active", completed_matches: 0, total_matches: 2, played_matches: 0, not_played_matches: 0, pending_matches: 2, can_advance: false, can_add_player: true, can_remove_player: true, add_player_effective_round: 3,
      players: key === "a" ? [...current.side_a, ...current.side_b].map(p => member(p.id, p.name, {on_court: true})).concat(member("move", "Darla"), member("variant", "D. Simmons")) : [member("other", "Other Player")]})),
    eligible_queue: [waiting], blocked_queue: [], reserved_queue: [], players: [], identity_review: [], identity_pending_count: 0
  }};
}
let detail, requests, receipts, conflictNext, loseNext, holdNext, held, listReads, detailReads;
function reset() { stored.clear(); detail = fixture(); requests = []; receipts = new Map(); conflictNext = false; loseNext = false; holdNext = false; held = null; listReads = 0; detailReads = 0; }
const response = data => ({ok: true, json: async () => clone(data)});
global.fetch = async (url, options = {}) => {
  if (url.includes("/players?")) return response({players: []});
  if (!options.body) {
    if (url.endsWith("/round-robin-hubs")) { listReads++; return response({ok: true, hubs: [detail.hub], sessions: []}); }
    detailReads++; return response(detail);
  }
  const body = JSON.parse(options.body); requests.push(clone(body));
  if (receipts.has(body.idempotency_key)) { assert.deepEqual(body, receipts.get(body.idempotency_key).body, "Retries preserve the entire request"); return response(receipts.get(body.idempotency_key).value); }
  if (conflictNext) { conflictNext = false; detail.hub.version++; return {ok: false, status: 409, json: async () => ({detail: "Board changed. Review before saving."})}; }
  assert.equal(body.expected_version, detail.hub.version, "Each new command targets the reviewed board");
  const p = body.payload;
  if (["score_and_release", "resolve_game"].includes(body.action)) {
    const group = detail.workspace.groups.find(g => g.session_key === p.session_key);
    const notPlayed = body.action === "resolve_game" || p.score_a === null && p.score_b === null;
    if (body.action === "resolve_game") assert.equal(p.resolution, "not_played");
    else assert.equal(p.score_a === null, p.score_b === null, "Half blank is not a result");
    group.completed_matches++; group.pending_matches--; group[notPlayed ? "not_played_matches" : "played_matches"]++;
    detail.workspace.courts.forEach(c => { if (c.current?.match_id === p.match_id) c.current = null; if (c.next?.match_id === p.match_id) c.next = null; });
    for (const key of ["eligible_queue", "blocked_queue", "reserved_queue"]) detail.workspace[key] = detail.workspace[key].filter(m => m.match_id !== p.match_id);
  } else if (body.action === "transfer_player") {
    const source = detail.workspace.groups.find(g => g.session_key === p.source_session_key);
    const destination = detail.workspace.groups.find(g => g.session_key === p.destination_session_key);
    const person = source.players.find(person => person.participant_id === p.participant_id);
    assert.ok(person && destination && source !== destination);
    person.active = false; person.can_remove = false; person.can_rejoin = true; person.membership_status = "departed";
    destination.players.push({...clone(person), participant_id: "moved-darla", active: true, can_remove: true, can_rejoin: false, membership_status: "active"});
  } else if (body.action === "configure_courts") {
    detail.workspace.courts = p.courts.map(c => ({...detail.workspace.courts.find(old => old.id === c.id), ...c}));
  } else if (body.action === "rejoin_player") {
    const person = detail.workspace.groups.find(g => g.session_key === p.session_key).players.find(person => person.participant_id === p.participant_id);
    person.active = true; person.membership_status = "active"; person.can_rejoin = false;
  } else if (body.action === "process_checkins") detail.registration.automatic_pending_count = 0;
  else assert.ok(["resolve_identities", "move_game", "edit_match_players", "pause_player", "release_game", "assign_game"].includes(body.action), body.action);
  detail.hub.version++;
  const saved = clone(detail); receipts.set(body.idempotency_key, {body: clone(body), value: saved});
  if (loseNext) { loseNext = false; throw new Error("Response lost after commit"); }
  if (holdNext) { holdNext = false; return new Promise(resolve => { held = () => resolve(response(saved)); }); }
  return response(saved);
};
async function mount() { let tree; await act(async () => { tree = create(h(Panel, {apiBase: "http://test.local", clubId: "club", initialHubId: "event-1"})); }); return tree; }
async function click(tree, label, index = 0) { const node = buttons(tree, label)[index]; assert.ok(node, `Button ${label}`); assert.ok(!node.props.disabled, `Enabled ${label}`); await act(async () => node.props.onClick()); }
async function change(tree, label, value) { await act(async () => tree.root.findByProps({"aria-label": label}).props.onChange({target: {value}})); }
async function submit(tree, id) { await act(async () => tree.root.findByProps({id}).props.onSubmit({preventDefault() {}})); }
async function check(tree, label, checked = true) { await act(async () => tree.root.findByProps({"aria-label": label}).props.onChange({target: {checked}})); }
async function unmount(tree) { await act(async () => tree.unmount()); }
async function enterScore(tree, a = "11", b = "10") { await click(tree, "Enter score"); await change(tree, "Side A score", a); await change(tree, "Side B score", b); }
const passed = [];
async function scenario(name, run) { reset(); await run(); passed.push(name); }
(async () => {
  await scenario("current match explicitly not played", async () => {
    const tree = await mount();
    const court = tree.root.findByProps({"aria-label": "Court 6"});
    await act(async () => court.findAllByType("button").find(b => text(b) === "Not played").props.onClick());
    assert.match(text(tree.toJSON()), /No score, win, loss, or rating result/);
    await click(tree, "Confirm not played");
    assert.deepEqual(requests[0].payload, {session_key: "a", match_id: "playing", court_id: "c1", resolution: "not_played"});
    await click(tree, "Round robins"); assert.match(text(tree.toJSON()), /0 played · 1 not played · 1 pending/); assert.match(text(tree.toJSON()), /visible queue is not a round limit/);
    await unmount(tree);
  });
  await scenario("queued and reserved not played", async () => {
    detail.workspace.reserved_queue = [game("reserved", "a", {status: "reserved", reserved_court_id: "c2"})];
    detail.workspace.courts[1].next = detail.workspace.reserved_queue[0];
    const tree = await mount(); await click(tree, "Not played"); await click(tree, "Confirm not played");
    assert.equal(requests[0].payload.match_id, "waiting"); assert.ok(!Object.hasOwn(requests[0].payload, "court_id"));
    await click(tree, "Match queue"); await click(tree, "Not played"); await click(tree, "Confirm not played");
    assert.equal(requests[1].payload.match_id, "reserved"); await unmount(tree);
  });
  await scenario("both blank resolves and half blank blocks", async () => {
    const tree = await mount(); await click(tree, "Enter score");
    assert.equal(button(tree, "Mark not played & release court").props.disabled, false);
    await change(tree, "Side A score", "11"); assert.equal(button(tree, "Save score & release court").props.disabled, true);
    await submit(tree, "hub-score-form"); assert.equal(requests.length, 0);
    await change(tree, "Side A score", ""); await submit(tree, "hub-score-form");
    assert.equal(requests[0].payload.score_a, null); assert.equal(requests[0].payload.score_b, null); await unmount(tree);
  });
  await scenario("Moneyball excludes not played", async () => {
    detail.workspace.groups[0].event_format = "moneyball_expectations";
    const tree = await mount(); assert.equal(button(tree, "Not played"), undefined); await click(tree, "Enter score");
    assert.equal(button(tree, "Save score & release court").props.disabled, true); await submit(tree, "hub-score-form"); assert.equal(requests.length, 0); await unmount(tree);
  });
  await scenario("11–10 stale score retains and reviews values", async () => {
    const tree = await mount(); await enterScore(tree); conflictNext = true; await submit(tree, "hub-score-form");
    assert.equal(tree.root.findByProps({"aria-label": "Side A score"}).props.value, "11");
    assert.equal(tree.root.findByProps({"aria-label": "Side B score"}).props.value, "10");
    assert.equal(button(tree, "Save score & release court").props.disabled, true);
    await click(tree, "Reviewed — keep these scores"); await submit(tree, "hub-score-form");
    assert.equal(requests[1].expected_version, 2); assert.notEqual(requests[0].idempotency_key, requests[1].idempotency_key); await unmount(tree);
  });
  await scenario("lost score response exact retry blocks automatic arrivals", async () => {
    const tree = await mount(); await enterScore(tree); loseNext = true; await submit(tree, "hub-score-form");
    detail.registration.automatic_pending_count = 1; await click(tree, "Refresh");
    assert.equal(requests.length, 1); assert.equal(tree.root.findByProps({"aria-label": "Side A score"}).props.disabled, true);
    await click(tree, "Retry saved action"); assert.deepEqual(requests[1], requests[0]); assert.equal(receipts.size, 1); await unmount(tree);
  });
  await scenario("atomic transfer conflict and response loss retain selection", async () => {
    const tree = await mount(); await click(tree, "Move player"); await change(tree, "From round robin", "a"); await change(tree, "Player to move", "move"); await change(tree, "To round robin", "b");
    conflictNext = true; await submit(tree, "hub-move-player-form"); assert.equal(tree.root.findByProps({"aria-label": "Player to move"}).props.value, "move");
    loseNext = true; await submit(tree, "hub-move-player-form");
    assert.deepEqual(requests[1].payload, {source_session_key: "a", participant_id: "move", destination_session_key: "b"});
    await click(tree, "Retry saved action"); assert.deepEqual(requests[2], requests[1]); assert.equal(detail.workspace.groups[1].players.filter(p => p.identity === "guest:move").length, 1); await unmount(tree);
  });
  await scenario("departed history and qualified cross-group court with explicit rejoin", async () => {
    const departed = member("old", "Old Name", {active: false, can_rejoin: true, can_remove: false, membership_status: "departed", inactive_from_round: 2});
    detail.workspace.groups[0].players.push(departed);
    detail.workspace.groups[0].players.find(p => p.participant_id === "move").current_court = {session_key: "b", group_title: "2.5/3.0", court_id: "c2", court_label: "Court 7"};
    const tree = await mount(); await click(tree, "Players");
    assert.equal(tree.root.findAllByProps({"data-participant-id": "old"}).length, 0);
    assert.match(text(tree.toJSON()), /Playing elsewhere on Court 7 · 2.5\/3.0/);
    const history = tree.root.findAllByType("input").find(n => n.props.type === "checkbox" && !n.props["aria-label"]);
    await act(async () => history.props.onChange({target: {checked: true}}));
    assert.match(text(tree.root.findByProps({"data-participant-id": "old"})), /Historical entry/);
    assert.doesNotMatch(text(tree.root.findByProps({"data-participant-id": "old"})), /Waiting|Paused/);
    await click(tree, "Rejoin round robin"); assert.deepEqual(requests[0].payload, {session_key: "a", participant_id: "old"}); await unmount(tree);
  });
  await scenario("arbitrary variants and same-group dedup require confirmation", async () => {
    const tree = await mount(); await click(tree, "Players"); await click(tree, "Link name variants");
    await check(tree, "Link Darla from 3.5/4.0 (move)"); await check(tree, "Link D. Simmons from 3.5/4.0 (variant)");
    assert.equal(button(tree, "Link selected entries").props.disabled, true); await submit(tree, "hub-link-players-form"); assert.equal(requests.length, 0);
    await check(tree, "Confirm selected entries are the same person"); await submit(tree, "hub-link-players-form");
    assert.deepEqual(requests[0].payload, {resolution: "same", participants: [{session_key: "a", participant_id: "move"}, {session_key: "a", participant_id: "variant"}]}); await unmount(tree);
  });
  await scenario("one committed identity allowed and second blocks after refresh", async () => {
    detail.workspace.groups[0].players.find(p => p.participant_id === "move").on_court = true;
    const tree = await mount(); await click(tree, "Players"); await click(tree, "Link name variants");
    await check(tree, "Link Darla from 3.5/4.0 (move)"); await check(tree, "Link D. Simmons from 3.5/4.0 (variant)"); await check(tree, "Confirm selected entries are the same person");
    assert.equal(button(tree, "Link selected entries").props.disabled, false);
    detail.workspace.groups[0].players.find(p => p.participant_id === "variant").reserved = true; detail.hub.version++;
    await click(tree, "Refresh"); assert.equal(button(tree, "Link selected entries").props.disabled, true); await submit(tree, "hub-link-players-form"); assert.equal(requests.length, 0); await unmount(tree);
  });
  await scenario("different profiles cannot merge", async () => {
    detail.workspace.groups[0].players.find(p => p.participant_id === "move").profile_id = 101;
    detail.workspace.groups[0].players.find(p => p.participant_id === "variant").profile_id = 102;
    const tree = await mount(); await click(tree, "Players"); await click(tree, "Link name variants"); await check(tree, "Link Darla from 3.5/4.0 (move)"); await check(tree, "Link D. Simmons from 3.5/4.0 (variant)"); await check(tree, "Confirm selected entries are the same person");
    assert.equal(button(tree, "Link selected entries").props.disabled, true); await unmount(tree);
  });
  await scenario("court move conflict retains target and explicit review", async () => {
    const tree = await mount(); await click(tree, "Move or remove");
    let select = tree.root.findByProps({"data-dialog": "Move or remove match"}).findByType("select");
    await act(async () => select.props.onChange({target: {value: "c2"}})); conflictNext = true; await click(tree, "Move match");
    select = tree.root.findByProps({"data-dialog": "Move or remove match"}).findByType("select"); assert.equal(select.props.value, "c2");
    assert.equal(button(tree, "Move match").props.disabled, true); await click(tree, "Reviewed — keep selected court"); await click(tree, "Move match"); assert.equal(requests[1].payload.target_court_id, "c2"); await unmount(tree);
  });
  await scenario("player editor review retains target replacement and guest name", async () => {
    const tree = await mount(); const court = tree.root.findByProps({"aria-label": "Court 6"}); await act(async () => court.findAllByType("button").find(b => text(b) === "Edit players").props.onClick());
    await change(tree, "Player to replace or swap", "playing-1"); await change(tree, "Replacement", "new");
    const input = tree.root.findByProps({"aria-label": "New player name or club profile"});
    await act(async () => input.props.onTextChange ? input.props.onTextChange("Gibran") : input.props.onChange({target: {value: "Gibran"}}));
    conflictNext = true; await submit(tree, "hub-edit-players-form"); await click(tree, "Reload current players");
    assert.equal(tree.root.findByProps({"aria-label": "Player to replace or swap"}).props.value, "playing-1"); assert.equal(tree.root.findByProps({"aria-label": "Replacement"}).props.value, "new"); assert.equal(tree.root.findByProps({"aria-label": "New player name or club profile"}).props.value, "Gibran");
    await submit(tree, "hub-edit-players-form"); assert.equal(requests[1].payload.name, "Gibran"); await unmount(tree);
  });
  await scenario("focus refresh queued during a mutation drains", async () => {
    const tree = await mount(); await enterScore(tree); holdNext = true; await submit(tree, "hub-score-form"); const reads = detailReads;
    await act(async () => window.dispatchEvent({type: "focus"})); assert.equal(detailReads, reads);
    await act(async () => held()); assert.ok(detailReads > reads); await unmount(tree);
  });
  await scenario("court setup conflict preserves draft until reviewed", async () => {
    const tree = await mount(); await click(tree, "Court setup"); await change(tree, "Court 2 name", "West Court");
    conflictNext = true; await click(tree, "Save court setup");
    assert.equal(tree.root.findByProps({"aria-label": "Court 2 name"}).props.value, "West Court");
    assert.equal(button(tree, "Save court setup").props.disabled, true); await click(tree, "Reviewed — keep court setup"); await click(tree, "Save court setup");
    assert.equal(requests[1].payload.courts[1].label, "West Court"); assert.equal(requests[1].expected_version, 2); await unmount(tree);
  });
  await scenario("same match ID in another group cannot satisfy current court review", async () => {
    const tree = await mount(); const court = tree.root.findByProps({"aria-label": "Court 6"});
    await act(async () => court.findAllByType("button").find(b => text(b) === "Not played").props.onClick());
    detail.workspace.eligible_queue.push(clone(detail.workspace.courts[0].current));
    detail.workspace.courts[0].current = game("playing", "b", {status: "on_court", court_id: "c1"}); detail.hub.version++;
    await click(tree, "Refresh"); assert.equal(button(tree, "Confirm not played").props.disabled, true);
    assert.match(text(tree.toJSON()), /The match or lineup changed/); await unmount(tree);
  });
  for (const oldUnmounted of [true, false]) await scenario(`late A cannot clear B recovery slot (${oldUnmounted ? "unmounted" : "still mounted"})`, async () => {
    const old = await mount(); await enterScore(old); holdNext = true; await submit(old, "hub-score-form");
    assert.ok(held); const releaseA = held; if (oldUnmounted) await unmount(old);
    const current = await mount(); await click(current, "Retry saved action");
    await click(current, "Court setup"); loseNext = true; await click(current, "Save court setup");
    const b = JSON.parse(stored.get("rr-hub-pending:club")); assert.equal(b.body.action, "configure_courts");
    await act(async () => releaseA()); assert.deepEqual(JSON.parse(stored.get("rr-hub-pending:club")), b, "A late reply must leave newer B persisted");
    assert.ok(button(current, "Retry saved action")); await unmount(current); if (!oldUnmounted) await unmount(old);
    const restored = await mount(); await click(restored, "Retry saved action"); assert.deepEqual(requests.at(-1), b.body); assert.equal(receipts.size, 2); assert.equal(stored.has("rr-hub-pending:club"), false); await unmount(restored);
  });
  console.log(`PASS ${passed.length} Hub operations scenarios: ${passed.join("; ")}`);
})().catch(error => { console.error(error); process.exitCode = 1; });
