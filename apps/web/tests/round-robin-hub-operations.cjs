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
  } else if (body.action === "assign_next_court") {
    const match = detail.workspace.eligible_queue.find(match => match.session_key === p.session_key && match.match_id === p.match_id);
    const court = detail.workspace.courts.find(court => !court.current && !court.next && court.open && match?.allowed_court_ids.includes(court.id));
    assert.ok(match && court, "Only a ready match with an eligible open court can start");
    court.current = {...match, status: "on_court", court_id: court.id};
    detail.workspace.eligible_queue = detail.workspace.eligible_queue.filter(item => item !== match);
  } else if (body.action === "process_checkins") detail.registration.automatic_pending_count = 0;
  else assert.ok(["resolve_identities", "move_game", "edit_match_players", "set_match_lineup", "remove_players", "create_extra_game", "finish_group", "pause_player", "release_game", "assign_game"].includes(body.action), body.action);
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
async function withRequestClock(run) {
  const originalSetTimeout = global.setTimeout, originalClearTimeout = global.clearTimeout;
  const pendingTimers = new Map();
  global.setTimeout = (callback, delay, ...args) => {
    if (delay !== 30_000) return originalSetTimeout(callback, delay, ...args);
    const timer = {}; pendingTimers.set(timer, () => callback(...args)); return timer;
  };
  global.clearTimeout = timer => { if (!pendingTimers.delete(timer)) originalClearTimeout(timer); };
  try { await run(() => { assert.ok(pendingTimers.size, "An outstanding request has a deadline"); for (const callback of [...pendingTimers.values()]) callback(); }); }
  finally { global.setTimeout = originalSetTimeout; global.clearTimeout = originalClearTimeout; }
  assert.equal(pendingTimers.size, 0, "Every settled request clears its deadline");
}
(async () => {
  await scenario("board renders before automatic arrivals and every blocked action is visibly disabled", async () => {
    detail.registration.automatic_pending_count = 1; holdNext = true;
    const tree = await mount();
    assert.ok(tree.root.findByProps({"aria-label": "Court 6"}), "The successful GET paints the court board before roster processing finishes");
    assert.doesNotMatch(text(tree.toJSON()), /Loading court board/);
    assert.match(text(tree.toJSON()), /Updating new arrivals/);
    assert.equal(button(tree, "Enter score").props.disabled, true);
    assert.equal(button(tree, "Fill open courts").props.disabled, true);
    assert.equal(requests[0].action, "process_checkins");
    assert.deepEqual(JSON.parse(stored.get("rr-hub-pending:club")).body, requests[0]);
    await act(async () => held());
    assert.equal(button(tree, "Enter score").props.disabled, false);
    await enterScore(tree); await submit(tree, "hub-score-form");
    assert.equal(requests[1].action, "score_and_release", "The next enabled staff action is sent, never silently dropped");
    await unmount(tree);
  });
  await scenario("automatic arrival timeout recovers through the exact saved command", async () => withRequestClock(async expire => {
    detail.registration.automatic_pending_count = 1; holdNext = true;
    const tree = await mount();
    const saved = clone(requests[0]);
    await act(async () => expire());
    assert.match(text(tree.toJSON()), /result is not yet confirmed/);
    assert.equal(button(tree, "Refresh").props.disabled, false);
    await click(tree, "Refresh");
    assert.equal(requests.length, 1, "Reading the board does not replace the uncertain automatic command");
    await click(tree, "Retry saved action");
    assert.deepEqual(requests[1], saved);
    assert.equal(receipts.size, 1);
    assert.equal(stored.size, 0);
    await act(async () => held());
    await unmount(tree);
  }));
  await scenario("automatic arrival updates preserve a score draft and require version review", async () => {
    const tree = await mount(); await enterScore(tree, "11", "10");
    detail.registration.automatic_pending_count = 1; holdNext = true;
    await click(tree, "Refresh");
    assert.equal(button(tree, "Save score & release court").props.disabled, true);
    await act(async () => held());
    assert.equal(tree.root.findByProps({"aria-label": "Side A score"}).props.value, "11");
    assert.equal(tree.root.findByProps({"aria-label": "Side B score"}).props.value, "10");
    assert.equal(button(tree, "Save score & release court").props.disabled, true);
    await click(tree, "Reviewed — keep these scores"); await submit(tree, "hub-score-form");
    assert.equal(requests[1].expected_version, 2);
    await unmount(tree);
  });
  await scenario("rejected automatic arrival update stops retrying the same snapshot", async () => {
    detail.registration.automatic_pending_count = 1;
    const originalFetch = global.fetch; let attempts = 0, tree;
    global.fetch = (url, options = {}) => options.body && JSON.parse(options.body).action === "process_checkins"
      ? (attempts++, Promise.resolve({ok: false, status: 422, json: async () => ({detail: "Review the new arrival's group."})}))
      : originalFetch(url, options);
    try {
      tree = await mount();
      assert.equal(attempts, 1); assert.equal(stored.size, 0);
      assert.equal(button(tree, "Enter score").props.disabled, false);
      await act(async () => tickIntervals());
      assert.equal(attempts, 1, "A repeated GET of the same version never creates an automatic retry loop");
      await click(tree, "Update arrivals"); assert.equal(attempts, 2);
    } finally { global.fetch = originalFetch; if (tree) await unmount(tree); }
  });
  await scenario("court polling and scoring avoid redundant event-list reads while discovery stays fresh", async () => {
    const tree = await mount(); const initialListReads = listReads, initialDetailReads = detailReads;
    await act(async () => tickIntervals());
    assert.equal(listReads, initialListReads); assert.equal(detailReads, initialDetailReads + 1);
    await enterScore(tree); await submit(tree, "hub-score-form");
    assert.equal(listReads, initialListReads, "The command snapshot is sufficient after a score");
    await act(async () => window.dispatchEvent({type: "focus"})); assert.equal(listReads, initialListReads + 1);
    await click(tree, "Round robins"); const discoveryReads = listReads;
    await act(async () => tickIntervals()); assert.equal(listReads, discoveryReads + 1);
    await unmount(tree);
  });
  await scenario("automatic arrival conflict refreshes without blindly resubmitting at a newer version", async () => {
    detail.registration.automatic_pending_count = 1; conflictNext = true;
    const tree = await mount();
    assert.equal(requests.length, 1); assert.equal(stored.size, 0);
    assert.equal(detail.hub.version, 2);
    await act(async () => tickIntervals());
    assert.equal(requests.length, 1, "A newer snapshot does not silently retry a rejected automatic write");
    await click(tree, "Update arrivals");
    assert.equal(requests.length, 2); assert.equal(requests[1].expected_version, 2);
    assert.notEqual(requests[1].idempotency_key, requests[0].idempotency_key);
    detail.registration.automatic_pending_count = 1;
    await click(tree, "Refresh");
    assert.equal(requests.length, 3, "Successful recovery restores automatic processing for later arrivals");
    await unmount(tree);
  });
  await scenario("next-round and queue cards name ready, playing, and waiting players and refresh their status", async () => {
    const ready = {ready: true, current_court: null, reserved_court: null, paused: false, waiting_round_number: null, identity_review: false, duplicate_in_match: false};
    const match = game("readiness", "a", {round_number: 4, blockers: ["A player is already assigned to a court."]});
    const players = [...match.side_a, ...match.side_b];
    const names = ["Ana", "Ben", "Cara", "Drew"];
    players.forEach((player, index) => Object.assign(player, {name: names[index], readiness: {...ready}}));
    players[1].readiness = {...ready, ready: false, current_court: {court_id: "c2", court_label: "Court 7", session_key: "b", group_title: "2.5/3.0"}};
    players[2].readiness = {...ready, ready: false, paused: true};
    players[3].readiness = {...ready, ready: false, waiting_round_number: 3};
    const played = game("finished", "a", {round_number: 4, result_status: "played", score_a: 11, score_b: 8});
    [...played.side_a, ...played.side_b].forEach(player => player.readiness = {...ready});
    Object.assign(detail.workspace.groups[0], {next_round_number: 4, next_round_matches: [match, played]});
    detail.workspace.eligible_queue = []; detail.workspace.blocked_queue = [match];
    const tree = await mount(); await click(tree, "Next round plan");
    const plan = () => tree.root.findByProps({"aria-label": "3.5/4.0 next round plan"});
    assert.match(text(plan()), /1 of 4 players ready/);
    for (const label of ["Ana: Ready", "Ben: Playing on Court 7 · 2.5/3.0", "Cara: Paused", "Drew: Waiting for round 3 game"])
      assert.ok(plan().findByProps({"aria-label": label}), label);
    assert.equal(plan().findAllByProps({"aria-label": "Player readiness"}).length, 1, "Completed games show their result, not current player readiness");
    await click(tree, "Match queue");
    assert.ok(tree.root.findByProps({"aria-label": "Ben: Playing on Court 7 · 2.5/3.0"}));
    await click(tree, "Next round plan");
    players.forEach(player => player.readiness = {...ready});
    detail.workspace.eligible_queue = [match]; detail.workspace.blocked_queue = []; match.blockers = [];
    detail.hub.version++;
    await click(tree, "Refresh");
    assert.match(text(plan()), /4 of 4 players ready/);
    for (const name of names) assert.ok(plan().findByProps({"aria-label": `${name}: Ready`}));
    assert.doesNotMatch(text(plan()), /Playing on Court 7|Waiting for round 3 game|Paused/);
    await unmount(tree);
  });
  await scenario("readiness labels distinguish reservations and identity blockers for singles", async () => {
    const match = game("singles", "a", {round_number: 4});
    match.side_a = match.side_a.slice(0, 1); match.side_b = match.side_b.slice(0, 1);
    const [a, b] = [match.side_a[0], match.side_b[0]];
    a.readiness = {ready: false, current_court: null, reserved_court: {court_id: "c1", court_label: "Court 6", session_key: "a", group_title: "3.5/4.0"}, paused: false, waiting_round_number: 3, identity_review: false, duplicate_in_match: false};
    b.readiness = {ready: false, current_court: null, reserved_court: null, paused: false, waiting_round_number: null, identity_review: true, duplicate_in_match: true};
    Object.assign(detail.workspace.groups[0], {next_round_number: 4, next_round_matches: [match]});
    const tree = await mount(); await click(tree, "Next round plan");
    assert.match(text(tree.toJSON()), /0 of 2 players ready/);
    assert.ok(tree.root.findByProps({"aria-label": `${a.name}: Reserved on Court 6`}));
    assert.ok(tree.root.findByProps({"aria-label": `${b.name}: Needs identity review · Duplicate player in match`}));
    await unmount(tree);
  });
  await scenario("whole next-round plan includes byes and completed games and dispatches a ready match early", async () => {
    const ready = game("next-ready", "a", {round_number: 4, plan_locked: true});
    const blocked = game("next-waiting", "a", {round_number: 4, blockers: ["Player is still on Court 6"]});
    const played = game("next-played", "a", {round_number: 4, result_status: "played", score_a: 11, score_b: 8});
    Object.assign(detail.workspace.groups[0], {next_round_number: 4, planned_through_round: 4, plan_revision: 7,
      next_round_matches: [ready, blocked, played], next_round_byes: [{id: "bye", name: "Esteban", identity: "guest:bye"}]});
    detail.workspace.eligible_queue = [ready]; detail.workspace.blocked_queue = [blocked];
    const tree = await mount(); await click(tree, "Next round plan");
    const plan = tree.root.findByProps({"aria-label": "3.5/4.0 next round plan"});
    assert.match(text(plan), /3 matches planned/); assert.match(text(plan), /Byes: Esteban/);
    assert.match(text(plan), /Played · 11–8/); assert.match(text(plan), /Teams set by staff/);
    assert.match(text(plan), /Player is still on Court 6/);
    assert.equal(plan.findAllByType("button").filter(node => text(node) === "Next open court").length, 1);
    await click(tree, "Next open court");
    assert.deepEqual(requests[0].payload, {session_key: "a", match_id: "next-ready"});
    assert.equal(detail.workspace.courts[0].current.round_number, 3, "Earlier round can still be playing");
    assert.equal(detail.workspace.courts[1].current.round_number, 4);
    assert.match(text(tree.toJSON()), /Playing on Court 7/);
    await unmount(tree);
  });
  await scenario("exact team editor submits reviewed teams through durable command recovery", async () => {
    const tree = await mount();
    const court = tree.root.findByProps({"aria-label": "Court 6"});
    await act(async () => court.findAllByType("button").find(node => text(node) === "Edit players").props.onClick());
    await change(tree, "Change players", "exact");
    await change(tree, "Team A player 1", "move");
    loseNext = true; await submit(tree, "hub-edit-players-form");
    assert.equal(requests[0].action, "set_match_lineup");
    assert.deepEqual(requests[0].payload, {session_key: "a", match_id: "playing", round_number: 3, court_id: "c1",
      side_a: ["move", "playing-2"], side_b: ["playing-3", "playing-4"]});
    assert.equal(button(tree, "Save both teams").props.disabled, true);
    await click(tree, "Retry saved action"); assert.deepEqual(requests[1], requests[0]);
    assert.equal(tree.root.findAllByProps({id: "hub-edit-players-form"}).length, 0);
    await unmount(tree);
  });
  await scenario("exact team edit on next-round plan targets that future round", async () => {
    const match = game("next-edit", "a", {round_number: 4});
    detail.workspace.groups[0].players.push(...[...match.side_a, ...match.side_b].map(player => member(player.id, player.name)));
    Object.assign(detail.workspace.groups[0], {next_round_number: 4, next_round_matches: [match], next_round_byes: []});
    detail.workspace.eligible_queue = [match];
    const tree = await mount(); await click(tree, "Next round plan"); await click(tree, "Edit players");
    await change(tree, "Change players", "exact"); await change(tree, "Team A player 1", "move");
    await submit(tree, "hub-edit-players-form");
    assert.equal(requests[0].action, "set_match_lineup");
    assert.deepEqual(requests[0].payload, {session_key: "a", match_id: "next-edit", round_number: 4, court_id: null,
      side_a: ["move", "next-edit-2"], side_b: ["next-edit-3", "next-edit-4"]});
    await unmount(tree);
  });
  await scenario("batch departures submit once and preserve the exact selection after response loss", async () => {
    const tree = await mount(); await click(tree, "Remove player"); await change(tree, "Round robin", "a");
    await check(tree, "Remove Darla"); await check(tree, "Remove D. Simmons");
    loseNext = true; await submit(tree, "hub-remove-player-form");
    assert.equal(requests[0].action, "remove_players");
    assert.deepEqual(requests[0].payload, {session_key: "a", participant_ids: ["move", "variant"]});
    await click(tree, "Retry saved action"); assert.deepEqual(requests[1], requests[0]);
    assert.equal(tree.root.findAllByProps({id: "hub-remove-player-form"}).length, 0);
    await unmount(tree);
  });
  await scenario("cross-division exhibition creation has an explicit policy and retries the same teams", async () => {
    detail.workspace.groups[1].players.push(member("extra", "Fourth Player"));
    const tree = await mount(); await click(tree, "Add extra game");
    for (const [label, session, participant] of [
      ["Extra game team A player 1", "a", "move"], ["Extra game team A player 2", "a", "variant"],
      ["Extra game team B player 1", "b", "other"], ["Extra game team B player 2", "b", "extra"]
    ]) await change(tree, label, JSON.stringify([session, participant]));
    assert.match(text(tree.toJSON()), /Exhibition only/);
    loseNext = true; await submit(tree, "hub-extra-game-form");
    assert.equal(requests[0].action, "create_extra_game");
    assert.deepEqual(requests[0].payload, {side_a: [{session_key: "a", participant_id: "move"}, {session_key: "a", participant_id: "variant"}],
      side_b: [{session_key: "b", participant_id: "other"}, {session_key: "b", participant_id: "extra"}], standings_policy: "exhibition"});
    await click(tree, "Retry saved action"); assert.deepEqual(requests[1], requests[0]);
    assert.equal(tree.root.findAllByProps({id: "hub-extra-game-form"}).length, 0);
    assert.equal(button(tree, "Match queue").props["aria-selected"], true);
    await unmount(tree);
  });
  await scenario("finish-now selection sends an explicit zero boundary", async () => {
    Object.assign(detail.workspace.groups[0], {can_finish: true, finish_pending_matches: 0,
      finish_boundary_round: 0, finish_round_options: [{round_number: 0, pending_matches: 0}]});
    const tree = await mount(); await click(tree, "Finish round robin"); await change(tree, "Round robin to finish", "a");
    await submit(tree, "hub-finish-group-form");
    assert.equal(requests[0].action, "finish_group");
    assert.deepEqual(requests[0].payload, {session_key: "a", finish_after_round: 0});
    await unmount(tree);
  });
  await scenario("stalled board reads end loading and can be refreshed", async () => withRequestClock(async expire => {
    const originalFetch = global.fetch, signals = [];
    let tree;
    global.fetch = (_url, options) => { signals.push(options.signal); return new Promise(() => {}); };
    try {
      tree = await mount();
      assert.match(text(tree.toJSON()), /Loading court board/);
      await act(async () => expire());
      assert.ok(signals.every(signal => signal.aborted));
      assert.doesNotMatch(text(tree.toJSON()), /Loading court board/);
      assert.match(text(tree.toJSON()), /request timed out/);
      assert.equal(stored.size, 0, "A timed-out read never creates an uncertain write");
      global.fetch = originalFetch;
      await click(tree, "Refresh");
      assert.ok(tree.root.findByProps({"aria-label": "Court 6"}));
    } finally { global.fetch = originalFetch; if (tree) await unmount(tree); }
  }));
  for (const phase of ["headers", "body"]) await scenario(`stalled command ${phase} retains the exact request and safely replays`, async () => withRequestClock(async expire => {
    const originalFetch = global.fetch;
    let tree, releaseLate, signal, stallNext = true;
    global.fetch = async (url, options = {}) => {
      const result = await originalFetch(url, options);
      if (!options.body || !stallNext) return result;
      stallNext = false; signal = options.signal;
      const saved = await result.json();
      if (phase === "headers") return new Promise(resolve => { releaseLate = () => resolve(response(saved)); });
      return {...result, json: () => new Promise(resolve => { releaseLate = () => resolve(saved); })};
    };
    try {
      tree = await mount(); await enterScore(tree); await submit(tree, "hub-score-form");
      const saved = JSON.parse(stored.get("rr-hub-pending:club"));
      assert.equal(button(tree, "Checking…").props.disabled, true);
      await act(async () => expire());
      assert.equal(signal.aborted, true);
      assert.match(text(tree.toJSON()), /result is not yet confirmed/);
      assert.equal(button(tree, "Retry saved action").props.disabled, false);
      assert.equal(button(tree, "Refresh").props.disabled, false);
      assert.deepEqual(JSON.parse(stored.get("rr-hub-pending:club")), saved);
      await unmount(tree); tree = await mount();
      await click(tree, "Retry saved action");
      assert.deepEqual(requests.at(-1), saved.body, "Reload and retry preserve payload, expected version, and idempotency key");
      assert.equal(receipts.size, 1, "A committed score is never applied twice");
      assert.equal(stored.size, 0);
      await click(tree, "Court setup"); loseNext = true; await click(tree, "Save court setup");
      const newer = stored.get("rr-hub-pending:club");
      await act(async () => releaseLate());
      assert.equal(stored.get("rr-hub-pending:club"), newer, "Late timed-out responses cannot clear a newer pending action");
      await click(tree, "Retry saved action");
    } finally { global.fetch = originalFetch; if (tree) await unmount(tree); }
  }));
  await scenario("late initial event list preserves a new-event draft", async () => {
    const originalFetch = global.fetch;
    let releaseList, tree;
    global.fetch = (url, options = {}) => url.endsWith("/round-robin-hubs") && !options.body
      ? new Promise(resolve => { releaseList = () => resolve(response({ok: true, hubs: [detail.hub], sessions: []})); })
      : originalFetch(url, options);
    try {
      await act(async () => { tree = create(h(Panel, {apiBase: "http://test.local", clubId: "club"})); });
      assert.ok(releaseList);
      await click(tree, "New event");
      await act(async () => tree.root.findByProps({placeholder: "Friday Pickleball Night"}).props.onChange({target: {value: "Tonight's new event"}}));
      await act(async () => releaseList());
      assert.equal(tree.root.findByProps({placeholder: "Friday Pickleball Night"}).props.value, "Tonight's new event");
      assert.equal(tree.root.findByProps({"aria-label": "Select hub"}).props.value, "");
      assert.equal(detailReads, 0, "A delayed list must not open an existing event over the user's creation form");
    } finally {
      global.fetch = originalFetch;
      if (tree) await unmount(tree);
    }
  });
  await scenario("initial event list still selects an active event without user input", async () => {
    let tree;
    await act(async () => { tree = create(h(Panel, {apiBase: "http://test.local", clubId: "club"})); });
    assert.equal(tree.root.findByProps({"aria-label": "Select hub"}).props.value, detail.hub.id);
    assert.equal(tree.root.findAllByProps({placeholder: "Friday Pickleball Night"}).length, 0);
    await unmount(tree);
  });
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
