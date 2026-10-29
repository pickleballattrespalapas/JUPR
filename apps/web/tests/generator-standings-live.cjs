const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const h = React.createElement, root = path.resolve(__dirname, ".."), cache = new Map();
global.IS_REACT_ACT_ENVIRONMENT = true;
const stored = new Map(), navigations = [];
global.sessionStorage = { getItem: key => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
const stubs = {
  "next/link": ({ children, ...props }) => h("a", props, children),
  "next/navigation": { useRouter: () => ({ push: path => navigations.push(path), refresh() {} }) },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-staff" }) },
  "@/components/interaction": { InteractionDialog: ({ title, children, actions }) => h("section", { "data-dialog": title }, children, actions) },
  "@/components/GeneratorSessionDetails": props => h("div", { "data-session-details": true, "data-locked": props.locked }),
  "@/components/GeneratorSharing": () => null,
  "@/components/GeneratorSubmission": { __esModule: true, default: () => h("div", { "data-submission": true }), generatorResultLabel: () => "Not submitted" },
  "@/components/GeneratorMoneyballExpectations": () => null,
  "@/components/GeneratorMoneyball": () => null,
  "@/components/GeneratorPlayoff": () => h("div", { "data-playoff": true }),
  "@/components/PlayGeneratorStandingsTable": { __esModule: true, default: ({ rows }) => h("div", { "data-rows": rows }), standingsSortLabel: () => "Total wins" },
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
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports); return module.exports;
}
const Standings = load("@/app/admin/play-generators/GeneratorStandings").default;
const Finish = load("@/app/admin/round-robin-hub/HubFinishDialog").default;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const clone = value => JSON.parse(JSON.stringify(value));
const response = session => ({ ok: true, json: async () => ({ session: clone(session) }) });
const props = { apiBase: "https://api.test", clubId: "club", sessionKey: "session" };
function fixture() {
  const standing = { rank: 1, participantId: "p1", name: "Departed player", matches: 1, wins: 1, losses: 0, pointsFor: 11, pointsAgainst: 7, differential: 4 };
  return { session_key: "session", round_robin_hub_id: "hub", title: "Evening", status: "active", version: "v1", generator_kind: "round_robin", play_format: "doubles", scoring_mode: "scored", current_round_number: 1, total_rounds: 2,
    standings: [standing], provisional_standings: [{ ...standing, matches: 2, wins: 2, pointsFor: 22 }],
    standings_view: { completed_through_round: 1, scored_matches_in_partial_rounds: 1, pending_matches: [{ match_id: "last", round_number: 2, side_a: ["Yakelin", "Gaby"], side_b: ["Stephen", "Marianne"] }], pending_through_round: 2, finish_after_round: 2, final: false },
    event: { currentRoundNumber: 1, totalRounds: 2, rounds: [{ number: 1, status: "saved" }, { number: 2, status: "preview" }] } };
}
async function mount(component = Standings, value = props) { let tree; await act(async () => { tree = create(h(component, value)); }); return tree; }
async function click(tree, label) { const found = button(tree, label); assert.ok(found, label); assert.equal(Boolean(found.props.disabled), false, label); await act(async () => found.props.onClick()); }
async function unmount(tree) { await act(async () => tree.unmount()); }
async function clock(run) {
  const originalSet = global.setTimeout, originalClear = global.clearTimeout, timers = new Map();
  global.setTimeout = (callback, delay, ...args) => { if (delay !== 30_000) return originalSet(callback, delay, ...args); const id = {}; timers.set(id, callback); return id; };
  global.clearTimeout = id => { if (!timers.delete(id)) originalClear(id); };
  try { await run(() => { assert.ok(timers.size); for (const callback of [...timers.values()]) callback(); }); }
  finally { global.setTimeout = originalSet; global.clearTimeout = originalClear; }
  assert.equal(timers.size, 0, "Request timers are cleared");
}

(async () => {
  global.fetch = async () => response(fixture());
  let tree = await mount();
  assert.match(text(tree.toJSON()), /Completed rounds through Round 1/);
  assert.equal(tree.root.findAll(node => node.props["data-rows"])[0].props["data-rows"][0].matches, 1);
  await act(async () => tree.root.findByType("input").props.onChange({ target: { checked: true } }));
  assert.match(text(tree.toJSON()), /Provisional standings — all recorded games/);
  assert.equal(tree.root.findAll(node => node.props["data-rows"])[0].props["data-rows"][0].matches, 2);
  assert.match(text(tree.toJSON()), /Yakelin \/ Gaby vs Stephen \/ Marianne/);
  assert.match(text(tree.toJSON()), /Last round: 2/);
  await unmount(tree);

  for (const phase of ["headers", "body"]) await clock(async expire => {
    global.fetch = async () => phase === "headers" ? new Promise(() => {}) : { ok: true, json: () => new Promise(() => {}) };
    tree = await mount();
    await act(async () => expire());
    assert.match(text(tree.toJSON()), /request timed out/i);
    global.fetch = async () => response(fixture());
    await click(tree, "Refresh standings");
    assert.match(text(tree.toJSON()), /Evening standings/);
    await unmount(tree);
  });

  await clock(async expire => {
    stored.clear();
    let current = fixture(), firstBody, lateReply, posts = 0;
    global.fetch = async (_url, options) => {
      if (!options.body) return response(current);
      posts += 1;
      if (posts === 1) {
        firstBody = options.body;
        current = { ...current, version: "v2", current_round_number: 2 };
        return new Promise(resolve => { lateReply = () => resolve(response(current)); });
      }
      assert.equal(options.body, firstBody, "Retry reuses version and operation key exactly");
      return response(current);
    };
    tree = await mount();
    await click(tree, "Next scheduled round");
    await act(async () => expire());
    assert.match(text(tree.toJSON()), /not yet confirmed/);
    assert.equal(button(tree, "Next scheduled round").props.disabled, true);
    await unmount(tree);
    tree = await mount();
    await click(tree, "Retry saved action");
    assert.equal(posts, 2);
    assert.equal(stored.size, 0);
    await act(async () => lateReply());
    assert.equal(stored.size, 0);
    await unmount(tree);
  });

  let releaseFirst, reads = 0;
  global.fetch = async () => ++reads === 1 ? new Promise(resolve => { releaseFirst = () => resolve(response({ ...fixture(), title: "Old" })); }) : response({ ...fixture(), title: "Newest" });
  tree = await mount();
  await click(tree, "Refresh standings");
  await act(async () => releaseFirst());
  assert.match(text(tree.toJSON()), /Newest standings/);
  assert.doesNotMatch(text(tree.toJSON()), /Old standings/);
  await unmount(tree);

  global.fetch = async () => response({ ...fixture(), status: "completed", exhibition_only: true });
  tree = await mount();
  assert.match(text(tree.toJSON()), /Extra game · exhibition/);
  assert.equal(tree.root.findByProps({ "data-session-details": true }).props["data-locked"], true, "Exhibition settings cannot switch to rated play");
  assert.equal(tree.root.findAllByProps({ "data-submission": true }).length, 0);
  assert.equal(tree.root.findAllByProps({ "data-playoff": true }).length, 0);
  await unmount(tree);

  stored.clear();
  let finishDepartedWrite;
  const navigationCount = navigations.length;
  global.fetch = async (url, options) => {
    if (options.body) return new Promise(resolve => { finishDepartedWrite = () => resolve(response({ ...fixture(), title: "Old session result", version: "v2" })); });
    return response({ ...fixture(), session_key: url.endsWith("/other") ? "other" : "session", title: url.endsWith("/other") ? "Other session" : "Evening" });
  };
  tree = await mount();
  await click(tree, "Next scheduled round");
  assert.equal(stored.size, 1);
  await unmount(tree);
  tree = await mount(Standings, { ...props, sessionKey: "other" });
  await act(async () => finishDepartedWrite());
  assert.match(text(tree.toJSON()), /Other session standings/);
  assert.doesNotMatch(text(tree.toJSON()), /Old session result/);
  assert.equal(navigations.length, navigationCount, "Leaving an in-flight action prevents its late response from navigating the new session");
  assert.equal(stored.size, 1, "Leaving preserves the first session's exact recovery request");
  await unmount(tree);
  stored.clear();

  const group = { session_key: "a", title: "Evening", status: "active", current_round_number: 5, can_finish: true, finish_pending_matches: 1, finish_boundary_round: 5,
    finish_round_options: [{ round_number: 5, pending_matches: 1 }, { round_number: 6, pending_matches: 3 }] };
  const submitted = [];
  const finishProps = { groups: [group], busy: false, disabled: false, pending: false, error: "", onClose() {}, onRetry() {}, onSubmit: (...values) => submitted.push(values) };
  tree = await mount(Finish, finishProps);
  await act(async () => tree.root.findByProps({ "aria-label": "Round robin to finish" }).props.onChange({ target: { value: "a" } }));
  await act(async () => tree.root.findByProps({ "aria-label": "Last round of the night" }).props.onChange({ target: { value: "6" } }));
  assert.match(text(tree.toJSON()), /3 matches still need/);
  await act(async () => tree.root.findByProps({ id: "hub-finish-group-form" }).props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(submitted, [["a", 6]]);
  await act(async () => tree.update(h(Finish, { ...finishProps, groups: [{ ...group, finish_round_options: [{ round_number: 7, pending_matches: 2 }] }] })));
  assert.equal(button(tree, "Finish round robin").props.disabled, true);
  assert.match(text(tree.toJSON()), /A later round has started or the schedule changed/);
  await unmount(tree);
  console.log("PASS: live standings scope, missing scores, header/body deadlines, stale reads, exact replay after reload, exhibition limits, and reviewed last-round selection");
})().catch(error => { console.error(error); process.exitCode = 1; });
