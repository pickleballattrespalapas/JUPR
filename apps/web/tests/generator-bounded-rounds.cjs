const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map(), h = React.createElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
const navigation = [];
let refreshes = 0;
const router = { push: url => navigation.push(url), replace() {}, refresh() { refreshes++; } };
const Link = ({ children, ...props }) => h("a", props, children);
const stubs = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel, description, onConfirm, disabled }) => h("button", { onClick: onConfirm, disabled, "data-description": description }, triggerLabel) },
  "@/components/GeneratorSharing": () => null,
  "@/components/GeneratorSubmission": { __esModule: true, default: () => null, generatorResultLabel: () => "Unrated" },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: () => null },
};
function load(name, parent = root) {
  if (name === "@/components/PublicControls") return require("./helpers/public-controls-module.cjs")(name);
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts", path.join(base, "index.ts")].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const clone = value => JSON.parse(JSON.stringify(value));
const props = { apiBase: "http://test.local", clubId: "club", generatorKind: "round_robin", sessionKey: "session", roundNumber: 1 };
function fixture() {
  const match = number => ({ id: `r${number}-c1`, court: 1, sideA: ["p1", "p2"], sideB: ["p3", "p4"], scoreA: null, scoreB: null });
  return { session_key: "session", title: "Evening RR", status: "active", version: 1, generator_kind: "round_robin", play_format: "doubles", scoring_mode: "scored", current_round_number: 1,
    event: { name: "Evening RR", generatorKind: "round_robin", playFormat: "doubles", status: "active", currentRoundNumber: 1, totalRounds: 5, courtCount: 1, adaptiveFromRound: 2,
      participants: Array.from({ length: 8 }, (_, i) => ({ id: `p${i+1}`, name: `Player ${i+1}`, roster_order: i+1 })),
      rounds: [1, 2, 3, 4, 5].map(number => ({ number, status: number === 1 ? "active" : "preview", generationPending: number > 2, matches: number > 2 ? [] : [match(number)], byeParticipantIds: [] }))
    }
  };
}
async function check(file) {
  const Runner = load(file).default;
  const storage = new Map([["public-generator-edit:club:session", "organizer"]]);
  global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, val) => storage.set(key, val), removeItem: key => storage.delete(key) };
  global.window = { location: { hash: "", pathname: "/test", search: "" }, history: { replaceState() {} } };
  let session = fixture(), fail = true, writes = [], reads = 0, tree;
  global.fetch = async (url, options = {}) => {
    if (options.body) {
      const body = JSON.parse(options.body); writes.push({url, body});
      if (fail) return { ok: false, status: 503, json: async () => ({detail: "temporarily busy"}) };
      session.version++;
      if (url.endsWith("/roster")) {
        session.event.participants.filter(p => body.participant_ids.includes(p.id)).forEach(p => p.inactive_from_round = 2);
      } else {
        assert.ok(url.endsWith("/rounds/3/select"));
        session.current_round_number = session.event.currentRoundNumber = 3;
        const round = session.event.rounds[2]; round.generationPending = false; round.status = "active";
        round.matches = [{id:"r3-c1",court:1,sideA:["p1","p2"],sideB:["p3","p4"],scoreA:null,scoreB:null}];
      }
    } else reads++;
    return { ok: true, status: 200, json: async () => ({session: clone(session)}) };
  };
  await act(async () => { tree = create(h(Runner, props)); });
  assert.match(text(tree.toJSON()), /Matchups generated when selected/);
  const change = tree.root.findAllByType("select").find(node => node.props.value === "add");
  await act(async () => change.props.onChange({target:{value:"remove"}}));
  for (const name of ["Player 7", "Player 8"]) await act(async () => tree.root.findByProps({"aria-label":`Remove ${name}`}).props.onChange({target:{checked:true}}));
  const input = () => tree.root.findByProps({"aria-label":"r1-c1 side A score"});
  await act(async () => input().props.onChange({target:{value:"11"}}));
  await act(async () => button(tree, "Apply change for next round").props.onClick());
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0].body.participant_ids, ["p7", "p8"]);
  assert.equal(writes[0].body.action, "remove");
  assert.equal(tree.root.findByProps({"aria-label":"Remove Player 7"}).props.checked, true, "Failed batch keeps selected departures");
  assert.equal(input().props.value, "11", "Failed batch keeps score draft");
  fail = false;
  await act(async () => button(tree, "Apply change for next round").props.onClick());
  assert.equal(writes.length, 2, "Retry sends one atomic batch");
  assert.equal(input().props.value, "11", "Confirmed roster update preserves current-round draft");
  assert.equal(reads, 1, "Applying a confirmed mutation does not reload session data");
  assert.equal(refreshes, 0, "Applying a mutation does not refresh server layouts");
  assert.equal(tree.root.findAllByProps({"aria-label":"Remove Player 7"}).length, 0, "Already departed players disappear from departure choices");
  await act(async () => tree.unmount());
  await act(async () => { tree = create(h(Runner, {...props, roundNumber:3})); });
  assert.ok(button(tree, "Prepare and select this round"));
  assert.equal(button(tree, "Save round scores"), undefined, "Pending round is not scoreable");
  await act(async () => button(tree, "Prepare and select this round").props.onClick());
  assert.equal(writes.length, 3);
  assert.ok(button(tree, "Save round scores"), "Selection makes the materialized round playable");
  assert.equal(button(tree, "Prepare and select this round"), undefined);
  assert.match(navigation.at(-1), /rounds\/3$/);
  assert.equal(refreshes, 0);
  await act(async () => tree.unmount());
  session = fixture(); session.round_robin_hub_id = "hub";
  await act(async () => { tree = create(h(Runner, props)); });
  await act(async () => tree.root.findAllByType("select").find(node => node.props.value === "add").props.onChange({target:{value:"remove"}}));
  assert.equal(tree.root.findAllByType("legend").filter(node => text(node) === "Players leaving").length, 0, "Hub changes retain their own single-player orchestration");
  await act(async () => tree.unmount());
}
(async () => {
  await check("@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner");
  await check("@/app/admin/play-generators/GeneratorRoundRunner");
  console.log("PASS public/admin deferred round preparation, atomic departure batch, failure draft preservation, hub isolation and no redundant route refresh.");
})().catch(error => { console.error(error); process.exitCode = 1; });
