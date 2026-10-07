const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const h = React.createElement;
const router = { push() {}, replace() {}, refresh() {} };
const Link = ({ children }) => h("a", null, children);
const stubs = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "@/components/interaction": {},
  "@/components/ConfirmAction": { ConfirmAction: () => null },
  "@/components/GeneratorSubmission": { __esModule: true, default: () => null, generatorResultLabel: () => "Unrated" },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: () => null },
};
function load(name, parent = root) {
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
function initialSession() {
  const participants = Array.from({ length: 8 }, (_, i) => ({ id: `p-${i + 1}`, name: `Original ${i + 1}`, roster_order: i + 1, active_from_round: 1 }));
  participants.push(...Array.from({ length: 3 }, (_, i) => ({ id: `p-new-${i + 1}`, name: `Arrival ${i + 1}`, roster_order: i + 9, active_from_round: 2 })));
  return {
    session_key: "session", title: "Late arrivals", status: "active", version: 4,
    generator_kind: "round_robin", play_format: "doubles", scoring_mode: "scored", rating_mode: "unrated",
    current_round_number: 1, total_rounds: 3,
    event: { name: "Late arrivals", generatorKind: "round_robin", playFormat: "doubles", status: "active", currentRoundNumber: 1, totalRounds: 3, courtCount: 2, participants,
      rounds: [{ number: 1, status: "active", byeParticipantIds: [], matches: [
        { id: "r1-c1", court: 1, sideA: ["p-1", "p-2"], sideB: ["p-3", "p-4"], scoreA: null, scoreB: null },
        { id: "r1-c2", court: 2, sideA: ["p-5", "p-6"], sideB: ["p-7", "p-8"], scoreA: null, scoreB: null }
      ] }]
    }
  };
}

async function checkRunner(file) {
  const Runner = load(file).default;
  let session = initialSession();
  const originalGames = clone(session.event.rounds[0].matches), requests = [];
  global.sessionStorage = { getItem: () => "test-organizer", setItem() {} };
  global.window = { location: { hash: "", pathname: "/test", search: "" }, history: { replaceState() {} } };
  global.fetch = async (url, options = {}) => {
    if (options.body) {
      const body = JSON.parse(options.body);
      requests.push(body);
      assert.equal(Number(body.expected_version), session.version);
      if (body.action === "add") {
        session.event.participants.push({ id: "p-new-4", name: body.name, roster_order: 12, active_from_round: 2 });
      } else if (body.action === "seat_arrivals") {
        assert.deepEqual(body.participant_ids, ["p-new-1", "p-new-2", "p-new-3", "p-new-4"]);
        assert.equal(body.court_number, 3);
        for (const player of session.event.participants) if (body.participant_ids.includes(player.id)) player.active_from_round = 1;
        session.event.rounds[0].matches.push({ id: "r1-c3", court: 3, sideA: body.participant_ids.slice(0, 2), sideB: body.participant_ids.slice(2), scoreA: null, scoreB: null });
        session.event.courtCount = 3;
      } else if (url.endsWith("/scores")) {
        assert.equal(body.scores.length, 3);
        assert.deepEqual(body.scores[0], { match_id: "r1-c1", score_a: 9, score_b: 7 });
        session.event.rounds[0].status = "saved";
      } else throw new Error(`Unexpected mutation ${url}`);
      session.version += 1;
    }
    return { ok: true, status: 200, json: async () => ({ ok: true, session: clone(session) }) };
  };
  let tree;
  await act(async () => { tree = create(h(Runner, { apiBase: "http://test.local", clubId: "club", generatorKind: "round_robin", sessionKey: "session", roundNumber: 1 })); });
  assert.match(text(tree.toJSON()), /Joining Round 2/);
  assert.equal(button(tree, "Start late-arrival game on Court 3"), undefined, "Three arrivals cannot start a doubles game");
  const scoreInput = label => tree.root.findByProps({ "aria-label": label });
  await act(async () => scoreInput("r1-c1 side A score").props.onChange({ target: { value: "9" } }));
  const nameLabel = tree.root.findAllByType("label").find(node => text(node).startsWith("New player name"));
  await act(async () => nameLabel.findByType("input").props.onChange({ target: { value: "Arrival 4" } }));
  await act(async () => button(tree, "Add player for next round").props.onClick());
  assert.equal(scoreInput("r1-c1 side A score").props.value, "9", "Adding a player preserves typed scores");
  assert.deepEqual(session.event.rounds[0].matches, originalGames);
  assert.equal(tree.root.findAllByProps({ type: "checkbox" }).length, 4, "Only queued arrivals can be selected");
  assert.equal(tree.root.findByProps({ "aria-label": "Spare court" }).props.value, 3);
  await act(async () => button(tree, "Start late-arrival game on Court 3").props.onClick());
  assert.equal(scoreInput("r1-c1 side A score").props.value, "9", "Opening a spare court preserves typed scores");
  assert.equal(scoreInput("r1-c3 side A score").props.value, "");
  assert.deepEqual(session.event.rounds[0].matches.slice(0, 2), originalGames);
  assert.equal(tree.root.findAllByProps({ "aria-label": "Late arrivals" }).length, 0);
  for (const [id, value] of [["r1-c1 side B score", "7"], ["r1-c2 side A score", "11"], ["r1-c2 side B score", "4"], ["r1-c3 side A score", "11"], ["r1-c3 side B score", "6"]]) {
    await act(async () => scoreInput(id).props.onChange({ target: { value } }));
  }
  await act(async () => button(tree, "Save round scores").props.onClick());
  assert.equal(requests.length, 3);
  await act(async () => tree.unmount());
}

(async () => {
  await checkRunner("@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner");
  await checkRunner("@/app/admin/play-generators/GeneratorRoundRunner");
  console.log("PASS organizer and admin late arrivals: queue, spare court, preserved drafts, three-game score submission.");
})().catch(error => { console.error(error); process.exit(1); });
