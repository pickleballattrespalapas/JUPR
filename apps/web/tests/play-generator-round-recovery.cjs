const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map(), h = React.createElement;
const navigation = [];
const router = { push: url => navigation.push(url), replace() {}, refresh() {} };
const Link = ({ children, ...props }) => h("a", props, children);
const stubs = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel, description, onConfirm, disabled }) => h("button", { onClick: onConfirm, disabled, "data-description": description }, triggerLabel) },
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
const props = { apiBase: "http://test.local", clubId: "club", generatorKind: "round_robin", sessionKey: "session", roundNumber: 7 };
function initialSession() {
  const participants = Array.from({ length: 4 }, (_, i) => ({ id: `p-${i + 1}`, name: `Player ${i + 1}`, roster_order: i + 1, active_from_round: 1 }));
  return {
    session_key: "session", title: "Evening RR", status: "active", version: 4,
    generator_kind: "round_robin", play_format: "doubles", scoring_mode: "scored", rating_mode: "unrated",
    current_round_number: 8, total_rounds: 8,
    event: { name: "Evening RR", generatorKind: "round_robin", playFormat: "doubles", status: "active", currentRoundNumber: 8, totalRounds: 8, courtCount: 1, participants,
      rounds: [7, 8].map(number => ({ number, status: number === 7 ? "skipped" : "active", byeParticipantIds: [], matches: [
        { id: `r${number}-c1`, court: 1, sideA: ["p-1", "p-4"], sideB: ["p-2", "p-3"], scoreA: null, scoreB: null }
      ] }))
    }
  };
}
async function checkRunner(file, isPublic) {
  const Runner = load(file).default;
  let session = initialSession(), organizer = "test-organizer";
  let requests = [];
  navigation.length = 0;
  global.sessionStorage = { getItem: () => organizer, setItem() {} };
  global.window = { location: { hash: "", pathname: "/test", search: "" }, history: { replaceState() {} } };
  global.fetch = async (url, options = {}) => {
    if (options.body) {
      const body = JSON.parse(options.body);
      requests.push({ url, body });
      assert.equal(Number(body.expected_version), session.version);
      if (isPublic) assert.equal(body.edit_token, organizer);
      const row = session.event.rounds[0];
      if (url.endsWith("/reopen")) { row.status = "active"; session.status = "active"; }
      else if (url.endsWith("/scores")) {
        assert.equal(body.scores[0].score_a, 11); assert.equal(body.scores[0].score_b, 7);
        row.status = "saved"; row.matches[0].scoreA = 11; row.matches[0].scoreB = 7;
      } else if (url.endsWith("/played")) row.status = "played";
      else if (url.endsWith("/advance")) { session.current_round_number = 9; session.event.currentRoundNumber = 9; session.status = "active"; }
      else if (url.endsWith("/complete")) session.status = "completed";
      else throw new Error(`Unexpected mutation ${url}`);
      session.version += 1;
    }
    return { ok: true, status: 200, json: async () => ({ ok: true, session: clone(session) }) };
  };
  let tree;
  const mount = async extra => { await act(async () => { tree = create(h(Runner, { ...props, ...extra })); }); };
  const unmount = async () => { await act(async () => tree.unmount()); };
  await mount();
  const originalGames = clone(session.event.rounds);
  assert.ok(button(tree, "Play this round"));
  await act(async () => button(tree, "Play this round").props.onClick());
  assert.equal(session.current_round_number, 8, "Reopening an old round leaves latest round selected on server");
  assert.deepEqual(session.event.rounds.map(r => r.matches), originalGames.map(r => r.matches));
  assert.ok(tree.root.findAllByType("a").some(a => text(a) === "Current round" && a.props.href.endsWith("/rounds/8")));
  for (const [side, value] of [["A", "11"], ["B", "7"]]) await act(async () => tree.root.findByProps({ "aria-label": `r7-c1 side ${side} score` }).props.onChange({ target: { value } }));
  await act(async () => button(tree, "Save round scores").props.onClick());
  assert.deepEqual(requests.map(r => r.url.split("/").pop()), ["reopen", "scores"]);
  assert.equal(button(tree, "Save round scores"), undefined);
  assert.equal(session.event.rounds[1].status, "active");
  await unmount();

  session = initialSession(); session.scoring_mode = "unscored"; session.event.rounds[0].status = "active"; requests = [];
  await mount();
  await act(async () => button(tree, isPublic ? "Mark round played" : "Round Played").props.onClick());
  assert.equal(session.current_round_number, 8);
  assert.ok(navigation.at(-1).endsWith("/rounds/8"), "Saving a reopened old round returns to current round without advancing it");
  assert.equal(requests.length, 1);
  await unmount();

  session = initialSession(); session.status = "completed"; requests = [];
  await mount();
  assert.ok(button(tree, "Play this round"), "Completed sessions still allow skipped rounds to reopen");
  await act(async () => button(tree, "Keep playing").props.onClick());
  assert.ok(navigation.at(-1).endsWith("/rounds/9"));
  await unmount();

  session = initialSession(); requests = [];
  await mount({ roundNumber: 8 });
  assert.ok(!text(tree.toJSON()).includes("Round 8 of 8"));
  const finish = button(tree, "Finish session");
  assert.equal(finish.props.disabled, false);
  assert.match(finish.props["data-description"], /unfinished round will not count/);
  await act(async () => tree.root.findByProps({ "aria-label": "r8-c1 side A score" }).props.onChange({ target: { value: "3" } }));
  assert.equal(button(tree, "Finish session").props.disabled, true, "Unsaved score draft must not be discarded on finish");
  await unmount();

  for (const locked of [{ submission: { status: "pending" } }]) {
    session = { ...initialSession(), status: "completed", ...locked };
    await mount();
    assert.equal(button(tree, "Play this round"), undefined);
    assert.equal(button(tree, "Keep playing"), undefined);
    await unmount();
  }
  if (isPublic) {
    organizer = ""; session = initialSession(); await mount();
    assert.equal(button(tree, "Play this round"), undefined, "Public viewers have no replay mutation control");
    await unmount();
  }
}
(async () => {
  await checkRunner("@/app/admin/play-generators/GeneratorRoundRunner", false);
  await checkRunner("@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner", true);
  console.log("PASS admin/public RR skipped round recovery, scoring older rounds, resume, finish draft protection, and locks.");
})().catch(error => { console.error(error); process.exit(1); });
