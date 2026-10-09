const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const React = require("react");
const ts = require("typescript");
const { create, act } = require("react-test-renderer");
const h = React.createElement;
const Link = ({ children, ...props }) => h("a", props, children);
const routes = [];
const router = { push: path => routes.push(path), refresh() {} };

function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const playoffModule = load("components/GeneratorPlayoff.tsx", { "@/components/PublicClubLink": Link });
const seeds = Array.from({ length: 13 }, (_, i) => ({ seed: i + 1, participantId: `p-${i + 1}`, name: `Player ${i + 1}` }));
const match = (label, a, b) => ({ label, sideA: a.map(i => `p-${i}`), sideB: b.map(i => `p-${i}`) });
const options = {
  sourceRound: 3, seeds,
  formats: [
    { format: "groups_of_four", reason: null, matches: [match("Places 1–4", [1, 4], [2, 3]), match("Places 5–8", [5, 8], [6, 7]), match("Places 9–12", [9, 12], [10, 11])], sitOutParticipantIds: ["p-13"] },
    { format: "top_eight", reason: null, matches: [match("Semifinal 1", [1, 8], [4, 5]), match("Semifinal 2", [2, 7], [3, 6])], sitOutParticipantIds: seeds.slice(8).map(s => s.participantId) }
  ]
};
const text = tree => JSON.stringify(tree.toJSON());
const button = (tree, label) => tree.root.findAllByType("button").find(node => node.children.includes(label));

async function choiceAndVisibility() {
  const calls = [];
  let tree;
  const props = { event: {}, options, canManage: true, locked: false, busy: false, onStart: async f => calls.push(f), roundHref: r => `/rounds/${r}` };
  await act(async () => { tree = create(h(playoffModule.default, props)); });
  assert.equal(button(tree, "Play Final"), undefined);
  await act(async () => button(tree, "Playoff").props.onClick());
  assert.match(text(tree), /#1 Player 1 \/ #4 Player 4/);
  assert.match(text(tree), /#5 Player 5 \/ #8 Player 8/);
  assert.match(text(tree), /#13 Player 13/);
  await act(async () => button(tree, "Start playoff").props.onClick());
  assert.deepEqual(calls, ["groups_of_four"]);
  await act(async () => tree.root.findByType("select").props.onChange({ target: { value: "top_eight" } }));
  assert.match(text(tree), /#1 Player 1 \/ #8 Player 8/);
  assert.match(text(tree), /#4 Player 4 \/ #5 Player 5/);
  await act(async () => button(tree, "Start playoff").props.onClick());
  assert.deepEqual(calls, ["groups_of_four", "top_eight"]);
  for (const changed of [{ canManage: false }, { locked: true }]) {
    await act(async () => tree.update(h(playoffModule.default, { ...props, ...changed })));
    assert.equal(button(tree, "Playoff"), undefined);
  }
  await act(async () => tree.update(h(playoffModule.default, { ...props, options: { ...options, formats: options.formats.map(o => ({ ...o, reason: "Save or skip the current round." })) } })));
  assert.equal(button(tree, "Start playoff").props.disabled, true);
  await act(async () => tree.unmount());
}

async function standingsRequest(admin, fail = false) {
  const mocks = {
    "next/link": Link, "@/components/PublicClubLink": Link, "next/navigation": { useRouter: () => router },
    "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "admin-token" }) },
    "@/components/GeneratorPlayoff": playoffModule,
    "@/components/GeneratorSubmission": load("components/GeneratorSubmission.tsx"),
    "@/components/PlayGeneratorStandingsTable": { __esModule: true, default: () => h("table"), standingsSortLabel: () => "Total wins" }
  };
  const Component = load(admin ? "app/admin/play-generators/GeneratorStandings.tsx" : "app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx", mocks).default;
  let session = { session_key: "session", title: "RR", status: "active", version: admin ? "2026-10-09T02:00:00+00:00" : 4,
    generator_kind: "round_robin", play_format: "doubles", scoring_mode: "scored", current_round_number: 3,
    total_rounds: 5, event: { rounds: [{ number: 3, status: "saved" }] }, playoff_options: options, standings: [] };
  const requests = [];
  global.sessionStorage = { getItem: () => "organizer-token", setItem() {} };
  global.window = { location: { hash: "", pathname: "/standings", search: "" }, history: { replaceState() {} } };
  global.fetch = async (url, init = {}) => {
    if (init.body) {
      const body = JSON.parse(init.body);
      requests.push({ url, init, body });
      if (fail) return { ok: false, status: 409, json: async () => ({ detail: "This session changed." }) };
      session = { ...session, current_round_number: 4, event: { ...session.event, playoff: { format: body.playoff_format, seeds, sitOutParticipantIds: [] } } };
    }
    return { ok: true, status: 200, json: async () => ({ session }) };
  };
  const before = routes.length;
  let tree;
  await act(async () => { tree = create(h(Component, { apiBase: "http://test", clubId: "club", sessionKey: "session" })); });
  await act(async () => button(tree, "Playoff").props.onClick());
  await act(async () => tree.root.findByType("select").props.onChange({ target: { value: "top_eight" } }));
  await act(async () => button(tree, "Start playoff").props.onClick());
  assert.equal(requests[0].body.playoff_format, "top_eight");
  assert.equal(requests[0].body.expected_version, session.version);
  assert.equal(requests[0].body.idempotency_key, admin
    ? "generator-playoff:session:2026-10-09T02:00:00_00:00:top_eight"
    : "generator-playoff:session:4:top_eight");
  assert.match(requests[0].body.idempotency_key, /^[A-Za-z0-9._:-]{8,160}$/);
  if (admin) assert.equal(requests[0].init.headers.Authorization, "Bearer admin-token");
  else assert.equal(requests[0].body.edit_token, "organizer-token");
  if (fail) {
    assert.equal(routes.length, before);
    assert.match(text(tree), /This session changed/);
    await act(async () => button(tree, "Start playoff").props.onClick());
    assert.deepEqual(requests[0].body, requests[1].body, "A retry keeps the same key, version, and format");
  } else assert.match(routes.at(-1), /\/rounds\/4$/);
  await act(async () => tree.unmount());
}

(async () => {
  await choiceAndVisibility();
  for (const admin of [false, true]) { await standingsRequest(admin); await standingsRequest(admin, true); }
  console.log("PASS playoff choices, seed previews, spectators, locked results, organizer/admin requests, navigation, and safe retries.");
})().catch(error => { console.error(error); process.exit(1); });
