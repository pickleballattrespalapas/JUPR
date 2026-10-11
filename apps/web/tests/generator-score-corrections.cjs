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
  let session, organizer = 'test-organizer', requests, failure = false;
  global.sessionStorage = { getItem: () => organizer, setItem() {} };
  global.window = { location: { hash: '', pathname: '/test', search: '' }, history: { replaceState() {} } };
  const savedSession = () => {
    const value = initialSession(), first = value.event.rounds[0];
    value.event.participants.push(...Array.from({ length: 4 }, (_, i) => ({ id: `p-${i + 5}`, name: `Player ${i + 5}`, roster_order: i + 5 })));
    first.status = 'saved'; first.matches[0].scoreA = 11; first.matches[0].scoreB = 8;
    first.matches.push({ id: 'r7-c2', court: 2, sideA: ['p-5', 'p-6'], sideB: ['p-7', 'p-8'], scoreA: 11, scoreB: 6 });
    return value;
  };
  global.fetch = async (url, options = {}) => {
    if (options.body) {
      const body = JSON.parse(options.body);
      requests.push({ url, body });
      assert.equal(options.method, 'PATCH');
      assert.ok(url.endsWith('/rounds/7/scores'));
      assert.equal(Number(body.expected_version), session.version);
      if (isPublic) assert.equal(body.edit_token, organizer);
      else assert.equal(options.headers.get('Authorization'), 'Bearer test-admin');
      assert.deepEqual(body.scores, [
        { match_id: 'r7-c1', score_a: 8, score_b: 11 },
        { match_id: 'r7-c2', score_a: 11, score_b: 6 },
      ]);
      if (failure) return { ok: false, status: 409, json: async () => ({ detail: 'This session changed. Refresh before continuing.' }) };
      session.event.rounds[0].matches[0].scoreA = 8; session.event.rounds[0].matches[0].scoreB = 11;
      session.version += 1;
    }
    return { ok: true, status: 200, json: async () => ({ ok: true, session: clone(session) }) };
  };
  let tree;
  const mount = async (extra = {}) => { await act(async () => { tree = create(h(Runner, { ...props, ...extra })); }); };
  const unmount = async () => { await act(async () => tree.unmount()); };
  const enter = async (side, value) => act(async () => tree.root.findByProps({ 'aria-label': `r7-c1 side ${side} score` }).props.onChange({ target: { value } }));
  for (const status of ['active', 'completed']) {
    session = savedSession(); session.status = status; session.event.status = status; requests = [];
    await mount();
    assert.ok(button(tree, 'Edit scores'), 'Earlier saved rounds offer score correction');
    assert.equal(tree.root.findAllByProps({ 'aria-label': 'r7-c1 side A score' }).length, 0);
    await act(async () => button(tree, 'Edit scores').props.onClick());
    assert.equal(tree.root.findByProps({ 'aria-label': 'r7-c1 side A score' }).props.value, '11');
    assert.equal(tree.root.findByProps({ 'aria-label': 'r7-c1 side B score' }).props.value, '8');
    assert.equal(button(tree, 'Skip round'), undefined, 'Recorded games cannot be skipped from the editor');
    await enter('A', '4');
    await act(async () => button(tree, 'Cancel').props.onClick());
    assert.equal(requests.length, 0, 'Cancel never writes to the API');
    assert.ok(text(tree.toJSON()).includes('11–8'));
    await act(async () => button(tree, 'Edit scores').props.onClick());
    assert.equal(tree.root.findByProps({ 'aria-label': 'r7-c1 side A score' }).props.value, '11');
    await enter('A', '8'); await enter('B', '11');
    const laterRound = clone(session.event.rounds[1]);
    if (status === 'active') {
      failure = true;
      await act(async () => button(tree, 'Save score changes').props.onClick());
      assert.ok(button(tree, 'Save score changes'), 'Failed save keeps the editor open');
      assert.equal(tree.root.findByProps({ 'aria-label': 'r7-c1 side A score' }).props.value, '8');
      assert.equal(session.event.rounds[0].matches[0].scoreA, 11);
      failure = false;
    }
    await act(async () => button(tree, 'Save score changes').props.onClick());
    assert.equal(button(tree, 'Save score changes'), undefined);
    assert.ok(button(tree, 'Edit scores'));
    assert.ok(text(tree.toJSON()).includes('8–11'));
    const playerOne = tree.root.findAllByType('tr').find(row => row.findAllByType('td')[0]?.children.join('') === 'Player 1');
    assert.equal(text(playerOne.findAllByType('td')[1]), '0', 'Corrected loss updates results table');
    assert.equal(text(playerOne.findAllByType('td')[2]), '1');
    assert.deepEqual(session.event.rounds[1], laterRound);
    assert.equal(session.current_round_number, 8); assert.equal(session.status, status);
    await unmount();
  }
  for (const patch of [
    { submission: { status: 'pending' } }, { submission: { status: 'approved' } },
    isPublic ? { results_locked: true } : { official_publish: { published_match_ids: ['r7-c1'] } },
  ]) {
    session = { ...savedSession(), ...patch }; requests = [];
    await mount(); assert.equal(button(tree, 'Edit scores'), undefined); await unmount();
  }
  session = savedSession(); session.event.playoff = { format: 'top_eight' };
  await mount(); assert.equal(button(tree, 'Edit scores'), undefined, 'Seeded playoff locks earlier rounds'); await unmount();
  session = savedSession(); session.generator_kind = 'ladder'; session.event.generatorKind = 'ladder';
  await mount({ generatorKind: 'ladder' }); assert.equal(button(tree, 'Edit scores'), undefined, 'Dependent ladder rounds stay fixed'); await unmount();
  if (isPublic) {
    organizer = ''; session = savedSession(); await mount();
    assert.equal(button(tree, 'Edit scores'), undefined, 'Public viewers cannot edit'); await unmount();
  }
}
(async () => {
  await checkRunner('@/app/admin/play-generators/GeneratorRoundRunner', false);
  await checkRunner('@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner', true);
  console.log('PASS saved score correction: prefill, cancel, save, standings, errors, completed sessions, organizer access, and locks.');
})().catch(error => { console.error(error); process.exit(1); });
