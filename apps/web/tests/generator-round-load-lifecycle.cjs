const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), { act, create } = require('react-test-renderer');
const h = React.createElement, root = path.resolve(__dirname, '..'), cache = new Map();
global.IS_REACT_ACT_ENVIRONMENT = true;

const Link = ({ children, ...props }) => h('a', props, children);
const SharingProbe = () => null;
const router = { push() {}, replace() {}, refresh() {} };
const mocks = {
  'next/link': Link, '@/components/PublicClubLink': Link,
  'next/navigation': { useRouter: () => router },
  '@/components/interaction': { actionSuccess: () => ({ status: 'success' }), actionUncertain: () => ({ status: 'uncertain' }) },
  '@/components/ConfirmAction': { ConfirmAction: ({ triggerLabel, onConfirm, disabled }) => h('button', { onClick: onConfirm, disabled }, triggerLabel) },
  '@/components/GeneratorSharing': SharingProbe,
  '@/components/GeneratorSubmission': { __esModule: true, default: () => null, generatorResultLabel: () => 'Unrated' },
  '@/components/SearchablePlayerSelect': { __esModule: true, default: () => null }
};
function load(name, parent = root) {
  if (name === '@/components/PublicControls') return require('./helpers/public-controls-module.cjs')(name);
  if (Object.hasOwn(mocks, name)) return mocks[name];
  if (!name.startsWith('@/') && !name.startsWith('.')) return require(name);
  const base = name.startsWith('@/') ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + '.tsx', base + '.ts', path.join(base, 'index.ts')].find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
  if (!file) throw new Error(`Missing module ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function('require', 'module', 'exports', code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const RoundRunner = load('@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner').default;
const fixture = {
  session_key: 'session', title: 'Session load regression', status: 'active', version: '2026-10-20T01:00:00Z',
  generator_kind: 'round_robin', play_format: 'doubles', scoring_mode: 'scored', rating_mode: 'unrated', current_round_number: 1, total_rounds: 7,
  event: {
    generatorKind: 'round_robin', playFormat: 'doubles', scoringMode: 'scored', status: 'active', currentRoundNumber: 1, totalRounds: 7, courtCount: 2,
    participants: Array.from({ length: 8 }, (_, i) => ({ id: `p${i + 1}`, name: `Player ${i + 1}`, roster_order: i + 1 })),
    rounds: [{ number: 1, status: 'active', byeParticipantIds: [], matches: [
      { id: 'r1-c1', court: 1, sideA: ['p1', 'p2'], sideB: ['p3', 'p4'], scoreA: null, scoreB: null },
      { id: 'r1-c2', court: 2, sideA: ['p5', 'p6'], sideB: ['p7', 'p8'], scoreA: null, scoreB: null }
    ] }]
  }
};
const props = { apiBase: 'https://api.test', clubId: 'club', generatorKind: 'round_robin', sessionKey: 'session', roundNumber: 1, shared: true };
const storageKey = key => `shared-generator-edit:club:${key}`;
const labels = ['r1-c1 side A score', 'r1-c1 side B score', 'r1-c2 side A score', 'r1-c2 side B score'];
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const values = tree => labels.map(label => tree.root.findByProps({ 'aria-label': label }).props.value);
const reply = (status = 200, payload = { session: structuredClone(fixture) }) => ({ ok: status < 400, status, json: async () => payload });
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function setup() {
  const pending = [], storage = new Map([[storageKey('session'), 'operator-session'], [storageKey('next-session'), 'operator-next-session']]), removed = [];
  global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => { removed.push(key); storage.delete(key); } };
  global.window = { location: { hash: '', pathname: '/synthetic', search: '' }, history: { replaceState() {} } };
  // Deliberately ignore AbortSignal: completion after cancellation must still be harmless,
  // including cancellation while response.json() is pending.
  global.fetch = (url, options = {}) => {
    const request = { ...deferred(), url, options };
    pending.push(request);
    return request.promise;
  };
  return { pending, storage, removed };
}
const render = nextProps => h(React.StrictMode, null, h(RoundRunner, nextProps));
async function mount() {
  const state = setup();
  // React 18's test renderer does not replay StrictMode effects. Replace the
  // initial route while its fetch is pending to exercise real cleanup/setup.
  await act(async () => { state.tree = create(render({ ...props, roundNumber: 2 })); });
  assert.equal(state.pending.length, 1);
  await act(async () => state.tree.update(render(props)));
  assert.equal(state.pending.length, 2, 'A pending initial route load is replaced by the current route');
  assert.equal(state.pending[0].options.signal.aborted, true, 'Cleanup aborts the obsolete initial load');
  assert.equal(state.pending[1].options.signal.aborted, false);
  return state;
}
async function typeScores(tree) {
  for (let i = 0; i < labels.length; i++) {
    await act(async () => tree.root.findByProps({ 'aria-label': labels[i] }).props.onChange({ target: { value: i % 2 ? '7' : '11' } }));
  }
  assert.deepEqual(values(tree), ['11', '7', '11', '7']);
}

async function lateMountSuccessPreservesDrafts() {
  const { tree, pending } = await mount();
  await act(async () => pending[1].resolve(reply()));
  await typeScores(tree);
  await act(async () => pending[0].resolve(reply()));
  assert.deepEqual(values(tree), ['11', '7', '11', '7'], 'A canceled mount response must not erase typed scores');
  assert.equal(button(tree, 'Save round scores').props.disabled, false);
  await act(async () => { void button(tree, 'Save round scores').props.onClick(); });
  const body = JSON.parse(pending[2].options.body);
  assert.deepEqual(body.scores.map(row => [row.score_a, row.score_b]), [[11, 7], [11, 7]], 'The next real save submits the entered scores');
  assert.equal(body.edit_token, 'operator-session');
  await act(async () => pending[2].resolve(reply()));
  await act(async () => tree.unmount());
}

async function canceledErrorCannotFinishCurrentLoad() {
  const { tree, pending, removed } = await mount();
  await act(async () => pending[0].reject(new Error('obsolete request failed')));
  assert.ok(text(tree.toJSON()).includes('Loading session…'), 'An obsolete error must not end the current loading state');
  assert.ok(!text(tree.toJSON()).includes('obsolete request failed'));
  assert.deepEqual(removed, []);
  await act(async () => pending[1].resolve(reply()));
  assert.equal(button(tree, 'Save round scores').props.disabled, false);
  await act(async () => tree.unmount());
}

async function canceledForbiddenCannotRevokeButCurrentForbiddenCan() {
  const { tree, pending, storage, removed } = await mount();
  const body = deferred();
  await act(async () => pending[0].resolve({ ok: false, status: 403, json: () => body.promise }));
  await act(async () => pending[1].resolve(reply()));
  await typeScores(tree);
  await act(async () => body.resolve({ detail: 'obsolete handoff response' }));
  assert.deepEqual(removed, [], 'Check cancellation after parsing the response, before the 403 side effect');
  assert.equal(storage.get(storageKey('session')), 'operator-session');
  assert.deepEqual(values(tree), ['11', '7', '11', '7']);
  assert.ok(!text(tree.toJSON()).includes('Only the organizer'));
  await act(async () => { void button(tree, 'Save round scores').props.onClick(); });
  assert.equal(JSON.parse(pending[2].options.body).edit_token, 'operator-session');
  await act(async () => pending[2].resolve(reply(403, { detail: 'handoff revoked' })));
  assert.deepEqual(removed, [storageKey('session')], 'A genuine current 403 still removes the revoked token');
  assert.equal(storage.has(storageKey('session')), false);
  assert.ok(text(tree.toJSON()).includes('Only the organizer'));
  await act(async () => tree.unmount());
}

async function refreshReplacementAndContextCleanup() {
  const { tree, pending, removed, storage } = await mount();
  await act(async () => pending[1].resolve(reply()));
  await act(async () => { void tree.root.findByType(SharingProbe).props.onChanged(); });
  const oldRefresh = pending[2];
  const oldBody = deferred();
  await act(async () => oldRefresh.resolve({ ok: false, status: 403, json: () => oldBody.promise }));
  await act(async () => { void tree.root.findByType(SharingProbe).props.onChanged(); });
  assert.equal(oldRefresh.options.signal.aborted, true, 'A new sharing refresh aborts the previous refresh');
  await act(async () => oldBody.resolve({ detail: 'obsolete refresh' }));
  assert.equal(button(tree, 'Saving…').props.disabled, true, 'An obsolete refresh cannot finish a newer load');
  assert.deepEqual(removed, []);
  await act(async () => pending[3].resolve(reply()));
  await typeScores(tree);
  await act(async () => { void tree.root.findByType(SharingProbe).props.onChanged(); });
  const contextRefresh = pending[4];
  await act(async () => tree.update(render({ ...props, sessionKey: 'next-session' })));
  assert.equal(contextRefresh.options.signal.aborted, true, 'Context cleanup also cancels a sharing-triggered refresh');
  assert.equal(pending.length, 6);
  const next = structuredClone(fixture);
  next.session_key = 'next-session'; next.title = 'Next session';
  await act(async () => pending[5].resolve(reply(200, { session: next })));
  assert.deepEqual(values(tree), ['', '', '', ''], 'Drafts from a different session must not carry over even when match IDs repeat');
  await typeScores(tree);
  await act(async () => contextRefresh.resolve(reply(403, { detail: 'old context forbidden' })));
  await act(async () => pending[0].resolve(reply()));
  assert.ok(text(tree.toJSON()).includes('Next session'), 'Older loads cannot replace the new context');
  assert.deepEqual(values(tree), ['11', '7', '11', '7']);
  assert.deepEqual(removed, []);
  assert.equal(storage.get(storageKey('next-session')), 'operator-next-session');
  assert.equal(button(tree, 'Save round scores').props.disabled, false);
  await act(async () => { void tree.root.findByType(SharingProbe).props.onChanged(); });
  await act(async () => tree.unmount());
  assert.equal(pending[6].options.signal.aborted, true, 'Unmount aborts the latest sharing-triggered load');
  await act(async () => pending[6].resolve(reply(403, { detail: 'after unmount' })));
  assert.deepEqual(removed, [], 'An unmounted load cannot revoke session credentials');
}

(async () => {
  await lateMountSuccessPreservesDrafts();
  await canceledErrorCannotFinishCurrentLoad();
  await canceledForbiddenCannotRevokeButCurrentForbiddenCan();
  await refreshReplacementAndContextCleanup();
  console.log('PASS round load lifecycle: replaced initial loads preserve drafts, stale errors and 403s, latest refresh, context cleanup, and genuine token revocation.');
})().catch(error => { console.error(error); process.exitCode = 1; });
