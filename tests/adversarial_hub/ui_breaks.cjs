// Deterministic component regressions: only mock transport, no live API calls.
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const root = path.resolve(__dirname, '../../apps/web');
const requireWeb = require('node:module').createRequire(path.join(root, 'package.json'));
const ts = requireWeb('typescript'), React = requireWeb('react');
const { act, create } = requireWeb('react-test-renderer'), h = React.createElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
let accessToken = 'admin-first';
const cache = new Map(), routes = [];
const Link = ({ children, ...props }) => h('a', props, children);
const SharingProbe = () => null;
const mocks = {
  'next/link': Link, '@/components/PublicClubLink': Link,
  'next/navigation': { useRouter: () => ({ push: value => routes.push(value), replace: value => routes.push(value), refresh() {} }) },
  '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken }) },
  '@/components/GeneratorSharing': SharingProbe,
  '@/components/GeneratorSessionDetails': () => null,
  '@/components/GeneratorLateArrivals': () => null,
  '@/components/GeneratorSubmission': { __esModule: true, default: () => null, generatorResultLabel: () => 'Unrated' },
  '@/components/SearchablePlayerSelect': () => null,
  '@/components/interaction': { actionSuccess: () => ({ status: 'success' }), actionUncertain: () => ({ status: 'uncertain' }) },
  '@/components/ConfirmAction': { ConfirmAction: ({ triggerLabel, onConfirm, disabled }) => h('button', { onClick: onConfirm, disabled }, triggerLabel) },
};
function load(name, parent = root) {
  if (Object.hasOwn(mocks, name)) return mocks[name];
  if (name === '@/components/PublicControls') return require(path.join(root, 'tests/helpers/public-controls-module.cjs'))(name);
  if (!name.startsWith('@/') && !name.startsWith('.')) return requireWeb(name);
  const base = name.startsWith('@/') ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, `${base}.tsx`, `${base}.ts`, path.join(base, 'index.ts')].find(file => fs.existsSync(file) && fs.statSync(file).isFile());
  if (!file) throw new Error(`Missing ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function('require', 'module', 'exports', code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const Admin = load('@/app/admin/play-generators/GeneratorRoundRunner').default;
const Public = load('@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner').default;
const clone = value => structuredClone(value);
const fixture = () => ({
  session_key: 'session', title: 'RR regression', status: 'active', version: 1, generator_kind: 'round_robin', play_format: 'doubles', scoring_mode: 'scored', current_round_number: 1, total_rounds: 7,
  event: { name: 'RR regression', generatorKind: 'round_robin', playFormat: 'doubles', scoringMode: 'scored', status: 'active', currentRoundNumber: 1, totalRounds: 7, courtCount: 2,
    participants: Array.from({ length: 8 }, (_, i) => ({ id: `p${i + 1}`, name: `Player ${i + 1}`, roster_order: i + 1 })),
    rounds: [{ number: 1, status: 'active', matches: [
      { id: 'same-1', court: 1, sideA: ['p1', 'p2'], sideB: ['p3', 'p4'], scoreA: null, scoreB: null },
      { id: 'same-2', court: 2, sideA: ['p5', 'p6'], sideB: ['p7', 'p8'], scoreA: null, scoreB: null },
    ] }],
  },
});
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const reply = (session, status = 200, detail = undefined) => ({ ok: status < 400, status, json: async () => detail === undefined ? { session: clone(session) } : { detail } });
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const input = (tree, side = 'A', id = 'same-1') => tree.root.findByProps({ 'aria-label': `${id} side ${side} score` });
async function enter(state, a = '11', b = '7', id = 'same-1') {
  await act(async () => { input(state.tree, 'A', id).props.onChange({ target: { value: a } }); input(state.tree, 'B', id).props.onChange({ target: { value: b } }); });
}
async function click(state, label) { await act(async () => { const found = button(state.tree, label); assert.ok(found, `Missing ${label}`); assert.equal(found.props.disabled, false, `${label} should be enabled`); void found.props.onClick(); }); }
async function sharing(state) { await act(async () => { void state.tree.root.findByType(SharingProbe).props.onChanged(); }); }
async function unmount(state) { await act(async () => state.tree.unmount()); }
async function setup(mode = 'admin', options = {}) {
  accessToken = 'admin-first';
  const storage = options.storage || new Map([['public-generator-edit:club:session', 'public-token'], ['shared-generator-edit:club:session', 'shared-token']]);
  global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  global.window = { location: { hash: '', pathname: '/test', search: '' }, history: { replaceState() {} } };
  const state = { session: fixture(), requests: [], storage, mode, ...options };
  const Component = mode === 'admin' ? Admin : Public;
  state.props = { apiBase: 'https://mock.invalid', clubId: 'club', generatorKind: 'round_robin', sessionKey: 'session', roundNumber: 1, ...(mode === 'shared' ? { shared: true } : {}), ...(options.props || {}) };
  state.transport = options.transport || (() => reply(state.session));
  global.fetch = (url, options = {}) => {
    assert.ok(url.startsWith('https://mock.invalid/'), 'External hosts are prohibited');
    const request = { url, options, body: options.body ? JSON.parse(options.body) : null };
    state.requests.push(request);
    return Promise.resolve().then(() => state.transport(request));
  };
  state.render = props => h(Component, props || state.props);
  await act(async () => { state.tree = create(state.render()); });
  state.update = async patch => { state.props = { ...state.props, ...patch }; await act(async () => state.tree.update(state.render())); };
  state.writes = () => state.requests.filter(row => row.body);
  return state;
}
const tests = [];
const test = (id, run) => tests.push({ id, run });

test('stale-admin-success', async () => {
  const old = deferred(); const state = await setup('admin', { transport: () => old.promise });
  state.transport = () => reply(state.session); accessToken = 'admin-second'; await state.update({});
  await enter(state); await act(async () => old.resolve(reply({ ...fixture(), title: 'Obsolete' })));
  assert.equal(input(state.tree).props.value, '11'); assert.ok(!text(state.tree.toJSON()).includes('Obsolete')); await unmount(state);
});
test('stale-admin-error', async () => {
  const old = deferred(); const state = await setup('admin', { transport: () => old.promise });
  state.transport = () => reply(state.session); accessToken = 'admin-second'; await state.update({}); await enter(state);
  await act(async () => old.reject(new Error('obsolete failure'))); assert.equal(input(state.tree).props.value, '11'); assert.ok(!text(state.tree.toJSON()).includes('obsolete failure')); await unmount(state);
});
test('auth-refresh-draft', async () => {
  const state = await setup(); await enter(state); const refresh = deferred(); state.transport = () => refresh.promise;
  accessToken = 'admin-refreshed'; await state.update({}); await act(async () => refresh.resolve(reply(state.session)));
  assert.equal(input(state.tree).props.value, '11'); assert.equal(input(state.tree, 'B').props.value, '7'); await unmount(state);
});
test('context-admin-repeat-ids', async () => {
  const state = await setup(); await enter(state); const old = deferred(); state.transport = () => old.promise; await sharing(state);
  state.session = { ...fixture(), session_key: 'next', title: 'Next' }; state.transport = () => reply(state.session); await state.update({ sessionKey: 'next' });
  assert.equal(input(state.tree).props.value, ''); await enter(state, '9', '11'); await act(async () => old.resolve(reply(fixture())));
  assert.equal(input(state.tree).props.value, '9'); assert.ok(text(state.tree.toJSON()).includes('Next')); await unmount(state);
});
test('lineup-refresh-drops-old-draft', async () => {
  const state = await setup(); await enter(state); state.session.event.rounds[0].matches[0].sideA = ['p1', 'p8']; await sharing(state);
  assert.equal(input(state.tree).props.value, '', 'Same match ID with different players must not inherit scores'); await unmount(state);
});
test('deferred-busy-refresh', async () => {
  const initial = deferred(); const state = await setup('admin', { transport: () => initial.promise });
  await state.update({ refreshKey: 1 }); state.transport = () => reply({ ...state.session, version: 2, title: 'Fresh after busy' });
  await act(async () => initial.resolve(reply(state.session))); assert.equal(state.requests.length, 2); assert.ok(text(state.tree.toJSON()).includes('Fresh after busy')); await unmount(state);
});
test('deferred-dirty-refresh', async () => {
  const state = await setup(); await enter(state); await state.update({ refreshKey: 1 }); assert.equal(state.requests.length, 1);
  state.session.title = 'Fresh after clean'; await enter(state, '', ''); assert.equal(state.requests.length, 2); assert.ok(text(state.tree.toJSON()).includes('Fresh after clean')); await unmount(state);
});
for (const mode of ['public', 'shared']) test(`clear-token-${mode}`, async () => {
  const state = await setup(mode); await enter(state); state.session.session_key = 'missing-token'; await state.update({ sessionKey: 'missing-token' });
  assert.equal(button(state.tree, 'Save round scores'), undefined); assert.equal(state.tree.root.findAllByProps({ 'aria-label': 'same-1 side A score' }).length, 0); await unmount(state);
});
for (const mode of ['admin', 'public']) {
  test(`retry-${mode}`, async () => {
    const state = await setup(mode); await enter(state); const lost = deferred(); state.transport = () => lost.promise;
    const save = button(state.tree, 'Save round scores').props.onClick;
    await act(async () => { void save(); void save(); }); assert.equal(state.writes().length, 1);
    await act(async () => lost.reject(new Error('lost response'))); assert.equal(input(state.tree).props.disabled, true);
    const nameField = state.tree.root.findAllByType('label').find(node => text(node).startsWith('New player name')).findByType('input');
    await act(async () => nameField.props.onChange({ target: { value: 'Late Arrival' } }));
    assert.equal(button(state.tree, 'Add player for next round').props.disabled, true, 'Uncertain score outcome also blocks roster writes');
    const first = state.writes()[0]; state.transport = () => reply(state.session); await click(state, 'Retry score save');
    assert.equal(state.writes()[1].options.body, first.options.body); assert.equal(state.writes()[1].url, first.url);
    assert.ok(![...state.storage.keys()].some(key => key.startsWith('generator-score-save:'))); await unmount(state);
  });
  test(`validation-${mode}`, async () => {
    const state = await setup(mode); await enter(state); state.transport = request => request.body ? reply(null, 422, 'Correct the score') : reply(state.session);
    await click(state, 'Save round scores'); const firstKey = state.writes()[0].body.idempotency_key;
    assert.equal(input(state.tree).props.disabled, false); assert.equal(input(state.tree).props.value, '11'); await enter(state, '11', '8'); state.transport = () => reply(state.session); await click(state, 'Save round scores');
    assert.notEqual(state.writes()[1].body.idempotency_key, firstKey); assert.equal(state.writes()[1].body.scores[0].score_b, 8); await unmount(state);
  });
  test(`conflict-${mode}`, async () => {
    const state = await setup(mode); await enter(state); state.transport = () => reply(null, 409, { code: 'RECOVERY_REQUIRED', recovery_required: true, message: 'Recovery required' });
    await click(state, 'Save round scores'); assert.ok(button(state.tree, 'Retry score save')); const first = state.writes()[0];
    state.session.version = 2; state.transport = request => request.body ? reply(null, 409, 'Session changed') : reply(state.session);
    await click(state, 'Retry score save'); assert.equal(state.writes()[1].options.body, first.options.body); assert.equal(input(state.tree).props.disabled, false); assert.equal(input(state.tree).props.value, '11');
    state.transport = () => reply(state.session); await click(state, 'Save round scores'); assert.equal(state.writes()[2].body.expected_version, 2); assert.notEqual(state.writes()[2].body.idempotency_key, first.body.idempotency_key); await unmount(state);
  });
  test(`blank-contract-${mode}`, async () => {
    const state = await setup(mode); await enter(state, '11', ''); await click(state, 'Save round scores'); assert.equal(state.writes().length, 0); assert.ok(text(state.tree.toJSON()).includes('Enter both scores'));
    await enter(state); state.transport = request => {
      if (request.body) { state.session.event.rounds[0].status = 'saved'; state.session.event.rounds[0].matches[0].scoreA = 11; state.session.event.rounds[0].matches[0].scoreB = 7; state.session.event.rounds[0].matches[1].status = 'not_played'; }
      return reply(state.session);
    }; await click(state, 'Save round scores'); assert.deepEqual(state.writes()[0].body.scores[1], { match_id: 'same-2', score_a: null, score_b: null }); assert.ok(text(state.tree.toJSON()).includes('Not played')); await unmount(state);
  });
}
test('retry-remount', async () => {
  for (const mode of ['admin', 'public']) {
    let state = await setup(mode); await enter(state); state.transport = () => Promise.reject(new Error('lost response')); await click(state, 'Save round scores');
    const original = state.writes()[0]; const storage = state.storage; await unmount(state);
    const locked = fixture(); locked.event.rounds[0].status = 'saved'; locked.submission = { status: 'pending' };
    locked.capabilities = { can_manage: false, can_score: false, unavailable_reason: 'Results submitted.' };
    state = await setup(mode, { storage, session: locked });
    await click(state, 'Retry score save'); assert.equal(state.writes()[0].options.body, original.options.body); assert.equal(state.writes()[0].url, original.url); await unmount(state);
  }
});
test('moneyball-and-capability', async () => {
  const moneyball = fixture(); moneyball.event.eventFormat = 'moneyball_expectations'; let state = await setup('admin', { session: moneyball });
  await enter(state); await click(state, 'Save round scores'); assert.equal(state.writes().length, 0); assert.ok(text(state.tree.toJSON()).includes('Moneyball requires')); await unmount(state);
  const restricted = fixture(); restricted.capabilities = { can_manage: false, can_score: false, unavailable_reason: 'Use the organizer link.' }; state = await setup('admin', { session: restricted });
  assert.equal(button(state.tree, 'Save round scores'), undefined); assert.ok(text(state.tree.toJSON()).includes('Use the organizer link.')); await unmount(state);
});
test('clear-saved-round-reason', async () => {
  const session = fixture(); session.event.rounds[0].status = 'saved'; session.event.rounds[0].matches.forEach(match => { match.scoreA = 11; match.scoreB = 8; });
  const state = await setup('admin', { session }); assert.equal(button(state.tree, 'Clear results and skip round').props.disabled, true);
  await act(async () => state.tree.root.findByProps({ 'aria-label': 'Reason for clearing saved round' }).props.onChange({ target: { value: 'Games were not played' } }));
  await click(state, 'Clear results and skip round'); const request = state.writes()[0]; assert.ok(request.url.endsWith('/rounds/1/skip')); assert.equal(request.body.reason, 'Games were not played'); assert.ok(request.body.idempotency_key); assert.equal(request.body.expected_version, 1); await unmount(state);
});
test('late-unmounted-save-preserves-new-pending', async () => {
  const old = await setup(); await enter(old); const delayed = deferred(); old.transport = () => delayed.promise; await click(old, 'Save round scores'); const storage = old.storage;
  await unmount(old); const current = await setup('admin', { storage }); await click(current, 'Retry score save');
  await enter(current, '11', '9'); current.transport = () => Promise.reject(new Error('second lost response')); await click(current, 'Save round scores');
  const second = [...storage.entries()].find(([key]) => key.startsWith('generator-score-save:')); assert.ok(second);
  await act(async () => delayed.resolve(reply(fixture()))); assert.equal(storage.get(second[0]), second[1]); assert.ok(button(current.tree, 'Retry score save')); await unmount(current);
});
(async () => {
  for (const { id, run } of tests) { await run(); console.log(JSON.stringify({ id, classification: 'GUARD_PASSED' })); }
  assert.equal(tests.length, 21); console.log(JSON.stringify({ summary: { total: tests.length, guardsPassed: tests.length, network: 'Mock transport only; external hosts prohibited' } }));
})().catch(error => { console.error(error); process.exitCode = 1; });
