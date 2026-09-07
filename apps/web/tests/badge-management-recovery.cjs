const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
const load = (file, overrides = {}) => {
  const output = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', output)(name => overrides[name] || require(name), mod, mod.exports);
  return mod.exports;
};
const storage = new Map();
global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
global.crypto = require('node:crypto').webcrypto;
const guards = load('lib/useAuthenticatedAutoLoad.ts');
const Panel = load('app/admin/badges/BadgeManagementPanel.tsx', {
  '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'test-token', session: { user: { id: 'admin-test' } } }) },
  '@/lib/useAuthenticatedAutoLoad': guards,
}).default;
const options = { write_enabled: true, players: [{ id: 1, name: 'Test Player' }], seasons: [], recent_awards: [], badges: [{ id: 'good_sport', name: 'Good Sport', available: true, criteria: { honest_calls: 'Honest calls, even when they cost a point.' } }] };
let posts = [], loseFirstResponse = true;
global.fetch = async (url, request) => {
  if (request.method === 'POST') {
    posts.push(JSON.parse(request.body));
    if (loseFirstResponse) { loseFirstResponse = false; throw new Error('Connection lost after save.'); }
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  }
  return { ok: true, status: 200, json: async () => options };
};
let tree;
const mount = async () => { await act(async () => { tree = create(React.createElement(Panel, { apiBase: 'https://staging.invalid', clubId: 'test-club' })); }); };
const fill = async () => {
  await act(async () => {
    tree.root.findAllByType('select')[0].props.onChange({ target: { value: '1' } });
    tree.root.findByProps({ type: 'checkbox' }).props.onChange({ target: { checked: true } });
    tree.root.findByType('textarea').props.onChange({ target: { value: 'Called their own ball out.' } });
    tree.root.findAllByProps({ type: 'date' })[0].props.onChange({ target: { value: '2026-01-01' } });
  });
};
const submit = async () => { await act(async () => { tree.root.findAllByType('form')[0].props.onSubmit({ preventDefault() {} }); }); };
(async () => {
  await mount();
  await fill();
  await submit();
  assert.equal(posts.length, 1);
  assert.equal(storage.size, 1, 'Unknown response retains the exact request');
  const original = posts[0];
  await act(async () => tree.unmount());
  await mount();
  const retry = tree.root.findAllByType('button').find(node => node.children.includes('Retry save'));
  assert.ok(retry, 'A refresh restores the pending save');
  await act(async () => { retry.props.onClick(); });
  assert.deepEqual(posts[1], original, 'Retry cannot invent a second contribution ID');
  assert.equal(storage.size, 0, 'Verified success clears pending recovery');
  await fill();
  await submit();
  assert.notEqual(posts[2].operation_id, original.operation_id, 'Separate contribution gets its own ID');
  assert.equal(posts[2].badge_id, 'good_sport');
  assert.deepEqual(posts[2].criteria, ['honest_calls']);
  await act(async () => tree.unmount());
  options.program = { revision: 8, applied_revision: 8, pending_ties: [{ source_key: 'live:round-robin', source_fingerprint: 'a'.repeat(64), event_name: 'Thursday Round Robin', completed_at: '2026-01-01', leaders: [{player_id:1,name:'Test Player',wins:5,differential:20,points:70},{player_id:2,name:'Other Player',wins:5,differential:20,points:70}] }] };
  global.FormData = class { constructor(form) { this.form = form; } get(key) { return this.form[key]; } };
  loseFirstResponse = true;
  await mount();
  assert.ok(tree.root.findAllByType('legend').some(node => node.children.includes('Thursday Round Robin')));
  await act(async () => { tree.root.findAllByType('form')[0].props.onSubmit({preventDefault(){},currentTarget:{winner_player_id:'1'}}); });
  const decision = posts.at(-1);
  assert.equal(decision.source_key, 'live:round-robin');
  assert.equal(decision.source_fingerprint, 'a'.repeat(64));
  await act(async () => tree.unmount());
  await mount();
  const retryDecision = tree.root.findAllByType('button').find(node => node.children.includes('Retry save'));
  assert.ok(retryDecision, 'Round-robin decision survives refresh');
  await act(async () => { retryDecision.props.onClick(); });
  assert.deepEqual(posts.at(-1), decision, 'Winner retry preserves its original operation and result version');
  await act(async () => tree.unmount());
  console.log('badge management recovery, separate contributions, and round-robin decisions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
