const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const ts = require('typescript'), React = require('react');
const { create, act } = require('react-test-renderer');
function load(file, mocks) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const m = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => Object.hasOwn(mocks, n) ? mocks[n] : require(n), m, m.exports);
  return m.exports;
}
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const fixture = (id = 'beta', status = 'draft') => ({
  club: { id, slug: `${id}-club`, name: `${id} Club`, tagline: '', support_email: '', is_active: false, onboarding_status: status, updated_at: '2026-09-08T05:00:00.123456Z' },
  setup: { missing: ['Contact email'], can_submit: true }
});
let clubId = 'beta', role = 'administrator', identity = 'user-b', accessToken = 'token-b';
let current = fixture(), requests = [], pending;
const events = [];
global.window = { dispatchEvent: e => events.push(e.type) };
global.fetch = async (url, options) => {
  requests.push({ url, options });
  if (options.method === 'PUT') return new Promise(resolve => { pending = resolve; });
  return reply(current);
};
const Page = load('app/admin/club-settings/page.tsx', {
  'next/link': ({ children, ...p }) => React.createElement('a', p, children),
  '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://staging.test' },
  '@/lib/adminWorkspace': { ADMIN_WORKSPACE_DETAILS_CHANGE: 'club-details' },
  '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId }) },
  '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken, loading: false, session: { user: { id: identity }, capabilities: { assignments: [{ club_id: clubId, role }] } } }) },
  './settings.module.css': {}
}).default;
let tree;
const form = () => tree.root.findByType('form');
const button = label => tree.root.findAllByType('button').find(b => b.children.includes(label));
const input = type => type === 'name' ? tree.root.findAllByType('input')[0] : tree.root.findAllByType('input')[1];
(async () => {
  await act(async () => { tree = create(React.createElement(Page)); });
  assert.equal(requests[0].url, 'https://staging.test/admin/clubs/beta/settings');
  assert.equal(input('name').props.value, 'beta Club');
  assert.equal(button('Save and submit for review').props.disabled, true);
  await act(async () => { input('name').props.onChange({ target: { value: 'New club name' } }); input('email').props.onChange({ target: { value: 'club@example.test' } }); });
  accessToken = 'refreshed-token';
  await act(async () => tree.update(React.createElement(Page)));
  assert.equal(requests.length, 1, 'Token refresh does not discard an unsaved form or refetch');
  assert.equal(input('name').props.value, 'New club name');
  await act(async () => { void form().props.onSubmit({ preventDefault() {} }); void form().props.onSubmit({ preventDefault() {} }); });
  assert.equal(requests.length, 2, 'Duplicate saves are blocked while pending');
  let sent = JSON.parse(requests[1].options.body);
  assert.equal(sent.expected_updated_at, '2026-09-08T05:00:00.123456Z', 'Database revision keeps microseconds');
  assert.equal(sent.name, 'New club name');
  assert.equal(requests[1].options.headers.Authorization, 'Bearer refreshed-token');
  assert.ok(!('club_id' in sent) && !('actor_id' in sent));
  await act(async () => pending(reply({ detail: 'Settings changed. Reload before saving.' }, 409)));
  assert.equal(tree.root.findByType('fieldset').props.disabled, true);
  assert.equal(input('name').props.value, 'New club name', 'Conflict preserves the draft for inspection');
  assert.ok(button('Reload club settings'));
  current = fixture(); current.club.updated_at = '2026-09-08T05:01:00.654321Z';
  await act(async () => button('Reload club settings').props.onClick());
  assert.equal(input('name').props.value, 'beta Club');
  await act(async () => input('email').props.onChange({ target: { value: 'club@example.test' } }));
  await act(async () => button('Save and submit for review').props.onClick({ currentTarget: { form: { reportValidity: () => true } } }));
  sent = JSON.parse(requests.at(-1).options.body);
  assert.equal(sent.submit_for_review, true);
  assert.equal(sent.expected_updated_at, current.club.updated_at);
  current = fixture('beta', 'ready_for_review'); current.club.support_email = 'club@example.test';
  await act(async () => pending(reply(current)));
  assert.ok(JSON.stringify(tree.toJSON()).includes('submitted for Super Admin review'));
  assert.equal(button('Save and submit for review').props.disabled, true);
  assert.deepEqual(events, ['club-details'], 'Successful save refreshes club names in navigation');
  await act(async () => input('name').props.onChange({ target: { value: 'Another edit' } }));
  await act(async () => { void form().props.onSubmit({ preventDefault() {} }); });
  const oldSave = pending, oldSignal = requests.at(-1).options.signal;
  clubId = 'alpha'; identity = 'user-a'; current = fixture('alpha');
  await act(async () => tree.update(React.createElement(Page)));
  assert.ok(oldSignal.aborted, 'Changing account or club aborts the old save response');
  await act(async () => oldSave(reply(fixture('beta'))));
  assert.equal(input('name').props.value, 'alpha Club');
  assert.deepEqual(events, ['club-details'], 'An old save cannot report success in another club');
  role = 'operator'; const before = requests.length;
  await act(async () => tree.update(React.createElement(Page)));
  assert.equal(tree.root.findAllByType('form').length, 0);
  assert.equal(requests.length, before, 'Operators do not fetch settings');
  await act(async () => tree.unmount());
  console.log('Club settings: scoped load/save, submission, stale edits, duplicate clicks, token refresh, account changes and operator denial passed.');
})().catch(e => { console.error(e); process.exitCode = 1; });
