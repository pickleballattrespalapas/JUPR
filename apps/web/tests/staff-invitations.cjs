const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const ts = require('typescript'), React = require('react');
const { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const m = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => Object.hasOwn(mocks, n) ? mocks[n] : require(n), m, m.exports);
  return m.exports;
}
const helpers = load('lib/staffInvitations.ts');
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const id = '00000000-0000-4000-8000-000000000001';
const invitation = { id, club_id: 'beta', email: 'staff@example.test', role: 'operator', scopes: [{ kind: 'program_type', program_type: 'leagues', resource_id: '' }], status: 'pending', expires_at: '2099-01-01', access_expires_at: null };
const link = ({ children, ...p }) => React.createElement('a', p, children);
const button = (tree, label) => tree.root.findAllByType('button').find(b => b.children.includes(label));
global.window = { location: { origin: 'https://staging.example.test', hash: '', href: '' } };
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => id }, configurable: true });
Object.defineProperty(global, 'navigator', { value: {}, configurable: true });

async function staffScreen() {
  let clubId = 'beta', identity = 'admin-b', role = 'administrator', requests = [], finish;
  let rows = [], invitations = [];
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    return reply(url.endsWith('/targets') ? { targets: [] } : url.endsWith('/invitations') ? { invitations } : { staff: rows });
  };
  const Page = load('app/admin/staff/page.tsx', {
    'next/link': link,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'admin-token', loading: false, session: { user: { id: identity }, capabilities: { assignments: [{ club_id: clubId, role }] } } }) },
    '@/lib/adminPlayerEditorApi': { getAdminPlayerEditorApiBaseUrl: () => 'https://api.example.test' },
    '@/lib/staffInvitations': helpers,
    '@/components/ConfirmAction': { ConfirmAction: ({ triggerLabel, onConfirm }) => React.createElement('button', { onClick: onConfirm }, triggerLabel) }
  }).default;
  let tree;
  await act(async () => { tree = create(React.createElement(Page)); });
  assert.ok(requests.every(r => r.url.includes('/clubs/beta/')));
  await act(async () => tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'Staff@Example.Test' } }));
  await act(async () => { const form = tree.root.findByType('form'); void form.props.onSubmit({ preventDefault() {} }); void form.props.onSubmit({ preventDefault() {} }); });
  let writes = requests.filter(r => r.options.method);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].options.method, 'POST');
  assert.ok(writes[0].url.endsWith('/staff/invitations'));
  const sent = JSON.parse(writes[0].options.body);
  assert.equal(sent.email, 'staff@example.test');
  assert.equal(sent.invitation_id, id);
  await act(async () => finish(reply({ detail: 'Could not confirm save. Reload.' }, 503)));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(JSON.parse(requests.at(-1).options.body).invitation_id, id, 'Retry keeps the request ID');
  invitations = [invitation];
  await act(async () => finish(reply({ invitation })));
  assert.ok(button(tree, 'Copy invitation link'));
  assert.equal(tree.root.findByProps({ readOnly: true }).props.value, `https://staging.example.test/admin/accept-invitation?invitation=${id}`);
  await act(async () => button(tree, 'Copy invitation link').props.onClick());
  assert.ok(JSON.stringify(tree.toJSON()).includes('Select and copy'), 'Clipboard fallback works');
  await act(async () => { void button(tree, 'Cancel invitation').props.onClick(); });
  assert.equal(requests.at(-1).url, `https://api.example.test/admin/clubs/beta/staff/invitations/${id}/cancel`);
  invitations = [{ ...invitation, status: 'cancelled' }];
  rows = [{ email: 'current@example.test', role: 'administrator', scopes: [], expires_at: null, revoked_at: null }];
  await act(async () => finish(reply({ invitation: invitations[0] })));
  assert.equal(button(tree, 'Copy invitation link'), undefined);
  await act(async () => button(tree, 'Edit access').props.onClick());
  assert.equal(tree.root.findByProps({ type: 'email' }).props.disabled, true);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(requests.at(-1).options.method, 'PUT', 'Existing access stays editable');
  const signal = requests.at(-1).options.signal, oldFinish = finish;
  clubId = 'alpha'; identity = 'admin-a'; rows = []; invitations = [];
  await act(async () => tree.update(React.createElement(Page)));
  assert.equal(signal.aborted, true, 'Club/account change aborts an old mutation');
  await act(async () => oldFinish(reply({ ok: true })));
  assert.equal(tree.root.findByProps({ type: 'email' }).props.value, '');
  assert.ok(!JSON.stringify(tree.toJSON()).includes('current@example.test'));
  role = 'operator';
  await act(async () => tree.update(React.createElement(Page)));
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => tree.unmount());
}

async function recipientScreen() {
  let requests = [], authorized = [], selected = [], finish;
  const raw = { access_token: 'recipient-token', expires_at: Date.now() + 600000, user: { id: 'recipient', email: invitation.email } };
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (url.endsWith('/sign-in')) return reply({ message: 'Sign-in email is disabled in this test environment.' });
    if (url.endsWith('/accept')) return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/workspaces')) return reply({ workspaces: [{ club_id: 'beta', club_slug: 'beta', club_name: 'Beta Club', roles: ['operator'] }, { club_id: 'alpha', club_slug: 'alpha', club_name: 'Alpha Club', roles: ['administrator'] }] });
    return reply({ invitation, club: { id: 'beta', slug: 'beta', name: 'Beta Club' } });
  };
  const Page = load('app/admin/accept-invitation/AcceptInvitation.tsx', {
    'next/link': link, 'next/navigation': { useSearchParams: () => new URLSearchParams({ invitation: id }) },
    '@/lib/adminAuthClient': {
      getAdminApiBaseUrl: () => 'https://api.example.test', consumeStaffInvitationSession: async () => null,
      signInWithPassword: async () => raw, refreshAdminSession: async s => s,
      authorizeAndSaveAdminSession: async s => { authorized.push(s); return s; }
    },
    '@/lib/adminWorkspace': { selectAdminWorkspace: w => selected.push(w) },
    '@/lib/staffInvitations': helpers, './invitation.module.css': {}
  }).default;
  let tree;
  await act(async () => { tree = create(React.createElement(Page)); });
  await act(async () => tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: invitation.email } }));
  await act(async () => button(tree, 'Email me a sign-in link').props.onClick());
  assert.ok(JSON.stringify(tree.toJSON()).includes('disabled in this test environment'));
  assert.equal(authorized.length, 0);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.ok(JSON.stringify(tree.toJSON()).includes('Beta Club'));
  assert.equal(tree.root.findAllByType('form').length, 0);
  assert.equal(authorized.length, 0, 'Sign-in/review do not grant or persist admin access');
  await act(async () => { void button(tree, 'Accept invitation').props.onClick(); void button(tree, 'Accept invitation').props.onClick(); });
  assert.equal(requests.filter(r => r.url.endsWith('/accept')).length, 1);
  await act(async () => finish(reply({ detail: 'Invitation cancelled.' }, 409)));
  assert.equal(authorized.length, 0, 'Rejected acceptance never activates the session');
  assert.equal(selected.length, 0);
  await act(async () => button(tree, 'Accept invitation').props.onClick());
  await act(async () => finish(reply({ invitation: { ...invitation, status: 'accepted' } })));
  assert.equal(authorized.length, 1);
  assert.equal(selected[0].club_id, 'beta', 'Acceptance opens the invited club, preserving other assignments');
  await act(async () => tree.unmount());
}

async function authLink() {
  process.env.NEXT_PUBLIC_SUPABASE_URL = 'https://auth.example.test';
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'public-test-key';
  let cleaned = '', stored = false, calls = [];
  global.document = { title: 'Invitation' };
  window.location.hash = '#staff_token_hash=test-credential';
  window.location.href = `https://staging.example.test/admin/accept-invitation?invitation=${id}#staff_token_hash=test-credential`;
  window.history = { replaceState: (_state, _title, url) => { cleaned = url; } };
  window.localStorage = { setItem: () => { stored = true; } };
  global.fetch = async (url, options) => {
    assert.equal(cleaned, `/admin/accept-invitation?invitation=${id}`, 'Credential removed before network request');
    calls.push({ url, options });
    return reply({ access_token: 'test-session', user: { email: invitation.email } });
  };
  const auth = load('lib/adminAuthClient.ts');
  const session = await auth.consumeStaffInvitationSession();
  assert.equal(calls[0].url, 'https://auth.example.test/auth/v1/verify');
  assert.deepEqual(JSON.parse(calls[0].options.body), { token_hash: 'test-credential', type: 'email' });
  assert.equal(session.access_token, 'test-session');
  assert.equal(stored, false, 'Pending invitation sessions are not persisted');
}

(async () => {
  await staffScreen(); await recipientScreen(); await authLink();
  console.log('Staff invitations: scoped creation, retries, links, cancellation, editing, account changes, explicit acceptance and private email sign-in passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
