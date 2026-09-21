const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const base = 'app/admin/interclub/registrations/';
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const registration = load('lib/interclubRegistration.ts'), types = load('lib/interclubPlayerPool.ts');
const windowHelpers = load('lib/interclubRegistrationWindow.ts');
const windowHook = load('lib/useRegistrationWindow.ts', { './interclubRegistrationWindow': windowHelpers });
const resource = load(base + 'usePoolResource.ts', { '@/lib/interclubRegistration': registration });
const common = load(base + 'PoolPanelCommon.tsx', { './playerPool.module.css': {} });
const mocks = { '@/lib/interclubRegistrationWindow': windowHelpers, '@/lib/useRegistrationWindow': windowHook, '@/lib/interclubPlayerPool': types, './usePoolResource': resource, './PoolPanelCommon': common, './playerPool.module.css': {} };
const email = load(base + 'PoolInvitationEmail.tsx', mocks);
const bulk = load(base + 'PoolBulkAdd.tsx', mocks);
const panels = load(base + 'PlayerPoolPanels.tsx', { ...mocks, './PoolInvitationEmail': email, './PoolBulkAdd': bulk });
const nodeText = node => typeof node === 'string' ? node : node.children.map(nodeText).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => nodeText(node) === label);
const text = tree => JSON.stringify(tree.toJSON());
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const root = 'https://api.test/admin/clubs/beta/interclub/registrations/season-1';
const emailRoot = 'https://api.test/admin/clubs/beta/interclub/player-pools/season-1/emails';
const openWindow = { opens_at: '2020-01-01T00:00:00Z', closes_at: '2099-01-01T00:00:00Z', revision: 1, status: 'open', can_register: true, meet_planning_open: false };
const closedWindow = { ...openWindow, closes_at: '2020-02-01T00:00:00Z', status: 'closed', can_register: false, meet_planning_open: true };
const season = { registration: openWindow, id: 'season-1', details: { name: 'Coastal League', divisions: ['3.5'], timezone: 'America/Mazatlan' } };
const member = { id: 'member-1', club_id: 'beta', season_id: 'season-1', name: 'Alex Example', email: 'alex@example.invalid', divisions: ['3.5'], notes: 'Away in January', player_id: null, status: 'active', revision: 2 };
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa' }, configurable: true });

async function seasonPool() {
  let requests = [], finish, pool = { registration: openWindow, signup: { revision: 0, open: true, url: 'https://site.test/interclub/signup/share' }, members: [member], email_mode: 'dry_run' };
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    if (url.includes('/players?')) return reply({ players: [{ id: 'local-1', name: 'Alex Example', starting_rating: 3.5 }], next_offset: null });
    return reply(pool);
  };
  let tree;
  const props = { root, accessToken: 'token-1', clubName: 'Beta Club', season };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  assert.equal(button(tree, 'Invite club players by email').props.disabled, false);
  assert.ok(text(tree).includes('Away in January'));
  assert.equal(button(tree, 'Open season signup'), undefined, 'Clubs cannot open season registration');
  assert.equal(button(tree, 'Close season signup'), undefined, 'Clubs cannot close season registration');
  assert.equal(requests.filter(row => row.options.method).length, 0, 'Opening the pool does not alter commissioner registration dates');
  assert.ok(text(tree).includes('https://site.test/interclub/signup/share'));
  assert.equal(tree.root.findByProps({ children: 'Open signup page' }).props.href, pool.signup.url);
  assert.ok(button(tree, 'Add players'), 'Admin can add verbal commitments without opening public signup');
  assert.equal(button(tree, 'Invite club players by email').props.disabled, false);
  await act(async () => button(tree, 'Link club player').props.onClick());
  assert.ok(requests.at(-1).url.startsWith(root + '/players?q=Alex%20Example'), 'Player links search only the represented club');
  const count = requests.length;
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, accessToken: 'token-2' })));
  assert.equal(requests.length, count, 'Token refresh retains current state');
  await act(async () => button(tree, 'Link Alex Example').props.onClick());
  assert.equal(requests.at(-1).options.headers.Authorization, 'Bearer token-2');
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { expected_revision: 2, player_id: 'local-1', status: 'active' });
  await act(async () => finish(reply({ detail: 'Signup changed. Reload.' }, 409)));
  assert.ok(text(tree).includes('Signup changed'));
  assert.equal(button(tree, 'Withdraw from season pool').props.disabled, true, 'Conflict blocks further changes');
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  await act(async () => button(tree, 'Withdraw from season pool').props.onClick());
  const signal = requests.at(-1).options.signal, oldFinish = finish;
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, root: root.replace('/beta/', '/gamma/') })));
  assert.equal(signal.aborted, true, 'Changing club aborts pending profile mutations');
  await act(async () => oldFinish(reply({ detail: 'STALE CLUB ERROR' }, 400)));
  assert.ok(!text(tree).includes('STALE CLUB ERROR'));
  await act(async () => tree.unmount());
}

async function availability() {
  const meet = { id: 'meet-1', starts_at: '2099-02-20T18:00:00Z' };
  const responses = [
    { id: 'r1', member_id: 'm1', name: 'Available Player', email: 'one@example.invalid', player_id: 'p1', status: 'available', member_status: 'active', divisions: ['3.5'], response_url: 'https://site.test/interclub/respond#token=private-one' },
    { id: 'r2', member_id: 'm2', name: 'Withdrawn Player', email: 'two@example.invalid', player_id: 'p2', status: 'available', member_status: 'withdrawn', divisions: ['3.5'], response_url: 'https://site.test/interclub/respond#token=private-two' },
    { id: 'r3', member_id: 'm3', name: 'Waiting Player', email: 'three@example.invalid', player_id: null, status: 'invited', member_status: 'active', divisions: ['3.5'], response_url: 'https://site.test/interclub/respond#token=private-three' },
  ];
  let state = { settings: { revision: 0, open: false, deadline: null }, responses, email_mode: 'dry_run' }, requests = [], observed;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method === 'PUT') { state = { ...state, settings: { revision: 1, open: true, deadline: '2099-02-19T12:00:00Z' } }; return reply(state); }
    return reply(state);
  };
  let tree; const props = { meetRoot: root + '/meets/meet-1', accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: closedWindow }, meet, onResponses: rows => { observed = rows; } };
  await act(async () => { tree = create(React.createElement(panels.MeetAvailability, props)); });
  assert.equal(observed, responses);
  assert.ok(text(tree).includes('1 available · 0 maybe · 0 unavailable · 1 not replied'), 'Withdrawn members do not count toward available players');
  assert.ok(!text(tree).includes('token=private-two'), 'Withdrawn player capability is not offered for sharing');
  await act(async () => tree.root.findByProps({ type: 'datetime-local' }).props.onChange({ target: { value: '2099-02-19T12:00' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(JSON.parse(requests.at(-1).options.body).expected_revision, 0);
  assert.equal(JSON.parse(requests.at(-1).options.body).open, true);
  assert.equal(button(tree, 'Choose players to invite').props.disabled, false);
  await act(async () => tree.root.findByType('select').props.onChange({ target: { value: 'available' } }));
  assert.ok(text(tree).includes('Available Player') && !text(tree).includes('Withdrawn Player') && !text(tree).includes('Waiting Player'));
  assert.equal(requests.filter(row => row.options.method).length, 1, 'Opening replies does not send invitations');
  await act(async () => tree.unmount());
}

async function commissionerRegistrationGates() {
  const scheduled = { ...openWindow, opens_at: '2098-01-01T00:00:00Z', status: 'scheduled', can_register: false };
  for (const registrationWindow of [undefined, scheduled, closedWindow, openWindow]) {
    let reads = 0;
    global.fetch = async () => { reads++; return reply({ registration: registrationWindow, signup: { open: true, revision: 1, url: 'https://site.test/interclub/signup/share' }, members: [member, { ...member, id: 'withdrawn', name: 'Withdrawn Player', status: 'withdrawn' }], email_mode: 'dry_run' }); };
    let tree;
    await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: registrationWindow } })); });
    const open = registrationWindow === openWindow;
    assert.equal(button(tree, 'Add players').props.disabled, !open, 'Only commissioner-open registration permits admin additions');
    assert.equal(button(tree, 'Invite club players by email').props.disabled, !open);
    assert.ok(tree.root.findByProps({ children: 'Open signup page' }), 'The signup page stays shareable in every phase');
    assert.equal(button(tree, 'Open season signup'), undefined);
    assert.equal(button(tree, 'Close season signup'), undefined);
    assert.equal(button(tree, 'Withdraw from season pool').props.disabled, false, 'Existing players can still withdraw outside registration');
    assert.equal(button(tree, 'Link club player').props.disabled, false, 'Administrators can still correct profile links');
    await act(async () => tree.root.findByType('select').props.onChange({ target: { value: 'all' } }));
    assert.equal(button(tree, 'Restore season signup').props.disabled, !open, 'Restoring a withdrawn player requires open registration');
    if (!registrationWindow) assert.ok(text(tree).includes('commissioner has not set registration dates yet'));
    await act(async () => tree.unmount());
    if (!open) continue;
    reads = 0;
    await act(async () => { tree = create(React.createElement(panels.MeetAvailability, { meetRoot: root + '/meets/meet-1', accessToken: 'token', clubName: 'Beta Club', season, meet: { id: 'meet-1', starts_at: '2099-02-20T18:00:00Z' } })); });
    assert.equal(reads, 0, 'Meet settings and availability data are not opened during registration');
    assert.equal(tree.root.findAllByType('form').length, 0);
    await act(async () => tree.unmount());
  }
}

async function resumedRegistrationBoundary() {
  const originalNow = Date.now, previousWindow = global.window;
  let clock = Date.parse('2050-01-01T00:00:00Z'), reads = 0, tree;
  Date.now = () => clock;
  const scheduled = { ...openWindow, opens_at: new Date(clock + 3_600_000).toISOString(), status: 'scheduled', can_register: false };
  const opened = { ...scheduled, status: 'open', can_register: true };
  const focus = new Set();
  global.window = { addEventListener: (event, callback) => { if (event === 'focus') focus.add(callback); }, removeEventListener: (event, callback) => focus.delete(callback) };
  global.fetch = async () => { reads++; return reply({ registration: scheduled, signup: { revision: 1, open: false, url: 'https://site.test/interclub/signup/share' }, members: [], email_mode: 'dry_run' }); };
  try {
    const props = { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: scheduled } };
    await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
    assert.equal(button(tree, 'Add players').props.disabled, true);
    clock = Date.parse(scheduled.opens_at) + 1;
    await act(async () => { for (const listener of [...focus]) listener(); });
    assert.equal(reads, 2, 'Returning after a suspended opening timer rechecks the pool at the missed boundary');
    await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, season: { ...season, registration: opened } })));
    assert.equal(button(tree, 'Add players').props.disabled, false, 'A parent open snapshot supersedes scheduled pool data at the same revision');
    const closed = { ...opened, status: 'closed', can_register: false, meet_planning_open: true };
    assert.equal(windowHelpers.latestRegistrationWindow(opened, closed), closed, 'Equal-revision phases cannot regress from closed to open');
    const revised = { ...scheduled, revision: 2 };
    assert.equal(windowHelpers.latestRegistrationWindow(closed, revised), revised, 'A new commissioner revision overrides prior phase progression');
  } finally {
    if (tree) await act(async () => tree.unmount());
    Date.now = originalNow;
    if (previousWindow === undefined) delete global.window; else global.window = previousWindow;
  }
}

async function invitationEmail() {
  let requests = [], finish, prepared = 0;
  const candidate = { id: 'local-player', name: 'Alex Example', email: 'alex@example.invalid', available: true };
  const audience = { candidates: [candidate, { id: 'no-email', name: 'No Contact', email: '', available: false, unavailable_reason: 'No verified contact' }], delivery_mode: 'dry_run', defaults: { subject: 'Join our season', message: 'Can you play this season?' } };
  const preview = { recipient_count: 1, recipients: [candidate], preview: { subject: 'Join our season', text: 'Hi Alex, join us.', html: '<p>ignored</p>' }, preview_fingerprint: 'fingerprint', delivery_mode: 'dry_run', send_available: true };
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (!options.method) return reply(audience);
    if (url.endsWith('/preview')) return reply(preview);
    if (url === emailRoot) return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/recipients/0/send')) return reply({ index: 0, status: 'dry_run', detail: 'Test invitation prepared; no email was sent.', links: [{ name: 'Alex', url: 'https://site.test/interclub/signup/share' }] });
    throw new Error('Unexpected request: ' + url);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(email.InvitationEmail, { root: emailRoot, accessToken: 'token', kind: 'season', onPrepared: () => prepared++ })); });
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' })[1].props.disabled, true);
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' })[0].props.onChange());
  await act(async () => button(tree, 'Preview invitations').props.onClick());
  assert.ok(button(tree, 'Prepare test invitations'));
  assert.equal(requests.filter(row => row.url.endsWith('/send')).length, 0, 'Preview does not deliver or prepare private invitations');
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Updated message' } }));
  assert.equal(button(tree, 'Prepare test invitations'), undefined, 'Editing after preview invalidates the reviewed message');
  await act(async () => button(tree, 'Preview invitations').props.onClick());
  await act(async () => { void button(tree, 'Prepare test invitations').props.onClick(); void button(tree, 'Prepare test invitations').props.onClick(); });
  assert.equal(requests.filter(row => row.url === emailRoot).length, 1, 'Repeated send click cannot create two batches');
  assert.equal(JSON.parse(requests.at(-1).options.body).preview_fingerprint, 'fingerprint');
  assert.deepEqual(JSON.parse(requests.at(-1).options.body).recipient_ids, ['local-player']);
  await act(async () => finish(reply({ operation_key: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', recipients: [{ index: 0, name: 'Alex', email: 'alex@example.invalid', status: 'pending' }], pending_count: 1, delivery_mode: 'dry_run' })));
  assert.equal(prepared, 1);
  assert.ok(text(tree).includes('Test invitation prepared; no email was sent.'));
  assert.ok(text(tree).includes('https://site.test/interclub/signup/share'), 'Dry-run results expose a working authenticated test link');
  assert.equal(requests.filter(row => row.url.endsWith('/send')).length, 1);
  assert.equal(button(tree, 'Send invitations'), undefined);
  await act(async () => tree.unmount());
}
async function recoverSavedInvitations() {
  const key = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', storage = new Map([[`pcs_interclub_invitation:${emailRoot}:meet:meet-1`, key]]);
  global.window = { sessionStorage: { getItem: k => storage.get(k), setItem: (k, value) => storage.set(k, value), removeItem: k => storage.delete(k) } };
  const audience = { candidates: [], delivery_mode: 'dry_run', defaults: { subject: 'Meet invitation', message: 'Please reply' } };
  const saved = { operation_key: key, pending_count: 1, delivery_mode: 'dry_run', recipients: [
    { index: 0, name: 'Already done', status: 'dry_run', detail: 'No email sent' },
    { index: 1, name: 'Uncertain', status: 'uncertain', detail: 'Check before sending again' },
    { index: 2, name: 'Pending', status: 'pending' },
  ] };
  let requests = [], prepared = 0;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (url.includes('/audience?')) return reply(audience);
    if (url.endsWith('/recipients/2/send')) return reply({ index: 2, status: 'dry_run', detail: 'Test invitation prepared; no email was sent.' });
    return reply(saved);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(email.InvitationEmail, { root: emailRoot, accessToken: 'token', kind: 'meet', meetId: 'meet-1', onPrepared: () => prepared++ })); });
  assert.ok(requests[0].url.endsWith('/audience?kind=meet&meet_id=meet-1'));
  assert.equal(tree.root.findByType('fieldset').props.disabled, true, 'Saved batch prevents accidentally starting a new one');
  await act(async () => button(tree, 'Check saved invitations').props.onClick());
  await act(async () => button(tree, 'Continue pending invitations').props.onClick());
  assert.deepEqual(requests.filter(row => row.options.method).map(row => row.url), [`${emailRoot}/${key}/recipients/2/send`], 'Recovery resumes only pending recipients, never uncertain or completed sends');
  assert.equal(prepared, 1);
  assert.equal(storage.size, 0);
  await act(async () => tree.unmount());
  delete global.window;
}

async function bulkAdd() {
  assert.deepEqual(types.parsePoolPlayerList('Name, Email\nAlex Garcia\nPat Jones, pat@example.com\nSam Lee <sam@example.com>\n"Doe, Jane",jane@example.com').members,
    [{ name: 'Alex Garcia' }, { name: 'Pat Jones', email: 'pat@example.com' }, { name: 'Sam Lee', email: 'sam@example.com' }, { name: 'Doe, Jane', email: 'jane@example.com' }]);
  assert.equal(types.parsePoolPlayerList('missing@example.com').errors.length, 1);
  assert.equal(types.parsePoolPlayerList('Name, bad@address').errors.length, 1);
  assert.equal(types.parsePoolPlayerList('Name, first@example.com, second@example.com').errors.length, 1);
  assert.deepEqual(types.sortedPoolDivisions(['4.0', '3.0', '4.5', '3.5']), ['3.0', '3.5', '4.0', '4.5']);
  assert.equal(types.poolRating(3.55555), '3.56');
  assert.equal(types.poolRating(null), 'Not rated');
  let requests = [], added, finish;
  const player = { id: 'p1', name: 'Club Player', rating: 3.5523, gender: 'female', eligible_divisions: ['3.5'] };
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (!options.method) return reply({ players: [player, { ...player, id: 'existing', name: 'Already Added' }], next_offset: null });
    const entries = JSON.parse(options.body).members;
    if (url.endsWith('/bulk-preview')) {
      const rows = entries.map((entry, index) => ({ ...entry, index, email: entry.email || '', candidates: entry.name === 'Ambiguous Name' ? [{ ...player, id: 'p2', name: 'Ambiguous Name' }] : [], rating: entry.player_id ? player.rating : null, gender: null, eligible_divisions: entry.player_id ? ['3.5'] : [], status: entry.name === 'Already Added' ? 'duplicate' : entry.name === 'Ambiguous Name' && !Object.hasOwn(entry, 'player_id') ? 'ambiguous' : entry.player_id ? 'matched' : 'new' }));
      return reply({ rows, ready_count: rows.filter(row => ['new', 'matched'].includes(row.status)).length, duplicate_count: rows.filter(row => row.status === 'duplicate').length, ambiguous_count: rows.filter(row => row.status === 'ambiguous').length });
    }
    if (url.endsWith('/bulk-add')) return new Promise(resolve => { finish = resolve; });
    throw new Error('Unexpected bulk request ' + url);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(bulk.PoolBulkAdd, { registration: openWindow, root, accessToken: 'token', divisions: ['3.5'], members: [{ ...member, player_id: 'existing' }], onClose() {}, onAdded: result => { added = result; } })); });
  const choices = tree.root.findAllByProps({ type: 'checkbox' });
  assert.equal(choices[1].props.disabled, true, 'Already pooled profiles cannot be selected twice');
  await act(async () => choices[0].props.onChange());
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'New Player\nAmbiguous Name\nAlready Added' } }));
  await act(async () => button(tree, 'Preview 4 players').props.onClick());
  assert.deepEqual(JSON.parse(requests.at(-1).options.body).members[0], { player_id: 'p1', name: 'Club Player', divisions: ['3.5'] });
  assert.equal(button(tree, 'Add 2 players to pool').props.disabled, true, 'Ambiguous names must be resolved before saving');
  assert.ok(text(tree).includes('Already in pool · skipped'));
  assert.ok(text(tree).includes('New signup · needs player link'));
  await act(async () => tree.root.findByType('select').props.onChange({ target: { value: 'p2' } }));
  assert.equal(JSON.parse(requests.at(-1).options.body).members[2].player_id, 'p2', 'Ambiguous row resolves to the selected club profile');
  assert.equal(button(tree, 'Add 3 players to pool').props.disabled, false);
  await act(async () => button(tree, 'Preview 4 players').props.onClick());
  assert.equal(Object.hasOwn(JSON.parse(requests.at(-1).options.body).members[2], 'player_id'), false, 'Pasted names omit player_id to request automatic profile matching');
  assert.ok(text(tree).includes('None of these — add as a new signup'));
  await act(async () => tree.root.findByType('select').props.onChange({ target: { value: '__new__' } }));
  assert.equal(JSON.parse(requests.at(-1).options.body).members[2].player_id, null, 'Explicit none-of-these choice opts out of automatic profile linking');
  assert.equal(button(tree, 'Add 3 players to pool').props.disabled, false, 'An unmatched new signup can be saved after declining profile suggestions');
  assert.ok(text(tree).includes('Email invitations are optional.'), 'Verbal availability can be used directly in lineups');
  const addButton = button(tree, 'Add 3 players to pool');
  await act(async () => { void addButton.props.onClick(); void addButton.props.onClick(); });
  assert.equal(requests.filter(row => row.url.endsWith('/bulk-add')).length, 1, 'Repeated clicks create one batch');
  assert.ok(!Object.hasOwn(JSON.parse(requests.at(-1).options.body).members[1], 'email'), 'Email remains optional for new verbal commitments');
  await act(async () => finish(reply({ added_count: 3, skipped_count: 1, pool: { members: [], signup: {}, email_mode: 'dry_run' } })));
  assert.equal(added.added_count, 3);
  assert.equal(requests.some(row => row.url.includes('/emails')), false, 'Adding commitments does not invoke email delivery');
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Different Player' } }));
  assert.equal(button(tree, 'Add 3 players to pool'), undefined, 'Changing a batch clears the reviewed preview');
  const beforeClose = requests.length, originalNow = Date.now;
  try {
    Date.now = () => Date.parse('2100-01-01T00:00:00Z');
    await act(async () => button(tree, 'Preview 2 players').props.onClick());
    assert.equal(requests.length, beforeClose, 'A stale bulk preview click is blocked after the closing boundary');
    assert.ok(text(tree).includes('Season registration has closed.'));
  } finally { Date.now = originalNow; }
  await act(async () => tree.unmount());
}
(async () => { await seasonPool(); await availability(); await commissionerRegistrationGates(); await resumedRegistrationBoundary(); await invitationEmail(); await recoverSavedInvitations(); await bulkAdd(); console.log('PASS interclub player pool: scoped signup/linking, commissioner registration phases, suspended boundary recovery, revision conflicts, stale responses, availability, bulk player matching and optional emails, email preview and dry-run delivery'); })().catch(error => { console.error(error); process.exit(1); });
