const playerSearchModules = require("./helpers/player-search-modules.cjs");
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const base = 'app/admin/interclub/registrations/';
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : (playerSearchModules(name) || require(name)), module, module.exports);
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
const newPlayerFields = load(base + 'PoolNewPlayerFields.tsx', mocks);
const newPlayerForm = load(base + 'PoolCreatePlayer.tsx', { ...mocks, './PoolNewPlayerFields': newPlayerFields });
const late = load(base + 'PoolLateRequest.tsx', { ...mocks, './PoolNewPlayerFields': newPlayerFields });
const panels = load(base + 'PlayerPoolPanels.tsx', { ...mocks, './PoolInvitationEmail': email, './PoolBulkAdd': bulk, './PoolLateRequest': late, './PoolCreatePlayer': newPlayerForm });
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
    global.fetch = async () => { reads++; return reply({ registration: registrationWindow, can_request_late: true, signup: { open: true, revision: 1, url: 'https://site.test/interclub/signup/share' }, members: [member, { ...member, id: 'withdrawn', name: 'Withdrawn Player', status: 'withdrawn' }], email_mode: 'dry_run' }); };
    let tree;
    await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: registrationWindow } })); });
    const open = registrationWindow === openWindow;
    assert.equal(button(tree, 'Add players').props.disabled, !open, 'Only commissioner-open registration permits admin additions');
    assert.equal(button(tree, 'Invite club players by email').props.disabled, !open);
    assert.equal(!!button(tree, 'Create new player'), open, 'Regular inline creation is only exposed while registration is open');
    assert.equal(!!button(tree, 'Request late player'), registrationWindow === closedWindow, 'Late requests require a server-confirmed closed registration window');
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

async function latePlayerRequests() {
  const player = { id: '101', name: 'Late Traveler', rating: 2.9, league_rating: null, gender: 'female', eligible_divisions: ['3.0', '3.5'] };
  const existing = { ...member, name: 'Existing Member', player_id: '102', approval_status: 'approved' };
  let pool = { registration: closedWindow, can_request_late: true, signup: { open: false, revision: 1, url: 'https://site.test/signup' }, members: [existing], email_mode: 'dry_run' };
  let requests = [], finish, tree;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method === 'POST') return new Promise(resolve => { finish = resolve; });
    if (url.includes('/pool/players?')) return reply({ players: [player, { ...player, id: '102', name: 'Existing Member' }], next_offset: null });
    return reply(pool);
  };
  const props = { root, accessToken: 'token-1', clubName: 'Beta Club', season: { ...season, registration: closedWindow } };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  assert.equal(button(tree, 'Add players').props.disabled, true);
  assert.equal(button(tree, 'Invite club players by email').props.disabled, true);
  await act(async () => button(tree, 'Request late player').props.onClick());
  assert.ok(requests.at(-1).url.startsWith(root + '/pool/players?'), 'Closed registration still allows club-scoped profile lookup for late requests');
  assert.equal(requests.at(-1).options.method, undefined, 'Opening late search is read-only');
  assert.equal(button(tree, 'Submit late player request').props.disabled, true);
  let choices = tree.root.findAllByProps({ type: 'radio' });
  assert.equal(choices.length, 1, 'Only a player without a season signup has a selectable radio');
  assert.ok(tree.root.findByProps({ 'aria-label': 'View Existing Member in player pool' }), 'Existing players get a useful action instead of a disabled selector');
  assert.ok(text(tree).includes('Already registered and approved. No late request is needed.'));
  await act(async () => choices[0].props.onChange());
  assert.equal(button(tree, 'Submit late player request').props.disabled, false, 'Selecting an eligible profile is enough; notes are optional');
  assert.equal(tree.root.findByType('textarea').props.required, undefined);
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: '   ' } }));
  assert.equal(button(tree, 'Submit late player request').props.disabled, false, 'Whitespace-only optional notes do not block submission');
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: '  Arriving after registration closes.  ' } }));
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, accessToken: 'token-2' })));
  assert.ok(text(tree).includes('Selected: '));
  assert.equal(tree.root.findByType('textarea').props.value, '  Arriving after registration closes.  ', 'Token refresh preserves the request draft');
  const form = tree.root.findByType('form');
  await act(async () => { form.props.onSubmit({ preventDefault() {} }); form.props.onSubmit({ preventDefault() {} }); });
  const writes = requests.filter(row => row.options.method);
  assert.equal(writes.length, 1, 'Repeated submit creates one request');
  assert.equal(writes[0].url, root + '/pool/late-requests');
  assert.equal(writes[0].options.headers.Authorization, 'Bearer token-2');
  assert.deepEqual(JSON.parse(writes[0].options.body), { player_id: 101, reason: 'Arriving after registration closes.' }, 'Late requests carry no email, consent, or automatic approval');
  const pending = { ...member, id: 'late-1', name: player.name, email: '', player_id: player.id, approval_status: 'pending', late_join: true, late_request_reason: 'Arriving after registration closes.', rating: 2.9, eligible_divisions: player.eligible_divisions };
  pool = { ...pool, members: [existing, pending] };
  await act(async () => finish(reply({ member: pending, pool })));
  assert.ok(text(tree).includes('Pending · cannot play'));
  assert.ok(text(tree).includes('Arriving after registration closes.'));
  assert.ok(text(tree).includes('Pending commissioner approval; this player cannot play yet.'));
  assert.equal(button(tree, 'Submit late player request'), undefined);
  await act(async () => button(tree, 'Request late player').props.onClick());
  choices = tree.root.findAllByProps({ type: 'radio' });
  assert.equal(choices.length, 0, 'The new pending player is not offered for another request');
  assert.ok(text(tree).includes('Awaiting commissioner approval. A request already exists.'));
  assert.ok(text(tree).includes('Everyone on this page already has a season signup. View their status below, or search for another player.'));
  await act(async () => button(tree, 'Close late player request').props.onClick());
  pool = { ...pool, members: [existing, { ...pending, approval_status: 'approved', revision: 3, approval_reason: 'Visitor approved' }] };
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  assert.ok(text(tree).includes('Approved for this season’s player pool.'));
  assert.ok(text(tree).includes('Visitor approved'));
  pool = { ...pool, members: [existing, { ...pending, approval_status: 'rejected', revision: 4, approval_reason: 'No late entry this season' }] };
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  assert.ok(text(tree).includes('Rejected · cannot play'));
  assert.ok(text(tree).includes('No late entry this season'));
  pool = { ...pool, can_request_late: false };
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  assert.equal(button(tree, 'Request late player').props.disabled, true, 'The server can close late requests when the season ends');
  assert.ok(!requests.some(row => row.url.includes('/emails') || row.url.includes('/bulk-')), 'Late entry does not send email or reopen normal signup');
  await act(async () => tree.unmount());
}

async function lateRequestRecoveryAndScope() {
  const player = { id: '101', name: 'Late Traveler', rating: 3.2, gender: 'female', eligible_divisions: ['3.5'] };
  const pool = { registration: closedWindow, can_request_late: true, signup: { open: false, revision: 1, url: null }, members: [], email_mode: 'dry_run' };
  let requests = [], finish, tree;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    return reply(url.includes('/pool/players?') ? { players: [player], next_offset: null } : pool);
  };
  const props = { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: closedWindow } };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  await act(async () => button(tree, 'Request late player').props.onClick());
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'New arrival' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  await act(async () => finish(reply({ detail: 'This player is already in the pool. Reload.' }, 409)));
  assert.equal(button(tree, 'Submit late player request').props.disabled, true, 'An uncertain or duplicate response requires reload');
  assert.ok(text(tree).includes('already in the pool'));
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, refreshKey: 1 })));
  assert.equal(button(tree, 'Submit late player request').props.disabled, true, 'A sibling refresh cannot clear the manual recovery required after an uncertain write');
  assert.ok(text(tree).includes('already in the pool'));
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'New arrival' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  const signal = requests.at(-1).options.signal, finishOld = finish;
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, root: root.replace('/beta/', '/gamma/') })));
  assert.equal(signal.aborted, true, 'Changing club aborts an outstanding late request');
  await act(async () => finishOld(reply({ member: { ...member, name: 'OLD CLUB PLAYER' }, pool })));
  assert.ok(!text(tree).includes('OLD CLUB PLAYER'));
  assert.equal(tree.root.findAllByType('form').length, 0, 'The old club draft does not follow the administrator');
  await act(async () => tree.unmount());
}

async function fillNewPlayer(tree, values) {
  for (const [label, value] of Object.entries(values)) await act(async () => tree.root.findByProps({ 'aria-label': label }).props.onChange({ target: { value } }));
}

async function inlinePlayerCreation() {
  for (const isLate of [false, true]) {
    const registration = isLate ? closedWindow : openWindow;
    let pool = { registration, can_request_late: isLate, signup: { open: !isLate, revision: 1, url: null }, members: [], email_mode: 'dry_run' };
    let requests = [], finish, tree;
    global.fetch = async (url, options) => {
      requests.push({ url, options });
      if (options.method) return new Promise(resolve => { finish = resolve; });
      return reply(url.includes('/pool/players?') ? { players: [], next_offset: null } : pool);
    };
    const props = { root, accessToken: 'token-1', clubName: 'Beta Club', season: { ...season, registration } };
    await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
    if (isLate) await act(async () => button(tree, 'Request late player').props.onClick());
    await act(async () => button(tree, 'Create new player').props.onClick());
    const submitLabel = isLate ? 'Create player and request late entry' : 'Create player and add to pool';
    assert.equal(button(tree, submitLabel).props.disabled, true, 'Creating a player requires a name and reviewed starting rating');
    assert.equal(tree.root.findByProps({ 'aria-label': 'Starting JUPR' }).props.value, '', 'Never silently assign a starting rating');
    assert.equal(tree.root.findByProps({ 'aria-label': 'Gender (optional)' }).props.required, undefined);
    assert.equal(tree.root.findByProps({ 'aria-label': 'Email (optional)' }).props.required, undefined);
    assert.equal(tree.root.findByType('textarea').props.required, undefined, 'Notes are optional for both regular and late entry');
    await fillNewPlayer(tree, { 'Player name': '  New Traveler  ', 'Starting JUPR': '0' });
    assert.equal(button(tree, submitLabel).props.disabled, true);
    await fillNewPlayer(tree, { 'Starting JUPR': '7.01' });
    assert.equal(button(tree, submitLabel).props.disabled, true);
    await fillNewPlayer(tree, { 'Starting JUPR': '3.45' });
    assert.equal(button(tree, submitLabel).props.disabled, false, 'Name and reviewed rating suffice; gender, email and notes are optional');
    if (!isLate) {
      await fillNewPlayer(tree, { 'Email (optional)': 'invalid-email' });
      assert.equal(button(tree, submitLabel).props.disabled, true);
      await fillNewPlayer(tree, { 'Gender (optional)': 'female', 'Email (optional)': '  traveler@example.invalid  ' });
    } else {
      await act(async () => button(tree, 'Choose existing player').props.onClick());
      assert.equal(tree.root.findAllByProps({ 'aria-label': 'Player name' }).length, 0);
      await act(async () => button(tree, 'Create new player').props.onClick());
      assert.equal(tree.root.findByProps({ 'aria-label': 'Player name' }).props.value, '  New Traveler  ', 'Switching entry modes preserves the new-player draft');
    }
    await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, accessToken: 'token-2' })));
    assert.equal(tree.root.findByProps({ 'aria-label': 'Starting JUPR' }).props.value, '3.45', 'Token refresh preserves new-player fields');
    const form = tree.root.findByType('form');
    await act(async () => { form.props.onSubmit({ preventDefault() {} }); form.props.onSubmit({ preventDefault() {} }); });
    const writes = requests.filter(row => row.options.method);
    assert.equal(writes.length, 1, 'Profile creation and season registration use one guarded atomic request');
    assert.equal(writes[0].url, root + (isLate ? '/pool/late-requests' : '/pool/create-player'));
    assert.equal(writes[0].options.headers.Authorization, 'Bearer token-2');
    const body = JSON.parse(writes[0].options.body);
    assert.match(body.request_id, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
    assert.deepEqual(body.new_player, { name: 'New Traveler', starting_jupr: 3.45, ...(!isLate ? { gender: 'female', email: 'traveler@example.invalid' } : {}) });
    assert.equal(body.reason, '', 'A new player can register without notes');
    assert.ok(!Object.hasOwn(body, 'player_id') && !Object.hasOwn(body, 'email_consent'), 'The payload neither invents a profile ID nor claims consent');
    const created = { ...member, id: 'created-1', name: 'New Traveler', email: isLate ? '' : 'traveler@example.invalid', player_id: '999', approval_status: isLate ? 'pending' : 'approved', late_join: isLate, late_request_reason: '', late_requested_at: isLate ? '2026-09-22T18:00:00Z' : null, rating: 3.45 };
    pool = { ...pool, members: [created] };
    await act(async () => finish(reply({ member: created, pool })));
    assert.equal(button(tree, submitLabel), undefined);
    if (isLate) {
      assert.ok(text(tree).includes('Pending · cannot play'), 'A new player without notes still waits for commissioner approval');
      assert.equal(button(tree, 'Change linked player'), undefined, 'Late identity stays fixed even when the optional request reason is blank');
    } else assert.ok(text(tree).includes('club player created and added to the season pool.'));
    assert.ok(!requests.some(row => row.url.includes('/emails') || row.url.includes('/bulk-') || row.url.includes('/players/editor')), 'Inline creation does not send email or chain a separate player write');
    await act(async () => tree.unmount());
  }
}

async function blankLateNotesAndCreateRecovery() {
  const player = { id: '601', name: 'Existing Late Traveler', rating: 3.2, gender: 'female', eligible_divisions: ['3.5'] };
  let pool = { registration: closedWindow, can_request_late: true, signup: { open: false, revision: 1, url: null }, members: [], email_mode: 'dry_run' };
  let requests = [], finish, tree;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    return reply(url.includes('/pool/players?') ? { players: [player], next_offset: null } : pool);
  };
  const props = { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: closedWindow } };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  await act(async () => button(tree, 'Request late player').props.onClick());
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { player_id: 601, reason: '' }, 'Existing profiles also support a late request with no reason');
  const pending = { ...member, id: 'late-601', name: player.name, player_id: player.id, late_join: true, approval_status: 'pending', late_request_reason: '', late_requested_at: '2026-09-22T18:00:00Z' };
  pool = { ...pool, members: [pending] };
  await act(async () => finish(reply({ member: pending, pool })));
  assert.ok(text(tree).includes('Pending · cannot play'));
  assert.equal(button(tree, 'Change linked player'), undefined);
  await act(async () => button(tree, 'Request late player').props.onClick());
  await act(async () => button(tree, 'Create new player').props.onClick());
  const fields = { 'Player name': 'Retry Traveler', 'Starting JUPR': '3.5' };
  await fillNewPlayer(tree, fields);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  const firstCreation = JSON.parse(requests.at(-1).options.body);
  await act(async () => finish(reply({ detail: 'Creation result could not be confirmed. Reload.' }, 503)));
  assert.equal(button(tree, 'Create player and request late entry').props.disabled, true, 'An uncertain creation blocks another attempt until the pool is checked');
  await act(async () => button(tree, 'Reload player pool').props.onClick());
  await act(async () => button(tree, 'Create new player').props.onClick());
  await fillNewPlayer(tree, fields);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), firstCreation, 'Retrying the same draft after reload reuses its request UUID');
  const signal = requests.at(-1).options.signal, oldFinish = finish;
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, root: root.replace('/beta/', '/gamma/') })));
  assert.equal(signal.aborted, true, 'A club change aborts a pending atomic creation');
  await act(async () => oldFinish(reply({ member: { ...pending, name: 'STALE CREATED PLAYER' }, pool })));
  assert.ok(!text(tree).includes('STALE CREATED PLAYER'));
  await act(async () => tree.unmount());
}

async function inlineCreationClosingBoundary() {
  const pool = { registration: openWindow, can_request_late: false, signup: { open: true, revision: 1, url: null }, members: [], email_mode: 'dry_run' };
  let requests = [], tree;
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply(pool); };
  const props = { root, accessToken: 'token', clubName: 'Beta Club', season };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  await act(async () => button(tree, 'Create new player').props.onClick());
  await fillNewPlayer(tree, { 'Player name': 'Boundary Traveler', 'Starting JUPR': '3.5' });
  const staleSubmit = tree.root.findByType('form').props.onSubmit, originalNow = Date.now;
  try {
    Date.now = () => Date.parse('2100-01-01T00:00:00Z');
    await act(async () => staleSubmit({ preventDefault() {} }));
    assert.equal(requests.filter(row => row.options.method).length, 0, 'An open-phase creation handler rechecks the clock before any mutation');
    assert.ok(text(tree).includes('Season registration has closed.'));
  } finally { Date.now = originalNow; await act(async () => tree.unmount()); }
}

async function existingLatePlayerGuidance() {
  const statusCases = [
    { id: 'approved', name: 'Approved Player', player_id: 201, approval_status: 'approved', message: 'Already registered and approved. No late request is needed.' },
    { id: 'pending', name: 'Pending Player', player_id: '202', approval_status: 'pending', message: 'Awaiting commissioner approval. A request already exists.' },
    { id: 'rejected', name: 'Rejected Player', player_id: '203', approval_status: 'rejected', message: 'The existing request was rejected. View the signup for the decision.' },
    { id: 'withdrawn', name: 'Withdrawn Player', player_id: '204', status: 'withdrawn', approval_status: 'approved', message: 'This signup was withdrawn. A new late request cannot replace it.' },
    { id: 'unknown', name: 'Existing Player', player_id: '205', approval_status: undefined, message: 'A season signup already exists. View it to check its status.' },
  ];
  const members = statusCases.map(({ message, ...row }) => ({ ...member, ...row }));
  const players = statusCases.map((row, index) => ({ id: index === 1 ? Number(row.player_id) : String(row.player_id), name: row.name, rating: 3.5, gender: 'female', eligible_divisions: ['3.5'] }));
  let requests = [], viewed = [], submitted = [], tree;
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply({ players, next_offset: null }); };
  const props = { root, accessToken: 'token', members, disabled: false, onClose() {}, onViewMember: row => viewed.push(row), onRequest: async body => { submitted.push(body); } };
  await act(async () => { tree = create(React.createElement(late.PoolLateRequest, props)); });
  assert.equal(tree.root.findAllByProps({ type: 'radio' }).length, 0, 'Existing signups of every status use readable status rows, including numeric/string ID matches');
  assert.ok(text(tree).includes('Everyone on this page already has a season signup. View their status below, or search for another player.'));
  assert.equal(button(tree, 'Submit late player request').props.disabled, true);
  for (const [index, row] of statusCases.entries()) {
    assert.ok(text(tree).includes(row.message), `${row.id} membership gets accurate next-step guidance`);
    const view = tree.root.findByProps({ 'aria-label': `View ${row.name} in player pool` });
    assert.equal(nodeText(view), 'View in player pool');
    await act(async () => view.props.onClick());
    assert.equal(viewed.at(-1), members[index], 'The view action passes the matched signup, not the directory profile');
  }
  assert.equal(requests.filter(row => row.options.method).length, 0, 'Viewing status does not send requests or change existing eligibility');
  assert.equal(submitted.length, 0);
  await act(async () => tree.unmount());

  const sameNameNewPlayer = { ...players[0], id: '999' };
  global.fetch = async () => reply({ players: [players[0], sameNameNewPlayer], next_offset: null });
  await act(async () => { tree = create(React.createElement(late.PoolLateRequest, props)); });
  assert.equal(tree.root.findAllByProps({ type: 'radio' }).length, 1, 'A different player ID stays selectable even when the name and rating match an existing signup');
  assert.ok(!text(tree).includes('Everyone on this page already has a season signup.'));
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Different player with the same name' } }));
  assert.equal(button(tree, 'Submit late player request').props.disabled, false);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(submitted, [{ player_id: 999, reason: 'Different player with the same name' }]);
  await act(async () => tree.unmount());
}

async function viewWithdrawnPlayerFromLateSearch() {
  const withdrawn = { ...member, id: 'withdrawn-1', name: 'Withdrawn Traveler', player_id: '301', status: 'withdrawn', approval_status: 'approved' };
  const approved = { ...member, id: 'approved-1', name: 'Other Player', player_id: '302', approval_status: 'approved' };
  const pool = { registration: closedWindow, can_request_late: true, signup: { open: false, revision: 1, url: null }, members: [withdrawn, approved], email_mode: 'dry_run' };
  const directoryPlayer = { id: 301, name: withdrawn.name, rating: 3.45, gender: 'female', eligible_divisions: ['3.5'] };
  let tree, requests = [], focused = [], nodes = [];
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply(url.includes('/pool/players?') ? { players: [directoryPlayer], next_offset: null } : pool); };
  const props = { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: closedWindow } };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props), { createNodeMock: element => {
    const summary = React.Children.toArray(element.props.children).find(child => child?.type === 'summary');
    const label = element.props['aria-label'] || summary?.props['aria-label'];
    const node = { label, open: false, focus: () => focused.push(label), scrollIntoView() {}, querySelector: () => ({ focus: () => focused.push(label) }) };
    nodes.push(node); return node;
  } }); });
  await act(async () => tree.root.findByProps({ placeholder: 'Name or email' }).props.onChange({ target: { value: 'Other Player' } }));
  assert.equal(tree.root.findAllByProps({ 'aria-label': `Manage ${withdrawn.name}` }).length, 0, 'Withdrawn players are hidden by the default active filter and current query');
  await act(async () => button(tree, 'Request late player').props.onClick());
  const readsBeforeView = requests.length;
  await act(async () => tree.root.findByProps({ 'aria-label': `View ${withdrawn.name} in player pool` }).props.onClick());
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Request a late player' }).length, 0, 'View in player pool closes the late request form');
  assert.equal(tree.root.findByProps({ placeholder: 'Name or email' }).props.value, withdrawn.name, 'The pool search jumps to the selected signup');
  assert.equal(tree.root.findByType('select').props.value, 'all', 'The jump reveals withdrawn players by switching to all signups');
  const manage = tree.root.findByProps({ 'aria-label': `Manage ${withdrawn.name}` });
  assert.ok(manage.parent.props.open || nodes.some(node => node.label === `Manage ${withdrawn.name}` && node.open), 'The selected player’s Manage details open');
  assert.ok(focused.includes(`Manage ${withdrawn.name}`), 'The destination receives keyboard focus');
  assert.equal(button(tree, 'Restore season signup').props.disabled, true, 'Viewing the existing signup does not bypass the closed registration window');
  assert.equal(requests.length, readsBeforeView, 'Jumping to an already-loaded signup needs no new request');
  assert.equal(requests.filter(row => row.options.method).length, 0, 'The view action never restores or re-registers the player');
  await act(async () => tree.unmount());
}

async function backgroundLateRequestRefresh() {
  const player = { id: '101', name: 'Late Traveler', rating: 3.2, gender: 'female', eligible_divisions: ['3.5'] };
  const pool = { registration: closedWindow, can_request_late: true, signup: { open: false, revision: 1, url: null }, members: [], email_mode: 'dry_run' };
  let tree, reads = 0, finishRead, finishWrite, notified = 0;
  global.fetch = async (url, options) => {
    if (options.method) return new Promise(resolve => { finishWrite = resolve; });
    if (url.includes('/pool/players?')) return reply({ players: [player], next_offset: null });
    if (reads++) return new Promise(resolve => { finishRead = resolve; });
    return reply(pool);
  };
  const props = { root, accessToken: 'token', clubName: 'Beta Club', season: { ...season, registration: closedWindow }, refreshKey: 0, onLateRequested: () => { notified++; } };
  await act(async () => { tree = create(React.createElement(panels.SeasonPlayerPool, props)); });
  await act(async () => button(tree, 'Request late player').props.onClick());
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Preserve this request draft' } }));
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, refreshKey: 1 })));
  assert.equal(tree.root.findByType('textarea').props.value, 'Preserve this request draft', 'A commissioner decision refresh keeps the request form mounted');
  await act(async () => finishRead(reply(pool)));
  assert.equal(tree.root.findByType('textarea').props.value, 'Preserve this request draft');
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  await act(async () => tree.update(React.createElement(panels.SeasonPlayerPool, { ...props, refreshKey: 2 })));
  const pending = { ...member, id: 'late-1', name: player.name, player_id: player.id, late_join: true, approval_status: 'pending', late_request_reason: 'Preserve this request draft' };
  await act(async () => finishWrite(reply({ member: pending, pool: { ...pool, members: [pending] } })));
  assert.equal(notified, 1, 'A saved request prompts commissioner review to refresh once');
  await act(async () => finishRead(reply(pool)));
  assert.ok(text(tree).includes('Pending · cannot play'), 'An older background response cannot erase a newly submitted request');
  await act(async () => tree.unmount());
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
(async () => { await seasonPool(); await latePlayerRequests(); await inlinePlayerCreation(); await blankLateNotesAndCreateRecovery(); await inlineCreationClosingBoundary(); await existingLatePlayerGuidance(); await viewWithdrawnPlayerFromLateSearch(); await lateRequestRecoveryAndScope(); await backgroundLateRequestRefresh(); await availability(); await commissionerRegistrationGates(); await resumedRegistrationBoundary(); await invitationEmail(); await recoverSavedInvitations(); await bulkAdd(); console.log('PASS interclub player pool: scoped signup/linking, atomic inline player creation and optional late notes, existing signup guidance and focused pool navigation, commissioner review status, commissioner registration phases, suspended boundary recovery, revision conflicts, stale responses, availability, bulk player matching and optional emails, email preview and dry-run delivery'); })().catch(error => { console.error(error); process.exit(1); });
