const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => Object.hasOwn(mocks, n) ? mocks[n] : require(n), module, module.exports);
  return module.exports;
}
const registration = load('lib/interclubRegistration.ts'), helpers = load('lib/interclubSetup.ts');
const Link = ({ children, ...props }) => React.createElement('a', props, children);
const Panel = load('app/admin/interclub/ClubInvitationPanel.tsx', { './setup.module.css': {} }).default;
const Wizard = load('app/admin/interclub/InterclubSetupWizard.tsx', { 'next/link': Link, '@/lib/interclubRegistration': registration, '@/lib/interclubSetup': helpers, './setup.module.css': {}, './ClubInvitationPanel': Panel }).default;
const sid = '00000000-0000-4000-8000-000000000011';
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => sid }, configurable: true });
global.window = new EventTarget();
window.location = { origin: "https://staging.example.test" };
const clubs = [{ id: 'alpha', name: 'Tres Palapas', slug: 'alpha' }, { id: 'beta', name: 'Visiting Club', slug: 'beta' }];
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const nodeText = node => typeof node === 'string' ? node : node.children.map(nodeText).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(b => nodeText(b) === label);
const text = tree => JSON.stringify(tree.toJSON());
async function click(tree, label) { await act(async () => button(tree, label).props.onClick()); }
async function fill(tree, label, value) { await act(async () => tree.root.findByProps({ 'aria-label': label }).props.onChange({ target: { value } })); }
function newProps(initialSeason = helpers.newSeason()) { return { api: 'https://api.test', club: clubs[0], accessToken: 'token-1', initialSeason, choices: clubs, onSaved() {}, onOpened() {}, onClose() {} }; }

async function completeJourney() {
  let stored, opened = null, finish, requests = [], saved = [], closed = 0;
  global.fetch = async (url, options) => {
    if (url.endsWith("/club-invitations")) return reply({ invitations: [], clubs: [] });
    requests.push({ url, options });
    if (options.method === 'PUT') {
      const body = JSON.parse(options.body);
      assert.equal(body.expected_revision, stored?.revision || 0);
      stored = { id: sid, revision: body.expected_revision + 1, draft: body.draft };
      return reply({ season: stored });
    }
    if (options.method === 'POST') return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/setup')) return reply({ seasons: stored ? [stored] : [] });
    return opened ? reply({ season: opened }) : reply({}, 404);
  };
  let props = { ...newProps(), choices: [clubs[0]], onSaved: s => saved.push(s), onClose: () => closed++, onOpened: s => { opened = s; } }, tree;
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  assert.equal(button(tree, 'Open club invitations'), undefined);
  await click(tree, 'Save and continue');
  assert.ok(text(tree).includes('Enter a season name.')); assert.equal(requests.length, 0);
  await fill(tree, 'Season name', 'Southern BCS');
  await fill(tree, 'Season start date', '2099-01-01'); await fill(tree, 'Season end date', '2099-03-31');
  props = { ...props, accessToken: 'token-2' };
  await act(async () => tree.update(React.createElement(Wizard, props)));
  assert.equal(tree.root.findByProps({ 'aria-label': 'Season name' }).props.value, 'Southern BCS');
  const continueButton = button(tree, 'Save and continue');
  await act(async () => { continueButton.props.onClick(); continueButton.props.onClick(); });
  assert.equal(requests.length, 1, 'Duplicate continue saves once');
  assert.equal(requests[0].options.headers.Authorization, 'Bearer token-2');
  assert.equal(stored.draft.setup_step, 1);
  assert.ok(button(tree, 'Invite a new club'));
  assert.ok(!text(tree).includes('PCS administration'));
  await click(tree, 'Save and continue'); assert.equal(requests.length, 1);
  props = { ...props, choices: clubs }; await act(async () => tree.update(React.createElement(Wizard, props)));
  await act(async () => tree.root.findByProps({ 'aria-label': 'Tres Palapas' }).props.onChange());
  await act(async () => tree.root.findByProps({ 'aria-label': 'Visiting Club' }).props.onChange());
  await click(tree, 'Save and continue');
  assert.equal(stored.draft.setup_step, 2);
  assert.equal(tree.root.findByProps({ 'aria-label': '3.5 maximum rating' }).props.value, '');
  await fill(tree, '3.5 maximum rating', '3.999'); await fill(tree, '3.5 team composition', '2');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Include 4.0 division' }).props.onChange());
  await click(tree, 'Save and exit');
  assert.equal(closed, 1); assert.equal(stored.draft.registration_rules['3.5'].max_rating, 3.999);
  await act(async () => tree.unmount()); props = { ...props, initialSeason: stored };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  assert.equal(tree.root.findByProps({ 'aria-label': '3.5 maximum rating' }).props.value, 3.999, 'Rules survive exit and reload');
  await click(tree, 'Save and continue'); await click(tree, 'Add a meet'); await click(tree, 'Save and exit');
  assert.equal(stored.draft.meets[0].starts_at, null, 'Unfinished meet can be resumed');
  await act(async () => tree.unmount()); props = { ...props, initialSeason: stored };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  await fill(tree, 'Meet 1 host', 'alpha');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Meet 1: Tres Palapas' }).props.disabled, true);
  await act(async () => tree.root.findByProps({ 'aria-label': 'Meet 1: Visiting Club' }).props.onChange());
  await fill(tree, 'Meet 1 date and time', '2099-04-01T09:00'); await click(tree, 'Save and continue');
  assert.ok(text(tree).includes('within the season dates'));
  await fill(tree, 'Meet 1 date and time', '2099-01-10T09:00');
  await click(tree, 'Add a meet'); await fill(tree, 'Meet 2 host', 'alpha');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Meet 2: Visiting Club' }).props.onChange());
  await fill(tree, 'Meet 2 date and time', '2099-01-10T09:00'); await click(tree, 'Save and continue');
  assert.ok(text(tree).includes('already scheduled at another meet'));
  await click(tree, 'Remove meet 2'); await click(tree, 'Save and continue');
  assert.equal(stored.draft.meets[0].starts_at, '2099-01-10T16:00:00.000Z');
  assert.equal(stored.draft.setup_step, 4); assert.ok(text(tree).includes('What happens next?'));
  assert.equal(button(tree, 'Open club invitations').props.disabled, true);
  assert.equal(requests.filter(r => r.options.method === 'POST').length, 0, 'No invitations during intermediate saves');
  await act(async () => tree.root.findByProps({ 'aria-label': 'I have reviewed the season setup' }).props.onChange({ target: { checked: true } }));
  const openButton = button(tree, 'Open club invitations');
  await act(async () => { openButton.props.onClick(); openButton.props.onClick(); });
  const writes = requests.filter(r => r.options.method === 'POST'); assert.equal(writes.length, 1);
  assert.deepEqual(JSON.parse(writes[0].options.body), { expected_revision: stored.revision, rules: stored.draft.registration_rules });
  await act(async () => finish(reply({ season: { id: sid, organizer_club_id: 'alpha', details: stored.draft, rules: stored.draft.registration_rules } })));
  assert.ok(text(tree).includes('Club invitations are open.'));
  assert.ok(tree.root.findAllByType('a').some(a => a.props.href === `/admin/interclub/registrations?season=${sid}`));
  assert.equal(tree.root.findAllByType('input').length, 0);
  await act(async () => tree.unmount());
}

async function conflictsAndContext() {
  const original = { ...helpers.newSeason(), revision: 2 }; original.draft.name = 'Saved season';
  let requests = [], conflict = true, deferred, saveCount = 0;
  global.fetch = async (url, options) => {
    if (url.endsWith("/club-invitations")) return reply({ invitations: [], clubs: [] });
    requests.push({ url, options });
    if (options.method === 'PUT') return conflict ? reply({ detail: 'Setup changed. Reload.' }, 409) : new Promise(resolve => { deferred = resolve; });
    return url.endsWith('/setup') ? reply({ seasons: [original] }) : reply({}, 404);
  };
  let tree; const props = { ...newProps(original), onSaved: () => saveCount++ };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  await fill(tree, 'Season name', 'Unsaved correction'); await click(tree, 'Save and exit');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Season name' }).props.value, 'Unsaved correction');
  assert.equal(button(tree, 'Save and continue').props.disabled, true);
  await click(tree, 'Reload saved setup');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Season name' }).props.value, 'Saved season');
  conflict = false;
  await click(tree, 'Save and exit'); const signal = requests.at(-1).options.signal, before = saveCount;
  await act(async () => tree.unmount()); assert.equal(signal.aborted, true);
  await act(async () => deferred(reply({ season: { ...original, revision: 3 } })));
  assert.equal(saveCount, before, 'Old club/account request cannot publish a saved result after unmount');
  global.fetch = async url => url.endsWith('/club-invitations') ? reply({ invitations: [], clubs: [] }) : reply({ season: { id: sid, details: { name: 'Opened season' } } });
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  assert.equal(tree.root.findAllByType('input').length, 0, 'Opened season cannot be edited as a draft');
  assert.ok(text(tree).includes('Manage this season'));
  await act(async () => tree.unmount());
}

async function inviteDuringClubSelection() {
  const initial = { ...helpers.newSeason(), revision: 1 };
  initial.draft = { ...initial.draft, name: 'Southern BCS', start_date: '2099-01-01', end_date: '2099-03-31', club_ids: ['alpha'], setup_step: 1 };
  const added = { id: 'la-ribera', name: 'La Ribera', slug: 'la-ribera' };
  let stored = initial, invitations = [], requests = [], finish, saved = [], mode = 'normal', copied;
  Object.defineProperty(global, 'navigator', { configurable: true, value: { clipboard: { writeText: async value => { copied = value; } } } });
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (url.includes('/club-choices?')) return reply({ clubs: [clubs[0], ...(invitations.length ? [added] : [])], next_offset: null });
    if (url.endsWith('/club-invitations') && options.method === 'POST') {
      const body = JSON.parse(options.body); assert.equal(body.expected_revision, stored.revision);
      stored = { id: sid, revision: stored.revision + 1, draft: { ...body.draft, club_ids: [...body.draft.club_ids, added.id] } };
      invitations = [{ id: body.invitation_id, club_id: added.id, club_name: added.name, email: body.email, revision: 1, status: 'pending', expires_at: '2099-01-01T00:00:00Z' }];
      return new Promise(resolve => { finish = () => resolve(mode === 'uncertain' ? reply({ detail: 'Could not confirm invitation.' }, 503) : reply({ season: stored, invitation: invitations[0], club: added })); });
    }
    if (url.endsWith('/club-invitations')) return reply({ invitations, clubs: invitations.length ? [added] : [] });
    if (url.includes('/club-invitations/') && options.method === 'POST') {
      const body = JSON.parse(options.body); assert.equal(body.expected_revision, invitations[0].revision);
      invitations = [{ ...invitations[0], revision: invitations[0].revision + 1, email: body.email, status: body.action === 'cancel' ? 'cancelled' : 'pending' }];
      return reply({ invitation: invitations[0] });
    }
    if (options.method === 'PUT') { const body = JSON.parse(options.body); stored = { id: sid, revision: stored.revision + 1, draft: body.draft }; return reply({ season: stored }); }
    if (url.endsWith('/setup')) return reply({ seasons: [stored] });
    return mode === 'opened' ? reply({ season: { id: sid, details: stored.draft } }) : reply({}, 404);
  };
  let tree, props = { ...newProps(initial), choices: [clubs[0]], onSaved: s => saved.push(s) };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  await click(tree, 'Save and continue'); assert.ok(text(tree).includes('at least two'));
  assert.equal(requests.filter(r => ['PUT', 'POST'].includes(r.options.method)).length, 0);
  await fill(tree, 'Find an existing club', 'La Ribera'); assert.ok(text(tree).includes('No clubs match'));
  await click(tree, 'Invite a new club');
  await fill(tree, 'New club name', 'La Ribera'); await fill(tree, 'New club administrator email', ' NEW@EXAMPLE.TEST ');
  props = { ...props, accessToken: 'latest-token' }; await act(async () => tree.update(React.createElement(Wizard, props)));
  const form = tree.root.findByType('form');
  await act(async () => { void form.props.onSubmit({ preventDefault() {} }); void form.props.onSubmit({ preventDefault() {} }); });
  assert.equal(requests.filter(r => r.options.method === 'POST').length, 1, 'Duplicate submission creates one club');
  assert.equal(requests.at(-1).options.headers.Authorization, 'Bearer latest-token');
  assert.equal(button(tree, 'Saving…').props.disabled, true); await act(async () => finish());
  assert.ok(tree.root.findByProps({ 'aria-label': 'La Ribera' }).props.checked);
  assert.deepEqual(saved.at(-1).draft.club_ids, ['alpha', added.id]);
  await click(tree, 'Copy invitation link');
  assert.equal(copied, `https://staging.example.test/admin/accept-invitation?invitation=${sid}&kind=club`);
  await click(tree, 'Save and exit'); await act(async () => tree.unmount());
  props = { ...props, initialSeason: stored };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  assert.ok(tree.root.findByProps({ 'aria-label': 'La Ribera' }).props.checked, 'Invited club and link survive reopen with old directory cache');
  await click(tree, 'Update or renew invitation'); await fill(tree, 'Update email for La Ribera', 'corrected@example.test');
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(invitations[0].email, 'corrected@example.test');
  await click(tree, 'Cancel invitation'); assert.ok(text(tree).includes('Invitation cancelled'));
  await click(tree, 'Update or renew invitation');
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(invitations[0].status, 'pending');
  await click(tree, 'Save and continue'); assert.ok(tree.root.findByProps({ 'aria-label': '3.5 maximum rating' }));
  assert.ok(!requests.some(r => r.url.endsWith('/open') || r.url.includes('/admin/platform')));
  await act(async () => tree.unmount());
  // Reload must recover an invitation committed before a lost response.
  stored = initial; invitations = []; mode = 'uncertain'; props = { ...props, initialSeason: initial };
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  await click(tree, 'Invite a new club'); await fill(tree, 'New club name', 'La Ribera'); await fill(tree, 'New club administrator email', 'new@example.test');
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  await act(async () => finish());
  assert.equal(tree.root.findByProps({ 'aria-label': 'New club name' }).props.value, 'La Ribera');
  assert.equal(button(tree, 'Save and continue').props.disabled, true);
  const count = requests.filter(r => r.options.method === 'POST').length;
  await click(tree, 'Reload saved setup'); assert.ok(tree.root.findByProps({ 'aria-label': 'La Ribera' }).props.checked);
  assert.equal(requests.filter(r => r.options.method === 'POST').length, count); await act(async () => tree.unmount());
  mode = 'opened';
  await act(async () => { tree = create(React.createElement(Wizard, { ...props, initialSeason: stored })); });
  assert.equal(button(tree, 'Invite a new club'), undefined, 'New clubs cannot change an opened season');
  assert.ok(button(tree, 'Copy invitation link'), 'Pending onboarding invitations stay accessible after the season opens');
  await click(tree, 'Cancel invitation');
  assert.equal(invitations[0].status, 'cancelled'); await act(async () => tree.unmount());
  stored = initial; invitations = []; mode = 'normal';
  await act(async () => { tree = create(React.createElement(Wizard, props)); });
  await click(tree, 'Invite a new club'); await fill(tree, 'New club name', 'La Ribera'); await fill(tree, 'New club administrator email', 'new@example.test');
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  const signal = requests.at(-1).options.signal, before = saved.length;
  await act(async () => tree.unmount()); assert.equal(signal.aborted, true); await act(async () => finish());
  assert.equal(saved.length, before, 'Club change aborts the old invitation result');
}

function timezoneChecks() {
  assert.equal(helpers.meetUtcTime('2027-01-10T09:00', 'America/Mazatlan'), '2027-01-10T16:00:00.000Z');
  assert.equal(helpers.meetUtcTime('2027-07-10T09:00', 'America/New_York'), '2027-07-10T13:00:00.000Z');
  assert.equal(helpers.meetLocalTime('2027-07-10T13:00:00Z', 'America/New_York'), '2027-07-10T09:00');
  assert.throws(() => helpers.meetUtcTime('2027-03-14T02:30', 'America/New_York'), /clock change/);
  assert.throws(() => helpers.meetUtcTime('2027-11-07T01:30', 'America/New_York'), /clock change/);
}
(async () => { timezoneChecks(); await completeJourney(); await conflictsAndContext(); await inviteDuringClubSelection(); console.log('Interclub wizard: inline club creation/selection, invitation links and renewal, saved progress, uncertain responses, duplicate actions, stale context and opened-season management passed.'); })().catch(e => { console.error(e); process.exitCode = 1; });
