const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const registration = load('lib/interclubRegistration.ts');
const competition = load('lib/interclubCompetition.ts', { './interclubRegistration': registration });
const setup = load('lib/interclubSetup.ts'), registrationWindow = load('lib/interclubRegistrationWindow.ts');
const Link = ({ children, ...props }) => React.createElement('a', props, children);
let currentClub = 'alpha', requestIds = 0;
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => `00000000-0000-4000-8000-${String(++requestIds).padStart(12, '0')}` }, configurable: true });
const common = { '@/lib/interclubCompetition': competition, '@/lib/interclubSetup': setup, '@/lib/interclubRegistrationWindow': registrationWindow, '@/lib/adminWorkspace': { readBrowserWorkspace: () => ({ clubId: currentClub }) }, './competition.module.css': {} };
const Form = load('app/admin/interclub/competition/ScheduleMeet.tsx', common).default;
const Panel = load('app/admin/interclub/registrations/SeasonMeetSchedule.tsx', { ...common, 'next/link': Link, '../InterclubWorkflow': load('app/admin/interclub/InterclubWorkflow.tsx', { 'next/link': Link, './workflow.module.css': {} }), '../competition/ScheduleMeet': Form, './registrations.module.css': {} }).default;
const nodeText = node => typeof node === 'string' ? node : node.children.map(nodeText).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => nodeText(node) === label);
const text = tree => JSON.stringify(tree.toJSON());
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const closed = { opens_at: '2000-01-01T00:00:00Z', closes_at: '2000-02-01T00:00:00Z', revision: 1, status: 'closed', can_register: false, meet_planning_open: true };
const season = { id: 'season-1', organizer_club_id: 'alpha', registration: closed, details: { name: 'Coastal League', start_date: '2099-01-01', end_date: '2099-03-31', timezone: 'America/Mazatlan', divisions: ['3.5'], club_ids: ['alpha', 'beta'] } };
const meet = { id: 'meet-1', season_id: season.id, host_club_id: 'alpha', club_ids: ['alpha', 'beta'], starts_at: '2099-01-16T17:00:00Z', roster_deadline: '2099-01-16T17:00:00Z', duration_minutes: 180, courts: 4, revision: 7, competition_phase: 'regular', schedule_editable: true, schedule_deadline_editable: true, courts_editable: true };
const context = { season, is_organizer: true, clubs: [{ id: 'alpha', name: 'Alpha Club' }, { id: 'beta', name: 'Beta Club' }], meets: [meet], batches: [] };
const base = { root: 'https://api.test/admin/clubs/alpha/interclub/competition/season-1', clubId: 'alpha', accessToken: 'token-1', context, disabled: false };
const input = (tree, label) => tree.root.findByProps({ 'aria-label': label });
const set = async (tree, label, value) => act(async () => input(tree, label).props.onChange({ target: { value } }));
const submit = async tree => act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
async function createAndTimezone() {
  const requests = []; let finish, saved, tree;
  global.fetch = (url, options) => new Promise(resolve => { requests.push({ url, options }); finish = resolve; });
  const props = { ...base, onScheduled: value => { saved = value; } };
  await act(async () => { tree = create(React.createElement(Form, props)); });
  assert.equal(input(tree, 'Competition').props.value, 'regular');
  await set(tree, 'Host club', 'alpha');
  await act(async () => input(tree, 'Beta Club').props.onChange({ target: { checked: true } }));
  await set(tree, 'Meet date and time', '2099-02-10T09:30');
  assert.equal(input(tree, 'Roster deadline').props.value, '2099-02-10T09:30');
  await act(async () => tree.update(React.createElement(Form, { ...props, accessToken: 'token-2' })));
  await act(async () => { tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  assert.equal(requests.length, 1, 'Double submit creates one request');
  assert.equal(requests[0].options.headers.Authorization, 'Bearer token-2');
  const body = JSON.parse(requests[0].options.body);
  assert.equal(body.starts_at, '2099-02-10T16:30:00.000Z', 'Dates follow the season timezone, independently of the computer');
  assert.equal(body.roster_deadline, body.starts_at); assert.equal(body.competition_phase, 'regular');
  assert.deepEqual(body.club_ids, ['alpha', 'beta']); assert.match(body.request_id, /^[\da-f-]{36}$/);
  await act(async () => finish(reply({ meet: { ...meet, ...body, id: 'meet-2' } })));
  assert.equal(saved.id, 'meet-2');
  await submit(tree);
  assert.equal(JSON.parse(requests[1].options.body).request_id, body.request_id, 'Unchanged resubmission uses its idempotency key');
  await act(async () => finish(reply({ detail: 'Choose another date.' }, 422)));
  await set(tree, 'Meet date and time', '2099-02-11T09:30'); await submit(tree);
  assert.notEqual(JSON.parse(requests[2].options.body).request_id, body.request_id);
  await act(async () => finish(reply({ meet: { ...meet, id: 'meet-3' } })));
  await act(async () => tree.unmount());
}
async function ordinaryEditAndLockedFields() {
  let tree, requests = [], savedMessage;
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply({ meet: { ...meet, revision: 8 }, availability_reset_count: 4, publication_review_required: true }); };
  await act(async () => { tree = create(React.createElement(Form, { ...base, meet, onScheduled: (_, message) => { savedMessage = message; } })); });
  assert.equal(input(tree, 'Meet date and time').props.value, '2099-01-16T10:00');
  assert.equal(tree.root.findAllByType('select').length, 0, 'Editing fixes host/clubs/phase');
  assert.ok(text(tree).includes('Existing lineups are kept'));
  await set(tree, 'Meet date and time', '2099-01-15T08:00'); await submit(tree);
  assert.ok(requests[0].url.endsWith('/meets/meet-1/schedule')); assert.equal(requests[0].options.method, 'PUT');
  assert.deepEqual(JSON.parse(requests[0].options.body), { expected_revision: 7, starts_at: '2099-01-15T15:00:00.000Z', roster_deadline: '2099-01-15T15:00:00.000Z', courts: 4, duration_minutes: 180 });
  assert.ok(savedMessage.includes('Reopen availability') && savedMessage.includes('publish the updated public schedule'));
  await act(async () => tree.unmount()); requests = [];
  const locked = { ...meet, roster_deadline: '2000-01-16T17:00:00Z', schedule_deadline_editable: false, courts_editable: false };
  await act(async () => { tree = create(React.createElement(Form, { ...base, meet: locked, onScheduled() {} })); });
  assert.equal(input(tree, 'Roster deadline').props.disabled, true); assert.equal(input(tree, 'Courts').props.disabled, true);
  await set(tree, 'Meet date and time', '2099-01-20T10:00'); await submit(tree);
  assert.equal(JSON.parse(requests[0].options.body).roster_deadline, locked.roster_deadline, 'A frozen rating cutoff is preserved exactly, even when past');
  await act(async () => tree.unmount());
}
async function authorityAndUncertainSave() {
  const requests = []; let tree;
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply({ detail: 'This meet changed. Reload it.' }, 409); };
  for (const overrides of [{ context: { ...context, is_organizer: false } }, { context: { ...context, season: { ...season, registration: { ...closed, status: 'open', meet_planning_open: false } } } }, { meet: { ...meet, schedule_editable: false, schedule_locked_reason: 'Scores have been entered.' } }]) {
    await act(async () => { tree = create(React.createElement(Form, { ...base, meet, onScheduled() {}, ...overrides })); });
    assert.equal(tree.root.findByType('fieldset').props.disabled, true); await submit(tree);
    assert.equal(requests.length, 0, 'Unauthorized and played-meet forms never write'); await act(async () => tree.unmount());
  }
  const props = { ...base, meet, onScheduled() {} };
  await act(async () => { tree = create(React.createElement(Form, props)); });
  await set(tree, 'Meet date and time', '2099-02-01T08:00'); await submit(tree);
  assert.equal(requests.length, 1); assert.equal(input(tree, 'Meet date and time').props.value, '2099-02-01T08:00');
  assert.equal(tree.root.findByType('fieldset').props.disabled, true); await submit(tree); assert.equal(requests.length, 1);
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(Form, props)); }); currentClub = 'beta'; await submit(tree);
  assert.equal(requests.length, 1, 'Stale club tabs cannot write'); assert.ok(text(tree).includes('selected club changed')); currentClub = 'alpha';
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(Form, props)); }); await set(tree, 'Meet date and time', '2099-02-01T08:00');
  await act(async () => tree.update(React.createElement(Form, { ...props, meet: { ...meet, revision: 8 } })));
  assert.equal(input(tree, 'Meet date and time').props.value, '2099-02-01T08:00'); assert.equal(tree.root.findByType('fieldset').props.disabled, true);
  await act(async () => tree.unmount());
}
async function schedulePanelGate() {
  let reads = [], tree, changed; const selected = [];
  const played = { ...meet, id: 'played', starts_at: '2000-01-10T17:00:00Z', schedule_editable: false, schedule_locked_reason: 'Scores have been entered. Use the results workflow to protect recorded games.' };
  global.fetch = async (url, options) => { reads.push({ url, options }); return reply({ ...context, meets: [meet, played] }); };
  const seasonData = { season, is_organizer: true, clubs: context.clubs, meets: [meet, played] };
  const props = { root: base.root, clubId: 'alpha', accessToken: 'token', seasonData, meetPlanningOpen: false, disabled: false, onSaved: value => { changed = value; }, onSelectMeet: (id, step) => selected.push({ id, step }) };
  await act(async () => { tree = create(React.createElement(Panel, props)); });
  assert.equal(reads.length, 0, 'No operational context fetch before registration closes'); assert.equal(button(tree, 'Add meet').props.disabled, true); assert.equal(tree.root.findAllByType(Form).length, 0);
  await act(async () => tree.update(React.createElement(Panel, { ...props, meetPlanningOpen: true })));
  assert.equal(reads.length, 1); assert.ok(reads.every(read => !read.url.includes('/meets/')), 'Schedule viewing never enters score operations');
  const edits = tree.root.findAllByType('button').filter(node => nodeText(node) === 'Edit date');
  assert.equal(edits[0].props.disabled, true); assert.ok(text(tree).includes(played.schedule_locked_reason)); assert.equal(edits[1].props.disabled, false);
  await act(async () => edits[1].props.onClick()); assert.equal(tree.root.findByType(Form).props.meet.id, meet.id);
  await act(async () => tree.root.findByType(Form).props.onScheduled({ ...meet, revision: 8 }, 'Meet schedule saved.'));
  assert.equal(changed.revision, 8); assert.ok(text(tree).includes('Meet schedule saved.')); assert.equal(tree.root.findAllByType(Form).length, 0); assert.equal(reads.length, 2);
  const links = tree.root.findAllByType('a'); let prevented = 0;
  for (const label of ['Choose players for this meet', 'Meet availability', 'Choose players']) {
    await act(async () => links.find(link => nodeText(link) === label).props.onClick({ preventDefault() { prevented++; } }));
  }
  assert.deepEqual(selected, [{ id: meet.id, step: 'lineups' }, { id: meet.id, step: 'availability' }, { id: meet.id, step: 'lineups' }], 'Schedule links update the existing workspace instead of only changing its URL');
  assert.equal(prevented, 3);
  links.find(link => nodeText(link) === 'Choose players').props.onClick({ ctrlKey: true, preventDefault() { throw new Error('New-tab navigation must remain available'); } });
  assert.equal(selected.length, 3);
  await act(async () => button(tree, 'Add meet').props.onClick()); assert.equal(tree.root.findByType(Form).props.meet, undefined);
  await act(async () => tree.unmount());
}
async function visibleSeasonBounds() {
  let tree; const requests = [];
  global.fetch = async (url, options) => { requests.push({ url, options }); return reply({ meet }); };
  await act(async () => { tree = create(React.createElement(Form, { ...base, meet, onScheduled() {} })); });
  assert.equal(input(tree, 'Meet date and time').props.min, '2099-01-01T00:00');
  assert.equal(input(tree, 'Meet date and time').props.max, '2099-03-31T23:59');
  await set(tree, 'Meet date and time', '2098-12-31T10:00'); await submit(tree);
  assert.equal(requests.length, 0); assert.match(nodeText(tree.root.findByProps({ role: 'alert' })), /Jan 1, 2099.*Mar 31, 2099.*America\/Mazatlan/);
  await set(tree, 'Meet date and time', '2099-03-31T23:00'); await submit(tree);
  assert.equal(requests.length, 0, 'A meet ending after the season is rejected before sending');
  assert.match(nodeText(tree.root.findByProps({ role: 'alert' })), /ends on Apr 1, 2099/);
  await set(tree, 'Meet date and time', '2099-03-31T20:00'); await submit(tree);
  assert.equal(requests.length, 1, 'Valid local dates remain allowed even when the UTC date is the next day');
  await act(async () => tree.unmount());
}
(async () => { await createAndTimezone(); await ordinaryEditAndLockedFields(); await authorityAndUncertainSave(); await schedulePanelGate(); await visibleSeasonBounds(); console.log('PASS interclub meet schedule: commissioner/registration/played locks, regular meet creation, season timezone, visible season bounds, idempotency, ordinary date changes, protected cutoffs/courts, conflict recovery, and schedule navigation'); })().catch(error => { console.error(error); process.exitCode = 1; });
