const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => Object.hasOwn(mocks, n) ? mocks[n] : require(n), module, module.exports);
  return module.exports;
}
const helpers = load('lib/interclubRegistration.ts');
const Link = ({ children, ...p }) => React.createElement('a', p, children);
const button = (tree, label) => tree.root.findAllByType('button').find(b => b.children.includes(label));
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const sid = '00000000-0000-4000-8000-000000000001', tid = '00000000-0000-4000-8000-000000000002';
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => tid }, configurable: true });
const season = { id: sid, organizer_club_id: 'alpha', source_revision: 2, roster_deadline: '2099-01-01T00:00:00Z', details: {
  name: 'Coastal League', start_date: '2099-01-10', end_date: '2099-03-31', timezone: 'America/Mazatlan', divisions: ['3.5'], club_ids: ['beta', 'gamma'], meets: []
}, rules: { '3.5': { min_rating: null, max_rating: 3.75, women_required: 2 } } };
const lineup = [1, 2, 3, 4].map(n => ({ entry_id: `entry-${n}`, player_id: String(n), name: `Player ${n}`, starting_rating: 3.5 }));
const team = { id: tid, club_id: 'beta', season_id: sid, name: 'Beta Blue', division: '3.5', revision: 1, withdrawn: false, status: 'needs_exception', roster: lineup, issues: [{ code: 'rating_above_maximum', message: 'Player exceeds rating limit.' }], late_change: true, decision_reason: null };

async function opening() {
  let requests = [], finish, exists = false;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    return options.method ? new Promise(resolve => { finish = resolve; }) : reply({}, exists ? 200 : 404);
  };
  const Open = load('app/admin/interclub/OpenRegistration.tsx', { 'next/link': Link, '@/lib/interclubRegistration': helpers }).default;
  const props = { api: 'https://api.test', clubId: 'alpha', accessToken: 'token', seasonId: sid, revision: 2, divisions: ['3.5'] };
  let tree;
  await act(async () => { tree = create(React.createElement(Open, props)); });
  assert.equal(tree.root.findByProps({ 'aria-label': '3.5 maximum rating' }).props.value, '', 'Division labels do not imply a rating limit');
  await act(async () => {
    tree.root.findByProps({ type: 'datetime-local' }).props.onChange({ target: { value: '2099-01-01T09:00' } });
    tree.root.findByProps({ 'aria-label': '3.5 maximum rating' }).props.onChange({ target: { value: '3.75' } });
    tree.root.findByProps({ 'aria-label': '3.5 team composition' }).props.onChange({ target: { value: '2' } });
  });
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  const writes = requests.filter(r => r.options.method);
  assert.equal(writes.length, 1);
  assert.ok(writes[0].url.includes('/clubs/alpha/'));
  const body = JSON.parse(writes[0].options.body);
  assert.equal(body.expected_revision, 2);
  assert.deepEqual(body.rules, season.rules);
  await act(async () => finish(reply({ detail: 'Draft changed. Reload.' }, 409)));
  assert.equal(tree.root.findAllByType('form').length, 0, 'Conflicting open cannot be repeated blindly');
  exists = true;
  await act(async () => button(tree, 'Check again').props.onClick());
  assert.ok(tree.root.findAllByType('a').some(a => a.props.href === `/admin/interclub/registrations?season=${sid}`));
  await act(async () => tree.unmount());
}

async function clubsAndRosters() {
  let clubId = 'beta', identity = 'b', role = 'administrator', token = 'token-1';
  let requests = [], finish, teams = [], ownStatus = 'invited';
  const participation = () => ({ season_id: sid, club_id: 'beta', status: ownStatus, revision: 2 });
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/players?')) return reply({ players: [...lineup, { name: 'Player 5', player_id: '5', starting_rating: 3.4 }].map(p => ({ id: p.player_id, name: p.name, starting_rating: p.starting_rating })), next_offset: null });
    if (url.endsWith('/history')) return reply({ history: [{ ...team, submitted_at: '2099-01-01T00:00:00Z' }] });
    return reply({ season, is_organizer: clubId === 'alpha', own_participation: clubId === 'beta' ? participation() : null,
      participations: [participation(), { season_id: sid, club_id: 'gamma', status: 'invited', revision: 1 }],
      clubs: ['alpha', 'beta', 'gamma'].map(id => ({ id, name: `${id} Club`, slug: id })), teams, next_team_offset: null });
  };
  const Page = load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: token, loading: false, session: { user: { id: identity }, capabilities: { assignments: [{ club_id: clubId, role }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: ({ triggerLabel, onConfirm }) => React.createElement('button', { onClick: onConfirm }, triggerLabel) },
    './registrations.module.css': {}
  }).default;
  let tree;
  await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid })); });
  assert.equal(button(tree, 'Add a four-player team'), undefined);
  assert.ok(!requests.some(r => r.url.includes('/players')));
  await act(async () => { void button(tree, 'Accept season invitation').props.onClick(); void button(tree, 'Accept season invitation').props.onClick(); });
  assert.equal(requests.filter(r => r.options.method).length, 1);
  assert.ok(requests.at(-1).url.endsWith('/participations/beta'));
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { action: 'accept', expected_revision: 2 });
  ownStatus = 'accepted';
  await act(async () => finish(reply({ participation: participation() })));
  assert.equal(button(tree, 'Add a four-player team').props.disabled, false);
  await act(async () => button(tree, 'Add a four-player team').props.onClick());
  assert.ok(requests.at(-1).url.includes(`/clubs/beta/interclub/registrations/${sid}/players`));
  const name = () => tree.root.findAllByType('input').find(i => i.props.maxLength === 80 && !i.props.type);
  await act(async () => name().props.onChange({ target: { value: 'Beta Blue' } }));
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, true);
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' }).slice(0, 4).forEach(i => i.props.onChange()));
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, false);
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' })[4].props.disabled, true, 'Fifth player cannot be selected');
  const beforeRefresh = requests.length; token = 'token-2';
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(requests.length, beforeRefresh, 'Token refresh preserves unsaved roster');
  assert.equal(name().props.value, 'Beta Blue');
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  const write = requests.at(-1);
  assert.equal(write.options.method, 'PUT');
  assert.equal(write.options.headers.Authorization, 'Bearer token-2');
  assert.deepEqual(JSON.parse(write.options.body), { expected_revision: 0, name: 'Beta Blue', division: '3.5', player_ids: ['1', '2', '3', '4'] });
  await act(async () => finish(reply({ detail: 'Roster changed. Reload.' }, 409)));
  assert.equal(name().props.value, 'Beta Blue');
  assert.equal(tree.root.findByType('fieldset').props.disabled, true, 'Conflict preserves and disables the draft');
  teams = [team];
  await act(async () => button(tree, 'Reload season').props.onClick());
  assert.equal(tree.root.findAllByType('form').length, 0);
  assert.equal(button(tree, 'Approve exception'), undefined, 'Represented club cannot approve its own exception');
  await act(async () => button(tree, 'Edit roster').props.onClick());
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  const oldFinish = finish, oldSignal = requests.at(-1).options.signal;
  clubId = 'alpha'; identity = 'a';
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(oldSignal.aborted, true, 'Switching accounts/clubs cancels old responses');
  await act(async () => oldFinish(reply({ detail: 'OLD REQUEST FAILURE' }, 503)));
  assert.ok(!JSON.stringify(tree.toJSON()).includes('OLD REQUEST FAILURE'));
  assert.equal(button(tree, 'Edit roster'), undefined, 'Organizer cannot edit another club roster');
  assert.equal(button(tree, 'Approve exception').props.disabled, true);
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Approved for this roster' } }));
  await act(async () => button(tree, 'Approve exception').props.onClick());
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { expected_revision: 1, approve: true, reason: 'Approved for this roster' });
  teams = [{ ...team, status: 'exception_approved', decision_reason: 'Approved for this roster' }];
  await act(async () => finish(reply({ team: teams[0] })));
  assert.equal(button(tree, 'Approve exception'), undefined);
  await act(async () => button(tree, 'Roster history').props.onClick());
  assert.ok(JSON.stringify(tree.toJSON()).includes('Roster history (latest 50 versions)'));
  role = 'operator'; const requestCount = requests.length;
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(requests.length, requestCount);
  assert.equal(tree.root.findAllByType('textarea').length, 0);
  await act(async () => tree.unmount());
}

(async () => { await opening(); await clubsAndRosters(); console.log('Interclub registration: explicit rules, acceptance, scoped player selection, four-player rosters, stale saves, role separation, history and token/account changes passed.'); })()
  .catch(e => { console.error(e); process.exitCode = 1; });
