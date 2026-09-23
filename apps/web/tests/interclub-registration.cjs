const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => n === '@/lib/interclubRegistrationWindow' ? load('lib/interclubRegistrationWindow.ts') : n === '@/lib/useRegistrationWindow' ? load('lib/useRegistrationWindow.ts', { './interclubRegistrationWindow': load('lib/interclubRegistrationWindow.ts') }) : n === './SeasonRegistrationWindow' ? load('app/admin/interclub/registrations/SeasonRegistrationWindow.tsx', { '@/lib/interclubRegistration': helpers, './registrations.module.css': {} }) : n === '@/lib/interclubSetup' ? load('lib/interclubSetup.ts') : n === '../InterclubWorkflow' ? load('app/admin/interclub/InterclubWorkflow.tsx', { 'next/link': Link, './workflow.module.css': {} }) : n === './SeasonMeetSchedule' ? { default: () => null, __esModule: true } : n === './SeasonEligibilityApprovals' ? { default: p => React.createElement('section', { 'data-eligibility-root': p.root, 'data-refresh-key': p.refreshKey, onDecision: p.onDecision }), __esModule: true } : n === './PlayerPoolPanels' ? { SeasonPlayerPool: p => React.createElement('section', { 'data-pool-root': p.root, 'data-refresh-key': p.refreshKey, onLateRequested: p.onLateRequested }), MeetAvailability: p => React.createElement('section', { 'data-availability-root': p.meetRoot, onResponses: p.onResponses }) } : Object.hasOwn(mocks, n) ? mocks[n] : require(n), module, module.exports);
  return module.exports;
}
const helpers = load('lib/interclubRegistration.ts');
const Link = ({ children, ...p }) => React.createElement('a', p, children);
const button = (tree, label) => tree.root.findAllByType('button').find(b => b.children.includes(label));
const content = tree => JSON.stringify(tree.toJSON());
const textContent = tree => {
  const visit = node => typeof node === 'string' ? node : Array.isArray(node) ? node.map(visit).join('') : node ? visit(node.children || []) : '';
  return visit(tree.toJSON());
};
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const mid = '00000000-0000-4000-8000-000000000003', mid2 = '00000000-0000-4000-8000-000000000004';
const sid = '00000000-0000-4000-8000-000000000001', tid = '00000000-0000-4000-8000-000000000002';
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => tid }, configurable: true });
const closedRegistration = { opens_at: '2000-01-01T00:00:00Z', closes_at: '2000-02-01T00:00:00Z', revision: 1, status: 'closed', can_register: false, meet_planning_open: true };
const season = { registration: closedRegistration, id: sid, organizer_club_id: 'alpha', source_revision: 2, roster_deadline: '2099-01-01T00:00:00Z', details: {
  name: 'Coastal League', start_date: '2099-01-10', end_date: '2099-03-31', timezone: 'America/Mazatlan', divisions: ['3.5'], club_ids: ['beta', 'gamma'], meets: []
}, rules: { '3.5': { min_rating: null, max_rating: 3.75, women_required: 2 } } };
const lineup = [1, 2, 3, 4].map(n => ({ entry_id: `entry-${n}`, player_id: String(n), name: `Player ${n}`, starting_rating: 3.5, gender: n <= 2 ? 'female' : 'male' }));
const meet = { id: mid, season_id: sid, host_club_id: 'beta', club_ids: ['beta', 'gamma'], starts_at: '2099-01-10T18:00:00Z', roster_deadline: '2099-01-10T12:00:00Z', revision: 2, roster_open: true, deadline_editable: false, courts: 4 };
const secondMeet = { ...meet, id: mid2, starts_at: '2099-02-10T18:00:00Z', revision: 1, deadline_editable: true };
const team = { meet_id: mid, id: tid, club_id: 'beta', season_id: sid, name: 'Beta Blue', division: '3.5', revision: 1, withdrawn: false, status: 'needs_exception', roster: lineup, issues: [{ code: 'rating_above_maximum', message: 'Player exceeds rating limit.' }], late_change: true, decision_reason: null };

async function clubsAndRosters() {
  const originalWindow = global.window;
  global.window = Object.assign(new EventTarget(), { location: { hash: '#invitation-title', search: `?season=${sid}` } });
  let clubId = 'beta', identity = 'b', role = 'administrator', token = 'token-1';
  let requests = [], finish, teams = [], ownStatus = 'invited';
  const participation = () => ({ season_id: sid, club_id: 'beta', status: ownStatus, revision: 2 });
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/players?')) return reply({ players: [...lineup, { name: 'Player 5', player_id: '5', starting_rating: 3.4, gender: 'female' }].map(p => ({ id: p.player_id, name: p.name, starting_rating: p.starting_rating, gender: p.gender })), next_offset: null });
    if (url.endsWith('/history')) return reply({ history: [{ ...team, submitted_at: '2099-01-01T00:00:00Z' }] });
    if (url.includes(`/meets/${mid2}`)) return reply({ meet: secondMeet, teams: [], next_team_offset: null });
    if (url.includes(`/meets/${mid}`)) return reply({ meet, teams, next_team_offset: null });
    return reply({ season, meets: [meet, secondMeet], is_organizer: clubId === 'alpha', own_participation: clubId === 'beta' ? participation() : null,
      participations: [participation(), { season_id: sid, club_id: 'gamma', status: 'invited', revision: 1 }],
      clubs: ['alpha', 'beta', 'gamma'].map(id => ({ id, name: `${id} Club`, slug: id })), teams: [], next_team_offset: null });
  };
  const Page = load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: token, loading: false, session: { user: { id: identity }, capabilities: { assignments: [{ club_id: clubId, role }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: ({ triggerLabel, onConfirm }) => React.createElement('button', { onClick: onConfirm }, triggerLabel) },
    './registrations.module.css': {}
  }).default;
  let tree; const focused = [], scrolled = [];
  await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid }), { createNodeMock: element => ({
    focus() { focused.push(element.props.id); }, scrollIntoView() { scrolled.push(element.props.id); }
  }) }); });
  assert.equal(button(tree, 'Add a team for this meet'), undefined);
  assert.equal(scrolled[0], 'invitation-title', 'The notification scrolls to the loaded invitation');
  assert.equal(focused[0], 'invitation-title', 'The invitation receives keyboard focus');
  assert.ok(requests.every(request => !request.options.method), 'Opening the invitation does not accept it');
  assert.ok(!requests.some(r => r.url.includes('/players')));
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0, 'Pending invitation does not show meet roster controls');
  assert.ok(!requests.some(r => r.url.includes('/meets/')), 'Pending invitation does not fetch meet details');
  assert.ok(textContent(tree).includes('Coastal League') && textContent(tree).includes('alpha Club'));
  assert.ok(textContent(tree).includes('Below 4.0') && textContent(tree).includes('Players may play up'), 'Existing seasons explain the upper limit and playing up without imposing a lower floor');
  const seasonReadsBeforeAcceptance = requests.filter(r => r.url.endsWith(`/registrations/${sid}`)).length;
  await act(async () => { void button(tree, 'Accept invitation').props.onClick(); void button(tree, 'Accept invitation').props.onClick(); });
  assert.equal(requests.filter(r => r.options.method).length, 1);
  assert.equal(button(tree, 'Accepting…').props.disabled, true);
  assert.ok(requests.at(-1).url.endsWith('/participations/beta'));
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { action: 'accept', expected_revision: 2 });
  ownStatus = 'accepted';
  await act(async () => finish(reply({ participation: participation() })));
  assert.equal(requests.filter(r => r.url.endsWith(`/registrations/${sid}`)).length, seasonReadsBeforeAcceptance, 'Acceptance updates immediately without reloading the season');
  assert.ok(textContent(tree).includes('beta Club has joined'));
  assert.ok(button(tree, 'Next: check meet availability'));
  assert.equal(focused.at(-1), 'participation-confirmed', 'Acceptance brings keyboard focus to the confirmation');
  assert.equal(tree.root.findByProps({ id: 'season-player-pool' }).props.hidden, false, 'Accepted clubs open directly on their player pool');
  assert.equal(tree.root.findAll(n => n.props['data-pool-root'])[0].props['data-pool-root'], `https://api.test/admin/clubs/beta/interclub/registrations/${sid}`);
  await act(async () => tree.root.findByProps({ 'aria-label': 'Lineups' }).props.onClick({ preventDefault() {} }));
  assert.equal(focused.at(-1), 'meet-rosters');
  assert.equal(scrolled.at(-1), 'meet-rosters', 'Next action brings the meet controls into view');
  assert.equal(button(tree, 'Accept invitation'), undefined);
  assert.equal(button(tree, 'Add a team for this meet').props.disabled, false);
  assert.ok(tree.root.findByProps({ 'aria-label': 'Run meet' }).props.href.includes(`meet=${mid}`), 'Operations keeps the selected meet');
  assert.ok(tree.root.findByProps({ 'aria-label': 'Run meet' }).props.href.includes(`season=${sid}`), 'Operations keeps the selected season');
  await act(async () => button(tree, 'Add a team for this meet').props.onClick());
  assert.ok(requests.at(-1).url.includes(`/clubs/beta/interclub/registrations/${sid}/meets/${mid}/players`));
  const name = () => tree.root.findAllByType('input').find(i => i.props.maxLength === 80 && !i.props.type);
  await act(async () => name().props.onChange({ target: { value: 'Beta Blue' } }));
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, true);
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' }).slice(0, 4).forEach(i => i.props.onChange()));
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, false);
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' })[4].props.disabled, true, 'Fifth player cannot be selected');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Meet availability' }).props.onClick({ preventDefault() {} }));
  assert.equal(name().props.value, 'Beta Blue', 'Switching to availability preserves the lineup draft');
  await act(async () => tree.root.findAll(n => n.props['data-availability-root'])[0].props.onResponses([
    { member_id: 'm1', player_id: '1', name: 'Player 1', status: 'available', member_status: 'withdrawn' },
    { member_id: 'm2', player_id: '2', name: 'Player 2', status: 'available', member_status: 'active' },
    { member_id: 'm5', player_id: '5', name: 'Player 5', status: 'unavailable', member_status: 'active' },
  ]));
  assert.ok(textContent(tree).includes('Sending availability invitations is optional.'));
  await act(async () => button(tree, 'Choose lineups now').props.onClick());
  assert.equal(name().props.value, 'Beta Blue', 'Returning to lineups retains the draft and players');
  const availableFilter = () => tree.root.findAllByType('label').find(label => label.children.includes('Show only players who said they are available')).findByType('input');
  await act(async () => availableFilter().props.onChange({ target: { checked: true } }));
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' }).length, 6, 'Availability filter preserves four selected players');
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' })[1].props.onChange());
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' }).length, 5, 'Withdrawn pool member is not suggested by their historical available reply');
  await act(async () => availableFilter().props.onChange({ target: { checked: false } }));
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' }).length, 7, 'Clearing availability filter allows manual substitutes');
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' })[1].props.onChange());
  const beforeRefresh = requests.length; token = 'token-2';
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(requests.length, beforeRefresh, 'Token refresh preserves unsaved roster');
  assert.equal(name().props.value, 'Beta Blue');
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  const write = requests.at(-1);
  assert.ok(write.url.includes(`/meets/${mid}/teams/`), 'Roster writes identify the selected meet');
  assert.equal(write.options.method, 'PUT');
  assert.equal(write.options.headers.Authorization, 'Bearer token-2');
  assert.deepEqual(JSON.parse(write.options.body), { expected_meet_revision: 2, expected_revision: 0, name: 'Beta Blue', division: '3.5', player_ids: ['2', '3', '4', '1'] });
  await act(async () => finish(reply({ detail: 'Roster changed. Reload.' }, 409)));
  assert.equal(name().props.value, 'Beta Blue');
  assert.equal(tree.root.findByType('fieldset').props.disabled, true, 'Conflict preserves and disables the draft');
  teams = [team];
  await act(async () => button(tree, 'Reload meet').props.onClick());
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
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { expected_meet_revision: 2, expected_revision: 1, approve: true, reason: 'Approved for this roster' });
  teams = [{ ...team, status: 'exception_approved', decision_reason: 'Approved for this roster' }];
  await act(async () => finish(reply({ team: teams[0] })));
  assert.equal(button(tree, 'Approve exception'), undefined);
  await act(async () => button(tree, 'Roster history').props.onClick());
  assert.ok(JSON.stringify(tree.toJSON()).includes('Roster history (latest 50 versions)'));
  // The next meet starts empty and has its own organizer deadline.
  await act(async () => tree.root.findByProps({ 'aria-label': 'Meet' }).props.onChange({ target: { value: mid2 } }));
  assert.ok(JSON.stringify(tree.toJSON()).includes('No teams submitted for this meet yet.'));
  assert.ok(tree.root.findByProps({ 'aria-label': 'Run meet' }).props.href.includes(`meet=${mid2}`), 'Changing meets updates all workflow links');
  assert.equal(button(tree, 'Roster history'), undefined, 'First-meet roster does not carry into the next meet');
  await act(async () => tree.root.findByProps({ type: 'datetime-local' }).props.onChange({ target: { value: '2099-02-10T09:00' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.ok(requests.at(-1).url.endsWith(`/meets/${mid2}/deadline`));
  assert.equal(JSON.parse(requests.at(-1).options.body).expected_revision, 1);
  await act(async () => finish(reply({ meet: secondMeet })));
  // Club roster drafts and in-flight saves are discarded when selecting a different meet.
  clubId = 'beta'; identity = 'b';
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  await act(async () => button(tree, 'Edit roster').props.onClick());
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  const firstMeetFinish = finish, firstMeetSignal = requests.at(-1).options.signal;
  await act(async () => tree.root.findByProps({ 'aria-label': 'Meet' }).props.onChange({ target: { value: mid2 } }));
  assert.equal(firstMeetSignal.aborted, true);
  await act(async () => firstMeetFinish(reply({ detail: 'OLD MEET FAILURE' }, 503)));
  assert.ok(!JSON.stringify(tree.toJSON()).includes('OLD MEET FAILURE'));
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => button(tree, 'Add a team for this meet').props.onClick());
  assert.equal(tree.root.findAllByProps({ type: 'checkbox' }).filter(i => i.props.checked).length, 0, 'Next meet starts with no assumed players');
  assert.equal(name().props.value, 'beta Club 3.5');
  assert.ok(requests.at(-1).url.includes(`/meets/${mid2}/players`));
  await act(async () => tree.root.findByProps({ 'aria-label': 'Missing pairing forfeit' }).props.onChange({ target: { checked: true } }));
  assert.ok(textContent(tree).includes('missing pairing will forfeit all three games'));
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' }).slice(0, 2).forEach(i => i.props.onChange()));
  await act(async () => name().props.onChange({ target: { value: 'Beta Pair Only' } }));
  assert.equal(button(tree, 'Submit two-player roster with forfeit').props.disabled, false);
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { expected_meet_revision: 1, expected_revision: 0, name: 'Beta Pair Only', division: '3.5', player_ids: ['1', '2'], missing_pairing_forfeit: true });
  await act(async () => finish(reply({ team: { ...team, meet_id: mid2, name: 'Beta Pair Only', roster: lineup.slice(0, 2), status: 'eligible' } })));
  secondMeet.roster_open = false; secondMeet.deadline_editable = false;
  await act(async () => button(tree, 'Reload meet').props.onClick());
  assert.equal(button(tree, 'Add a team for this meet'), undefined, 'Past-meet rosters are history');
  role = 'operator'; const requestCount = requests.length;
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(requests.length, requestCount);
  assert.equal(tree.root.findAllByType('textarea').length, 0);
  await act(async () => tree.unmount());
  global.window = originalWindow;
}

async function invitationResponses() {
  let clubId = 'beta', organizer = false, status = 'invited', revision = 2;
  let requests = [], finish;
  const participation = () => ({ season_id: sid, club_id: clubId, status, revision });
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/meets/')) return reply({ meet, teams: [], next_team_offset: null });
    return reply({ season, meets: [meet], is_organizer: organizer, own_participation: participation(),
      participations: [participation()], clubs: ['alpha', 'beta', 'gamma'].map(id => ({ id, name: `${id} Club`, slug: id })), teams: [], next_team_offset: null });
  };
  const Page = load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'token', loading: false, session: { user: { id: 'staff' }, capabilities: { assignments: [{ club_id: clubId, role: 'administrator' }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: () => null }, './registrations.module.css': {}
  }).default;
  let tree;
  const mount = async () => { await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid })); }); };
  await mount();
  const respond = async label => { await act(async () => { void button(tree, label).props.onClick(); }); };
  await respond('Accept invitation');
  await act(async () => finish(reply({ detail: 'Invitation cannot be accepted right now.' }, 400)));
  assert.ok(textContent(tree).includes('Invitation cannot be accepted right now.'));
  assert.equal(button(tree, 'Accept invitation').props.disabled, false, 'A rejected request can be retried');
  assert.equal(button(tree, 'Next: check meet availability'), undefined, 'Failed acceptance must not appear joined');
  assert.ok(!requests.some(r => r.url.includes('/meets/')));
  await respond('Accept invitation');
  await act(async () => finish(reply({ detail: 'Invitation changed. Reload before responding.' }, 409)));
  assert.equal(button(tree, 'Accept invitation').props.disabled, true);
  const writes = requests.filter(r => r.options.method).length;
  await respond('Accept invitation');
  assert.equal(requests.filter(r => r.options.method).length, writes, 'Stale invitation revision blocks another write');
  revision = 3;
  await act(async () => button(tree, 'Reload season').props.onClick());
  await respond('Decline invitation');
  assert.equal(button(tree, 'Declining…').props.disabled, true);
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { action: 'decline', expected_revision: 3 });
  status = 'declined'; revision = 4;
  await act(async () => finish(reply({ participation: participation() })));
  assert.equal(button(tree, 'Accept invitation'), undefined);
  assert.equal(button(tree, 'Next: check meet availability'), undefined);
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0);
  assert.ok(!requests.some(r => r.url.includes('/meets/')), 'Declining does not load roster details');
  await act(async () => tree.unmount());
  await mount();
  assert.equal(button(tree, 'Accept invitation'), undefined, 'Reopened declined invitation stays closed');
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0);
  await act(async () => tree.unmount());

  status = 'accepted';
  await mount();
  assert.ok(textContent(tree).includes('beta Club has joined'), 'Reopening an accepted invitation shows its saved outcome');
  assert.ok(button(tree, 'Next: check meet availability'));
  assert.equal(button(tree, 'Accept invitation'), undefined);
  await act(async () => tree.unmount());

  status = 'cancelled';
  await mount();
  assert.equal(button(tree, 'Accept invitation'), undefined);
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0, 'Cancelled invitation has no roster controls');
  await act(async () => tree.unmount());

  status = 'invited'; requests = [];
  await mount();
  await respond('Accept invitation');
  const staleFinish = finish, signal = requests.at(-1).options.signal;
  clubId = 'gamma';
  await act(async () => tree.update(React.createElement(Page, { initialSeasonId: sid })));
  assert.equal(signal.aborted, true, 'Club change cancels in-flight invitation acceptance');
  await act(async () => staleFinish(reply({ participation: { season_id: sid, club_id: 'beta', status: 'accepted', revision: 5 } })));
  assert.ok(!textContent(tree).includes('beta Club has joined'), 'Old club acceptance does not overwrite the selected club');
  assert.equal(button(tree, 'Accept invitation').props.disabled, false);
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0);
  await act(async () => tree.unmount());

  await mount();
  await respond('Accept invitation');
  await act(async () => finish(reply({ participation: { season_id: sid, club_id: 'beta', status: 'accepted', revision: 5 } })));
  assert.equal(button(tree, 'Accept invitation').props.disabled, true, 'Mismatched club response requires reloading');
  assert.ok(textContent(tree).includes('Could not confirm your club’s response.'));
  assert.equal(button(tree, 'Next: check meet availability'), undefined, 'Another club’s success cannot unlock roster controls');
  await act(async () => tree.unmount());

  clubId = 'alpha'; organizer = true;
  await mount();
  assert.ok(textContent(tree).includes('Club responses'));
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 1, 'Organizer can manage meets before accepting its own participation');
  assert.ok(button(tree, 'Accept invitation'));
  await act(async () => tree.unmount());
}

async function deepLinkContext() {
  const reads = [];
  const otherClubsMeet = { ...meet, id: 'other-clubs-meet', club_ids: ['alpha', 'gamma'] };
  global.fetch = async (url, options) => {
    reads.push(url); assert.equal(options.method, undefined);
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/meets/')) return reply({ meet: { ...secondMeet, roster_open: true }, teams: [], next_team_offset: null });
    return reply({ season: { ...season, details: { ...season.details, divisions: ['4.0', '3.0', '3.5'] } }, meets: [otherClubsMeet, meet, secondMeet], is_organizer: true,
      own_participation: { season_id: sid, club_id: 'beta', status: 'accepted', revision: 1 }, participations: [],
      clubs: [{ id: 'beta', name: 'Beta Club', slug: 'beta' }], teams: [], next_team_offset: null });
  };
  const Page = load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: 'beta' }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'token', loading: false, session: { user: { id: 'b' }, capabilities: { assignments: [{ club_id: 'beta', role: 'administrator' }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: () => null }, './registrations.module.css': {}
  }).default;
  const Route = load('app/admin/interclub/registrations/page.tsx', { './RegistrationWorkspace': { default: Page, __esModule: true } }).default;
  let tree;
  await act(async () => { tree = create(React.createElement(Route, { searchParams: { season: sid, meet: mid2, step: 'lineups' } })); });
  assert.equal(tree.root.findByProps({ 'aria-label': 'Meet' }).props.value, mid2, 'A link from operations opens the requested meet instead of the first meet');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Lineups' }).props['aria-current'], 'step');
  assert.equal(tree.root.findByProps({ id: 'season-player-pool' }).props.hidden, true);
  assert.equal(tree.root.findByProps({ id: 'meet-rosters' }).props.hidden, false);
  assert.ok(reads.some(url => url.endsWith(`/meets/${mid2}`)) && !reads.some(url => url.endsWith(`/meets/${mid}`)), 'Only the requested meet is loaded');
  const rulesTable = tree.root.findAllByType('table').find(table => table.findAllByType('caption').some(caption => caption.children.includes('Season eligibility rules')));
  assert.deepEqual(rulesTable.findByType('tbody').findAllByType('tr').map(row => row.findAllByType('td')[0].children[0]), ['3.0', '3.5', '4.0'], 'Skill levels display in numerical order');
  for (const label of ['Player pool', 'Meet availability', 'Run meet', 'Approve results']) {
    const url = new URL(tree.root.findByProps({ 'aria-label': label }).props.href, 'https://example.test');
    assert.equal(url.searchParams.get('season'), sid);
    assert.equal(url.searchParams.get('meet'), mid2);
  }
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(Route, { searchParams: { season: sid } })); });
  assert.equal(tree.root.findByProps({ 'aria-label': 'Meet' }).props.value, mid, 'Organizer defaults to its own next meet, not the first visible meet for other clubs');
  await act(async () => button(tree, 'Next: check meet availability').props.onClick());
  assert.equal(tree.root.findByProps({ 'aria-label': 'Meet' }).props.value, mid, 'Pool next action opens the own-club meet shown in the confirmation');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Meet availability' }).props['aria-current'], 'step');
  await act(async () => tree.unmount());
}

function phaseWorkspace(commissioner = false) {
  return load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: commissioner ? 'alpha' : 'beta' }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'token', loading: false, session: { user: { id: 'staff' }, capabilities: { assignments: [{ club_id: commissioner ? 'alpha' : 'beta', role: 'administrator' }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: () => null }, './registrations.module.css': {}
  }).default;
}

async function registrationPhaseLocks() {
  const windows = [undefined,
    { opens_at: null, closes_at: null, revision: 0, status: 'unconfigured', can_register: false, meet_planning_open: false },
    { opens_at: '2099-01-01T00:00:00Z', closes_at: '2099-02-01T00:00:00Z', revision: 1, status: 'scheduled', can_register: false, meet_planning_open: false },
    { opens_at: '2000-01-01T00:00:00Z', closes_at: '2099-02-01T00:00:00Z', revision: 1, status: 'open', can_register: true, meet_planning_open: false },
    { ...closedRegistration, meet_planning_open: false },
  ];
  for (const registration of windows) {
    const reads = [], selectedSeason = { ...season, registration };
    global.fetch = async (url, options) => {
      reads.push(url); assert.equal(options.method, undefined);
      if (url.endsWith('/registrations')) return reply({ seasons: [selectedSeason] });
      return reply({ season: selectedSeason, meets: [meet], is_organizer: false, own_participation: { status: 'accepted' }, participations: [], clubs: [], teams: [], next_team_offset: null });
    };
    let tree; const Page = phaseWorkspace();
    await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid, initialMeetId: mid, initialStep: 'availability' })); });
    assert.equal(tree.root.findByProps({ id: 'season-player-pool' }).props.hidden, false, 'A locked meet deep link returns to the player pool');
    assert.equal(tree.root.findByProps({ 'aria-label': 'Player pool' }).props['aria-current'], 'step');
    assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0, 'Meet selector is not mounted before registration closes');
    assert.equal(tree.root.findAllByProps({ id: 'meet-rosters' }).length, 0, 'Locked meet forms are absent, not merely hidden');
    assert.equal(tree.root.findAllByProps({ 'aria-disabled': 'true' }).length, 4);
    assert.equal(button(tree, 'Next: check meet availability'), undefined);
    assert.equal(button(tree, 'Set registration dates'), undefined, 'A participating club cannot set commissioner dates');
    assert.equal(button(tree, 'Edit registration dates'), undefined);
    assert.ok(!reads.some(url => url.includes('/meets/')), 'A locked deep link never fetches meet settings or availability');
    assert.ok(textContent(tree).includes('Meet planning opens after registration closes.'));
    await act(async () => tree.unmount());
  }
}

async function commissionerWindowEditor() {
  const Window = load('app/admin/interclub/registrations/SeasonRegistrationWindow.tsx', { '@/lib/interclubRegistration': helpers, './registrations.module.css': {} }).default;
  const unconfigured = { opens_at: null, closes_at: null, revision: 0, status: 'unconfigured', can_register: false, meet_planning_open: false };
  const requests = []; let finish, saved;
  global.fetch = (url, options) => new Promise(resolve => { requests.push({ url, options }); finish = resolve; });
  const props = { root: `https://api.test/registrations/${sid}`, accessToken: 'token', season: { ...season, registration: unconfigured }, commissioner: false, firstMeetAt: '2027-01-11T17:00:00Z', onSaved: value => { saved = value; }, onReload() {} };
  let tree;
  await act(async () => { tree = create(React.createElement(Window, props)); });
  assert.equal(tree.root.findAllByType('button').length, 0, 'Other clubs see the dates without date controls');
  await act(async () => tree.update(React.createElement(Window, { ...props, commissioner: true })));
  await act(async () => button(tree, 'Set registration dates').props.onClick());
  const input = label => tree.root.findByProps({ 'aria-label': label });
  assert.equal(input('Registration opens').props.value, '', 'No opening date is invented');
  assert.equal(input('Registration closes').props.value, '', 'No closing date is invented');
  assert.equal(input('Registration closes').props.max, '2027-01-11T10:00', 'Closing limit uses the league timezone and real first meet');
  await act(async () => {
    input('Registration opens').props.onChange({ target: { value: '2027-01-10T09:00' } });
    input('Registration closes').props.onChange({ target: { value: '2027-01-10T08:00' } });
  });
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(requests.length, 0); assert.ok(textContent(tree).includes('Registration must close after it opens.'));
  await act(async () => input('Registration closes').props.onChange({ target: { value: '2027-01-12T10:00' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(requests.length, 0); assert.ok(textContent(tree).includes('Registration must close by the first scheduled meet.'));
  await act(async () => input('Registration closes').props.onChange({ target: { value: '2027-01-11T10:00' } }));
  await act(async () => { tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  assert.equal(requests.length, 1, 'Repeated save creates one commissioner mutation');
  assert.ok(requests[0].url.endsWith('/registration-window'));
  assert.deepEqual(JSON.parse(requests[0].options.body), { expected_revision: 0, opens_at: '2027-01-10T16:00:00.000Z', closes_at: '2027-01-11T17:00:00.000Z' });
  await act(async () => finish(reply({ season: { ...season, registration: { ...unconfigured, ...JSON.parse(requests[0].options.body), status: 'scheduled', revision: 1 } } })));
  assert.equal(saved.registration.revision, 1);
  assert.ok(textContent(tree).includes('Registration dates saved for every club in this league.'));
  const configuredProps = { ...props, commissioner: true, season: { ...season, registration: saved.registration } };
  await act(async () => tree.update(React.createElement(Window, configuredProps)));
  await act(async () => button(tree, 'Edit registration dates').props.onClick());
  await act(async () => input('Registration opens').props.onChange({ target: { value: '2027-01-09T09:00' } }));
  await act(async () => tree.update(React.createElement(Window, { ...configuredProps, season: { ...configuredProps.season, registration: { ...saved.registration, revision: 2, opens_at: '2027-01-08T16:00:00Z' } } })));
  assert.equal(input('Registration opens').props.value, '2027-01-09T09:00', 'A concurrent commissioner edit preserves the local draft');
  assert.equal(tree.root.findByType('fieldset').props.disabled, true, 'Changed dates require a deliberate reload before overwriting');
  await act(async () => tree.unmount());
}

async function savedClosedWindowLoadsMeets() {
  let registration = { opens_at: null, closes_at: null, revision: 0, status: 'unconfigured', can_register: false, meet_planning_open: false };
  const reads = [], Page = phaseWorkspace(true);
  global.fetch = async (url, options) => {
    reads.push(url);
    if (options.method === 'PUT') {
      const body = JSON.parse(options.body);
      registration = { opens_at: body.opens_at, closes_at: body.closes_at, revision: 1, status: 'closed', can_register: false, meet_planning_open: true };
      return reply({ season: { ...season, registration } });
    }
    if (url.endsWith('/registrations')) return reply({ seasons: [{ ...season, registration }] });
    if (url.includes('/meets/')) return reply({ meet, teams: [], next_team_offset: null });
    return reply({ season: { ...season, registration }, meets: registration.meet_planning_open ? [meet] : [], first_meet_at: meet.starts_at,
      is_organizer: true, own_participation: { status: 'accepted' }, participations: [], clubs: [], teams: [], next_team_offset: null });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid, initialMeetId: mid })); });
  await act(async () => button(tree, 'Set registration dates').props.onClick());
  await act(async () => {
    tree.root.findByProps({ 'aria-label': 'Registration opens' }).props.onChange({ target: { value: '2000-01-01T09:00' } });
    tree.root.findByProps({ 'aria-label': 'Registration closes' }).props.onChange({ target: { value: '2000-02-01T09:00' } });
  });
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.ok(reads.some(url => url.endsWith(`/meets/${mid}`)), 'Saving a closed window refreshes the full season and loads its existing meets');
  assert.equal(tree.root.findAllByType('a').filter(node => node.props['aria-label'] === 'Lineups').length, 1);
  await act(async () => tree.unmount());
}

async function serverConfirmedWindowBoundary() {
  const originalNow = Date.now, originalWindow = global.window;
  let now = Date.parse('2026-09-21T18:00:00Z'), serverClosed = false;
  Date.now = () => now; global.window = new EventTarget();
  const reads = [], Page = phaseWorkspace(); let tree;
  global.fetch = async url => {
    reads.push(url);
    const registration = { opens_at: '2026-09-21T17:00:00Z', closes_at: '2026-09-21T19:00:00Z', revision: 1, status: serverClosed ? 'closed' : 'open', can_register: !serverClosed, meet_planning_open: serverClosed };
    const currentSeason = { ...season, registration };
    if (url.endsWith('/registrations')) return reply({ seasons: [currentSeason] });
    if (url.includes('/meets/')) return reply({ meet, teams: [], next_team_offset: null });
    return reply({ season: currentSeason, meets: serverClosed ? [meet] : [], is_organizer: false, own_participation: { status: 'accepted' }, participations: [], clubs: [], teams: [], next_team_offset: null });
  };
  try {
    await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid, initialMeetId: mid, initialStep: 'lineups' })); });
    now = Date.parse('2026-09-21T19:01:00Z');
    await act(async () => window.dispatchEvent(new Event('focus')));
    assert.ok(!reads.some(url => url.includes('/meets/')), 'Passing the local closing time cannot unlock meet forms without server confirmation');
    serverClosed = true;
    await act(async () => window.dispatchEvent(new Event('focus')));
    assert.ok(reads.some(url => url.endsWith(`/meets/${mid}`)), 'Fresh server-confirmed closure unlocks the selected meet');
    assert.equal(tree.root.findByProps({ 'aria-label': 'Player pool' }).props['aria-current'], 'step', 'Earlier locked deep link stays on the pool until the admin selects a meet step');
    await act(async () => tree.unmount());
  } finally { Date.now = originalNow; global.window = originalWindow; }
}

async function phaseRefreshCannotUndoAcceptance() {
  const originalWindow = global.window; global.window = new EventTarget();
  let status = 'invited', defer = false, finish, oldSignal, tree;
  const Page = phaseWorkspace();
  const details = () => ({ season, meets: [meet], is_organizer: false, own_participation: { season_id: sid, club_id: 'beta', status, revision: status === 'invited' ? 1 : 2 }, participations: [], clubs: [], teams: [], next_team_offset: null });
  global.fetch = async (url, options) => {
    if (options.method) { status = 'accepted'; return reply({ participation: details().own_participation }); }
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/meets/')) return reply({ meet, teams: [], next_team_offset: null });
    if (defer) { const old = details(); oldSignal = options.signal; return new Promise(resolve => { finish = () => resolve(reply(old)); }); }
    return reply(details());
  };
  try {
    await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid })); });
    defer = true;
    await act(async () => window.dispatchEvent(new Event('focus')));
    await act(async () => button(tree, 'Accept invitation').props.onClick());
    assert.equal(oldSignal.aborted, true, 'Accepting participation cancels older background phase reads');
    await act(async () => finish());
    assert.equal(button(tree, 'Accept invitation'), undefined, 'An older phase response cannot undo accepted participation');
    assert.equal(tree.root.findByProps({ id: 'season-player-pool' }).props.hidden, false);
    await act(async () => tree.unmount());
  } finally { global.window = originalWindow; }
}

async function updatedPlayersAndSchedulePreserveLineupDraft() {
  const originalWindow = global.window; global.window = new EventTarget();
  let currentMeet = { ...meet, host_club_id: 'alpha', club_ids: ['alpha', 'beta'], deadline_editable: false };
  let people = lineup.map(player => ({ id: player.player_id, name: player.name, starting_rating: player.starting_rating, gender: player.gender }));
  const reads = [], Page = phaseWorkspace(true); let tree;
  global.fetch = async (url, options) => {
    reads.push(url);
    assert.equal(options.method, undefined, 'This scenario only refreshes confirmed data');
    if (url.endsWith('/registrations')) return reply({ seasons: [season] });
    if (url.includes('/players?')) return reply({ players: people, next_offset: null });
    if (url.includes('/meets/')) return reply({ meet: currentMeet, teams: [], next_team_offset: null });
    return reply({ season, meets: [currentMeet], is_organizer: true, own_participation: { status: 'accepted' }, participations: [], clubs: [{ id: 'alpha', name: 'Alpha Club' }], teams: [], next_team_offset: null });
  };
  try {
    await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid, initialMeetId: mid, initialStep: 'lineups' })); });
    await act(async () => button(tree, 'Add a team for this meet').props.onClick());
    const name = () => tree.root.findAllByType('input').find(input => input.props.maxLength === 80 && !input.props.type);
    await act(async () => name().props.onChange({ target: { value: 'Draft lineup' } }));
    await act(async () => tree.root.findAllByProps({ type: 'checkbox' }).slice(0, 4).forEach(input => input.props.onChange()));
    people = [...people, { id: '5', name: 'Newly Approved Player', starting_rating: 3.2, gender: 'female' }];
    const beforePlayers = reads.filter(url => url.includes('/players?')).length;
    const approvals = () => tree.root.findAll(node => node.props['data-eligibility-root'])[0];
    const pool = () => tree.root.findAll(node => node.props['data-pool-root'])[0];
    await act(async () => approvals().props.onDecision());
    assert.equal(pool().props['data-refresh-key'], 1);
    assert.equal(reads.filter(url => url.includes('/players?')).length, beforePlayers + 1, 'An approval refreshes an already-open player picker');
    assert.ok(textContent(tree).includes('Newly Approved Player'));
    assert.equal(name().props.value, 'Draft lineup');
    assert.equal(tree.root.findAllByProps({ type: 'checkbox' }).filter(input => input.props.checked).length, 4, 'Approval refresh preserves current selections');
    await act(async () => pool().props.onLateRequested());
    assert.equal(approvals().props['data-refresh-key'], 1, 'A new request refreshes commissioner approvals');
    currentMeet = { ...currentMeet, revision: currentMeet.revision + 1, starts_at: '2099-01-11T18:00:00Z' };
    await act(async () => window.dispatchEvent(new Event('focus')));
    assert.equal(name().props.value, 'Draft lineup', 'Schedule changes preserve an unsaved lineup');
    assert.equal(button(tree, 'Submit four-player roster').props.disabled, true);
    assert.ok(textContent(tree).includes('This meet’s schedule changed. Your lineup draft is kept below.'));
    await act(async () => button(tree, 'Reload meet').props.onClick());
    assert.equal(name(), undefined, 'Only the explicit reload discards the stale draft');
    await act(async () => tree.unmount());
  } finally { global.window = originalWindow; }
}

async function loadFailuresCanBeRetried() {
  const requests = [];
  global.fetch = (url, options) => new Promise((resolve, reject) => requests.push({ url, options, resolve, reject }));
  const Page = load('app/admin/interclub/registrations/RegistrationWorkspace.tsx', {
    'next/link': Link, '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/interclubRegistration': helpers,
    '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: 'beta' }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'token', loading: false, session: { user: { id: 'b' }, capabilities: { assignments: [{ club_id: 'beta', role: 'administrator' }] } } }) },
    '@/components/ConfirmAction': { ConfirmAction: () => null }, './registrations.module.css': {}
  }).default;
  let tree;
  const content = () => JSON.stringify(tree.toJSON());
  await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid })); });
  assert.ok(content().includes('Loading club invitations…'));
  assert.equal(button(tree, 'Refresh invitations').props.disabled, true);
  await act(async () => requests.at(-1).reject(new TypeError('Load failed')));
  assert.ok(content().includes('Unable to load club invitations. Try again.'));
  assert.ok(!content().includes('Loading club invitations…'));
  assert.ok(!content().includes('Your club has no season invitations yet.'));
  await act(async () => button(tree, 'Retry loading invitations').props.onClick());
  assert.equal(tree.root.findAllByProps({ role: 'alert' }).length, 0);
  await act(async () => requests.at(-1).resolve(reply({ seasons: [season] })));
  assert.ok(content().includes('Loading season…'));
  assert.equal(button(tree, 'Reload season').props.disabled, true);
  await act(async () => requests.at(-1).reject(new TypeError('Load failed')));
  assert.ok(content().includes('Unable to load this season. Try again.'));
  assert.ok(!content().includes('Loading season…'), 'A failed season request must stop the loading message');
  assert.equal(button(tree, 'Accept invitation'), undefined);
  await act(async () => button(tree, 'Retry loading season').props.onClick());
  assert.equal(tree.root.findAllByProps({ role: 'alert' }).length, 0);
  await act(async () => requests.at(-1).resolve(reply({ season, meets: [meet], is_organizer: false,
    own_participation: { season_id: sid, club_id: 'beta', status: 'accepted', revision: 1 }, participations: [],
    clubs: [{ id: 'beta', name: 'Beta Club', slug: 'beta' }], teams: [], next_team_offset: null })));
  assert.ok(button(tree, 'Next: check meet availability'));
  assert.equal(button(tree, 'Accept invitation'), undefined);
  assert.ok(content().includes('Loading meet…'));
  await act(async () => requests.at(-1).resolve(reply({ detail: 'This meet is temporarily unavailable.' }, 503)));
  assert.ok(content().includes('This meet is temporarily unavailable.'));
  assert.ok(!content().includes('Loading meet…'), 'A failed meet request must stop the loading message');
  await act(async () => button(tree, 'Retry loading meet').props.onClick());
  await act(async () => requests.at(-1).resolve(reply({ meet, teams: [], next_team_offset: null })));
  assert.ok(content().includes('No teams submitted for this meet yet.'));
  assert.equal(tree.root.findAllByProps({ role: 'alert' }).length, 0);
  // Lost access is still enforced, and old season details are removed on a failed reload.
  await act(async () => button(tree, 'Reload season').props.onClick());
  await act(async () => requests.at(-1).resolve(reply({ detail: 'Your club no longer has access to this season.' }, 403)));
  assert.ok(content().includes('Your club no longer has access to this season.'));
  assert.equal(button(tree, 'Accept invitation'), undefined);
  assert.ok(!content().includes('Loading season…'));
  assert.ok(requests.every(r => r.url.includes('/clubs/beta/') && r.options.headers.Authorization === 'Bearer token' && !r.options.method));
  await act(async () => tree.unmount());
}

async function guidedLineupChoices() {
  const people = [
    { id: '1', name: 'Play Up Woman', starting_rating: 2.9, gender: 'female' },
    { id: '2', name: 'Second Woman', starting_rating: 3.4999, gender: 'female' },
    { id: '3', name: 'First Man', starting_rating: 3.4, gender: 'male' },
    { id: '4', name: 'Second Man', starting_rating: 3.2, gender: 'male' },
    { id: '5', name: 'Higher Woman', starting_rating: 3.5, gender: 'female' },
    { id: '6', name: 'Profile Review', starting_rating: 3.1, gender: 'unknown' },
    { id: '7', name: 'Already Picked', starting_rating: 3.1, gender: 'male' },
  ];
  const selectedSeason = { ...season, details: { ...season.details, divisions: ['3.0', '3.5'] } };
  const assigned = { ...team, id: 'other-team', status: 'eligible', issues: [], roster: [{ ...people[6], player_id: '7' }] };
  const requests = [], Page = phaseWorkspace(); let tree;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    if (options.method) return reply({ team: { ...team, status: 'eligible' } });
    if (url.endsWith('/registrations')) return reply({ seasons: [selectedSeason] });
    if (url.includes('/players?')) return reply({ players: people, next_offset: null });
    if (url.includes('/meets/')) return reply({ meet, teams: url.includes('team_offset=100') ? [assigned] : [], next_team_offset: url.includes('team_offset=100') ? null : 100 });
    return reply({ season: selectedSeason, meets: [meet], is_organizer: false, own_participation: { status: 'accepted' }, participations: [], clubs: [{ id: 'beta', name: 'Beta Club' }], teams: [], next_team_offset: null });
  };
  await act(async () => { tree = create(React.createElement(Page, { initialSeasonId: sid })); });
  await act(async () => button(tree, 'Choose players for the next meet').props.onClick());
  assert.equal(tree.root.findByProps({ id: 'meet-rosters' }).props.hidden, false);
  assert.equal(tree.root.findByProps({ 'aria-label': 'Lineups' }).props['aria-current'], 'step');
  assert.ok(requests.some(request => request.url.includes('team_offset=100')), 'Assignments beyond the first page are loaded before selecting players');
  await act(async () => button(tree, 'Add a team for this meet').props.onClick());
  const select = name => tree.root.findByProps({ 'aria-label': name });
  const pick = name => select(`Select ${name}`);
  const choose = async name => act(async () => pick(name).props.onChange());
  assert.equal(pick('Play Up Woman').props.disabled, false, 'Lower-rated players can play up');
  assert.equal(pick('Second Woman').props.disabled, false, 'The exact unrounded rating below the ceiling remains eligible');
  for (const name of ['Higher Woman', 'Profile Review', 'Already Picked']) assert.equal(tree.root.findAllByProps({ 'aria-label': `Select ${name}` }).length, 0);
  await act(async () => select('Player eligibility filter').props.onChange({ target: { value: 'all' } }));
  for (const name of ['Higher Woman', 'Profile Review', 'Already Picked']) assert.equal(pick(name).props.disabled, true);
  assert.ok(textContent(tree).includes('Rating must be below 3.5 for 3.0'));
  assert.ok(textContent(tree).includes('Gender needs review in the season player pool'));
  assert.ok(textContent(tree).includes('Already selected for Beta Blue'));
  await act(async () => select('Lineup division').props.onChange({ target: { value: '3.5' } }));
  assert.equal(pick('Higher Woman').props.disabled, false);
  await choose('Play Up Woman'); await choose('Second Woman'); await choose('Higher Woman'); await choose('First Man');
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, true, 'Three women and one man cannot be submitted');
  assert.ok(textContent(tree).includes('3 of 2 women · 1 of 2 men'));
  await act(async () => select('Lineup division').props.onChange({ target: { value: '3.0' } }));
  assert.equal(pick('Higher Woman').props.checked, true, 'A division change preserves selections and explains problems');
  assert.ok(textContent(tree).includes('Higher Woman: Rating must be below 3.5 for 3.0'));
  await choose('Higher Woman'); await choose('Second Man');
  assert.equal(button(tree, 'Submit four-player roster').props.disabled, false);
  await act(async () => button(tree, 'Review season player pool').props.onClick());
  assert.equal(tree.root.findByProps({ id: 'season-player-pool' }).props.hidden, false);
  await act(async () => tree.root.findByProps({ 'aria-label': 'Lineups' }).props.onClick({ preventDefault() {} }));
  assert.equal(pick('Play Up Woman').props.checked, true, 'Opening the pool does not discard a lineup draft');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Choose meet players' }).props.onSubmit({ preventDefault() {} }));
  const body = JSON.parse(requests.find(request => request.options.method).options.body);
  assert.deepEqual(body.player_ids, ['1', '2', '3', '4']); assert.equal(body.division, '3.0'); assert.equal(body.name, 'Beta Club 3.0');
  await act(async () => tree.unmount());
}

(async () => { await clubsAndRosters(); await invitationResponses(); await deepLinkContext(); await registrationPhaseLocks(); await commissionerWindowEditor(); await savedClosedWindowLoadsMeets(); await serverConfirmedWindowBoundary(); await phaseRefreshCannotUndoAcceptance(); await updatedPlayersAndSchedulePreserveLineupDraft(); await loadFailuresCanBeRetried(); await guidedLineupChoices(); console.log('Interclub registration: invitation outcomes, commissioner window dates, phase gates and server-confirmed boundaries, meet-specific lineups, scoped players, stale saves, deep-link context, stage draft preservation, load failures, retries and guided eligible-player selection passed.'); })()
  .catch(e => { console.error(e); process.exitCode = 1; });
