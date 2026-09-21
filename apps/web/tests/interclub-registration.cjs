const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(n => n === '@/lib/interclubSetup' ? load('lib/interclubSetup.ts') : n === '../InterclubWorkflow' ? load('app/admin/interclub/InterclubWorkflow.tsx', { 'next/link': Link, './workflow.module.css': {} }) : n === './SeasonEligibilityApprovals' ? { default: () => null, __esModule: true } : n === './PlayerPoolPanels' ? { SeasonPlayerPool: p => React.createElement('section', { 'data-pool-root': p.root }), MeetAvailability: p => React.createElement('section', { 'data-availability-root': p.meetRoot, onResponses: p.onResponses }) } : Object.hasOwn(mocks, n) ? mocks[n] : require(n), module, module.exports);
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
const season = { id: sid, organizer_club_id: 'alpha', source_revision: 2, roster_deadline: '2099-01-01T00:00:00Z', details: {
  name: 'Coastal League', start_date: '2099-01-10', end_date: '2099-03-31', timezone: 'America/Mazatlan', divisions: ['3.5'], club_ids: ['beta', 'gamma'], meets: []
}, rules: { '3.5': { min_rating: null, max_rating: 3.75, women_required: 2 } } };
const lineup = [1, 2, 3, 4].map(n => ({ entry_id: `entry-${n}`, player_id: String(n), name: `Player ${n}`, starting_rating: 3.5 }));
const meet = { id: mid, season_id: sid, host_club_id: 'beta', club_ids: ['beta', 'gamma'], starts_at: '2099-01-10T18:00:00Z', roster_deadline: '2099-01-10T12:00:00Z', revision: 2, roster_open: true, deadline_editable: false, courts: 4 };
const secondMeet = { ...meet, id: mid2, starts_at: '2099-02-10T18:00:00Z', revision: 1, deadline_editable: true };
const team = { meet_id: mid, id: tid, club_id: 'beta', season_id: sid, name: 'Beta Blue', division: '3.5', revision: 1, withdrawn: false, status: 'needs_exception', roster: lineup, issues: [{ code: 'rating_above_maximum', message: 'Player exceeds rating limit.' }], late_change: true, decision_reason: null };

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
  assert.ok(!requests.some(r => r.url.includes('/players')));
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Meet' }).length, 0, 'Pending invitation does not show meet roster controls');
  assert.ok(!requests.some(r => r.url.includes('/meets/')), 'Pending invitation does not fetch meet details');
  assert.ok(textContent(tree).includes('Coastal League') && textContent(tree).includes('alpha Club'));
  assert.ok(textContent(tree).includes('3.5 to below 4.0') && !textContent(tree).includes('No minimum'), 'Existing seasons show the enforced skill band even if old saved rules omit its floor');
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
  assert.equal(name().props.value, '');
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

(async () => { await clubsAndRosters(); await invitationResponses(); await deepLinkContext(); await loadFailuresCanBeRetried(); console.log('Interclub registration: invitation outcomes, gated rosters, meet-specific lineups, deadlines, scoped players, stale saves, account/meet changes, deep-link context, stage draft preservation, load failures and retries passed.'); })()
  .catch(e => { console.error(e); process.exitCode = 1; });
