const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
process.env.NEXT_PUBLIC_JUPR_API_BASE_URL = 'https://api.test';
let uuid = 0;
Object.defineProperty(global, 'crypto', { value: { randomUUID: () => `request-${++uuid}` }, configurable: true });
const helpers = load('lib/interclubPlayerSignup.ts');
const windowHelpers = load('lib/interclubRegistrationWindow.ts');
const windowHook = load('lib/useRegistrationWindow.ts', { './interclubRegistrationWindow': windowHelpers });
const windowMocks = { '@/lib/interclubRegistrationWindow': windowHelpers, '@/lib/useRegistrationWindow': windowHook };
const styles = new Proxy({}, { get: (_, property) => property });
let currentSession = null;
const Signup = load('app/interclub/signup/[shareId]/SeasonSignup.tsx', {
  ...windowMocks, '@/lib/interclubPlayerSignup': helpers, '../../player-signup.module.css': styles,
  '@/lib/tournamentRegistrationProfile': load('lib/tournamentRegistrationProfile.ts'),
  '@/lib/adminAuthClient': { loadAdminSession: () => currentSession, adminSessionIsFresh: () => true },
}).default;
const Response = load('app/interclub/respond/PlayerResponse.tsx', { ...windowMocks, '@/lib/interclubPlayerSignup': helpers, '../player-signup.module.css': styles }).default;
const reply = (value, status = 200) => ({ ok: status < 400, status, json: async () => value });
const content = tree => JSON.stringify(tree.toJSON());
const button = (tree, text) => tree.root.findAllByType('button').find(node => node.children.includes(text));
const settleLookup = () => act(async () => new Promise(resolve => setTimeout(resolve, 275)));
const club = { id: 'cabo', name: 'Cabo Test Club' };
const openWindow = { opens_at: '2020-01-01T00:00:00Z', closes_at: '2099-01-01T00:00:00Z', revision: 1, status: 'open', can_register: true, meet_planning_open: false };
const closedWindow = { ...openWindow, closes_at: '2020-02-01T00:00:00Z', status: 'closed', can_register: false, meet_planning_open: true };
const season = { registration: openWindow, id: 'season', name: 'Coastal Season', start_date: '2099-11-01', end_date: '2100-03-31', timezone: 'America/Mazatlan', divisions: ['3.5', '4.0'] };
const member = { id: 'member-a', name: 'Jo Player', email: 'jo@example.test', divisions: ['3.5'], notes: 'Away in January', status: 'active', revision: 4 };
const review = { kind: 'season', club, season, member, can_respond: true, can_withdraw: true };
const meetReview = { ...review, season: { ...season, registration: closedWindow }, kind: 'meet', meet: { id: 'meet-a', host_club_id: 'la-paz', host_club_name: 'La Paz Test Club', starts_at: '2099-11-15T17:00:00Z' }, availability: { status: 'invited', revision: 2, deadline: '2099-11-13T17:00:00Z', open: true } };

function personalLink(token = 'private-token') {
  const replaced = [];
  global.window = { location: { hash: token ? `#token=${token}` : '', pathname: '/interclub/respond', search: '' }, history: { state: {}, replaceState(state, title, url) { replaced.push(url); window.location.hash = ''; } } };
  return replaced;
}

async function signupConsentRetryAndPrivacy() {
  let calls = [], finish;
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    if (options.method) return new Promise(resolve => { finish = resolve; });
    if (url.includes('/players')) return reply({ players: [], linked_player: null });
    return reply({ club, season, signup: { open: true }, meets: [] });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  assert.ok(content(tree).includes('does not commit you to every meet'));
  assert.equal(button(tree, 'Join the season player pool').props.disabled, true);
  await act(async () => {
    tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Jo Player' } });
    tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'jo@example.test' } });
    tree.root.findAllByProps({ type: 'checkbox' })[0].props.onChange({ target: { checked: true } });
    tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } });
  });
  await settleLookup();
  await act(async () => {
    void tree.root.findByType('form').props.onSubmit({ preventDefault() {} });
    void tree.root.findByType('form').props.onSubmit({ preventDefault() {} });
  });
  assert.equal(calls.filter(call => call.options.method).length, 1, 'Duplicate submits are blocked');
  const initial = JSON.parse(calls.at(-1).options.body);
  assert.deepEqual(initial, { name: 'Jo Player', email: 'jo@example.test', divisions: ['3.5'], notes: '', email_consent: true, request_id: 'request-1' });
  await act(async () => finish(reply({ detail: 'Try again.' }, 503)));
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  assert.equal(JSON.parse(calls.at(-1).options.body).request_id, initial.request_id, 'A safe retry reuses the same request ID');
  const manageUrl = 'https://web.test/interclub/respond#token=personal';
  await act(async () => finish(reply({ status: 'registered', manage_url: manageUrl })));
  assert.equal(tree.root.findAllByType('form').length, 0);
  assert.equal(tree.root.findByProps({ href: manageUrl }).children.join(''), 'Manage my season signup');
  assert.ok(content(tree).includes('This link is just for you'));
  await act(async () => tree.unmount());

  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  await act(async () => {
    tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Jo Player' } });
    tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'jo@example.test' } });
    tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } });
  });
  await settleLookup();
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  await act(async () => finish(reply({ status: 'already_registered' })));
  assert.ok(content(tree).includes('Your earlier signup is still saved'));
  assert.equal(tree.root.findAllByType('a').length, 0, 'Duplicate email/name does not expose a personal link');
  await act(async () => tree.unmount());
}

async function seasonEditingAndWithdrawal() {
  let calls = [], current = structuredClone(review), finish;
  const replaced = personalLink();
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    if (url.endsWith('/review')) return reply(current);
    return new Promise(resolve => { finish = resolve; });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.deepEqual(replaced, ['/interclub/respond'], 'Personal token is removed from address before lookup');
  assert.equal(window.location.hash, '');
  assert.ok(calls.every(call => !call.url.includes('private-token')));
  assert.equal(calls[0].options.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].options.body), { token: 'private-token' });
  assert.equal(calls[0].options.referrerPolicy, 'no-referrer');
  assert.equal(calls[0].options.credentials, 'omit');
  assert.equal(calls[0].options.cache, 'no-store');
  assert.equal(tree.root.findAllByProps({ type: 'password' }).length, 0);
  await act(async () => tree.root.findByType('textarea').props.onChange({ target: { value: 'Back in December' } }));
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  assert.deepEqual(JSON.parse(calls.at(-1).options.body), { token: 'private-token', action: 'update_season', expected_revision: 4, name: member.name, email: member.email, divisions: ['3.5'], notes: 'Back in December', status: 'active' });
  current.member = { ...current.member, notes: 'Back in December', revision: 5 };
  await act(async () => finish(reply(current)));
  assert.ok(content(tree).includes('Your season signup is updated'));
  await act(async () => button(tree, 'Leave the season player pool').props.onClick());
  assert.ok(button(tree, 'Stay in the pool'));
  await act(async () => button(tree, 'Yes, leave the player pool').props.onClick());
  assert.equal(JSON.parse(calls.at(-1).options.body).status, 'withdrawn');
  assert.equal(JSON.parse(calls.at(-1).options.body).expected_revision, 5);
  current.member = { ...current.member, status: 'withdrawn', revision: 6 };
  await act(async () => finish(reply(current)));
  assert.ok(button(tree, 'Rejoin the season player pool'));
  assert.equal(button(tree, 'Leave the season player pool'), undefined);
  await act(async () => tree.unmount());
}

async function signupClosingWhileFormIsOpen() {
  let open = true;
  global.fetch = async (url, options) => options.method ? reply({ detail: 'Closed' }, 409) : url.includes('/players') ? reply({ players: [], linked_player: null }) : reply({ club, season: { ...season, registration: open ? openWindow : closedWindow }, signup: { open }, meets: [] });
  let tree;
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  await act(async () => {
    tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Jo Player' } });
    tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'jo@example.test' } });
    tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } });
  });
  await settleLookup();
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(button(tree, 'Join the season player pool').props.disabled, true);
  assert.equal(tree.root.findByProps({ autoComplete: 'name' }).props.value, 'Jo Player');
  open = false;
  await act(async () => button(tree, 'Reload signup').props.onClick());
  assert.ok(content(tree).includes('Season registration has closed.'));
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => tree.unmount());
}

async function withdrawalWhenSeasonSignupIsClosed() {
  personalLink(); let lastBody;
  global.fetch = async (url, options) => {
    if (url.endsWith('/review')) return reply({ ...review, can_respond: false });
    lastBody = JSON.parse(options.body);
    return reply({ ...review, member: { ...member, status: 'withdrawn', revision: 5 }, can_respond: false, can_withdraw: false });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.equal(button(tree, 'Save my changes'), undefined);
  assert.equal(button(tree, 'Leave the season player pool').props.disabled, false, 'Closing recruitment still permits withdrawal');
  await act(async () => button(tree, 'Leave the season player pool').props.onClick());
  await act(async () => button(tree, 'Yes, leave the player pool').props.onClick());
  assert.equal(lastBody.status, 'withdrawn');
  assert.equal(lastBody.name, member.name);
  assert.equal(button(tree, 'Rejoin the season player pool'), undefined, 'Reactivation stays unavailable while recruitment is closed');
  assert.ok(content(tree).includes('You’ve left the player pool'));
  await act(async () => tree.unmount());
}

async function meetRepliesConflictAndExpiry() {
  let calls = [], current = structuredClone(meetReview), finish;
  personalLink('meet-token');
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    if (url.endsWith('/review')) return reply(current);
    return new Promise(resolve => { finish = resolve; });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.ok(content(tree).includes('Your administrator will choose the final teams'));
  assert.ok(content(tree).includes('La Paz Test Club'), 'Meet response shows the hosting club by name');
  assert.equal(button(tree, 'Save my response').props.disabled, true);
  await act(async () => tree.root.findByProps({ value: 'available', type: 'radio' }).props.onChange());
  await act(async () => {
    void tree.root.findByType('form').props.onSubmit({ preventDefault() {} });
    void tree.root.findByType('form').props.onSubmit({ preventDefault() {} });
  });
  assert.equal(calls.filter(call => call.url.endsWith('/respond')).length, 1);
  assert.deepEqual(JSON.parse(calls.at(-1).options.body), { token: 'meet-token', expected_revision: 2, action: 'respond_meet', status: 'available' });
  await act(async () => finish(reply({ detail: 'Changed' }, 409)));
  assert.equal(button(tree, 'Save my response').props.disabled, true);
  assert.equal(tree.root.findByProps({ value: 'available', type: 'radio' }).props.checked, true, 'Failed save retains choice');
  current.availability = { ...current.availability, status: 'maybe', revision: 3 };
  await act(async () => button(tree, 'Reload latest details').props.onClick());
  assert.equal(tree.root.findByProps({ value: 'maybe', type: 'radio' }).props.checked, true);
  await act(async () => tree.root.findByProps({ value: 'unavailable', type: 'radio' }).props.onChange());
  await act(async () => { void tree.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
  assert.equal(JSON.parse(calls.at(-1).options.body).expected_revision, 3);
  current.availability = { ...current.availability, status: 'unavailable', revision: 4 };
  await act(async () => finish(reply(current)));
  assert.ok(content(tree).includes('You’re still in the season player pool'));
  await act(async () => tree.unmount());

  personalLink(); current.can_respond = false;
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.ok(content(tree).includes('Responses are closed'));
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => tree.unmount());
  personalLink(); global.fetch = async () => reply({ detail: 'Expired' }, 404);
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.ok(content(tree).includes('This link is no longer available'));
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => tree.unmount());
}

async function missingLinkAndWrongResponseKind() {
  personalLink(''); let calls = 0;
  global.fetch = async () => { calls++; return reply(review); };
  let tree;
  await act(async () => { tree = create(React.createElement(Response)); });
  assert.equal(calls, 0);
  assert.ok(content(tree).includes('Open your full personal link'));
  await act(async () => tree.unmount());
  personalLink();
  global.fetch = async url => reply(url.endsWith('/review') ? review : meetReview);
  await act(async () => { tree = create(React.createElement(Response)); });
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.ok(content(tree).includes('Reload before making another change'));
  assert.equal(button(tree, 'Save my changes').props.disabled, true);
  assert.equal(tree.root.findAllByProps({ type: 'radio' }).length, 0, 'A mismatched response never opens a different invitation');
  await act(async () => tree.unmount());
  for (const file of ['app/interclub/respond/page.tsx', 'app/interclub/signup/[shareId]/page.tsx']) {
    const source = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
    assert.match(source, /index: false, follow: false/);
    assert.match(source, /referrer: "no-referrer"/);
  }
}

async function signupProfileMatchingAndSafeFallback() {
  const player = { id: 'player-1', name: 'Jo Player', rating: 3.456789, gender: 'Female', eligible_divisions: ['3.5'] };
  const duplicate = { ...player, id: 'player-2', rating: 4.01 };
  let players = [player], posts = [], lookup;
  global.fetch = async (url, options) => {
    if (options.method) { posts.push(JSON.parse(options.body)); return reply({ detail: 'Test stop' }, 503); }
    if (url.includes('/players')) return lookup ? lookup(url) : reply({ players, linked_player: null });
    return reply({ club, season: { ...season, divisions: ['4.5', '3.5', '4.0', '3.0'] }, signup: { open: true }, meets: [] });
  };
  let tree;
  const mount = async () => {
    if (tree) await act(async () => tree.unmount());
    await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
    await act(async () => {
      tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: '  JO   Player  ' } });
      tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'jo@example.test' } });
      tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } });
    });
  };
  const submit = () => act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  await mount();
  await submit();
  assert.equal(posts.length, 0, 'Submit waits for profile matching');
  await settleLookup();
  assert.equal(tree.root.findByProps({ 'aria-label': 'Club rating' }).props.readOnly, true);
  assert.equal(tree.root.findByProps({ 'aria-label': 'Club rating' }).props.value, '3.46');
  const divisionLabels = tree.root.findAllByProps({ type: 'checkbox' }).slice(0, 4).map(input => input.parent.findByType('span').children[0]);
  assert.deepEqual(divisionLabels, ['3.0', '3.5', '4.0', '4.5']);
  await submit();
  assert.equal(posts.at(-1).player_id, player.id, 'Unique normalized exact match is submitted automatically');
  assert.equal(Object.hasOwn(posts.at(-1), 'rating'), false, 'Browser does not submit a rating override');
  await act(async () => button(tree, 'This isn’t my profile').props.onClick());
  await submit();
  assert.equal(posts.at(-1).player_id, null, 'Explicit opt out prevents server automatic linking');

  players = [player, duplicate];
  await mount(); await settleLookup();
  const beforeAmbiguous = posts.length;
  await submit();
  assert.equal(posts.length, beforeAmbiguous, 'Ambiguous matches require an explicit choice');
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Club rating' }).length, 0);
  await act(async () => tree.root.findByProps({ type: 'radio', value: duplicate.id }).props.onChange());
  assert.equal(tree.root.findByProps({ autoComplete: 'name' }).props.value, 'Jo Player');
  await submit();
  assert.equal(posts.at(-1).player_id, duplicate.id);

  let finishOld;
  lookup = url => url.includes('Jo') || url.includes('JO') ? new Promise(resolve => { finishOld = resolve; }) : reply({ players: [], linked_player: null });
  await mount(); await settleLookup();
  await act(async () => tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Different Person' } }));
  await act(async () => finishOld(reply({ players: [player], linked_player: null })));
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Club rating' }).length, 0, 'Late response cannot select the prior person');
  await settleLookup(); await submit();
  assert.equal(posts.at(-1).name, 'Different Person');
  assert.equal(Object.hasOwn(posts.at(-1), 'player_id'), false, 'Changing name removes the old profile from signup');

  lookup = () => { throw Error('offline'); };
  await mount(); await settleLookup();
  assert.ok(button(tree, 'Continue without a profile'));
  const beforeFailure = posts.length;
  await submit(); assert.equal(posts.length, beforeFailure);
  await act(async () => button(tree, 'Continue without a profile').props.onClick());
  await submit();
  assert.equal(posts.at(-1).player_id, null, 'Lookup outages allow an explicit unmatched signup');
  await act(async () => tree.unmount());
}

async function signupExistingAccountPrefill() {
  const player = { id: 'account-player', name: 'Verified Player', rating: 4.2, gender: 'Male', eligible_divisions: ['4.0'] };
  currentSession = { access_token: 'existing-session-token', user: { email: 'verified@example.test' } };
  let calls = [], finishAccount;
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    if (url.endsWith('/players')) return reply({ players: [player], linked_player: player });
    if (url.includes('/players?')) return reply({ players: [], linked_player: player });
    return reply({ club, season, signup: { open: true }, meets: [] });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  assert.equal(tree.root.findByProps({ autoComplete: 'name' }).props.value, player.name);
  assert.equal(tree.root.findByProps({ type: 'email' }).props.value, 'verified@example.test');
  assert.equal(tree.root.findByProps({ 'aria-label': 'Club rating' }).props.value, '4.2');
  assert.equal(calls.find(call => call.url.endsWith('/players')).options.headers.Authorization, 'Bearer existing-session-token');
  await act(async () => tree.unmount());

  global.fetch = async url => url.endsWith('/players') ? new Promise(resolve => { finishAccount = resolve; }) : url.includes('/players?') ? reply({ players: [], linked_player: player }) : reply({ club, season, signup: { open: true }, meets: [] });
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  await act(async () => tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Someone Else' } }));
  await act(async () => finishAccount(reply({ players: [player], linked_player: player })));
  assert.equal(tree.root.findByProps({ autoComplete: 'name' }).props.value, 'Someone Else', 'Account prefill cannot overwrite typed identity');
  await settleLookup();
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Club rating' }).length, 0, 'Search does not select the account profile over a different typed name');
  await act(async () => tree.unmount());
  currentSession = null;
}

async function commissionerRegistrationWindow() {
  let windowValue, calls = [];
  global.fetch = async (url, options) => { calls.push({ url, options }); return reply({ club, season: { ...season, registration: windowValue }, signup: { open: true }, meets: [] }); };
  for (const value of [undefined, { ...openWindow, opens_at: '2098-01-01T00:00:00Z', status: 'scheduled', can_register: false }, closedWindow]) {
    windowValue = value;
    let tree;
    await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
    assert.equal(tree.root.findAllByType('form').length, 0, 'Legacy club signup.open does not override commissioner registration');
    if (!value) assert.ok(content(tree).includes('commissioner has not set registration dates yet'));
    else assert.ok(content(tree).includes('Season registration:'));
    await act(async () => tree.unmount());
  }
  const opens = Date.now() + 60, closes = opens + 3_600_000;
  calls = [];
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    const canRegister = Date.now() >= opens;
    return reply({ club, season: { ...season, registration: { ...openWindow, opens_at: new Date(opens).toISOString(), closes_at: new Date(closes).toISOString(), status: canRegister ? 'open' : 'scheduled', can_register: canRegister } }, signup: { open: true }, meets: [] });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  assert.equal(tree.root.findAllByType('form').length, 0);
  await act(async () => new Promise(resolve => setTimeout(resolve, 125)));
  assert.ok(calls.length >= 2, 'The opening boundary rechecks commissioner settings automatically');
  assert.equal(tree.root.findAllByType('form').length, 1, 'The form opens only after the server confirms the registration phase');
  await act(async () => tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } }));
  const beforeClose = calls.length, originalNow = Date.now;
  try {
    Date.now = () => closes + 1;
    await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
    assert.equal(calls.slice(beforeClose).some(call => call.options.method === 'POST'), false, 'An already-open form cannot submit after registration closes');
    assert.equal(tree.root.findAllByType('form').length, 0);
  } finally { Date.now = originalNow; }
  await act(async () => tree.unmount());
}

(async () => {
  await signupConsentRetryAndPrivacy();
  await signupClosingWhileFormIsOpen();
  await seasonEditingAndWithdrawal();
  await withdrawalWhenSeasonSignupIsClosed();
  await meetRepliesConflictAndExpiry();
  await missingLinkAndWrongResponseKind();
  await signupProfileMatchingAndSafeFallback();
  await signupExistingAccountPrefill();
  await commissionerRegistrationWindow();
  console.log('Interclub public player signup and response checks passed');
})().catch(error => { console.error(error); process.exit(1); });
