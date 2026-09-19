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
const styles = new Proxy({}, { get: (_, property) => property });
const Signup = load('app/interclub/signup/[shareId]/SeasonSignup.tsx', { '@/lib/interclubPlayerSignup': helpers, '../../player-signup.module.css': styles }).default;
const Response = load('app/interclub/respond/PlayerResponse.tsx', { '@/lib/interclubPlayerSignup': helpers, '../player-signup.module.css': styles }).default;
const reply = (value, status = 200) => ({ ok: status < 400, status, json: async () => value });
const content = tree => JSON.stringify(tree.toJSON());
const button = (tree, text) => tree.root.findAllByType('button').find(node => node.children.includes(text));
const club = { id: 'cabo', name: 'Cabo Test Club' };
const season = { id: 'season', name: 'Coastal Season', start_date: '2099-11-01', end_date: '2100-03-31', timezone: 'America/Mazatlan', divisions: ['3.5', '4.0'] };
const member = { id: 'member-a', name: 'Jo Player', email: 'jo@example.test', divisions: ['3.5'], notes: 'Away in January', status: 'active', revision: 4 };
const review = { kind: 'season', club, season, member, can_respond: true, can_withdraw: true };
const meetReview = { ...review, kind: 'meet', meet: { id: 'meet-a', host_club_id: 'la-paz', host_club_name: 'La Paz Test Club', starts_at: '2099-11-15T17:00:00Z' }, availability: { status: 'invited', revision: 2, deadline: '2099-11-13T17:00:00Z', open: true } };

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
  global.fetch = async (url, options) => options.method ? reply({ detail: 'Closed' }, 409) : reply({ club, season, signup: { open }, meets: [] });
  let tree;
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  await act(async () => {
    tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Jo Player' } });
    tree.root.findByProps({ type: 'email' }).props.onChange({ target: { value: 'jo@example.test' } });
    tree.root.findAllByProps({ type: 'checkbox' }).at(-1).props.onChange({ target: { checked: true } });
  });
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(button(tree, 'Join the season player pool').props.disabled, true);
  assert.equal(tree.root.findByProps({ autoComplete: 'name' }).props.value, 'Jo Player');
  open = false;
  await act(async () => button(tree, 'Reload signup').props.onClick());
  assert.ok(content(tree).includes('Signup is currently closed'));
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

(async () => {
  await signupConsentRetryAndPrivacy();
  await signupClosingWhileFormIsOpen();
  await seasonEditingAndWithdrawal();
  await withdrawalWhenSeasonSignupIsClosed();
  await meetRepliesConflictAndExpiry();
  await missingLinkAndWrongResponseKind();
  console.log('Interclub public player signup and response checks passed');
})().catch(error => { console.error(error); process.exit(1); });
