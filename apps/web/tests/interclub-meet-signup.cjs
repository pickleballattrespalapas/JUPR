const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const base = path.join(__dirname, '..'), cache = new Map();
function load(file) {
  const absolute = path.resolve(base, file);
  if (cache.has(absolute)) return cache.get(absolute);
  const code = ts.transpileModule(fs.readFileSync(absolute, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    if (name.endsWith('.css')) return new Proxy({}, { get: (_, key) => key });
    if (name.startsWith('@/') || name.startsWith('.')) {
      const local = name.startsWith('@/') ? path.join(base, name.slice(2)) : path.resolve(path.dirname(absolute), name);
      const found = [local, local + '.ts', local + '.tsx'].find(file => fs.existsSync(file) && fs.statSync(file).isFile());
      if (found) return load(found);
    }
    return require(name);
  }, module, module.exports);
  cache.set(absolute, module.exports); return module.exports;
}
process.env.NEXT_PUBLIC_JUPR_API_BASE_URL = 'https://api.test';
Object.defineProperty(global, 'crypto', { value: require('node:crypto').webcrypto, configurable: true });
const Panel = load('app/admin/interclub/registrations/MeetSignupPanel.tsx').default;
const Signup = load('app/interclub/meet-signup/MeetSignup.tsx').default;
const response = value => ({ ok: true, status: 200, json: async () => value });
const board = { club: { id: 'home', name: 'Home' }, season: { id: 's', name: 'Season', divisions: ['3.0', '3.5'], timezone: 'UTC' }, meet: { starts_at: '2099-01-01T00:00:00Z', roster_deadline: '2098-12-31T00:00:00Z', revision: 1 }, signup: { open: true, deadline: '2098-12-31T00:00:00Z', revision: 1, configured: true, url: 'https://web.test/share' }, entries: [] };
const player = (id, name, rating, gender = 'female') => ({ id, name, starting_rating: rating, eligibility_rating: rating, gender });
const props = { root: 'https://api.test/meet', accessToken: 'test', onAutomatic() {}, onChanged() {} };
const text = tree => JSON.stringify(tree.toJSON());
const button = (tree, label) => tree.root.findAllByType('button').find(node => node.children.includes(label));

async function adminSlotFlow() {
  const registered = { id: 'entry', player_id: 'signed', name: 'Registered', division: '3.0', gender: 'female', status: 'active', placement: 'confirmed', priority: 'in_band', rating: 3.1 };
  let current = { ...board, entries: [registered] }, payload, requests = [], tree;
  global.fetch = async (url, options) => {
    requests.push(url);
    if (options.method === 'POST') { payload = JSON.parse(options.body); return response({ ...current, entries: [...current.entries, { ...registered, id: 'new', player_id: payload.player_id, name: 'Z Highest', gender: 'unknown', declared_gender: payload.gender, placement: 'review', reason: 'An admin will review your registration and confirm your lineup placement.' }] }); }
    if (url.endsWith('/signup')) return response(current);
    if (url.includes('/players?offset=0')) return response({ players: [player('middle', 'Middle', 3.3), player('high', 'Too high', 3.5), player('man', 'Other gender', 3.49, 'male'), player('low', 'Play up', 2.93), player('signed', 'Registered', 3.4), player('assigned', 'Assigned', 3.45), player('unknown', 'Needs gender', 3.35, 'unknown'), player('unrated', 'Unrated', null)], next_offset: 100 });
    if (url.includes('/players?offset=100')) return response({ players: [player('top', 'Z Highest', 3.49)], next_offset: null });
    return response({ teams: [{ club_id: 'home', withdrawn: false, roster: [{ player_id: 'assigned' }] }], next_team_offset: null });
  };
  await act(async () => { tree = create(React.createElement(Panel, props)); });
  await act(async () => tree.root.findByProps({ 'aria-label': 'Add player to 3.0 women' }).props.onClick({ currentTarget: { focus() {} } }));
  const list = tree.root.findByProps({ 'aria-label': 'Eligible players by rating' });
  assert.deepEqual(list.findAllByType('strong').map(node => node.children.join('')), ['Z Highest', 'Needs gender', 'Middle', 'Play up']);
  assert.ok(requests.some(url => url.includes('offset=100')), 'All pages participate in rating order');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Choose Z Highest for 3.0 women' }).props.onClick());
  assert.equal(tree.root.findByProps({ 'aria-label': 'Gender' }).props.value, 'female');
  await act(async () => tree.root.findByProps({ 'aria-label': 'Gender' }).props.onChange({ target: { value: 'non_binary' } }));
  assert.ok(text(tree).includes('An admin will review'));
  await act(async () => tree.root.findByProps({ 'aria-label': 'Confirm player registration' }).props.onSubmit({ preventDefault() {} }));
  assert.equal(payload.player_id, 'top'); assert.equal(payload.gender, 'non_binary'); assert.equal(payload.division, '3.0');
  assert.ok(text(tree).includes('Waiting for admin review'));
  const review = tree.root.findByProps({ 'aria-label': 'Review placement for Z Highest' });
  assert.equal(review.findByType('select').props.value, '');
  await act(async () => tree.unmount());
  current = { ...board, signup: { ...board.signup, open: false } };
  await act(async () => { tree = create(React.createElement(Panel, props)); });
  assert.equal(tree.root.findAllByProps({ 'aria-label': 'Add player to 3.0 women' }).length, 0);
  await act(async () => tree.unmount());
}

async function publicGenderFlow() {
  let submitted, tree;
  global.fetch = async (url, options) => {
    if (options.method === 'POST') { submitted = JSON.parse(options.body); return response({ ...board, entry: { id: 'private', name: 'Player', division: '3.0', status: 'active', gender: 'unknown', declared_gender: submitted.gender, placement: 'review', reason: 'An admin will review your registration and confirm your lineup placement.' } }); }
    if (url.includes('/players')) return response({ players: [{ id: '3', name: 'Player', league_rating: 3.2, eligible_divisions: ['3.0'], gender: 'unknown' }] });
    return response(board);
  };
  await act(async () => { tree = create(React.createElement(Signup, { shareId: 'shared' })); });
  await act(async () => tree.root.findByProps({ autoComplete: 'name' }).props.onChange({ target: { value: 'Player' } }));
  await act(async () => new Promise(resolve => setTimeout(resolve, 275)));
  await act(async () => tree.root.findByProps({ type: 'radio' }).props.onChange());
  const select = tree.root.findByProps({ 'aria-label': 'Gender' });
  assert.equal(select.props.value, '', 'Unknown profile does not guess gender');
  assert.deepEqual(select.findAllByType('option').map(node => node.children.join('')), ['Choose gender', 'Woman', 'Man', 'Non-binary', 'Prefer not to say']);
  await act(async () => tree.root.findByProps({ type: 'checkbox' }).props.onChange({ target: { checked: true } }));
  assert.equal(button(tree, 'Sign me up for this meet').props.disabled, true);
  await act(async () => select.props.onChange({ target: { value: 'prefer_not_to_say' } }));
  assert.ok(text(tree).includes('An admin will review'));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.equal(submitted.gender, 'prefer_not_to_say');
  assert.ok(text(tree).includes('Waiting for admin review'));
  await act(async () => tree.unmount());
}

(async () => { await adminSlotFlow(); await publicGenderFlow(); console.log('Interclub meet slot registration and gender review checks passed'); })().catch(error => { console.error(error); process.exit(1); });
