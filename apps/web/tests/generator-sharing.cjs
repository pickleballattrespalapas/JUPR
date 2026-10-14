const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), { create, act } = require('react-test-renderer');
const { QRCodeSVG } = require('qrcode.react');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : (require('./helpers/public-controls-module.cjs')(name) || require(name)), module, module.exports);
  return module.exports;
}
const Sharing = load('components/GeneratorSharing.tsx').default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const copied = [], calls = [];
global.window = { location: { origin: 'https://club.test', hash: '', pathname: '/test', search: '' }, history: { replaceState() {} } };
Object.defineProperty(global, 'navigator', { configurable: true, value: { clipboard: { writeText: async url => copied.push(url) } } });
const reply = body => ({ ok: true, status: 200, json: async () => body });
global.fetch = async (url, options) => {
  calls.push({ url, body: options?.body && JSON.parse(options.body), headers: options?.headers });
  return reply({ sharing: { enabled: true, club_slug: 'resolved-slug', handoff_token: 'private-operator-token' } });
};

async function sharingControls() {
  let tree, changed = 0;
  const props = { apiBase: 'https://api.test', clubId: 'opaque-club-id', sessionKey: 'session', version: '2026-10-13T20:00:00Z', adminToken: 'admin-session', onChanged: async () => { changed++; } };
  await act(async () => { tree = create(React.createElement(Sharing, props)); });
  await act(async () => button(tree, 'Share standings').props.onClick());
  assert.equal(copied.at(-1), 'https://club.test/clubs/resolved-slug/round-robin-generator/shared/session/standings');
  assert.equal(calls.at(-1).headers.Authorization, 'Bearer admin-session');
  assert.equal(calls.at(-1).body.expected_version, props.version);
  const standings = copied.at(-1);
  assert.equal(tree.root.findByType(QRCodeSVG).props.value, standings);
  assert.equal(tree.root.findByProps({'aria-label':'Standings link'}).props.value, standings);
  assert.equal(tree.root.findByType('a').props.href, standings);
  await act(async () => button(tree, 'Hand off RR').props.onClick());
  assert.equal(copied.at(-1), 'https://club.test/clubs/resolved-slug/round-robin-generator/shared/session#edit=private-operator-token');
  assert.ok(!calls.at(-1).url.includes('private-operator-token'));
  assert.equal(tree.root.findByType(QRCodeSVG).props.value, standings, 'Courtside QR always stays view-only when handing off');
  await act(async () => button(tree, 'Revoke handoff link').props.onClick());
  assert.equal(calls.at(-1).body.action, 'revoke');
  assert.equal(tree.root.findAllByProps({'aria-label':'Private handoff link'}).length, 0);
  assert.equal(tree.root.findByType(QRCodeSVG).props.value, standings, 'Revoking handoff preserves courtside sharing');
  assert.equal(changed, 3);
  await act(async () => tree.update(React.createElement(Sharing, { ...props, locked: true })));
  assert.equal(button(tree, 'Hand off RR').props.disabled, true);
  tree.unmount();

  const count = calls.length;
  await act(async () => { tree = create(React.createElement(Sharing, { ...props, shared: true, adminToken: undefined })); });
  assert.equal(button(tree, 'Hand off RR'), undefined);
  assert.equal(button(tree, 'Revoke handoff link'), undefined);
  assert.equal(tree.root.findByType(QRCodeSVG).props.value, 'https://club.test/clubs/opaque-club-id/round-robin-generator/shared/session/standings', 'Viewer gets a link and QR immediately without making a write request');
  await act(async () => button(tree, 'Share standings').props.onClick());
  assert.equal(calls.length, count, 'Spectator sharing is read-only and never requests access');
  assert.ok(copied.at(-1).endsWith('/standings') && !copied.at(-1).includes('#'));
  tree.unmount();
}

async function currentRoundLanding() {
  const navigated = [];
  const Landing = load('components/SharedGeneratorLanding.tsx', { 'next/navigation': { useRouter: () => router } }).default;
  const router = { replace: url => navigated.push(url) };
  window.location.hash = '#edit=operator-token';
  global.fetch = async url => {
    assert.ok(url.includes('/shared-play-generators/'));
    assert.ok(!url.includes('operator-token'));
    return reply({ session: { current_round_number: 7 } });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Landing, { apiBase: 'https://api.test', clubSlug: 'slug', sessionKey: 'session' })); });
  assert.equal(navigated[0], '/clubs/slug/round-robin-generator/shared/session/rounds/7#edit=operator-token');
  tree.unmount();
}

async function sharedStandingsPollsWithoutOperationAccess() {
  window.location.hash = '';
  global.sessionStorage = { getItem: () => null, setItem() {} };
  let interval;
  const savedInterval = global.setInterval, savedClear = global.clearInterval;
  global.setInterval = (fn, delay) => { assert.equal(delay, 15000); interval = fn; return 123; };
  global.clearInterval = () => {};
  let current = { session_key: 'session', title: 'Evening RR', status: 'active', version: 'timestamp', generator_kind: 'round_robin', play_format: 'doubles', standings: [{ name: 'Joe', wins: 1 }], event: { rounds: [{ number: 1, status: 'saved' }] } };
  global.fetch = async (url, options) => {
    assert.ok(url.includes('/shared-play-generators/'));
    assert.ok(!options?.body, 'Spectator makes no write requests');
    return reply({ session: current });
  };
  const Link = ({ children, ...props }) => React.createElement('a', props, children);
  const Component = load('app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx', {
    '@/components/PublicClubLink': Link, 'next/navigation': { useRouter: () => ({}) },
    '@/components/GeneratorSharing': Sharing,
    '@/components/GeneratorSubmission': { __esModule: true, default: () => null, generatorResultLabel: () => 'Unrated' },
    '@/components/GeneratorMoneyball': load('components/GeneratorMoneyball.tsx'),
    '@/components/GeneratorPlayoff': { __esModule: true, default: () => null },
    '@/components/PlayGeneratorStandingsTable': { __esModule: true, default: ({ rows }) => React.createElement('p', {}, rows[0].wins), standingsSortLabel: () => 'Wins' }
  }).default;
  let tree;
  await act(async () => { tree = create(React.createElement(Component, { apiBase: 'https://api.test', clubId: 'slug', sessionKey: 'session', shared: true })); });
  assert.equal(button(tree, 'Next scheduled round'), undefined);
  assert.equal(button(tree, 'Hand off RR'), undefined);
  assert.ok(interval);
  current = { ...current, standings: [{ name: 'Joe', wins: 2 }] };
  await act(async () => interval());
  assert.ok(JSON.stringify(tree.toJSON()).includes('"2"'));
  await act(async () => tree.unmount());
  global.setInterval = savedInterval; global.clearInterval = savedClear;
}

(async () => {
  await sharingControls(); await currentRoundLanding(); await sharedStandingsPollsWithoutOperationAccess();
  console.log('PASS sharing controls, view-only standings link/QR, private fragment handoff, revocation, current-round routing, and live spectator standings.');
})().catch(error => { console.error(error); process.exitCode = 1; });
