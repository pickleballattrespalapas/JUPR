const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const { act, create } = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
process.env.JUPR_API_BASE_URL = 'https://api.fixture.invalid';
delete process.env.JUPR_INTERACTION_TEST_HARNESS;

let refreshes = 0, authorized = true, failure = null, currentPath = '/clubs/alpha/round-robin-generator';
const requests = [];
const site = { slug: 'alpha', club_id: 'alpha', document: {
  name: 'Alpha', description: 'Welcome', location: 'Baja', visibility: 'listed',
  accent: '#2563eb', display: { ratings: false }, page_visibility: { players: 'private' },
  logo_url: '/public/clubs/alpha/site/logo', pages: [
    { slug: 'home', title: 'Home', in_navigation: true },
    { slug: 'visiting', title: 'Visiting', in_navigation: true, blocks: [{ kind: 'text', heading: 'Hello', text: 'Visitors' }] },
  ],
} };
global.fetch = async (url, options) => {
  requests.push({ url, options });
  if (failure === 'timeout') throw new DOMException('Private timeout details', 'TimeoutError');
  if (failure) return { ok: false, status: failure, text: async () => 'Private upstream error' };
  return { ok: true, status: 200, json: async () => url.includes('/tournaments/admin/status') ? { enabled: true } : structuredClone(site) };
};
const mocks = {
  'next/link': { __esModule: true, default: ({ children, prefetch, ...props }) => React.createElement('a', props, children) },
  'next/headers': { headers: async () => new Headers({ 'x-pcs-club-path': currentPath }) },
  '@/components/interaction': { InteractionProvider: ({ children }) => children },
  'next/navigation': {
    useRouter: () => ({ refresh: () => refreshes++ }),
    usePathname: () => currentPath,
    notFound: () => { throw Error('NOT_FOUND'); },
  },
  '@/lib/shareMetadata': { requestShareMetadata: async () => ({}) },
  '@/lib/adminWorkspaceServer': { requireAdminWorkspace: async () => {
    if (!authorized) throw Error('LOGIN_REQUIRED');
    return { clubId: 'alpha' };
  } },
  '@/components/RememberPublicClub': { __esModule: true, default: () => null },
  './TournamentAdminPanel': { __esModule: true, default: () => React.createElement('div', null, 'Tournament controls') },
};
const modules = new Map();
function load(file) {
  if (modules.has(file)) return modules.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    if (mocks[name]) return mocks[name];
    if (name.endsWith('.css')) return { __esModule: true, default: {} };
    if (name.startsWith('@/') || name.startsWith('.')) {
      const relative = name.startsWith('@/') ? name.slice(2) : path.join(path.dirname(file), name);
      return load(relative + (fs.existsSync(path.join(__dirname, '..', relative + '.tsx')) ? '.tsx' : '.ts'));
    }
    return require(name);
  }, module, module.exports);
  modules.set(file, module.exports);
  return module.exports;
}

(async () => {
  const PublishedClubShell = load('components/PublishedClubShell.tsx').default;
  const props = { params: Promise.resolve({ clubSlug: 'alpha' }), children: React.createElement('p', null, 'Current court lineup') };
  let markup = renderToStaticMarkup(await PublishedClubShell(props));
  assert.match(markup, /Current court lineup/);
  assert.match(markup, /https:\/\/api.fixture.invalid\/public\/clubs\/alpha\/site\/logo/);
  assert.equal(requests.length, 1, 'Rendering the shell never fetches optional logo bytes');
  assert.ok(requests[0].url.endsWith('/site/shell'));
  assert.equal(requests[0].options.cache, 'no-store');
  failure = 'timeout';
  markup = renderToStaticMarkup(await PublishedClubShell(props));
  assert.match(markup, /Club page temporarily unavailable.*Try again/s);
  assert.doesNotMatch(markup, /Current court lineup|Private timeout/, 'Unknown publication never falls back to stale club contents');
  failure = 404;
  await assert.rejects(PublishedClubShell(props), /NOT_FOUND/);
  failure = null;
  assert.match(renderToStaticMarkup(await PublishedClubShell(props)), /Current court lineup/);

  const RootLayout = load('app/layout.tsx').default;
  const root = RootLayout({ children: React.createElement('p', null, 'Page remains available') });
  assert.equal(typeof root?.then, 'undefined', 'The outer root never waits for navigation data');
  const collect = node => !node || typeof node !== 'object' ? [] : [node, ...React.Children.toArray(node.props?.children).flatMap(collect)];
  const nodes = collect(root);
  const header = nodes.find(node => node.type?.name === 'HeaderNavigation');
  const footer = nodes.find(node => node.type?.name === 'FooterNavigation');
  for (const route of ['/clubs/alpha/round-robin-generator', '/admin/tournaments']) {
    currentPath = route;
    requests.length = 0;
    await header.type(header.props);
    await footer.type(footer.props);
    assert.equal(requests.length, 0, `${route} has no global club-site dependency`);
  }
  currentPath = '/faq';
  requests.length = 0;
  const globalHeader = await header.type(header.props);
  assert.ok(requests.every(request => request.url.endsWith('/site/shell')), 'Global navigation reads only compact published settings');
  assert.equal(globalHeader.props.clubDocument.page_visibility.players, 'private');
  failure = 'timeout';
  assert.equal((await header.type(header.props)).props.clubDocument, null, 'Navigation failures retain safe generic links');
  assert.match(renderToStaticMarkup(root), /Page remains available/, 'Navigation failure cannot hide route content');
  currentPath = '/clubs/alpha/round-robin-generator';
  failure = null;

  const Page = load('app/admin/tournaments/page.tsx').default;
  failure = 503;
  markup = renderToStaticMarkup(await Page());
  assert.match(markup, /Tournament Manager.*Try again.*Admin Tools.*Operations cockpit/s);
  assert.doesNotMatch(markup, /Tournament controls|Private upstream/);
  failure = 'timeout';
  assert.match(renderToStaticMarkup(await Page()), /Try again/);
  failure = null;
  assert.match(renderToStaticMarkup(await Page()), /Tournament controls/);
  authorized = false;
  const before = requests.length;
  await assert.rejects(Page(), /LOGIN_REQUIRED/);
  assert.equal(requests.length, before, 'Authentication still runs before reading tournament status');

  const Retry = load('components/SectionLoadError.tsx').default;
  let tree;
  await act(async () => { tree = create(React.createElement(Retry, { title: 'Unavailable', message: 'Try again.' })); });
  await act(async () => tree.root.findByType('button').props.onClick());
  assert.equal(refreshes, 1, 'Retry requests fresh server data without replaying writes');
  await act(async () => tree.unmount());

  const { loadSharingClub } = load('lib/shareClub.ts');
  requests.length = 0;
  let club = await loadSharingClub('alpha', '/clubs/alpha/players/42', async () => null);
  assert.equal(club.name, 'Alpha');
  assert.equal(club.discoverable, false, 'Fresh hidden-section setting applies to metadata');
  assert.ok(requests[0].url.endsWith('/site/shell'), 'Metadata for built-in pages uses only the compact shell');
  requests.length = 0;
  club = await loadSharingClub('alpha', '/clubs/alpha/pages/visiting', async () => null);
  assert.equal(club.page.description, 'Hello Visitors');
  assert.ok(requests[0].url.endsWith('/site'), 'Custom-page previews retain actual published content');
  console.log('Page resilience: compact shell, fail-closed publication, optional logo, local retry, navigation, auth and metadata scoping passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
