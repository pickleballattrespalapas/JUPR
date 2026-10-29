const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { act, create } = require('react-test-renderer');
const { renderToStaticMarkup } = require('react-dom/server');
global.IS_REACT_ACT_ENVIRONMENT = true;

let pending = false;
const location = new URL('https://fixture.invalid/admin/tournaments/registration?tournament=fixture&tournament_name=Summer&draw=draw-one&day=day-two');
const mocks = {
  'next/link': { __esModule: true,
    default: ({ children, prefetch, onNavigate, ...props }) => React.createElement('a', props, children),
    useLinkStatus: () => ({ pending }) },
  'next/navigation': { usePathname: () => location.pathname, useSearchParams: () => location.searchParams },
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
  const Link = load('components/NavigationLink.tsx').default;
  const html = props => renderToStaticMarkup(React.createElement(Link, { href: '/next', ...props }, 'Next page'));
  assert.doesNotMatch(html({}), /Opening page/);
  pending = true;
  assert.match(html({}), /role="status".*Opening page/s);
  assert.doesNotMatch(html({}), /disabled|aria-disabled/);
  pending = false;
  assert.match(html({ target: '_blank', rel: 'noopener', 'aria-label': 'Open another tab' }), /target="_blank".*rel="noopener".*aria-label="Open another tab"/);

  for (const [file, props] of [
    ['components/TournamentAdminNav.tsx', {}],
    ['components/TournamentPhaseNav.tsx', { phase: 'registration' }],
  ]) {
    const Component = load(file).default;
    const markup = renderToStaticMarkup(React.createElement(Component, props));
    const hrefs = [...markup.matchAll(/href="([^"]+)"/g)].map(match => match[1].replaceAll('&amp;', '&'));
    const selectedLinks = hrefs.filter(href => href.startsWith('/admin/tournaments/'));
    assert.ok(selectedLinks.length >= 5);
    for (const href of selectedLinks) {
      const query = new URL(href, location.origin).searchParams;
      assert.equal(query.get('tournament'), 'fixture');
      assert.equal(query.get('draw'), 'draw-one');
      assert.equal(query.get('day'), 'day-two');
    }
  }

  let retries = 0, tree;
  const ErrorPage = load('components/PageLoadError.tsx').default;
  await act(async () => { tree = create(React.createElement(ErrorPage, {
    error: new Error('Private server credentials must not appear'), retry: () => { retries++; },
  })); });
  assert.doesNotMatch(JSON.stringify(tree.toJSON()), /Private server/);
  await act(async () => tree.root.findByType('button').props.onClick());
  assert.equal(retries, 1, 'one retry click asks Next to refetch the failed server component');
  await act(async () => tree.unmount());

  process.env.JUPR_API_BASE_URL = 'https://fixture.invalid';
  delete process.env.JUPR_INTERACTION_TEST_HARNESS;
  const requests = [];
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ enabled: true, tournament_count: null }) };
  };
  const adminApi = load('lib/adminTournamentApi.ts');
  assert.equal((await adminApi.getAdminTournamentStatus('club one')).data.enabled, true);
  assert.equal(requests[0].url, 'https://fixture.invalid/admin/clubs/club%20one/tournaments/admin/status?include_count=false');
  assert.equal(requests[0].options.cache, 'no-store');
  assert.ok(requests[0].options.signal instanceof AbortSignal);

  const siteApi = load('lib/clubSiteServer.ts');
  await siteApi.publicSiteFetch('/public/clubs/fixture/site');
  assert.ok(requests.at(-1).options.signal instanceof AbortSignal);
  global.fetch = async () => ({ ok: false, status: 404 });
  assert.equal(await siteApi.publicSiteFetch('/public/clubs/fixture/site'), null, 'unpublish is not masked by a cached document');
  global.fetch = async () => { throw new DOMException('The operation was aborted due to timeout', 'TimeoutError'); };
  await assert.rejects(siteApi.publicSiteFetch('/public/clubs/fixture/site'), { name: 'TimeoutError' });
  await assert.rejects(adminApi.getAdminTournamentStatus('fixture'), /Tournament Manager is temporarily unavailable/);
  global.fetch = async () => ({ ok: false, status: 503, text: async () => 'Private upstream failure' });
  await assert.rejects(adminApi.getAdminTournamentStatus('fixture'), /Tournament Manager is temporarily unavailable/);
  console.log('Tournament navigation: pending feedback, route context, retry, bounded fresh reads, and count-free status passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
