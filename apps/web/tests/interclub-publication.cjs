const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const { renderToStaticMarkup } = require('react-dom/server');
function load(file, mocks) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const css = new Proxy({}, { get: (_, key) => key });
let club = 'alpha', fingerprint = 'a'.repeat(64), conflict = true, posts = [];
const Component = load('app/admin/interclub/publication/PublicationEditor.tsx', {
  'next/link': ({ children, href }) => React.createElement('a', { href }, children),
  'next/navigation': { useSearchParams: () => new URLSearchParams('season=season') },
  '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: 'alpha' }) },
  '@/lib/useAdminSession': { useAdminSession: () => ({ session: { user: { id: 'user' }, capabilities: { assignments: [{ club_id: 'alpha', role: 'administrator' }] } }, accessToken: 'test-token', loading: false }) },
  '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' },
  '@/lib/adminWorkspace': { readBrowserWorkspace: () => ({ clubId: club }) },
  '@/lib/interclubRegistration': { apiError: (value, fallback) => value.detail || fallback },
  '@/components/PublicInterclubLeague': () => React.createElement('p', null, 'League preview contents'),
  '@/components/ClubWebsite.module.css': css,
}).default;
const context = () => ({ season: { details: { name: 'Southern BCS' } }, clubs: [], meets: [], publication: { revision: 1, draft: { results: [] }, published: null }, preview: {}, preview_fingerprint: fingerprint });
global.fetch = async (url, options) => {
  if (!options.method) return { ok: true, json: async () => context() };
  posts.push(JSON.parse(options.body));
  return conflict ? { ok: false, status: 409, json: async () => ({ detail: 'Results changed. Reload and review.' }) } : { ok: true, status: 200, json: async () => ({ ...context().publication, revision: 2 }) };
};
const text = node => typeof node === 'string' ? node : node.children.map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
async function main() {
  let tree;
  await act(async () => { tree = create(React.createElement(Component)); });
  assert.equal(button(tree, 'Publish league').props.disabled, true, 'Publishing requires opening the loaded preview');
  await act(async () => button(tree, 'Preview league website').props.onClick());
  assert.equal(button(tree, 'Publish league').props.disabled, false);
  await act(async () => button(tree, 'Publish league').props.onClick());
  assert.equal(posts[0].preview_fingerprint, 'a'.repeat(64));
  assert.equal(button(tree, 'Publish league').props.disabled, true, 'Conflict blocks blind retry');
  fingerprint = 'b'.repeat(64);
  await act(async () => button(tree, 'Reload saved draft').props.onClick());
  assert.equal(button(tree, 'Publish league').props.disabled, true, 'Reload requires reviewing the new source snapshot');
  await act(async () => button(tree, 'Preview league website').props.onClick());
  conflict = false;
  await act(async () => button(tree, 'Publish league').props.onClick());
  assert.equal(posts[1].preview_fingerprint, 'b'.repeat(64));
  await act(async () => button(tree, 'Preview league website').props.onClick());
  club = 'beta';
  await act(async () => button(tree, 'Publish league').props.onClick());
  assert.equal(posts.length, 2, 'A stale club tab cannot publish');
  await act(async () => tree.unmount());
  const Results = load('components/PublicInterclubCompetition.tsx', { './ClubWebsite.module.css': css }).CompetitionResults;
  const markup = renderToStaticMarkup(React.createElement(Results, { names: { a: 'Alpha', b: 'Beta' }, results: [{ id: 'game', meet_id: 'meet', phase: 'regular', division: '3.5', club_a: 'a', club_b: 'b', weather: 'normal', pairings: [{ kind: 'women', games: [{ status: 'retired', a: 9, b: 4, winner: 'b' }] }] }] }));
  assert.ok(markup.includes('9–4') && markup.includes('Beta awarded the game'), 'Retirement shows actual score and conceded-game winner');
  console.log('Interclub publication review, conflict, stale-club and retirement display checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
