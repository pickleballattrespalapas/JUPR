const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), { act, create } = require('react-test-renderer');
const root = path.resolve(__dirname, '..'), cache = new Map();
const stubs = {
  '@/lib/useAdminSession': { useAdminSession: () => ({ loading: false, accessToken: 'local-test', session: { user: { email: 'staff@example.test' } } }), adminSessionLabel: () => 'Staff' },
  '@/components/ConfirmAction': { ConfirmAction: () => null },
  '@/components/interaction': {},
};
function load(name, parent = root) {
  if (stubs[name]) return stubs[name];
  if (!name.startsWith('@/') && !name.startsWith('.')) return require(name);
  const base = name.startsWith('@/') ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + '.tsx', base + '.ts', path.join(base, 'index.ts')].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function('require', 'module', 'exports', code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const h = React.createElement;
const players = [{ id: 200, name: 'Returning Player', rating: 2046, is_active: false }, { id: 201, name: 'Regular Player', rating: 1600, is_active: true }];
const nodeText = node => typeof node === 'string' ? node : (node?.children || []).map(nodeText).join('');
global.sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
(async () => {
  const Explorer = load('@/app/clubs/[clubSlug]/match-explorer/MatchExplorerForm').default;
  let tree;
  await act(async () => { tree = create(h(Explorer, { apiBase: null, clubSlug: 'club', players, contexts: ['OVERALL'] })); });
  assert.ok(tree.root.findAllByType('option').some(option => String(option.props.value) === '200'), 'Match Explorer retains the inactive profile ID');
  await act(async () => tree.unmount());

  const Moneyball = load('@/app/admin/moneyball/MoneyballPanel').default;
  await act(async () => { tree = create(h(Moneyball, { apiBase: null, clubId: 'club', status: { enabled: true, status: 'ready', players } })); });
  const picker = tree.root.findByType(load('@/components/SearchablePlayerSelect').default);
  await act(async () => picker.props.onValuesChange(['200']));
  assert.match(nodeText(tree.toJSON()), /P1: Returning Player/, 'Moneyball selects the existing inactive profile');
  await act(async () => tree.unmount());

  const Live = load('@/app/clubs/[clubSlug]/live/PublicLiveCreator').default;
  await act(async () => { tree = create(h(Live, { apiBase: null, clubSlug: 'club', players })); });
  const search = tree.root.findByProps({ placeholder: 'Type at least 2 letters, then add a player' });
  await act(async () => search.props.onChange({ target: { value: 'Returning' } }));
  const add = tree.root.findAllByType('button').find(button => nodeText(button).startsWith('Add'));
  assert.ok(add && !add.props.disabled, 'An inactive profile can be added to live play');
  await act(async () => add.props.onClick());
  assert.ok(tree.root.findAllByType('textarea').some(node => String(node.props.value).includes('Returning Player')));
  await act(async () => tree.unmount());
  console.log('PASS inactive profiles: Match Explorer ID, Moneyball selection, live-play search and add.');
})().catch(error => { console.error(error); process.exit(1); });
