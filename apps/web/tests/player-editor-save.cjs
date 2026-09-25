const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), Module = require('node:module');
const React = require('react'), { act, create } = require('react-test-renderer');
const ts = require('typescript');
const h = React.createElement;
function load(file, overrides = {}) {
  const filename = path.resolve(__dirname, '..', file);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] ? { __esModule: true, ...overrides[name] } : original(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}
const Editor = load('app/admin/players/PlayerEditorPanel.tsx', {
  'next/link': { default: ({ children }) => h('span', null, children) },
  '@/components/SearchablePlayerSelect': { default: ({ onValueChange, children, ...props }) => h('select', { ...props, onChange: e => onValueChange(e.target.value) }, children) },
  '@/components/ConfirmAction': { ConfirmAction: () => null },
  '@/components/interaction': {
    InteractionDialog: ({ open, children }) => open ? h('div', { role: 'dialog' }, children) : null,
    StaticActionFeedback: ({ title, description }) => h('div', null, title, description),
  },
  '@/lib/useAuthenticatedAutoLoad': load('lib/useAuthenticatedAutoLoad.ts'),
  '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'test-session', loading: false, session: {} }), adminSessionLabel: () => 'Test admin' },
}).default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
let player, tree;
const writes = [];
global.fetch = async (url, options = {}) => {
  let payload;
  if (url.endsWith('/social-identities')) return { ok: false, status: 503, json: async () => ({ detail: 'Unavailable' }) };
  if (options.method === 'PATCH') {
    const body = JSON.parse(options.body); writes.push(body);
    player = { ...player, ...(body.name ? { name: body.name } : {}), ...(body.rating_jupr != null ? { rating_jupr: body.rating_jupr } : {}), state_fingerprint: 'b'.repeat(64) };
    payload = { ok: true, player };
  } else if (url.endsWith('/players/1')) payload = { player, league_ratings: [], match_reference_counts: { total: 9 } };
  else if (url.endsWith('/players')) payload = { players: [player], count: 1 };
  else throw new Error(`Unexpected request: ${url}`);
  return { ok: true, status: 200, json: async () => payload };
};
function input(label) {
  return tree.root.findAllByType('label').find(node => node.findAllByType('strong').some(strong => strong.children.join('') === label)).findByType('input');
}
const save = () => tree.root.findAllByType('button').find(node => node.children.join('') === 'Save player');
async function mount(starting) {
  player = { id: 1, name: 'Fixture Player', rating_jupr: 4.155, starting_jupr: starting, active: true, state_fingerprint: 'a'.repeat(64) };
  await act(async () => { tree = create(h(Editor, { apiBase: 'https://api.test', clubId: 'club', status: { enabled: true, merge_enabled: false, warnings: [] } })); });
  assert.match(text(tree.toJSON()), /can still edit player profiles/);
  assert.doesNotMatch(text(tree.toJSON()), /Merge player accounts/);
  const picker = tree.root.findAllByType('select').find(node => node.props['aria-label'] === 'Select player');
  assert.ok(picker, 'Player roster remains available when social identities fail');
  await act(async () => picker.props.onChange({ target: { value: '1' } }));
}
(async () => {
  await mount(3.503086419725);
  assert.equal(input('Overall JUPR').props.step, 0.001);
  await act(async () => input('Overall JUPR').props.onChange({ target: { value: '4.408' } }));
  await act(async () => save().props.onClick());
  assert.equal(writes.length, 1);
  assert.equal(writes[0].rating_jupr, 4.408);
  for (const field of ['starting_jupr', 'name', 'active']) assert.equal(field in writes[0], false, `Unchanged ${field} is not written`);
  assert.equal(writes[0].expected_state_fingerprint, 'a'.repeat(64));
  assert.match(text(tree.toJSON()), /Player saved: Fixture Player/);
  await act(async () => save().props.onClick());
  assert.equal(writes.length, 1, 'No-change saves do not send another write');
  await act(async () => tree.unmount());
  await mount(null);
  await act(async () => input('Overall JUPR').props.onChange({ target: { value: '4.408' } }));
  await act(async () => save().props.onClick());
  assert.equal(writes.length, 2, 'A missing historical starting rating does not block a current-rating correction');
  assert.equal('starting_jupr' in writes[1], false);
  await act(async () => tree.unmount());
  console.log('Player Editor: precise rating saves, unchanged history, no-op saves, and independent roster loading passed.');
})().catch(error => { console.error(error); process.exit(1); });
