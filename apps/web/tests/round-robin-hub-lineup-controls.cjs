const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), {act, create} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const h = React.createElement;
function Dialog({title, children, actions}) { return h('section', {'data-dialog': title}, children, actions); }
function load(file) {
  const source = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../app/admin/round-robin-hub', file), 'utf8'), {
    compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true}
  }).outputText;
  const module = {exports: {}};
  new Function('require', 'module', 'exports', source)(name => {
    if (name === '@/components/interaction') return {InteractionDialog: Dialog};
    if (name === '@/components/PlayerSearchInput') return props => h('input', props);
    if (name === '@/lib/playerSearch') return {fetchAllPublicPlayers: async () => []};
    if (name === '@/lib/adminRoundRobinHubApi') return {hubPlayerLabel: p => p.display_name || p.name};
    if (name.endsWith('.module.css')) return {};
    return require(name);
  }, module, module.exports);
  return module.exports.default;
}
const Edit = load('HubEditPlayersDialog.tsx'), Remove = load('HubRemovePlayerDialog.tsx');
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const change = async (tree, label, value) => act(async () => tree.root.findByProps({'aria-label': label}).props.onChange({target: {value}}));
const submit = async tree => act(async () => tree.root.findByType('form').props.onSubmit({preventDefault() {}}));
const player = (id, extra = {}) => ({participant_id: id, name: id, identity: id, active: true,
  active_from_round: 1, can_remove: true, remove_effective_round: 3, ...extra});
const players = ['a', 'b', 'c', 'd', 'bye'].map(id => player(id, id === 'bye' ? {on_bye: true} : {}));
const group = {session_key: 'rr', title: 'Level 3.5', current_round_number: 2, can_add_player: true,
  can_remove_player: true, players};
const match = {session_key: 'rr', match_id: 'game', round_number: 2,
  side_a: [{id: 'a', name: 'a'}, {id: 'b', name: 'b'}], side_b: [{id: 'c', name: 'c'}, {id: 'd', name: 'd'}]};

(async () => {
  let tree; const exact = [], batch = [];
  let editProps = {apiBase: 'http://localhost', clubId: 'club', match, group,
    stale: false, available: true, busy: false, disabled: false, pending: false, error: '',
    onClose() {}, onRetry() {}, onReview() {}, onSubmit() { throw new Error('Wrong action'); },
    onSubmitExact: payload => exact.push(payload)};
  await act(async () => { tree = create(h(Edit, editProps)); });
  await change(tree, 'Change players', 'exact');
  await change(tree, 'Team A player 1', 'c');
  assert.equal(tree.root.findByProps({'aria-label': 'Team B player 1'}).props.value, 'a', 'Choosing an existing slot exchanges positions');
  await change(tree, 'Team B player 2', 'bye');
  await submit(tree);
  assert.deepEqual(exact, [{side_a: ['c', 'b'], side_b: ['a', 'bye']}]);
  editProps = {...editProps, stale: true};
  await act(async () => tree.update(h(Edit, editProps)));
  assert(button(tree, 'Save both teams').props.disabled);
  await submit(tree);
  assert.equal(exact.length, 1, 'A stale exact lineup cannot be submitted');
  assert.equal(tree.root.findByProps({'aria-label': 'Team B player 2'}).props.value, 'bye', 'Review preserves the drafted teams');
  editProps = {...editProps, stale: false, group: {...group, players: players.map(p => p.participant_id === 'bye' ? {...p, identity: 'a'} : p)}};
  await act(async () => tree.update(h(Edit, editProps)));
  assert(button(tree, 'Save both teams').props.disabled);
  assert.match(text(tree.toJSON()), /four different people/);
  editProps = {...editProps, group: {...group, current_round_number: 1, players: players.map(p => p.participant_id === 'bye'
    ? {...p, on_bye: false, on_court: true, playing_elsewhere: true, current_court: {session_key: 'other', court_id: 'court-2'}} : p)}};
  await act(async () => tree.update(h(Edit, editProps)));
  assert.equal(tree.root.findByProps({'aria-label': 'Team B player 2'}).findAllByType('option').find(option => option.props.value === 'bye').props.disabled, false,
    'A future plan can include someone currently playing in another group');
  assert(!button(tree, 'Save both teams').props.disabled);
  assert.match(text(tree.toJSON()), /once all four are free/);
  await change(tree, 'Change players', 'replace');
  await change(tree, 'Player to replace or swap', 'a');
  assert(tree.root.findByProps({'aria-label': 'Replacement'}).findByProps({value: 'bye'}).props.disabled,
    'The ordinary current-lineup replacement guard remains conservative');
  await act(async () => tree.unmount());

  let removeProps = {groups: [{...group, players: players.map(p => ({...p, remove_affected_match_ids: ['future-1']}))}],
    busy: false, disabled: false, pending: false, error: '', onClose() {}, onRetry() {},
    onSubmit() { throw new Error('Expected batch action'); }, onSubmitMany: (...args) => batch.push(args)};
  await act(async () => { tree = create(h(Remove, removeProps)); });
  await change(tree, 'Round robin', 'rr');
  for (const id of ['a', 'bye']) await act(async () => tree.root.findByProps({'aria-label': `Remove ${id}`}).props.onChange({target: {checked: true}}));
  assert.match(text(tree.toJSON()), /From round 3/);
  assert.match(text(tree.toJSON()), /1 planned game may change/);
  await submit(tree);
  assert.deepEqual(batch, [['rr', ['a', 'bye']]]);
  removeProps = {...removeProps, pending: true};
  await act(async () => tree.update(h(Remove, removeProps)));
  assert(button(tree, 'Remove 2 players').props.disabled);
  await submit(tree);
  assert.equal(batch.length, 1);
  removeProps = {...removeProps, pending: false, groups: [{...group, players: [player('a'), player('different-id', {name: 'bye'})]}]};
  await act(async () => tree.update(h(Remove, removeProps)));
  assert(button(tree, 'Remove 2 players').props.disabled);
  assert.match(text(tree.toJSON()), /bye is no longer in this round robin/);
  await submit(tree);
  assert.equal(batch.length, 1, 'A same-name replacement never becomes a selected departure');
  await act(async () => tree.unmount());
  console.log('PASS exact lineup and batch departures: atomic payloads, swaps, identities, stale/pending guards, and departure preview.');
})().catch(error => {console.error(error); process.exitCode = 1;});
