const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), {act, create} = require('react-test-renderer');
const h = React.createElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
const source = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../app/admin/round-robin-hub/HubRemovePlayerDialog.tsx'), 'utf8'), {
  compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true}
}).outputText;
const compiled = {exports: {}};
function Dialog({title, description, children, actions, onRequestClose}) {
  return h('section', {'data-dialog': title, onRequestClose}, description, children, actions);
}
new Function('require', 'module', 'exports', source)(name => {
  if (name === '@/components/interaction') return {InteractionDialog: Dialog};
  if (name === '@/lib/adminRoundRobinHubApi') return {hubPlayerLabel: player => player.display_name || player.name};
  if (name.endsWith('.module.css')) return {};
  return require(name);
}, compiled, compiled.exports);
const Component = compiled.exports.default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const select = (tree, label) => tree.root.findByProps({'aria-label': label});
const player = (id, name, extra = {}) => ({participant_id: id, name, identity: id, profile_id: null, paused: false,
  wait_seconds: 0, on_court: false, reserved: false, active: true, can_remove: true, ...extra});
const group = (key, title, extra = {}) => ({session_key: key, title, current_round_number: 2, can_remove_player: true,
  players: [player('p10', 'Player 10'), player('p2', 'Player 2'), player('departed', 'Left', {active: false, can_remove: false}),
    player('future', 'Future player', {active: false, active_from_round: 3})], ...extra});
const submit = tree => tree.root.findByType('form').props.onSubmit({preventDefault(){}});

(async () => {
  const calls = []; let closes = 0, retries = 0, tree;
  let props = {groups: [group('rr12', 'Round robin 12'), group('rr2', 'Round robin 2')], busy: false, disabled: false,
    pending: false, error: '', onClose: () => closes++, onRetry: () => retries++, onSubmit: (...args) => calls.push(args)};
  const update = async patch => {props = {...props, ...patch}; await act(async () => tree.update(h(Component, props)));};
  const choose = async (label, value) => act(async () => select(tree, label).props.onChange({target: {value}}));
  await act(async () => {tree = create(h(Component, props));});
  assert.equal(select(tree, 'Round robin').props.value, '');
  assert.equal(select(tree, 'Player').props.disabled, true);
  assert.equal(button(tree, 'Remove from this round robin').props.disabled, true);
  assert.deepEqual(select(tree, 'Round robin').findAllByType('option').map(row => row.props.value), ['', 'rr2', 'rr12']);
  await choose('Round robin', 'rr2');
  assert.equal(select(tree, 'Player').props.value, '', 'Picking a round robin must not choose a player');
  assert.deepEqual(select(tree, 'Player').findAllByType('option').map(row => row.props.value), ['', 'future', 'departed', 'p2', 'p10']);
  assert.equal(select(tree, 'Player').findByProps({value:'departed'}).props.disabled, true);
  await choose('Player', 'future');
  await act(async () => submit(tree));
  assert.deepEqual(calls, [['rr2', 'future']], 'Future arrivals are removable by their exact participant ID');
  assert.match(text(tree.toJSON()), /waiting matches and byes/);

  await choose('Player', 'p2');
  await update({groups: [group('rr2', 'Round robin 2', {players: [player('p2', 'Player 2', {on_court: true})]})]});
  assert.match(text(tree.toJSON()), /finish their current match and won’t join the next round/);
  await update({groups: [group('rr2', 'Round robin 2', {players: [player('p2', 'Player 2', {reserved: true})]})]});
  assert.match(text(tree.toJSON()), /reserved match stays in place/);
  await update({pending: true, error: 'Checking the saved removal.'});
  assert.ok(select(tree, 'Round robin').props.disabled && select(tree, 'Player').props.disabled);
  assert.equal(button(tree, 'Cancel').props.disabled, true);
  await choose('Round robin', 'rr12');
  assert.equal(select(tree, 'Round robin').props.value, 'rr2');
  await act(async () => {submit(tree); tree.root.findByType(Dialog).props.onRequestClose(); button(tree, 'Retry saved action').props.onClick();});
  assert.equal(calls.length, 1); assert.equal(closes, 0); assert.equal(retries, 1);
  await update({busy: true});
  assert.equal(button(tree, 'Retry saved action').props.disabled, true);
  await update({busy: false, pending: false, error: ''});

  await update({groups: [group('rr2', 'Round robin 2', {players: [player('replacement', 'Player 2')]})]});
  assert.equal(select(tree, 'Player').props.value, 'p2');
  assert.match(text(tree.toJSON()), /This player is no longer in this round robin/);
  assert.equal(button(tree, 'Remove from this round robin').props.disabled, true);
  await act(async () => submit(tree));
  assert.equal(calls.length, 1, 'A replacement with the same name must never become the selected target');
  await update({groups: [group('rr12', 'Round robin 12')]});
  assert.equal(select(tree, 'Round robin').props.value, 'rr2');
  assert.match(text(tree.toJSON()), /This round robin is no longer in the event/);
  assert.equal(button(tree, 'Remove from this round robin').props.disabled, true);
  await choose('Round robin', 'rr12');
  assert.equal(select(tree, 'Player').props.value, '', 'An explicit group change clears the old player');
  await choose('Player', 'p2');
  await update({groups: [group('rr12', 'Round robin 12', {can_remove_player: false, remove_player_unavailable_reason: 'Moneyball lineups are fixed.'})]});
  assert.match(text(tree.toJSON()), /Moneyball lineups are fixed/);
  assert.equal(button(tree, 'Remove from this round robin').props.disabled, true);
  await update({groups: [group('rr12', 'Round robin 12', {players: [player('p2', 'Player 2', {active: false, can_remove: false, remove_unavailable_reason: 'This player has left the roster.'})]})]});
  assert.equal(select(tree, 'Player').props.value, 'p2');
  assert.match(text(tree.toJSON()), /This player has left the roster/);
  assert.equal(button(tree, 'Remove from this round robin').props.disabled, true);
  await act(async () => tree.unmount());
  await act(async () => {tree = create(h(Component, {...props, groups: [group('only', 'Only group')]}));});
  assert.equal(select(tree, 'Round robin').props.value, '', 'Even one round robin requires an explicit choice');
  await act(async () => tree.unmount());
  console.log('PASS hub removal dialog: explicit target, natural sorting, current/future roster, protected match copy, frozen exact retry, and stale selection safety.');
})().catch(error => {console.error(error); process.exitCode = 1;});
