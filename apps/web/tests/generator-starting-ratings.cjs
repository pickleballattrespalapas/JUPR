const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const React = require('react');
const ts = require('typescript');
const {create, act} = require('react-test-renderer');
const mocks = {
  'next/link': ({children, ...props}) => React.createElement('a', props, children),
  '@/lib/useAdminSession': {useAdminSession: () => ({accessToken:'admin-token'})},
  '@/components/ConfirmAction': {ConfirmAction: props => React.createElement('button', {disabled:props.disabled, onClick:props.onConfirm}, props.triggerLabel)},
  '@/components/interaction': {actionSuccess: (title, message) => ({title, message})},
  '@/components/SearchablePlayerSelect': ({children, onValueChange, ...props}) => React.createElement('select', {...props, onChange:e => onValueChange(e.target.value)}, children)
};
const source = fs.readFileSync(path.join(__dirname, '../app/admin/play-generators/submissions/GeneratorSubmissions.tsx'), 'utf8');
const code = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2022, module:ts.ModuleKind.CommonJS, jsx:ts.JsxEmit.ReactJSX, esModuleInterop:true}}).outputText;
const compiled = {exports:{}};
new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), compiled, compiled.exports);
const Component = compiled.exports.default;
const label = node => typeof node === 'string' ? node : (node?.children || []).map(label).join('');
const button = (tree, text) => tree.root.findAllByType('button').find(node => label(node) === text);
const input = (tree, name) => tree.root.findByProps({'aria-label':name});
const rating = (level, games = 0, editable = true) => ({rating_jupr:level, matches_played:games, can_set_starting_rating:editable && games === 0, state_fingerprint:'a'.repeat(64)});
const seed = {
  submissions:[{id:'review', session_key:'session', title:'Evening RR', version:3, status:'pending', organizer_name:'Joe', rating_mode:'rated', match_date:'2026-10-10', match_count:2,
    participants:[1,2,3,4].map(id => ({id:'p'+id, name:'Player '+id, player_id:id})),
    matches:[{id:'doubles', round_number:1, sideA:['p1','p2'], sideB:['p3','p4'], scoreA:11, scoreB:8}, {id:'singles', round_number:2, sideA:['p1'], sideB:['p2'], scoreA:11, scoreB:7}]}],
  players:[1,2,3,4].map(id => ({id, name:'Player '+id, ratings:{doubles:rating(3.5, id === 3 ? 12 : 0), singles:rating(3.25, 0, false)}}))
};
global.window = {location:{search:''}};
async function mount(queue = structuredClone(seed)) {
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({url, options});
    if (options.method === 'PATCH') {
      const body = JSON.parse(options.body);
      const id = Number(url.split('/').at(-1));
      const player = queue.players.find(p => p.id === id);
      player.ratings[body.match_format] = rating(body.rating_jupr);
      return {ok:true, json:async () => ({ok:true, player:structuredClone(player)})};
    }
    if (url.endsWith('/review')) return {ok:true, json:async () => ({status:'approved'})};
    return {ok:true, json:async () => structuredClone(queue)};
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Component, {clubId:'club', apiBase:'https://api.test'})); });
  await act(async () => tree.root.findAllByType('button').find(n => label(n).startsWith('Evening RR')).props.onClick());
  return {tree, calls};
}
async function main() {
  const {tree, calls} = await mount();
  assert.equal(input(tree, 'Player 1 doubles starting level').props.value, '3.5');
  assert.equal(tree.root.findAllByProps({'aria-label':'Player 1 singles starting level'}).length, 0, 'Singles baselines are not editable here');
  assert.equal(tree.root.findAllByProps({'aria-label':'Player 3 doubles starting level'}).length, 0, 'Established players cannot be reset');
  assert.equal(button(tree, 'Approve rated results').props.disabled, false);
  await act(async () => input(tree, 'Player 1 doubles starting level').props.onChange({target:{value:'4.25'}}));
  assert.equal(button(tree, 'Approve rated results').props.disabled, true, 'Unsaved levels block approval');
  await act(async () => input(tree, 'Player 2 doubles starting level').props.onChange({target:{value:'3.75'}}));
  await act(async () => button(tree, 'Save doubles starting level').props.onClick());
  assert.equal(button(tree, 'Approve rated results').props.disabled, true, 'Other edits stay pending');
  assert.equal(input(tree, 'Player 1 doubles starting level').props.value, '4.25');
  await act(async () => button(tree, 'Save doubles starting level').props.onClick());
  assert.equal(button(tree, 'Approve rated results').props.disabled, false);
  const patch = calls.find(c => c.options.method === 'PATCH');
  assert(patch.url.endsWith('/sessions/session/starting-ratings/1'));
  assert.deepEqual(JSON.parse(patch.options.body), {participant_id:'p1', match_format:'doubles', rating_jupr:4.25, expected_state_fingerprint:'a'.repeat(64)});
  assert.equal(patch.options.headers.Authorization, 'Bearer admin-token');
  await act(async () => input(tree, 'Player 1 doubles starting level').props.onChange({target:{value:''}}));
  assert.equal(button(tree, 'Save doubles starting level').props.disabled, true);
  assert.equal(button(tree, 'Approve rated results').props.disabled, true);
  await act(async () => input(tree, 'Player 1 doubles starting level').props.onChange({target:{value:'7.5'}}));
  assert.equal(button(tree, 'Save doubles starting level').props.disabled, true);
  assert.equal(button(tree, 'Reject').props.disabled, false, 'Rating edits never prevent rejection');
  await act(async () => tree.unmount());

  const missing = structuredClone(seed);
  missing.players[0].ratings.doubles.rating_jupr = null;
  const nullRating = await mount(missing);
  assert.equal(input(nullRating.tree, 'Player 1 doubles starting level').props.value, '');
  assert.equal(button(nullRating.tree, 'Approve rated results').props.disabled, true);
  await act(async () => nullRating.tree.unmount());

  const unrated = structuredClone(seed);
  unrated.submissions[0].rating_mode = 'unrated';
  const unscored = await mount(unrated);
  assert.equal(unscored.tree.root.findAllByType('input').filter(n => n.props.type === 'number').length, 0);
  assert.equal(button(unscored.tree, 'Approve unrated results').props.disabled, false);
  await act(async () => unscored.tree.unmount());
  console.log('PASS: doubles starting levels, established ratings, singles isolation, validation, save-before-approve, and unrated results');
}
main().catch(error => {console.error(error); process.exitCode=1;});
