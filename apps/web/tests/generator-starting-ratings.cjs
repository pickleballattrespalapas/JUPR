const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const React = require('react');
const ts = require('typescript');
const {create, act} = require('react-test-renderer');
let automaticallyReviewed = true;
const reviewPanels = new Map();
function RatingReview(props) {
  reviewPanels.set(props.matchFormat, props);
  React.useEffect(() => { props.onReady(automaticallyReviewed, []); }, [props.playerIds.join(','), props.matchFormat]);
  return null;
}
const mocks = {
  "@/components/StartingRatingReview": {__esModule:true, default:RatingReview},
  'next/link': ({children, ...props}) => React.createElement('a', props, children),
  '@/lib/useAdminSession': {useAdminSession: () => ({accessToken:'admin-token'})},
  '@/components/ConfirmAction': {ConfirmAction: props => React.createElement('button', {disabled:props.disabled, onClick:props.onConfirm}, props.triggerLabel)},
  '@/components/SearchablePlayerSelect': ({children, onValueChange, ...props}) => React.createElement('select', {...props, onChange:e => onValueChange(e.target.value)}, children)
};
function compile(file) {
  const source = fs.readFileSync(file, 'utf8');
  const code = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2022, module:ts.ModuleKind.CommonJS, jsx:ts.JsxEmit.ReactJSX, esModuleInterop:true}}).outputText;
  const compiled = {exports:{}};
  new Function('require', 'module', 'exports', code)(name => {
    if (Object.hasOwn(mocks, name)) return mocks[name];
    if (name === './GeneratorPlayerCreate') return compile(path.join(path.dirname(file), name + '.tsx'));
    if (name === '@/lib/playerSearch') return compile(path.join(__dirname, '../lib/playerSearch.ts'));
    return require(name);
  }, compiled, compiled.exports);
  return compiled.exports;
}
mocks['@/components/interaction'] = compile(path.join(__dirname, '../components/interaction/types.ts'));
const Component = compile(path.join(__dirname, '../app/admin/play-generators/submissions/GeneratorSubmissions.tsx')).default;
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
const stored = new Map();
global.sessionStorage = {getItem:key => stored.get(key) || null, setItem:(key,value) => stored.set(key,value), removeItem:key => stored.delete(key)};
const operations = new Map();
async function mount(queue = structuredClone(seed), options = {}) {
  const calls = [];
  let failReload = false;
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
    if (url.endsWith('/players/editor/players') && options.method === 'POST') {
      const body = JSON.parse(options.body);
      const player = {id:9, name:body.name, ratings:{doubles:rating(body.starting_jupr), singles:rating(3.5, 0, false)}};
      queue.players.push(player);
      const result = {ok:true, player:{id:9, name:body.name}};
      operations.set(body.idempotency_key, result);
      if (creationFailure === 'lost') throw new TypeError('Network response lost');
      if (creationFailure === 'reconcile') return {ok:false, status:409, json:async () => ({detail:{message:'Check saved player', recovery_required:true}})};
      failReload = reloadFailure;
      return {ok:true, json:async () => result};
    }
    if (url.includes('/players/editor/operations/')) {
      const key = decodeURIComponent(url.split('/operations/')[1].split('/')[0]);
      const result = operations.get(key);
      if (url.endsWith('/reconcile')) return {ok:true, json:async () => result};
      return {ok:true, json:async () => ({status:creationFailure === 'reconcile' ? 'recovery_required' : 'completed', result})};
    }
    if (failReload) { failReload = false; throw new TypeError('List reload unavailable'); }
    return {ok:true, json:async () => structuredClone(queue)};
  };
  const {creationFailure, reloadFailure = false} = options;
  let tree;
  await act(async () => { tree = create(React.createElement(Component, {clubId:'club', apiBase:'https://api.test'})); });
  await act(async () => tree.root.findAllByType('button').find(n => label(n).startsWith('Evening RR')).props.onClick());
  return {tree, calls, queue};
}
async function main() {
  automaticallyReviewed = false;
  const needsReview = await mount();
  assert.equal(button(needsReview.tree, 'Approve rated results').props.disabled, true, 'Rated publication waits for authoritative starting-rating review');
  assert.deepEqual(reviewPanels.get('doubles').playerIds, [1,2,3,4]);
  assert.deepEqual(reviewPanels.get('singles').playerIds, [1,2], 'Only actual singles participants need singles review');
  await act(async () => reviewPanels.get('doubles').onReady(true, []));
  assert.equal(button(needsReview.tree, 'Approve rated results').props.disabled, true, 'Mixed publication waits for both formats');
  await act(async () => reviewPanels.get('singles').onReady(true, []));
  assert.equal(button(needsReview.tree, 'Approve rated results').props.disabled, false);
  await act(async () => reviewPanels.get('doubles').onReady(false, []));
  assert.equal(button(needsReview.tree, 'Approve rated results').props.disabled, true, 'An invalidated review closes the gate again');
  await act(async () => {
    reviewPanels.get('doubles').onConfirmed([]);
    input(needsReview.tree, 'Player 1 doubles starting level').props.onChange({target:{value:'4.25'}});
  });
  assert.equal(input(needsReview.tree, 'Player 1 doubles starting level').props.value, '4.25', 'A review refresh preserves a starting-level edit made while the save completed');
  await act(async () => needsReview.tree.unmount());
  automaticallyReviewed = true;

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

  const missingProfile = () => { const q = structuredClone(seed); q.players = q.players.filter(p => p.id !== 4); return q; };
  async function openCreator(tree) {
    await act(async () => tree.root.findByProps({'aria-label':'Create new player for Player 4'}).props.onClick());
    assert.equal(input(tree, 'New player name for Player 4').props.value, 'Player 4');
    assert.equal(button(tree, 'Create and select player').props.disabled, true);
    await act(async () => input(tree, 'New player starting level for Player 4').props.onChange({target:{value:'7.5'}}));
    assert.equal(button(tree, 'Create and select player').props.disabled, true);
    await act(async () => input(tree, 'New player starting level for Player 4').props.onChange({target:{value:'3.75'}}));
  }
  async function submitCreator(tree) {
    await act(async () => tree.root.findByProps({'aria-label':'New club player for Player 4'}).props.onSubmit({preventDefault(){}}));
  }
  const creation = await mount(missingProfile());
  assert.equal(button(creation.tree, 'Approve rated results').props.disabled, true);
  await act(async () => input(creation.tree, 'Player 1 doubles starting level').props.onChange({target:{value:'4.5'}}));
  await openCreator(creation.tree);
  await submitCreator(creation.tree);
  assert.equal(input(creation.tree, 'Club profile for Player 4').props.value, 9);
  assert.equal(input(creation.tree, 'Player 4 doubles starting level').props.value, '3.75');
  assert.equal(input(creation.tree, 'Player 1 doubles starting level').props.value, '4.5', 'Other starting-level edits survive creation');
  assert.equal(button(creation.tree, 'Approve rated results').props.disabled, true, 'Other unsaved edits still block approval');
  const creationCall = creation.calls.find(c => c.url.endsWith('/players/editor/players'));
  assert.equal(JSON.parse(creationCall.options.body).starting_jupr, 3.75);
  assert.equal(creationCall.options.headers.Authorization, 'Bearer admin-token');
  assert.match(JSON.parse(creationCall.options.body).idempotency_key, /^generator-player:/);
  await act(async () => button(creation.tree, 'Save doubles starting level').props.onClick());
  assert.equal(button(creation.tree, 'Approve rated results').props.disabled, false);
  assert.equal(stored.size, 0, 'Completed creation is cleared from recovery storage');
  await act(async () => creation.tree.unmount());

  const duplicate = await mount(missingProfile());
  await openCreator(duplicate.tree);
  duplicate.queue.players.push({id:9, name:'Player 4', ratings:{doubles:rating(4.0), singles:rating(3.5, 0, false)}});
  await submitCreator(duplicate.tree);
  assert.equal(duplicate.calls.filter(c => c.options.method === 'POST').length, 0, 'Fresh duplicate preflight prevents creation');
  await act(async () => button(duplicate.tree, 'Use Player 4').props.onClick());
  assert.equal(input(duplicate.tree, 'Club profile for Player 4').props.value, 9);
  await act(async () => duplicate.tree.unmount());

  const interrupted = await mount(missingProfile(), {creationFailure:'lost'});
  await openCreator(interrupted.tree); await submitCreator(interrupted.tree);
  assert(button(interrupted.tree, 'Check player creation'));
  assert.equal(stored.size, 1, 'Interrupted creation survives refresh');
  const afterInterrupted = interrupted.queue;
  await act(async () => interrupted.tree.unmount());
  const resumed = await mount(afterInterrupted);
  assert.equal(button(resumed.tree, 'Approve rated results').props.disabled, true, 'A matched profile cannot bypass pending creation recovery');
  await submitCreator(resumed.tree);
  assert.equal(resumed.calls.filter(c => c.options.method === 'POST').length, 0, 'Completed operation recovery does not repeat the insert');
  assert.equal(button(resumed.tree, 'Approve rated results').props.disabled, false);
  assert.equal(resumed.queue.players.filter(p => p.name === 'Player 4').length, 1);
  await act(async () => resumed.tree.unmount());

  const reconcile = await mount(missingProfile(), {creationFailure:'reconcile'});
  await openCreator(reconcile.tree); await submitCreator(reconcile.tree); await submitCreator(reconcile.tree);
  assert.equal(reconcile.calls.filter(c => c.url.endsWith('/players/editor/players')).length, 1);
  assert(reconcile.calls.some(c => c.url.endsWith('/reconcile')));
  assert.equal(button(reconcile.tree, 'Approve rated results').props.disabled, false);
  await act(async () => reconcile.tree.unmount());

  const reload = await mount(missingProfile(), {reloadFailure:true});
  await openCreator(reload.tree); await submitCreator(reload.tree);
  assert(button(reload.tree, 'Use created player'));
  await submitCreator(reload.tree);
  assert.equal(reload.calls.filter(c => c.url.endsWith('/players/editor/players')).length, 1, 'A failed list refresh never creates a second profile');
  assert.equal(button(reload.tree, 'Approve rated results').props.disabled, false);
  await act(async () => reload.tree.unmount());
  console.log('PASS: inline creation, required starting level, preserved edits, duplicate prevention, refresh recovery, reconciliation, and single-insert retries');
  console.log('PASS: doubles starting levels, established ratings, singles isolation, validation, save-before-approve, and unrated results');
}
main().catch(error => {console.error(error); process.exitCode=1;});
