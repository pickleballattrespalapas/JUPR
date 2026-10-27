const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const React = require('react');
const ts = require('typescript');
const {create, act} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const mocks = {
  '@/lib/useAdminSession': {useAdminSession: () => ({accessToken:'admin-token'})},
  '@/components/ConfirmAction': {ConfirmAction: props => React.createElement('button', {disabled:props.disabled, onClick:props.onConfirm}, props.triggerLabel)},
  '@/components/SearchablePlayerSelect': ({children, onValueChange, ...props}) => React.createElement('select', {...props, onChange:e => onValueChange(e.target.value)}, children),
  './GeneratorPlayerCreate': () => null
};
function compile(relative) {
  const source = fs.readFileSync(path.join(__dirname, '..', relative), 'utf8');
  const code = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2022, module:ts.ModuleKind.CommonJS, jsx:ts.JsxEmit.ReactJSX, esModuleInterop:true}}).outputText;
  const compiled = {exports:{}};
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), compiled, compiled.exports);
  return compiled.exports;
}
mocks['@/components/interaction'] = compile('components/interaction/types.ts');
const Component = compile('app/admin/play-generators/submissions/GeneratorSubmissions.tsx').default;
const label = node => typeof node === 'string' ? node : (node?.children || []).map(label).join('');
const button = (tree, text) => tree.root.findAllByType('button').find(node => label(node) === text);
const input = (tree, name) => tree.root.findAll(node => typeof node.type === 'string' && node.props['aria-label'] === name)[0];
const reply = (body, status = 200) => ({ok:status < 400, status, json:async () => structuredClone(body)});
const seed = {
  submissions:[{id:'review', session_key:'session', title:'Evening RR', version:3, status:'pending', organizer_name:'Joe', rating_mode:'unrated', match_date:'2026-10-10', match_count:2,
    participants:[1,2,3,4].map(id => ({id:'p'+id, name:'Player '+id, player_id:id})),
    matches:[
      {id:'game1', round_number:1, sideA:['p1','p2'], sideB:['p3','p4'], scoreA:11, scoreB:8},
      {id:'game2', round_number:2, sideA:['p1','p3'], sideB:['p2','p4'], scoreA:11, scoreB:9}]}],
  players:[1,2,3,4,5].map(id => ({id, name:'Player '+id}))
};
global.window = {location:{search:''}};
async function mount(queue = structuredClone(seed), saveHandler) {
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({url, options});
    if (url.endsWith('/submission/scores')) {
      if (saveHandler) return saveHandler();
      const body = JSON.parse(options.body);
      const item = queue.submissions[0];
      for (const score of body.scores) {
        const match = item.matches.find(m => m.id === score.match_id && m.round_number === score.round_number);
        Object.assign(match, {scoreA:score.score_a, scoreB:score.score_b});
      }
      item.version += 1;
      return reply({ok:true, submission:item, corrected_games:body.scores.length});
    }
    if (url.endsWith('/review')) return reply({status:'approved'});
    return reply(queue);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Component, {clubId:'club', apiBase:'https://api.test'})); });
  await act(async () => tree.root.findAllByType('button').find(node => label(node).startsWith('Evening RR')).props.onClick());
  return {tree, calls};
}
const change = async (tree, name, value) => act(async () => input(tree, name).props.onChange({target:{value}}));
const click = async (tree, text) => act(async () => button(tree, text).props.onClick());
const scoreName = 'Round 1, game game1, Player 1 & Player 2 score';

async function correctThenApprove() {
  const {tree, calls} = await mount();
  await change(tree, 'Club profile for Player 4', '5');
  await act(async () => tree.root.findByProps({type:'date'}).props.onChange({target:{value:'2026-10-09'}}));
  await act(async () => tree.root.findByType('textarea').props.onChange({target:{value:'Review note'}}));
  await click(tree, 'Edit scores');
  assert(button(tree, 'Approve unrated results').props.disabled);
  assert(button(tree, 'Reject').props.disabled);
  assert(button(tree, 'Edit session details').props.disabled);
  assert(button(tree, 'Save score corrections').props.disabled);
  await change(tree, scoreName, '6');
  await change(tree, 'Filter scores by round', '2');
  assert.equal(input(tree, scoreName), undefined);
  assert(label(tree.root).includes('1 game changed across all rounds'));
  await change(tree, 'Round 2, game game2, Player 1 & Player 3 score', '4');
  await change(tree, 'Find a player in saved scores', 'does not exist');
  assert(label(tree.root).includes('No games match these filters.'));
  await click(tree, 'Save score corrections');
  const patch = calls.find(call => call.url.endsWith('/submission/scores'));
  assert.equal(patch.options.method, 'PATCH');
  assert.equal(patch.options.headers.Authorization, 'Bearer admin-token');
  assert.deepEqual(JSON.parse(patch.options.body), {expected_version:3, scores:[
    {round_number:1, match_id:'game1', score_a:6, score_b:8},
    {round_number:2, match_id:'game2', score_a:4, score_b:9}]});
  assert.equal(input(tree, 'Club profile for Player 4').props.value, 5);
  assert.equal(tree.root.findByProps({type:'date'}).props.value, '2026-10-09');
  assert.equal(tree.root.findByType('textarea').props.value, 'Review note');
  assert(label(tree.root).includes('Score corrections saved. This session is still awaiting approval.'));
  await change(tree, 'Find a player in saved scores', 'player 1');
  await change(tree, 'Filter scores by round', '');
  assert(label(tree.root).includes('6 – 8'));
  await click(tree, 'Approve unrated results');
  assert.deepEqual(JSON.parse(calls.find(call => call.url.endsWith('/review')).options.body), {
    expected_version:4, action:'approve', player_ids:{p1:1,p2:2,p3:3,p4:5}, match_date:'2026-10-09', reason:'Review note'});
  await act(async () => tree.unmount());
}
async function invalidAndCanceledEdits() {
  const {tree, calls} = await mount();
  await click(tree, 'Edit scores');
  for (const bad of ['', '-1', '100', '1.5', '8']) {
    await change(tree, scoreName, bad);
    assert(button(tree, 'Save score corrections').props.disabled, `Reject invalid score ${bad}`);
    await click(tree, 'Save score corrections');
  }
  assert(!calls.some(call => call.options.method === 'PATCH'));
  await change(tree, scoreName, '0');
  assert.equal(button(tree, 'Save score corrections').props.disabled, false, 'Zero is a valid score');
  await click(tree, 'Cancel score edits');
  assert.equal(button(tree, 'Approve unrated results').props.disabled, false);
  await click(tree, 'Edit scores');
  assert.equal(input(tree, scoreName).props.value, '11');
  await act(async () => tree.unmount());
}
async function failuresKeepDrafts() {
  for (const fail of [() => reply({detail:'This submission changed. Refresh before correcting scores.'}, 409), () => {throw new TypeError('Network unavailable');}]) {
    const {tree} = await mount(structuredClone(seed), fail);
    await click(tree, 'Edit scores');
    await change(tree, scoreName, '5');
    await click(tree, 'Save score corrections');
    assert.equal(input(tree, scoreName).props.value, '5');
    assert(label(tree.root).includes('Your edits are still here. Refresh to check the saved scores'));
    assert(button(tree, 'Approve unrated results').props.disabled);
    await act(async () => tree.unmount());
  }
}
async function lockedAndSwitchedSubmissions() {
  for (const status of ['approved','rejected','processing']) {
    const queue = structuredClone(seed); queue.submissions[0].status = status;
    const {tree} = await mount(queue);
    assert.equal(button(tree, 'Edit scores'), undefined);
    await act(async () => tree.unmount());
  }
  const queue = structuredClone(seed);
  queue.submissions.push({...structuredClone(seed.submissions[0]), id:'other', session_key:'other', title:'Morning RR'});
  const {tree} = await mount(queue);
  await click(tree, 'Edit scores');
  await change(tree, scoreName, '5');
  await act(async () => tree.root.findAllByType('button').find(node => label(node).startsWith('Morning RR')).props.onClick());
  assert.equal(button(tree, 'Save score corrections'), undefined);
  await click(tree, 'Edit scores');
  assert.equal(input(tree, scoreName).props.value, '11');
  await act(async () => tree.unmount());
}
async function oldWorkspaceResponseIsIgnored() {
  let finish;
  const {tree} = await mount(structuredClone(seed), () => new Promise(resolve => {finish = resolve;}));
  await click(tree, 'Edit scores');
  await change(tree, scoreName, '5');
  await click(tree, 'Save score corrections');
  await act(async () => tree.update(React.createElement(Component, {clubId:'other', apiBase:'https://api.test'})));
  await act(async () => finish(reply({submission:{...seed.submissions[0], title:'STALE SAVE'}})));
  assert(!label(tree.root).includes('STALE SAVE'));
  assert(!label(tree.root).includes('Score corrections saved.'));
  await act(async () => tree.unmount());
}
(async () => {
  await correctThenApprove(); await invalidAndCanceledEdits(); await failuresKeepDrafts();
  await lockedAndSwitchedSubmissions(); await oldWorkspaceResponseIsIgnored();
  console.log('PASS submission score edits: filters, bulk save, preserved mappings/date, current approval version, validation, cancel, failures, sealed states, and workspace isolation');
})().catch(error => {console.error(error); process.exitCode=1;});
