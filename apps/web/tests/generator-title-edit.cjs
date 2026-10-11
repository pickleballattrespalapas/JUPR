const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const React = require('react');
const ts = require('typescript');
const {create, act} = require('react-test-renderer');

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
const rating = level => ({rating_jupr:level, matches_played:0, can_set_starting_rating:true, state_fingerprint:'a'.repeat(64)});
const seed = {
  submissions:[{id:'review', session_key:'session', title:'Evenig RR', version:3, status:'pending', organizer_name:'Joe', rating_mode:'rated', match_date:'2026-10-10', match_count:1,
    participants:[1,2,3,4].map(id => ({id:'p'+id, name:'Player '+id, player_id:id})),
    matches:[{id:'game', round_number:1, sideA:['p1','p2'], sideB:['p3','p4'], scoreA:11, scoreB:8}]}],
  players:[1,2,3,4,5].map(id => ({id, name:'Player '+id, ratings:{doubles:rating(3.5)}}))
};
global.window = {location:{search:''}};

async function mount(queue = structuredClone(seed), saveError) {
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({url, options});
    if (url.endsWith('/submission/title')) {
      if (saveError) return reply({detail:saveError}, 409);
      const body = JSON.parse(options.body);
      const item = queue.submissions.find(item => url.includes('/sessions/' + item.session_key + '/'));
      item.title = body.title;
      item.version += 1;
      return reply({ok:true, title:item.title, version:item.version});
    }
    if (url.includes('/starting-ratings/')) {
      const body = JSON.parse(options.body);
      const player = queue.players.find(item => item.id === Number(url.split('/').at(-1)));
      player.ratings[body.match_format] = rating(body.rating_jupr);
      return reply({ok:true, player});
    }
    if (url.endsWith('/review')) return reply({status:'approved'});
    return reply(queue);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Component, {clubId:'club', apiBase:'https://api.test'})); });
  await select(tree, queue.submissions[0].title);
  return {tree, calls};
}
async function select(tree, title) {
  await act(async () => tree.root.findAllByType('button').find(node => label(node).startsWith(title)).props.onClick());
}
async function edit(tree, value) {
  if (!button(tree, 'Save title')) await act(async () => button(tree, 'Edit title').props.onClick());
  await act(async () => input(tree, 'Session title').props.onChange({target:{value}}));
}
async function save(tree) {
  await act(async () => tree.root.findByProps({'aria-label':'Edit session title'}).props.onSubmit({preventDefault(){}}));
}

async function preservesReviewDraftsAndUsesNewVersion() {
  const {tree, calls} = await mount();
  await act(async () => input(tree, 'Club profile for Player 4').props.onChange({target:{value:'5'}}));
  await act(async () => tree.root.findByProps({type:'date'}).props.onChange({target:{value:'2026-10-09'}}));
  await act(async () => tree.root.findByType('textarea').props.onChange({target:{value:'Review note'}}));
  await act(async () => input(tree, 'Player 1 doubles starting level').props.onChange({target:{value:'4.25'}}));
  await edit(tree, '  Evening   RR  ');
  assert.equal(button(tree, 'Approve rated results').props.disabled, true, 'Approval waits for the title editor to finish');
  assert.equal(button(tree, 'Reject').props.disabled, true, 'Rejecting cannot race with a title edit');
  await save(tree);
  assert.equal(label(tree.root.findByType('h2')), 'Evening RR · Rated');
  assert(tree.root.findAllByType('button').some(node => label(node).startsWith('Evening RR2026-10-10')), 'The queue card reflects the saved title');
  assert.equal(button(tree, 'Save title'), undefined);
  assert.equal(input(tree, 'Club profile for Player 4').props.value, 5, 'Profile mapping survives title save');
  assert.equal(tree.root.findByProps({type:'date'}).props.value, '2026-10-09', 'Date draft survives title save');
  assert.equal(tree.root.findByType('textarea').props.value, 'Review note', 'Review note survives title save');
  assert.equal(input(tree, 'Player 1 doubles starting level').props.value, '4.25', 'Starting-level draft survives title save');
  assert.equal(button(tree, 'Approve rated results').props.disabled, true, 'Unsaved levels still block approval after title save');
  const patch = calls.find(call => call.url.endsWith('/submission/title'));
  assert.equal(patch.options.method, 'PATCH');
  assert.equal(patch.options.headers.Authorization, 'Bearer admin-token');
  assert.deepEqual(JSON.parse(patch.options.body), {expected_version:3, title:'Evening RR'});
  await act(async () => button(tree, 'Save doubles starting level').props.onClick());
  assert.equal(button(tree, 'Approve rated results').props.disabled, false);
  await act(async () => button(tree, 'Approve rated results').props.onClick());
  const review = calls.find(call => call.url.endsWith('/review'));
  assert.deepEqual(JSON.parse(review.options.body), {
    expected_version:4, action:'approve', player_ids:{p1:1,p2:2,p3:3,p4:5}, match_date:'2026-10-09', reason:'Review note'
  }, 'Approval uses the new version and the preserved drafts');
  await act(async () => tree.unmount());
}

async function validatesAndCancelsDrafts() {
  const {tree, calls} = await mount();
  await act(async () => button(tree, 'Edit title').props.onClick());
  assert.equal(input(tree, 'Session title').props.value, 'Evenig RR');
  assert.equal(input(tree, 'Session title').props.maxLength, 160);
  assert.equal(button(tree, 'Save title').props.disabled, true, 'An unchanged title cannot be saved');
  for (const value of ['   ', ' Evenig  RR ', 'x'.repeat(161)]) {
    await edit(tree, value);
    assert.equal(button(tree, 'Save title').props.disabled, true);
    await save(tree);
  }
  assert.equal(calls.filter(call => call.options.method === 'PATCH').length, 0);
  await edit(tree, 'Discarded change');
  await act(async () => button(tree, 'Cancel').props.onClick());
  assert.equal(label(tree.root.findByType('h2')), 'Evenig RR · Rated');
  assert.equal(button(tree, 'Approve rated results').props.disabled, false);
  await act(async () => button(tree, 'Edit title').props.onClick());
  assert.equal(input(tree, 'Session title').props.value, 'Evenig RR', 'Cancel discards the title draft');
  await act(async () => tree.unmount());
}

async function failedSavePreservesInput() {
  const {tree, calls} = await mount(structuredClone(seed), 'This session changed. Refresh before saving.');
  await edit(tree, 'Evening RR');
  await save(tree);
  assert.equal(input(tree, 'Session title').props.value, 'Evening RR');
  assert.equal(label(tree.root.findByType('h2')), 'Evenig RR · Rated', 'Failed saves do not claim the new title');
  assert(label(tree.root).includes('This session changed. Refresh before saving.'));
  assert.equal(button(tree, 'Save title').props.disabled, false, 'A failed save remains retryable');
  assert.equal(button(tree, 'Approve rated results').props.disabled, true);
  assert.equal(calls.filter(call => call.options.method === 'PATCH').length, 1);
  await act(async () => tree.unmount());
}

async function switchingAndFinalizedSubmissions() {
  const queue = structuredClone(seed);
  queue.submissions.push({...structuredClone(seed.submissions[0]), id:'review-2', session_key:'session-2', title:'Morning RR'});
  const {tree} = await mount(queue);
  await edit(tree, 'Never saved');
  await select(tree, 'Morning RR');
  assert.equal(button(tree, 'Save title'), undefined, 'Switching submissions closes the previous editor');
  await act(async () => button(tree, 'Edit title').props.onClick());
  assert.equal(input(tree, 'Session title').props.value, 'Morning RR');
  await select(tree, 'Evenig RR');
  await act(async () => button(tree, 'Edit title').props.onClick());
  assert.equal(input(tree, 'Session title').props.value, 'Evenig RR', 'Unsaved title does not follow the next selection');
  await act(async () => tree.unmount());
  for (const status of ['approved', 'rejected', 'processing']) {
    const queue = structuredClone(seed);
    queue.submissions[0].status = status;
    const {tree} = await mount(queue);
    assert.equal(button(tree, 'Edit title'), undefined, status + ' submissions cannot be renamed');
    assert.equal(tree.root.findAllByProps({'aria-label':'Session title'}).length, 0);
    await act(async () => tree.unmount());
  }
}

(async () => {
  await preservesReviewDraftsAndUsesNewVersion();
  await validatesAndCancelsDrafts();
  await failedSavePreservesInput();
  await switchingAndFinalizedSubmissions();
  console.log('PASS: pending submission title edits, authenticated versioned save, preserved review drafts, validation, retry, cancel, and locked review states');
})().catch(error => {console.error(error); process.exitCode=1;});
