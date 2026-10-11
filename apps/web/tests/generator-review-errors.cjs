const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const React = require('react');
const ts = require('typescript');
const {create, act} = require('react-test-renderer');

function load(relative, mocks = {}) {
  const source = fs.readFileSync(path.join(__dirname, '..', relative), 'utf8');
  const code = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2022, module:ts.ModuleKind.CommonJS, jsx:ts.JsxEmit.ReactJSX, esModuleInterop:true}}).outputText;
  const compiled = {exports:{}};
  new Function('require', 'module', 'exports', code)(name => name.endsWith('.module.css') ? {} : Object.hasOwn(mocks, name) ? mocks[name] : require(name), compiled, compiled.exports);
  return compiled.exports;
}

// Exercise the actual confirmation lifecycle and error normalization. Only the
// DOM portal is replaced, so a plain Error reproduces the original hidden detail.
const interaction = load('components/interaction/types.ts');
const lifecycle = load('components/interaction/useActionLifecycle.ts', {'./types':interaction});
const feedback = load('components/interaction/ActionFeedback.tsx');
const provider = load('components/interaction/InteractionProvider.tsx', {
  './useActionLifecycle':lifecycle,
  './ActionFeedback':feedback,
  './InteractionDialog':{InteractionDialog: ({open, title, children, actions}) => open ? React.createElement('div', {role:'dialog'}, title, children, actions) : null}
});
const confirm = load('components/ConfirmAction.tsx', {'./interaction/InteractionProvider':provider});
const Component = load('app/admin/play-generators/submissions/GeneratorSubmissions.tsx', {
  '@/lib/useAdminSession':{useAdminSession: () => ({accessToken:'admin-token'})},
  '@/components/ConfirmAction':confirm,
  '@/components/interaction':interaction,
  '@/components/SearchablePlayerSelect':({children, onValueChange, ...props}) => React.createElement('select', {...props, onChange:event => onValueChange(event.target.value)}, children),
  './GeneratorPlayerCreate':() => null
}).default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (scope, label) => scope.findAllByType('button').find(node => text(node) === label);
const reply = (payload, status = 200) => ({ok:status < 400, status, json:async () => structuredClone(payload)});
const seed = {
  submissions:[{id:'submission', session_key:'session', title:'Evening RR', version:3, status:'pending', organizer_name:'Joe', rating_mode:'rated', match_date:'2026-10-10', match_count:1,
    participants:[1,2,3,4].map(id => ({id:'p'+id, name:'Player '+id, player_id:id})),
    matches:[{id:'game', round_number:1, sideA:['p1','p2'], sideB:['p3','p4'], scoreA:11, scoreB:8}]}],
  players:[1,2,3,4].map(id => ({id, name:'Player '+id, ratings:{doubles:{rating_jupr:3.5, matches_played:12, can_set_starting_rating:false, state_fingerprint:'a'.repeat(64)}}}))
};
global.window = {location:{search:''}, requestAnimationFrame:callback => { callback(); return 1; }};
global.document = {querySelector:() => null};

async function mount(reviewResponse) {
  const queue = structuredClone(seed);
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({url, options});
    return url.endsWith('/review') ? reviewResponse(queue, calls) : reply(queue);
  };
  let tree;
  await act(async () => { tree = create(React.createElement(provider.InteractionProvider, null, React.createElement(Component, {clubId:'club', apiBase:'https://api.test'}))); });
  await select(tree);
  return {tree, queue, calls};
}
async function select(tree) {
  await act(async () => tree.root.findAllByType('button').find(node => text(node).startsWith('Evening RR')).props.onClick());
}
async function approve(tree) {
  await act(async () => button(tree.root, 'Approve rated results').props.onClick({currentTarget:null}));
  await act(async () => button(tree.root.findByProps({role:'dialog'}), 'Approve rated results').props.onClick());
}
async function visibleFailure(reviewResponse, expectedDetail, unexpectedDetail) {
  const {tree, calls} = await mount(reviewResponse);
  await approve(tree);
  const dialog = tree.root.findByProps({role:'dialog'});
  const alert = dialog.findByProps({role:'alert'});
  assert(text(alert).includes(expectedDetail), 'The actual confirmation lifecycle retains the useful failure detail');
  assert(text(alert).includes('refresh the submissions before trying again'));
  assert(text(alert).includes('Resume rated approval'));
  assert(tree.root.findAllByProps({role:'status'}).some(node => text(node).includes(expectedDetail)), 'The page retains the same actionable error after closing');
  assert(!text(alert).includes('Your changes are still here'), 'The generic interaction fallback must not replace this error');
  if (unexpectedDetail) assert(!text(tree.root).includes(unexpectedDetail), 'Unexpected internal errors must not reach either visible message');
  assert.equal(calls.filter(call => call.url.endsWith('/review')).length, 1, 'Presenting a failure never repeats the write');
  await act(async () => tree.unmount());
}

async function interruptedApprovalCanResume() {
  let reviews = 0;
  const storedRequest = {expected_version:3, action:'approve', player_ids:{p1:1,p2:2,p3:3,p4:4}, match_date:'2026-10-09', reason:'Saved review note'};
  const {tree, calls} = await mount(queue => {
    reviews += 1;
    if (reviews === 1) {
      Object.assign(queue.submissions[0], {status:'processing', version:4, review:{request:storedRequest}});
      return reply({detail:'Unexpected database failure'}, 500);
    }
    queue.submissions = [];
    return reply({status:'approved'});
  });
  await approve(tree);
  await act(async () => button(tree.root.findByProps({role:'dialog'}), 'No, go back').props.onClick());
  await act(async () => button(tree.root, 'Refresh').props.onClick());
  await select(tree);
  assert(text(tree.root).includes('Approval interrupted — resume review'));
  assert.equal(button(tree.root, 'Approve rated results'), undefined, 'An in-progress approval cannot become a fresh approval');
  assert.equal(button(tree.root, 'Reject'), undefined, 'An in-progress approval cannot be rejected');
  await act(async () => button(tree.root, 'Resume rated approval').props.onClick());
  const requests = calls.filter(call => call.url.endsWith('/review'));
  assert.equal(requests.length, 2, 'Refresh is read-only and resuming sends only one request');
  assert.deepEqual(JSON.parse(requests[1].options.body), {...storedRequest, expected_version:4}, 'Resume uses the saved mapping, date, action and reason with the refreshed version');
  assert.equal(requests[1].options.headers.Authorization, 'Bearer admin-token');
  assert(text(tree.root).includes('Approved as rated.'));
  await act(async () => tree.unmount());
}

(async () => {
  await visibleFailure(() => reply({detail:'Choose a club profile for every player before approving.'}, 400), 'Choose a club profile for every player before approving.');
  await visibleFailure(() => reply({detail:'The player rating changed. Refresh before continuing.'}, 409), 'The player rating changed. Refresh before continuing.');
  const fallback = 'We couldn’t confirm whether the approval finished.';
  await visibleFailure(() => { throw new TypeError('Private transport diagnostic'); }, fallback, 'Private transport diagnostic');
  await visibleFailure(() => reply({detail:'Private database diagnostic'}, 500), fallback, 'Private database diagnostic');
  await visibleFailure(() => reply({detail:'Private recovery diagnostic'}, 503), fallback, 'Private recovery diagnostic');
  await visibleFailure(() => ({ok:false, status:500, json:async () => { throw new SyntaxError('Private response HTML'); }}), fallback, 'Private response HTML');
  await visibleFailure(() => ({ok:true, status:200, json:async () => { throw new SyntaxError('Private response HTML'); }}), fallback, 'Private response HTML');
  await visibleFailure(() => reply({detail:'Private unexpected response'}), fallback, 'Private unexpected response');
  await interruptedApprovalCanResume();
  console.log('PASS: real confirmation error normalization preserves 400/409 details, hides unexpected diagnostics, guides refresh, and resumes an interrupted approval with its saved request');
})().catch(error => {console.error(error); process.exitCode=1;});
