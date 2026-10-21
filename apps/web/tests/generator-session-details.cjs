const assert = require('node:assert/strict');
const React = require('react');
const {create, act} = require('react-test-renderer');
const Component = require('./helpers/public-controls-module.cjs')('@/components/GeneratorSessionDetails').default;
global.IS_REACT_ACT_ENVIRONMENT = true;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => text(node) === label);
const field = (tree, label) => tree.root.findByProps({'aria-label':label});
const baseline = {title:'MOneyball October 20', version:15, rating_mode:'unrated'};
const props = {apiBase:'https://api.test', clubId:'club', sessionKey:'session', session:baseline, ownerToken:'owner-secret', locked:false};

async function run() {
  let saved, tree;
  const calls = [];
  global.fetch = async (url, options) => {
    const body = JSON.parse(options.body); calls.push({url, options, body});
    return {ok:true, status:200, json:async () => ({session:{title:body.title, rating_mode:body.rating_mode, version:16}})};
  };
  await act(async () => {tree = create(React.createElement(Component, {...props, onSaved:next => {saved=next;}}));});
  await act(async () => button(tree, 'Edit session details').props.onClick());
  assert.equal(field(tree,'Session title').props.value, baseline.title);
  await act(async () => field(tree,'Session title').props.onChange({target:{value:' Moneyball   October 20 '}}));
  await act(async () => field(tree,'Session rating').props.onChange({target:{value:'rated'}}));
  await act(async () => field(tree,'Edit session details').props.onSubmit({preventDefault(){}}));
  assert.equal(saved.title, 'Moneyball October 20'); assert.equal(saved.rating_mode, 'rated');
  assert.equal(calls[0].body.edit_token, 'owner-secret'); assert.equal(calls[0].body.expected_version, 15);
  assert.equal(calls[0].options.method,'PATCH'); assert.ok(calls[0].body.idempotency_key);
  assert.equal(calls[0].url, 'https://api.test/clubs/club/play-generators/sessions/session/details');
  await act(async () => tree.unmount());

  // A background refresh must not silently authorize an old draft to overwrite
  // another organizer's newer settings.
  let error = 'This session changed. Refresh before editing its details.';
  global.fetch = async (url, options) => {calls.push({url, options, body:JSON.parse(options.body)}); return {ok:false,status:409,json:async()=>({detail:error})};};
  await act(async () => {tree=create(React.createElement(Component,{...props,ownerToken:undefined,adminToken:'admin',onSaved:()=>assert.fail('No save on conflict')}));});
  await act(async () => button(tree,'Edit session details').props.onClick());
  await act(async () => field(tree,'Session rating').props.onChange({target:{value:'rated'}}));
  await act(async () => tree.update(React.createElement(Component,{...props,session:{...baseline,version:16},ownerToken:undefined,adminToken:'admin',onSaved:()=>assert.fail()})));
  await act(async () => field(tree,'Edit session details').props.onSubmit({preventDefault(){}}));
  assert.equal(calls.at(-1).body.expected_version,15); assert.equal(calls.at(-1).options.headers.Authorization,'Bearer admin');
  assert.equal(calls.at(-1).body.edit_token,undefined); assert.ok(JSON.stringify(tree.toJSON()).includes(error));
  assert.equal(field(tree,'Session rating').props.value,'rated','The failed draft remains available');
  await act(async () => button(tree,'Cancel').props.onClick());
  await act(async () => tree.unmount());

  await act(async () => {tree=create(React.createElement(Component,{...props,scored:false,onSaved:()=>{}}));});
  await act(async () => button(tree,'Edit session details').props.onClick());
  assert.equal(tree.root.findByProps({value:'rated'}).props.disabled,true);
  await act(async () => field(tree,'Session title').props.onChange({target:{value:'  '}}));
  assert.equal(button(tree,'Save session details').props.disabled,true);
  await act(async () => tree.unmount());
  await act(async () => {tree=create(React.createElement(Component,{...props,locked:true,onSaved:()=>{}}));});
  assert.equal(button(tree,'Edit session details'),undefined);
  assert.ok(JSON.stringify(tree.toJSON()).includes('awaiting approval'));
  await act(async () => tree.unmount());
  await act(async () => {tree=create(React.createElement(Component,{...props,ownerToken:undefined,onSaved:()=>{}}));});
  assert.equal(tree.toJSON(),null,'Spectators receive no editing control');
  await act(async () => tree.unmount());
  console.log('PASS session details: owner/admin requests, rating edits, stale drafts, locks, validation');
}
run().catch(error => {console.error(error);process.exitCode=1;});
