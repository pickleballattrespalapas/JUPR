const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const code = ts.transpileModule(fs.readFileSync(path.join(__dirname,'../components/StartingRatingReview.tsx'),'utf8'), {compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
const moduleValue={exports:{}};
new Function('require','module','exports',code)(name=>name.endsWith('.module.css')?new Proxy({},{get:(_,key)=>String(key)}):require(name),moduleValue,moduleValue.exports);
const Component=moduleValue.exports.default;
const text=node=>typeof node==='string'?node:(node?.children||[]).map(text).join('');
const button=(tree,label)=>tree.root.findAllByType('button').find(node=>text(node)===label);
const seed=(id,extra={})=>({player_id:id,name:`Player ${id}`,rating_jupr:3,starting_jupr:3,rated_matches:0,has_rated_history:false,review_required:true,requested_rating:null,requested_jupr:null,requested_rating_differs:false,state_fingerprint:`fingerprint-${id}`,confirmed_at:null,can_edit_starting_rating:true,...extra});
const props={apiBase:'https://api.test',clubId:'club',accessToken:'token',playerIds:[1,2],requestedRatings:{'1':'3.5','2':'3.75–4.0'}};
(async()=>{
  let tree,ready=[],calls=[],rows=[seed(1,{requested_rating:'3.5',requested_jupr:3.5,requested_rating_differs:true}),seed(2,{rating_jupr:3.812,rated_matches:30,has_rated_history:true,review_required:false,requested_rating:'3.75–4.0'})];
  global.fetch=async(url,options)=>{
    const body=JSON.parse(options.body); calls.push({url,body,headers:options.headers});
    if(url.endsWith('/confirm')) rows=rows.map(row=>row.player_id===1?{...row,starting_jupr:body.reviews[0].rating_jupr,rating_jupr:body.reviews[0].rating_jupr,review_required:false,confirmed_at:'2026-10-30T10:00:00Z'}:row);
    return {ok:true,json:async()=>({players:structuredClone(rows)})};
  };
  await act(async()=>{tree=create(React.createElement(Component,{...props,onReady:(value)=>ready.push(value)}));});
  assert.equal(ready.at(-1),false);
  assert.equal(tree.root.findByProps({'aria-label':'Player 1 doubles starting rating'}).props.value,'3.5');
  assert.equal(tree.root.findAllByProps({'aria-label':'Player 2 doubles starting rating'}).length,0);
  assert.equal((text(tree.toJSON()).match(/requested level differs/g)||[]).length,1,'Only no-history mismatch is flagged');
  await act(async()=>tree.root.findByProps({'aria-label':'Player 1 doubles starting rating'}).props.onChange({target:{value:'3.25'}}));
  await act(async()=>button(tree,'Confirm starting rating (1)').props.onClick());
  const write=calls.find(call=>call.url.endsWith('/confirm'));
  assert.deepEqual(write.body,{match_format:'doubles',reviews:[{player_id:1,rating_jupr:3.25,expected_state_fingerprint:'fingerprint-1',requested_rating:'3.5'}]});
  assert.equal(write.headers.Authorization,'Bearer token');
  assert.equal(ready.at(-1),true);
  assert.equal(calls.filter(call=>call.url.endsWith('/preview')).length,2,'Readback verifies readiness after confirmation');
  await act(async()=>tree.unmount());

  // A range is never silently averaged or rounded into a new player's seed.
  rows=[seed(1,{requested_rating:'3.75–4.0',requested_rating_differs:true})];ready=[];
  await act(async()=>{tree=create(React.createElement(Component,{...props,playerIds:[1],onReady:value=>ready.push(value)}));});
  assert.equal(tree.root.findByProps({'aria-label':'Player 1 doubles starting rating'}).props.value,'');
  assert.equal(button(tree,'Confirm starting rating (1)').props.disabled,true);
  await act(async()=>tree.root.findByProps({'aria-label':'Player 1 doubles starting rating'}).props.onChange({target:{value:'7.1'}}));
  assert.equal(button(tree,'Confirm starting rating (1)').props.disabled,true);
  await act(async()=>tree.unmount());

  // An incomplete server preview must not enable rated play.
  rows=[];ready=[];
  await act(async()=>{tree=create(React.createElement(Component,{...props,playerIds:[1],onReady:value=>ready.push(value)}));});
  assert.equal(ready.at(-1),false);assert.match(text(tree.toJSON()),/every selected player/);
  await act(async()=>tree.unmount());

  // Late requests from a previous roster cannot turn the new selection ready.
  let resolveOld;ready=[];
  global.fetch=async(url,options)=>{
    const id=JSON.parse(options.body).player_ids[0];
    if(id===1) return new Promise(resolve=>{resolveOld=()=>resolve({ok:true,json:async()=>({players:[seed(1,{review_required:false})]})});});
    return {ok:true,json:async()=>({players:[seed(2)]})};
  };
  const onReady=value=>ready.push(value);
  await act(async()=>{tree=create(React.createElement(Component,{...props,playerIds:[1],onReady}));});
  await act(async()=>tree.update(React.createElement(Component,{...props,playerIds:[2],onReady})));
  await act(async()=>resolveOld());
  assert.equal(ready.at(-1),false);assert.equal(tree.root.findAllByProps({'aria-label':'Player 1 doubles starting rating'}).length,0);
  await act(async()=>tree.unmount());

  // An uncertain confirmation is recovered with a read, not an automatic repeat write.
  let committed=false;calls=[];ready=[];
  global.fetch=async(url,options)=>{
    calls.push({url});
    if(url.endsWith('/confirm')){committed=true;throw new TypeError('Connection lost');}
    return {ok:true,json:async()=>({players:[seed(1,{review_required:!committed,confirmed_at:committed?'2026-10-30T10:00:00Z':null})]})};
  };
  await act(async()=>{tree=create(React.createElement(Component,{...props,playerIds:[1],requestedRatings:{},onReady:value=>ready.push(value)}));});
  await act(async()=>button(tree,'Confirm starting rating (1)').props.onClick());
  assert.equal(ready.at(-1),false);assert.equal(button(tree,'Confirm starting rating (1)').props.disabled,true);
  await act(async()=>button(tree,'Recheck ratings').props.onClick());
  assert.equal(ready.at(-1),true);assert.equal(calls.filter(call=>call.url.endsWith('/confirm')).length,1);
  await act(async()=>tree.unmount());
  console.log('PASS Starting-rating review: explicit estimates, established history, ranges, exact fingerprints, authoritative readback, incomplete/stale responses, uncertain-write recovery.');
})().catch(error=>{console.error(error);process.exitCode=1;});
