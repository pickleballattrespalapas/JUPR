const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const root = path.resolve(__dirname, '..');
const h = React.createElement;
const reviews = new Map();
let imported = {playerIds:[2,3,4,5],requestedRatings:{'2':'3.5','3':'4.0'}};
const directory = Array.from({length:5},(_,index)=>({id:index+1,name:`Player ${index+1}`}));
const mocks = {
  '@/components/StartingRatingReview': props => { reviews.set(props.matchFormat,props); return h('button',{onClick:()=>props.onReady(true,[])},`Confirm ${props.matchFormat}`); },
  '@/components/CourtReserveRosterImport': props => h('button',{onClick:()=>props.onImport(imported)},'Use test import'),
  '@/app/admin/play-generators/submissions/GeneratorPlayerCreate': ()=>null,
  '@/lib/playerSearch': {fetchAllPublicPlayers:async()=>directory,matchesPlayerSearch:(name,q)=>name.includes(q),playerSearchKey:name=>name.toLowerCase()},
};
const cache = new Map();
function load(name) {
  if (mocks[name]) return mocks[name];
  if (!name.startsWith('@/')) return require(name);
  if (cache.has(name)) return cache.get(name);
  const file = path.join(root,name.slice(2)+'.tsx');
  const code = ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
  const module = {exports:{}};
  new Function('require','module','exports',code)(load,module,module.exports);
  cache.set(name,module.exports); return module.exports;
}
const Review = load('@/components/GeneratorRatingReview').default;
const Setup = load('@/components/GeneratorRosterSetup').default;
const label = n => typeof n === 'string' ? n : (n?.children||[]).map(label).join('');
const button = (tree,text)=>tree.root.findAllByType('button').find(n=>label(n)===text);

(async()=>{
  let tree; const ready=[];
  const props={apiBase:'https://api.test',clubId:'club',accessToken:'staff',playerIds:[1,2],playFormat:'doubles_singles',onReady:value=>ready.push(value)};
  await act(async()=>{tree=create(h(Review,props));});
  assert.equal(ready.at(-1),false);
  await act(async()=>button(tree,'Confirm doubles').props.onClick());
  assert.equal(ready.at(-1),false,'Mixed session still requires its independent singles review');
  await act(async()=>button(tree,'Confirm singles').props.onClick());
  assert.equal(ready.at(-1),true);
  await act(async()=>tree.update(h(Review,{...props,playerIds:[1,3]})));
  assert.equal(ready.at(-1),false,'A changed roster cannot reuse the previous confirmation');
  await act(async()=>button(tree,'Confirm doubles').props.onClick());
  await act(async()=>button(tree,'Confirm singles').props.onClick());
  await act(async()=>tree.update(h(Review,{...props,playerIds:[1,3],requestedRatings:{'3':'4.0'}})));
  assert.equal(ready.at(-1),false,'A new CourtReserve request is reviewed in its own context');
  await act(async()=>tree.unmount());

  let names='Player 1',links={'player 1':1},count=4,invalidated=0;const setupReady=[];
  function Wrapper(){
    const [text,setText]=React.useState(names),[ids,setIds]=React.useState(links),[target,setTarget]=React.useState(count);
    return h(Setup,{apiBase:'https://api.test',clubKey:'club',creationClubId:'club',ratingReviewAccessToken:'staff',ratedSetup:true,
      generatorKind:'round_robin',playFormat:'doubles',targetCount:target,participantText:text,linkedPlayerIds:ids,
      doublesCourtCount:1,singlesCourtCount:0,onTargetCountChange:value=>{count=value;setTarget(value)},
      onParticipantTextChange:value=>{names=value;setText(value)},onLinkedPlayerIdsChange:value=>{links=value;setIds(value)},
      onDoublesCourtCountChange:()=>{},onSinglesCourtCountChange:()=>{},onInvalidate:()=>invalidated++,onRatingReviewReadyChange:value=>setupReady.push(value)});
  }
  await act(async()=>{tree=create(h(Wrapper));});
  await act(async()=>button(tree,'Import CourtReserve roster').props.onClick());
  await act(async()=>button(tree,'Use test import').props.onClick());
  assert.equal(names,'Player 2\nPlayer 3\nPlayer 4\nPlayer 5');
  assert.deepEqual(links,{'player 2':2,'player 3':3,'player 4':4,'player 5':5});
  assert.equal(count,4);
  assert.deepEqual(reviews.get('doubles').requestedRatings,imported.requestedRatings,'Raw requested levels reach review rather than replacing calculated ratings');
  assert.equal(setupReady.at(-1),false);
  assert(invalidated>0,'Import invalidates prior matchup/expectation previews');
  await act(async()=>button(tree,'Confirm doubles').props.onClick());
  assert.equal(setupReady.at(-1),true);
  await act(async()=>tree.unmount());
  console.log('PASS generator review: mixed formats, roster/request invalidation, CourtReserve profile selection, requested levels and preview invalidation.');
})().catch(error=>{console.error(error);process.exitCode=1;});
