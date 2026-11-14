const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),ts=require('typescript');
const React=require('react'),{create,act}=require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT=true;
function load(file,mocks={}) {
 const output=ts.transpileModule(fs.readFileSync(path.join(__dirname,'..',file),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
 const module={exports:{}};new Function('require','module','exports',output)(n=>Object.hasOwn(mocks,n)?mocks[n]:require(n),module,module.exports);return module.exports;
}
const helpers=load('lib/managedInterclub.ts'),registration=load('lib/interclubRegistration.ts');
const competition=load('lib/interclubCompetition.ts',{'./interclubRegistration':registration});
const Editor=()=>null;
const Board=load('components/interclub/ManagedScoreboard.tsx',{'@/lib/interclubCompetition':competition,'@/lib/managedInterclub':helpers,
 '@/app/admin/interclub/competition/CourtSchedule':()=>null,'@/app/admin/interclub/competition/PrintPacket':()=>null,
 '@/app/admin/interclub/competition/ScoreEditor':Editor,'@/app/admin/interclub/competition/competition.module.css':{}}).default;
const timers=new Map();let counter=0;
global.window={addEventListener(){},removeEventListener(){},print(){},setTimeout(fn){timers.set(++counter,fn);return counter;},clearTimeout(id){timers.delete(id);}};
const game={id:'g1',status:'pending',a:null,b:null,winner:null,players_a:[],players_b:[],played_at:null,injury_reason:null};
const document={schema_version:1,meet_id:'m1',phase:'regular',format:'gender',schedule_mode:'staggered',weather:'normal',encounters:[{id:'e1',club_a:'a',club_b:'b',division:'3.5',rotation:1,tiebreak:null,pairings:[{id:'p1',kind:'women',court:1,players_a:['a1','a2'],players_b:['b1','b2'],games:[game]}]}]};
const meet={id:'m1',season_id:'s1',revision:1,state:'draft',document,details:{starts_at:'2099-01-01T17:00:00Z',teams:[],club_ids:['a','b'],host_club_id:'a',courts:2,phase:'regular',format:'gender'}};
const copy=x=>structuredClone(x);
const nodeText=n=>typeof n==='string'?n:n.children.map(nodeText).join('');
const button=(tree,label)=>tree.root.findAllByType('button').find(n=>nodeText(n)===label);
async function change(tree,a,b=null) {
 await act(async()=>{const draft=copy(tree.root.findByType(Editor).props.document);draft.encounters[0].pairings[0].games[0]={...game,a,b};tree.root.findByType(Editor).props.onChange(draft);});
}
async function tick(){await act(async()=>{const callbacks=[...timers.values()];timers.clear();for(const fn of callbacks)fn();});}
(async()=>{
 const club={rosters:[{division:'3.5',roster:[{entry_id:'kept',name:'Ana',gender:'female',starting_rating:3.6}]}]};
 assert.equal(helpers.readRoster('Ana | 3.7','',club)[0].entry_id,'kept');
 assert.equal(helpers.readRoster('Ana','',club)[0].starting_rating,null);
 assert.throws(()=>helpers.readRoster('Ana\nana','',club),/twice/);
 assert.throws(()=>helpers.readRoster('Ana | NaN','',club),/rating/);
 let finish,requests=[],dirty=[],tree;
 await act(async()=>{tree=create(React.createElement(Board,{meet:copy(meet),seasonName:'BCS',timezone:'UTC',clubName:x=>x,onDirtyChange:x=>dirty.push(x),
  onWrite:async(action,revision,draft)=>{requests.push({action,revision,draft});return new Promise(resolve=>{finish=resolve;});}}));});
 await change(tree,11);await tick();assert.equal(requests.length,1);
 await change(tree,11,7);
 await act(async()=>finish({...meet,revision:2,document:requests[0].draft}));
 assert.equal(tree.root.findByType(Editor).props.document.encounters[0].pairings[0].games[0].b,7,'Save response preserves newer typing');
 await tick();assert.equal(requests[1].revision,2,'Next save uses acknowledged revision');
 await act(async()=>finish({...meet,revision:3,document:requests[1].draft}));
 assert.equal(dirty.at(-1),false);assert.equal(button(tree,'Submit completed meet').props.disabled,false);
 await act(async()=>tree.unmount());
 // Revocation/conflict must stop auto retries and preserve the unsaved sheet.
 for(const status of [403,404,409]) {
  let attempts=0;
  await act(async()=>{tree=create(React.createElement(Board,{meet:copy(meet),seasonName:'BCS',timezone:'UTC',clubName:x=>x,
   onWrite:async()=>{attempts++;throw Object.assign(new Error('Reload or request a new link'),{status});}}));});
  await change(tree,11,7);await tick();await tick();assert.equal(attempts,1);
  assert.equal(tree.root.findByType(Editor).props.disabled,true);assert.equal(button(tree,'Save scores now').props.disabled,true);
  assert.ok(button(tree,'Download unsaved scores'));assert.equal(tree.root.findByType(Editor).props.document.encounters[0].pairings[0].games[0].b,7);
  await act(async()=>tree.unmount());
 }
 // Transient outages allow an explicit retry of the retained draft.
 let attempts=0;
 await act(async()=>{tree=create(React.createElement(Board,{meet:copy(meet),seasonName:'BCS',timezone:'UTC',clubName:x=>x,
  onWrite:async(_action,_revision,draft)=>{if(++attempts===1)throw new Error('Offline');return {...meet,revision:2,document:draft};}}));});
 await change(tree,11,7);await tick();assert.equal(tree.root.findByType(Editor).props.disabled,false);
 await act(async()=>button(tree,'Save scores now').props.onClick());assert.equal(attempts,2);
 assert.equal(button(tree,'Submit completed meet').props.disabled,false);await act(async()=>tree.unmount());
 console.log('Managed interclub roster identity, concurrent typing, conflicts, revoked links and outage recovery passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
