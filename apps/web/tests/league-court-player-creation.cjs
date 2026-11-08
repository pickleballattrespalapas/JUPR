const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const {create, act} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
if (!global.crypto) global.crypto = require('node:crypto').webcrypto;
const storage = new Map(), local = new Map();
global.localStorage = {getItem:key=>local.get(key)||null,setItem:(key,value)=>local.set(key,value),removeItem:key=>local.delete(key)};
global.sessionStorage = {getItem:key => storage.get(key) || null, setItem:(key,value) => storage.set(key,value), removeItem:key => storage.delete(key)};
global.addEventListener = () => {};
global.removeEventListener = () => {};
const root = path.join(__dirname, '..');
const cache = new Map();
const players = ['Katri Ulmonen','John Schaeffer','Carrie Brand','Gail Shea','Rosie Watson','Richard Harley','Tricia Newbanks','Existing Guest'].map((name,i)=>({id:i+1,name,rating:1320,rating_jupr:3.3}));
const Review = props => React.createElement('button', {onClick:()=>props.onReady(true)}, 'Confirm test starting estimates');
const dnd = {
  DragDropContext: ({children})=>React.createElement('div',null,children),
  Droppable: ({children})=>children({innerRef(){},droppableProps:{},placeholder:null},{}),
  Draggable: ({children})=>children({innerRef(){},draggableProps:{style:{}},dragHandleProps:{}},{})
};
const mocks = {
  '@hello-pangea/dnd': dnd,
  'next/link': {__esModule:true,default:({children,...props}) => React.createElement('a',props,children)},
  'next/dynamic': {__esModule:true,default:factory => String(factory).includes('StartingRatingReview') ? Review : String(factory).includes('LeagueCourtRosterEditor') ? compile(path.join(root,'app/admin/league-manager/live/LeagueCourtRosterEditor.tsx')).LeagueCourtRosterEditor : () => null},
  '@/lib/useAdminSession': {useAdminSession:() => ({accessToken:'token',session:{user:{id:'test-operator'}}}),adminSessionLabel:() => 'Admin'},
  '@/components/ConfirmAction': {ConfirmAction:props => React.createElement('button',{disabled:props.disabled,onClick:() => props.onConfirm(props.confirmationText)},props.triggerLabel)},
  '@/components/SearchablePlayerSelect': {__esModule:true,default:props => React.createElement('select',props)},
};
function compile(file) {
  if (cache.has(file)) return cache.get(file);
  const mod={exports:{}};
  const code=ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
  const localRequire = name => {
    if (mocks[name]) return mocks[name];
    if (name.endsWith('.css')) return {};
    if (name.startsWith('@/') || name.startsWith('.')) {
      const base=name.startsWith('@/') ? path.join(root,name.slice(2)) : path.resolve(path.dirname(file),name);
      const target=[base,`${base}.ts`,`${base}.tsx`,path.join(base,'index.ts')].find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
      if (target) return compile(target);
    }
    return require(name);
  };
  new Function('require','module','exports',code)(localRequire,mod,mod.exports);
  cache.set(file,mod.exports); return mod.exports;
}

const Live=compile(path.join(root,'app/admin/league-manager/live/LeagueLiveRoundPanel.tsx')).default;
const label=node=>typeof node==='string'?node:(node?.children||[]).map(label).join('');
const button=(tree,text)=>tree.root.findAllByType('button').find(node=>label(node)===text);
const control=(tree,name)=>tree.root.findAll(node=>typeof node.type==='string'&&node.props['aria-label']===name)[0];
async function click(tree,text){const b=button(tree,text);assert(b,`Missing ${text}`);assert(!b.props.disabled,`${text} disabled`);await act(async()=>b.props.onClick());}
async function change(node,value){assert(!node.props.disabled);await act(async()=>node.props.onChange({target:{value}}));}
const league='Sunday Ladder', base='/admin/clubs/club/league-manager';
const detail={ok:true,league:{league_name:league,league_type:'Individual',match_format:'doubles',rules_config:{operation:{participation_mode:'set'}},status:'active'},roster:players.map(p=>({player_id:p.id,player_name:p.name,in_league:p.id<8,rating:p.rating,matches_played:8})),standings:[],capabilities:{roster_mutable:true}};
const response=payload=>({ok:true,status:200,json:async()=>structuredClone(payload)});
let calls=[], created=[], operation, scenario, detailFailed=false;
global.fetch=async(url,options={})=>{
 const pathname=new URL(url).pathname,body=options.body?JSON.parse(options.body):null;
 calls.push({pathname,body,method:options.method});
 if(pathname===`${base}/leagues`)return response({leagues:[detail.league]});
 if(pathname===`${base}/leagues/${encodeURIComponent(league)}`)return response(detail);
 if(pathname===`${base}/live-sessions`)return response({sessions:[]});
 if(pathname===`${base}/live/roster-suggestion`){
   const roster=body.roster.map((p,i)=>({...p,status:'active',court_number:i<3?1:2,slot:i<3?i+1:i-2}));
   return response({ok:true,roster,active_roster:roster,bench:[],bench_player_ids:[],default_bench_player_ids:[],court_sizes:[3,4],fingerprint:'reviewed',courts:[1,2].map(n=>({round_number:1,court_number:n,format_type:'4-Player',player_names:roster.filter(p=>p.court_number===n).map(p=>p.player_name)}))});
 }
 if(pathname.endsWith('/players/editor/players')&&options.method==='POST'){
   assert.equal(body.source,'league_court_roster');
   if(scenario==='denied')return {ok:false,status:403,json:async()=>({detail:'Creation not permitted'})};
   const player={id:900+created.length,club_id:'club',name:body.name,rating:body.starting_jupr*400,rating_jupr:body.starting_jupr};created.push(player);operation={key:body.idempotency_key,player};
   if(scenario==='lost'&&created.length===1)throw new TypeError('Connection lost after save');
   return response({player});
 }
 if(pathname.endsWith('/players/editor/players')){
   if(scenario==='duplicate'&&!created.length)created.push({id:900,club_id:'club',name:'Kathy Meyer',rating:1280,rating_jupr:3.2});
   return response({players:[...players,...created]});
 }
 if(pathname.includes('/players/editor/operations/')){assert.equal(decodeURIComponent(pathname.split('/operations/')[1]),operation.key);return response({status:'completed',result:{player:operation.player}});}
 if(/\/players\/editor\/players\/\d+$/.test(pathname)){
   if(scenario==='detail-lost'&&!detailFailed){detailFailed=true;throw new TypeError('Detail failed');}
   return response({player:[...players,...created].find(p=>p.id===Number(pathname.split('/').at(-1)))});
 }
 if(pathname.endsWith('/round-robin/preview'))return response({ok:true,match_count:1,courts:body.courts.map(c=>({...c,matches:[]}))});
 throw Error(`Unexpected ${pathname}`);
};
const props={apiBase:'https://staging.test',clubId:'club',selectedLeagueName:league,leagueStatus:{enabled:true,warnings:[]},liveDomainStatus:{enabled:true,submit_enabled:true,player_create_enabled:true,round_robin_format_options:['4-Player','5-Player']},players};
async function mount(){let tree;await act(async()=>{tree=create(React.createElement(Live,props));});return tree;}
const desired='Kathy Meyer\nKathy Shipley\nKatri Ulmonen\nJohn Schaeffer\nCarrie Brand';
async function submit(tree){await act(async()=>control(tree,'New club player for Court 1').props.onSubmit({preventDefault(){}}));}
async function main(){
 for(scenario of ['success','lost','detail-lost','duplicate','denied']){
   calls=[];created=[];storage.clear();local.clear();detailFailed=false;
   let tree=await mount();
   await click(tree,'Continue to Players');await click(tree,'Confirm test starting estimates');await click(tree,'Build courts from 7 attendees');
   const otherCourt=control(tree,'Players, one per line for Court 2').props.value;
   await change(control(tree,'Players, one per line for Court 1'),desired);
   await act(async()=>control(tree,'Create new player for Court 1').props.onClick());
   assert.equal(control(tree,'New player name for Court 1').props.value,'Kathy Meyer');
   assert.equal(button(tree,'Create and select player').props.disabled,true,'Requires a starting estimate');
   await change(control(tree,'New player starting level for Court 1'),'3.2');await submit(tree);
   if(scenario==='denied'){
     assert.match(label(tree.root),/Creation not permitted/);assert.equal(created.length,0);
     assert.equal(control(tree,'Format for Court 1').props.value,'4-Player');
     await act(async()=>tree.unmount());continue;
   }
   if(scenario==='lost'||scenario==='detail-lost'){
     assert.equal(button(tree,'Back to Players').props.disabled,true,'Keep the court context while the write is uncertain');
     assert(control(tree,'New club player for Court 1'),'Keep recovery available');
     await act(async()=>tree.unmount());tree=await mount();
     await submit(tree);
     assert.equal(calls.filter(c=>c.method==='POST'&&c.pathname.endsWith('/players/editor/players')).length,1,'Recovery does not create duplicate profiles');
   }
   if(scenario==='duplicate'){
     assert.equal(calls.filter(c=>c.method==='POST'&&c.pathname.endsWith('/players/editor/players')).length,0);
     await click(tree,'Use Kathy Meyer');
   }
   assert(control(tree,'Select Kathy Meyer on Court 1, position 1')||control(tree,'Select Kathy Meyer on Court 1, position 4'));
   assert.equal(control(tree,'Players, one per line for Court 2').props.value,otherCourt);
   assert.equal(button(tree,'Validate courts and generate preview').props.disabled,true,'Review new player rating before preview');
   if(scenario==='success'||scenario==='duplicate')assert.equal(control(tree,'Players, one per line for Court 1').props.value,desired,'Keep the second missing name after creating the first');
   // Re-enter the desired text after a reload, since unapplied text is intentionally local to the editor.
   if(scenario==='lost'||scenario==='detail-lost')await change(control(tree,'Players, one per line for Court 1'),desired);
   await act(async()=>control(tree,'Create new player for Court 1').props.onClick());
   assert.equal(control(tree,'New player name for Court 1').props.value,'Kathy Shipley');
   await change(control(tree,'New player starting level for Court 1'),'3.25');await submit(tree);
   assert.equal(control(tree,'Format for Court 1').props.value,'5-Player');
   assert(control(tree,'Select Kathy Meyer on Court 1, position 1'));assert(control(tree,'Select Kathy Shipley on Court 1, position 2'));
   assert.equal(control(tree,'Players, one per line for Court 2').props.value,otherCourt);
   assert.equal(control(tree,'Create new player for Court 1').props.disabled,true,'Cannot create a sixth player for this court');
   await click(tree,'Confirm test starting estimates');await click(tree,'Validate courts and generate preview');
   const preview=calls.filter(c=>c.pathname.endsWith('/round-robin/preview')).at(-1);
   assert.deepEqual(preview.body.courts[0].player_names,desired.split('\n'));
   assert.equal(storage.size,0,'Completed creation recovery is cleared');
   await act(async()=>tree.unmount());tree=await mount();
   assert.equal(control(tree,'Players, one per line for Court 1').props.value,desired,'Added players and positions survive a refresh');
   await act(async()=>tree.unmount());
 }
 // Existing profile selection uses the same court placement without creating a new profile.
 scenario='existing';calls=[];created=[];storage.clear();local.clear();
 let tree=await mount();await click(tree,'Continue to Players');await click(tree,'Confirm test starting estimates');await click(tree,'Build courts from 7 attendees');
 await change(control(tree,'Add existing player to Court 1'),'8');
 assert(control(tree,'Select Existing Guest on Court 1, position 4'));
 assert.equal(calls.filter(c=>c.pathname.endsWith('/players/editor/players')&&c.method==='POST').length,0);
 await act(async()=>tree.unmount());
 console.log('PASS league court player creation: two missing profiles, existing profile reuse, rating review, same-court placement, five-player format, untouched other courts, saved setup, denied writes, and lost-response recovery.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
