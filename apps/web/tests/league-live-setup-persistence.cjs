const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const {create, act} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
if (!global.crypto) global.crypto = require('node:crypto').webcrypto;
const storage = new Map(), tabStorage = new Map();
let blockLocal = false, blockTab = false;
const store = (map, blocked) => ({getItem:key => {if(blocked()) throw Error('Blocked'); return map.get(key) || null;}, setItem:(key,value) => {if(blocked()) throw Error('Blocked');map.set(key,value);}, removeItem:key => {if(blocked()) throw Error('Blocked');map.delete(key);}});
global.localStorage = store(storage, () => blockLocal);
global.sessionStorage = store(tabStorage, () => blockTab);
let token = 'token', operatorId = 'joe';
global.addEventListener = () => {};
global.removeEventListener = () => {};
const root = path.join(__dirname, '..');
const cache = new Map();
const players = ['Alex','Blair','Casey','Devon','Late arrival'].map((name,i) => ({id:i+1,name,rating:i===0?1200:1400,rating_jupr:i===0?3:3.5}));
const reviewRows = players.slice(0,4).map(p => ({player_id:p.id,name:p.name,has_rated_history:p.id!==1,rating_jupr:p.id===1?3.75:p.rating_jupr,starting_jupr:p.id===1?3.75:p.rating_jupr,review_required:false}));
const Review = props => {React.useEffect(() => props.onReady(true, reviewRows), []); return React.createElement('div', {'data-review':JSON.stringify(props.requestedRatings)});};
const CourtEditor = props => React.createElement('div', null, props.courts.map((court, i) => React.createElement('input', {key:i,'aria-label':`Court ${court.court} order`, value:court.playerNames, onChange:event => props.onChange(props.courts.map((row,j) => i===j ? {...row,playerNames:event.target.value} : row))})));
const Import = props => React.createElement('button', {onClick:() => props.onImport({playerIds:[1,2,3,4],requestedRatings:{'1':'3.75','2':'4.0'}})}, 'Import test registrants');
const mocks = {
  'next/link': {__esModule:true,default:({children,...props}) => React.createElement('a',props,children)},
  'next/dynamic': {__esModule:true,default:factory => String(factory).includes('StartingRatingReview') ? Review : String(factory).includes('CourtReserveRosterImport') ? Import : String(factory).includes('LeagueCourtRosterEditor') ? CourtEditor : () => null},
  '@/lib/useAdminSession': {useAdminSession:() => ({accessToken:token,session:{user:{id:operatorId}}}),adminSessionLabel:() => 'Admin'},
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
const Live = compile(path.join(root,'app/admin/league-manager/live/LeagueLiveRoundPanel.tsx')).default;
const label = node => typeof node === 'string' ? node : (node?.children || []).map(label).join('');
const button = (tree,text) => tree.root.findAllByType('button').find(node => label(node) === text);
async function click(tree,text) {const node=button(tree,text);assert(node,`Missing ${text}`);assert(!node.props.disabled,`${text} is disabled`);await act(async()=>node.props.onClick());}
function input(tree, name) {const parent=tree.root.findAllByType('label').find(node => label(node).startsWith(name));assert(parent,`Missing input ${name}`);return parent.findByType('input');}
async function change(node,value) {assert(!node.props.disabled);await act(async()=>node.props.onChange({target:{value}}));}
const league='Sunday Ladder', otherLeague='Tuesday Ladder';
const detail = name => ({ok:true,league:{league_name:name,league_type:'Individual',match_format:'doubles',rules_config:{participation_mode:'flex'},status:'active'},roster:players.map(p=>({player_id:p.id,player_name:p.name,in_league:p.id<5,rating:p.rating,overall_rating:p.rating,rating_jupr:p.rating_jupr,overall_rating_jupr:p.rating_jupr,matches_played:8})),standings:[]});
const response = payload => ({ok:true,status:200,json:async()=>structuredClone(payload)});
let calls=[], sessions=[], delayDetail=null, loseCreateResponse=false, lastCreateResult=null;
global.fetch = async(url, options={}) => {
  const pathname=new URL(url).pathname, body=options.body?JSON.parse(options.body):null;
  calls.push({pathname,body,method:options.method || 'GET'});
  if(pathname.endsWith('/leagues')) return response({leagues:[detail(league).league,detail(otherLeague).league]});
  if(pathname.includes('/leagues/')) {if(delayDetail) await delayDetail; return response(detail(decodeURIComponent(pathname.split('/').at(-1))));}
  if(pathname.endsWith('/live-sessions')) {
    if(options.method==='POST') {
      assert([...storage.values()].some(value => value.includes(body.idempotency_key)), 'The exact session operation must be durable before the request leaves');
      const row={id:'saved-session',league_name:body.league_name,week_tag:body.week_tag,status:'active',total_rounds:body.total_rounds,current_round:body.current_round,updated_at:'2026-11-01T16:00:00Z',roster_json:body.roster,current_court_state_json:body.courts};
      sessions=[row];lastCreateResult={ok:true,session:row,courts:body.courts};
      if(loseCreateResponse) throw new TypeError('Network connection lost');
      return response(lastCreateResult);
    }
    return response({sessions});
  }
  if(pathname.endsWith('/roster-suggestion')) {
    const roster=body.roster.map((p,i)=>({...p,status:'active',court_number:1,slot:i+1}));
    return response({ok:true,roster,active_roster:roster,bench:[],bench_player_ids:[],default_bench_player_ids:[],court_sizes:[4],fingerprint:'reviewed',courts:[{round_number:1,court_number:1,format_type:'4-Player',player_names:roster.map(p=>p.player_name),players_json:roster}]});
  }
  if(pathname.endsWith('/round-robin/preview')) return response({ok:true,match_count:1,courts:body.courts.map(c=>({...c,matches:[]}))});
  if(pathname.includes('/live-operations/')) return response({ok:true,status:'completed',result:lastCreateResult});
  if(pathname.endsWith('/live/players')) {
    assert([...storage.values()].some(value=>value.includes(body.idempotency_key)), 'Player operation is retained before sending');
    throw new TypeError('Player creation response lost');
  }
  if(pathname.includes('/live/player-operations/')) return response({ok:true,status:'failed',recovery_required:false});
  throw Error(`Unexpected ${pathname}`);
};
const props={apiBase:'https://staging.test',clubId:'club',selectedLeagueName:league,leagueStatus:{enabled:true,warnings:[]},liveDomainStatus:{enabled:true,submit_enabled:true,player_create_enabled:true,round_robin_format_options:['4-Player']},players};
async function mount() {let tree;await act(async()=>{tree=create(React.createElement(Live, props));});return tree;}
async function reopen(tree) {await act(async()=>tree.unmount());tabStorage.clear();return mount();}
function courtInput(tree) {return tree.root.findByProps({'aria-label':'Court 1 order'});}
function currentStep(tree) {return label(tree.root.findByProps({'aria-current':'step'}));}
async function main() {
  let tree=await mount();
  await change(input(tree,'Week'),'Week 3');
  await change(input(tree,'Date'),'2026-11-01');
  await change(input(tree,'Total rounds'),'6');
  await change(input(tree,'Notes'),'Morning ladder — keep these players');
  tree=await reopen(tree);
  assert.equal(input(tree,'Week').props.value,'Week 3','Setup saves before even selecting players');
  assert.equal(input(tree,'Total rounds').props.value,'6');
  assert.equal(input(tree,'Date').props.value,'2026-11-01');
  assert.equal(input(tree,'Notes').props.value,'Morning ladder — keep these players');
  assert(label(tree.toJSON()).includes('Restored setup.'));
  await click(tree,'Continue to Players');
  await click(tree,'Import CourtReserve registrants');
  await click(tree,'Import test registrants');
  await change(tree.root.findByType('textarea'),'New Guest');
  await click(tree,'Resolve and add pasted players');
  await change(input(tree,'Starting JUPR'),'3.25');
  tree=await reopen(tree);
  assert(currentStep(tree).endsWith('Players'),'Returns to the player setup step');
  assert.equal(tree.root.findByType('textarea').props.value,'New Guest','Unfinished pasted names are protected');
  assert.equal(input(tree,'Starting JUPR').props.value,'3.25');
  assert(label(tree.toJSON()).includes('Session player order (4)'));
  assert.deepEqual(JSON.parse(tree.root.findAll(node=>node.props['data-review'])[0].props['data-review']),{'1':'3.75','2':'4.0'});
  await change(tree.root.findByType('textarea'),'');
  await click(tree,'Build courts from 4 attendees');
  const order='Devon\nCasey\nAlex\nBlair';
  await change(courtInput(tree),order);
  tree=await reopen(tree);
  assert(currentStep(tree).endsWith('Courts and Preview'));
  assert.equal(courtInput(tree).props.value,order,'Closing the tab before session creation preserves manual positions');
  assert.equal(calls.filter(call=>call.pathname.endsWith('/live-sessions') && call.method==='POST').length,0,'Autosaving setup must not create matches or active sessions');
  assert(![...storage.values()].some(value=>value.includes('access_token') || value.includes('refresh_token')));

  // Sleeping laptops can return with a refreshed access token. Delay the
  // replacement read to expose any effect that saves an empty loading state.
  let release;
  delayDetail=new Promise(resolve=>{release=resolve;});
  token='refreshed-token';
  await act(async()=>tree.update(React.createElement(Live,props)));
  assert([...storage.values()].some(value=>value.includes('Devon\\nCasey\\nAlex\\nBlair')),'Loading/auth reset cannot erase saved positions');
  delayDetail=null;
  await act(async()=>release());
  assert.equal(courtInput(tree).props.value,order,'Auth refresh restores the same setup');

  await click(tree,'Back to Players');await click(tree,'Back to Setup');
  const selectLeague=()=>tree.root.findAllByType('select').find(node=>node.props.value===league || node.props.value===otherLeague);
  await change(selectLeague(),otherLeague);
  assert.equal(input(tree,'Notes').props.value,'','Different leagues never inherit the same setup');
  await change(input(tree,'Notes'),'Tuesday setup');
  await change(selectLeague(),league);
  assert.equal(input(tree,'Notes').props.value,'Morning ladder — keep these players');
  await click(tree,'Continue to Players');
  tree=await reopen(tree);
  assert(currentStep(tree).endsWith('Players'));

  await act(async()=>tree.unmount());operatorId='other-admin';tree=await mount();
  assert.equal(input(tree,'Notes').props.value,'','Drafts belong to the signed-in operator');
  await act(async()=>tree.unmount());operatorId='joe';tree=await mount();
  assert(currentStep(tree).endsWith('Players'));
  await click(tree,'Discard setup and start again');
  assert.equal(input(tree,'Notes').props.value,'');
  tree=await reopen(tree);
  assert.equal(input(tree,'Week').props.value,'Week 1','Discarded setup cannot reappear on reload');
  await change(selectLeague(),otherLeague);
  assert.equal(input(tree,'Notes').props.value,'Tuesday setup','Discard is scoped to one league');
  await change(selectLeague(),league);

  blockLocal=true;
  await change(input(tree,'Notes'),'Tab fallback');
  assert(label(tree.toJSON()).includes('saved only in this tab'),'Never claim restart-safe saving when local storage is blocked');
  blockTab=true;
  await change(input(tree,'Notes'),'Cannot save');
  assert(label(tree.toJSON()).includes('Setup could not be saved'));
  blockLocal=false;blockTab=false;
  await change(input(tree,'Notes'),'Saved again');
  assert(label(tree.toJSON()).includes('automatically saved on this browser'));
  await click(tree,'Continue to Players');await click(tree,'Import CourtReserve registrants');await click(tree,'Import test registrants');
  await click(tree,'Build courts from 4 attendees');await click(tree,'Validate courts and generate preview');
  await click(tree,'Create session and continue');
  assert(![...storage.entries()].some(([key,value])=>key.includes('joe') && key.includes(league) && value.includes('Saved again')),'Successful server creation retires the setup draft');
  tree=await reopen(tree);
  assert.equal(input(tree,'Notes').props.value,'','A created session must not return as an unstarted duplicate draft');

  await click(tree,'Continue to Players');
  await change(tree.root.findByType('textarea'),'New Guest');await click(tree,'Resolve and add pasted players');
  await change(input(tree,'Starting JUPR'),'3.25');
  await click(tree,'Create missing players');
  tree=await reopen(tree);
  assert(button(tree,'Create missing players').props.disabled,'A lost player response survives closing the tab and prevents another batch');
  await click(tree,'Check and reconcile exact player operation');
  assert(!button(tree,'Create missing players').props.disabled,'Only a conclusive failed operation allows retry');
  await change(tree.root.findByType('textarea'),'');
  await click(tree,'Import CourtReserve registrants');await click(tree,'Import test registrants');
  await click(tree,'Build courts from 4 attendees');await click(tree,'Validate courts and generate preview');
  loseCreateResponse=true;
  await click(tree,'Create session and continue');
  const createCount=calls.filter(call=>call.method==='POST' && call.pathname.endsWith('/live-sessions')).length;
  tree=await reopen(tree);
  assert(button(tree,'Create session and continue').props.disabled,'A recovered draft cannot duplicate a session whose response was lost');
  await click(tree,'Check and reconcile exact operation');
  assert.equal(calls.filter(call=>call.method==='POST' && call.pathname.endsWith('/live-sessions')).length,createCount);
  assert(label(tree.toJSON()).includes('Its session is now loaded.'));
  assert(!button(tree,'Check and reconcile exact operation'));
  await act(async()=>tree.unmount());
  console.log('League early setup persistence: settings, import, partial names, court positions, restart, auth refresh, scoping, discard, failed storage, promotion, and lost-response recovery passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
