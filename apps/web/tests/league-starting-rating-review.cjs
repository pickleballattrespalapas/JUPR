const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const {create, act} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
if (!global.crypto) global.crypto = require('node:crypto').webcrypto;
const storage = new Map();
global.sessionStorage = {getItem:key => storage.get(key) || null, setItem:(key,value) => storage.set(key,value), removeItem:key => storage.delete(key)};
global.addEventListener = () => {};
global.removeEventListener = () => {};
const root = path.join(__dirname, '..');
const cache = new Map();
const players = ['Alex','Blair','Casey','Devon','Late arrival'].map((name,i) => ({id:i+1,name,rating:i===0?1200:1400,rating_jupr:i===0?3:3.5}));
const reviewRows = players.slice(0,4).map(p => ({player_id:p.id,name:p.name,has_rated_history:p.id!==1,rating_jupr:p.id===1?3.75:p.rating_jupr,starting_jupr:p.id===1?3.75:p.rating_jupr,review_required:false}));
const Review = props => React.createElement('button', {onClick:() => {props.onReady(true,reviewRows); props.onConfirmed?.(reviewRows);}}, 'Confirm test starting estimates');
const Import = props => React.createElement('button', {onClick:() => props.onImport({playerIds:[1,2,3,4],requestedRatings:{'1':'3.75','2':'4.0'}})}, 'Import test registrants');
const mocks = {
  'next/link': {__esModule:true,default:({children,...props}) => React.createElement('a',props,children)},
  'next/dynamic': {__esModule:true,default:factory => String(factory).includes('StartingRatingReview') ? Review : String(factory).includes('CourtReserveRosterImport') ? Import : () => null},
  '@/lib/useAdminSession': {useAdminSession:() => ({accessToken:'token',session:{}}),adminSessionLabel:() => 'Admin'},
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
const Roster=compile(path.join(root,'app/admin/league-manager/roster/LeagueRosterPanel.tsx')).default;
const label=node => typeof node==='string'?node:(node.children||[]).map(label).join('');
const button=(tree,text) => tree.root.findAllByType('button').find(node => label(node)===text);
async function click(tree,text) {const node=button(tree,text);assert(node,`Missing ${text}`);assert(!node.props.disabled,`${text} is disabled`);await act(async()=>node.props.onClick());}
const league='Sunday Ladder';
const base='/admin/clubs/club/league-manager';
const detail={ok:true,league:{league_name:league,league_type:'Individual',match_format:'doubles',rules_config:{participation_mode:'set'},status:'active'},roster:players.map(p=>({player_id:p.id,player_name:p.name,in_league:p.id<5,rating:p.rating,overall_rating:p.rating,rating_jupr:p.rating_jupr,overall_rating_jupr:p.rating_jupr,matches_played:p.id===1?0:8})),standings:[],capabilities:{roster_mutable:true}};
const response=payload=>({ok:true,status:200,json:async()=>structuredClone(payload)});
let calls=[];
global.fetch=async(url,options={})=>{
  const pathname=new URL(url).pathname, body=options.body?JSON.parse(options.body):null;
  calls.push({pathname,body});
  if(pathname===`${base}/leagues`) return response({leagues:[detail.league]});
  if(pathname===`${base}/leagues/${encodeURIComponent(league)}`) return response(detail);
  if(pathname===`${base}/live-sessions`) return response({sessions:[]});
  if(pathname===`${base}/live/roster-suggestion`) {
    const roster=body.roster.map((p,i)=>({...p,status:'active',court_number:1,slot:i+1}));
    return response({ok:true,roster,active_roster:roster,bench:[],bench_player_ids:[],court_sizes:[4],fingerprint:'reviewed',courts:[{round_number:1,court_number:1,format_type:'4-Player',player_names:roster.map(p=>p.player_name),players_json:roster}]});
  }
  if(pathname.endsWith('/roster/batch')) return response({ok:true,committed:true,updated_count:body.player_ids.length,detail});
  throw new Error(`Unexpected ${pathname}`);
};
async function main(){
  let tree;
  await act(async()=>{tree=create(React.createElement(Live,{apiBase:'https://staging.test',clubId:'club',selectedLeagueName:league,leagueStatus:{enabled:true,warnings:[]},liveDomainStatus:{enabled:true,submit_enabled:true,player_create_enabled:true,round_robin_format_options:['4-Player']},players}));});
  assert.equal(calls.filter(c=>c.pathname.endsWith('/roster-suggestion')).length,0,'Loading the roster must not seed courts before review');
  await click(tree,'Continue to Players');
  assert.equal(button(tree,'Build courts from 4 attendees').props.disabled,true);
  await click(tree,'Confirm test starting estimates');
  await click(tree,'Build courts from 4 attendees');
  const suggested=calls.find(c=>c.pathname.endsWith('/roster-suggestion'));
  assert.equal(suggested.body.roster.find(p=>p.player_id===1).rating,1500,'Confirmed seed is used by the new court plan');
  await click(tree,'Back to Players');
  const select=tree.root.findAllByType('select').find(node=>node.props['aria-label']==='Add an existing club player, including a non-roster player');
  await act(async()=>select.props.onValueChange('5'));
  assert.equal(button(tree,'Build courts from 5 attendees').props.disabled,true,'A late selected player invalidates previous review readiness');
  await act(async()=>tree.unmount());

  calls=[];
  detail.roster=detail.roster.map(row=>({...row,in_league:row.player_id<=2}));
  await act(async()=>{tree=create(React.createElement(Roster,{apiBase:'https://staging.test',clubId:'club',initialLeague:league,status:{enabled:true}}));});
  await click(tree,'Import CourtReserve roster');
  await click(tree,'Import test registrants');
  const saved=calls.find(c=>c.pathname.endsWith('/roster/batch'));
  assert.deepEqual(saved.body.player_ids,[3,4],'Import adds only missing members and never removes the season roster');
  assert.equal(saved.body.action,'activate');
  assert.equal(saved.body.starting_rating,null,'Imported requested values cannot replace calculated JUPR through roster writes');
  assert.equal(saved.body.confirmation_text,'SAVE LEAGUE ROSTER BATCH');
  await act(async()=>tree.unmount());
  console.log('League starting review and CourtReserve integration passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
