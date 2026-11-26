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

async function clubRegistrationPlanning() {
  const directory = 'app/admin/interclub/managed/';
  const setup = load('lib/interclubSetup.ts');
  const prefix = '/admin/test-season';
  const host = { id: 'host', name: 'Host club', captain: 'Old captain', contact: 'old@example.invalid', rosters: [] };
  const visitor = { id: 'visitor', name: 'Visiting club', captain: 'Visitor captain', contact: 'visitor@example.invalid', rosters: [] };
  const details = { name: 'Season', timezone: 'UTC', divisions: ['3.5'], clubs: [visitor, host], start_date: null, end_date: null, rules: '' };
  const initial = { season: { id: 'season', revision: 1, details }, meets: [], standings: { divisions: {}, qualification: [] }, club_cup: { standings: [] } };
  const full = { date: '2027-01-09', start_time: '09:30', end_time: '12:00', courts: 3 };
  const partial = { date: '2027-01-16', start_time: null, end_time: null, courts: null };
  const registration = { captain: 'Current captain', contact_method: 'whatsapp', contact: '+52 123', participation: 'host_travel',
    divisions: ['3.5'], travel_notes: 'Saturday mornings', host_offers: [full, partial] };
  const entries = [{ club_id: host.id, status: 'registered', revision: 1, registration }];
  const writes = [];
  const managedRequest = async (url, accessToken, method, body) => {
    assert.equal(accessToken, 'admin-session');
    if (method === 'GET' && url === prefix + '/club-entries') return { entries: [...entries] };
    writes.push({ url, method, body });
    if (method === 'POST' && url === prefix + '/club-entries/visitor/link') {
      entries.push({ club_id: visitor.id, status: 'invited', revision: 0, registration: null });
      return { token: 'private-test-token' };
    }
    assert.match(url, /^\/admin\/test-season\/meets\//);
    assert.equal(method, 'PUT');
    return { ...initial, meets: [] };
  };
  const mocks = { '@/lib/managedInterclub': { ...helpers, managedRequest }, '@/lib/interclubSetup': setup, '../setup.module.css': {} };
  const ClubRegistrations = load(directory + 'ClubRegistrations.tsx', mocks).default;
  const MeetPlanner = load(directory + 'MeetPlanner.tsx', mocks).default;
  const TeamsEditor = load(directory + 'TeamsEditor.tsx', mocks).default;
  const Workspace = load(directory + 'ManagedSeasonWorkspace.tsx', { ...mocks,
    './ClubRegistrations': ClubRegistrations, './MeetPlanner': MeetPlanner, './TeamsEditor': TeamsEditor,
    '@/components/PublicInterclubCompetition': { CompetitionStandings: () => null },
    '@/components/interclub/ManagedScoreboard': () => null, 'next/link': () => null,
  }).default;
  window.location = { origin: 'https://example.invalid' };
  let tree;
  await act(async () => { tree = create(React.createElement(Workspace, { root: prefix, accessToken: 'admin-session', initial, onClose() {} })); });
  const click = async label => { await act(async () => button(tree, label).props.onClick()); };
  const input = (area, label) => area.findAllByType('label').find(node => nodeText(node).startsWith(label)).findByType('input');
  const planner = () => tree.root.findByType(MeetPlanner);
  const draftClub = () => input(tree.root.findByType(TeamsEditor), 'Participating club');
  await act(async () => draftClub().props.onChange({ target: { value: 'Unsaved club' } }));
  await click('Club registrations');
  const cards = tree.root.findByType(ClubRegistrations).findAllByType('article');
  assert.match(nodeText(cards[0]), /Awaiting invitation.*Visitor captain.*visitor@example.invalid/);
  assert.match(nodeText(cards[1]), /Registered.*Current captain.*WhatsApp.*Saturday mornings/);
  assert.doesNotMatch(nodeText(cards[1]), /Old captain|old@example/);
  await click('Create registration link');
  assert.equal(input(cards[0], 'Private registration link').props.value, 'https://example.invalid/interclub/club-entry#token=private-test-token');
  assert.match(nodeText(cards[0]), /Invited/);
  const planAt = date => 'Plan meet at Host club on ' + setup.seasonDateLabel(date);
  await click(planAt(partial.date));
  assert.equal(input(planner(), 'Meet date').props.value, partial.date);
  assert.equal(input(planner(), 'Start time').props.value, '');
  assert.equal(input(planner(), 'Courts').props.value, '');
  await click('Cancel');
  await click('Club registrations');
  await click(planAt(full.date));
  assert.equal(input(planner(), 'Meet date').props.value, full.date);
  assert.equal(input(planner(), 'Start time').props.value, full.start_time);
  assert.equal(input(planner(), 'Duration').props.value, 150);
  assert.equal(input(planner(), 'Courts').props.value, full.courts);
  assert.equal(planner().findByProps({ 'aria-label': 'Host' }).props.value, host.id);
  assert.equal(input(planner(), 'Visiting club').props.checked, false);
  assert.equal(writes.length, 1, 'Selecting a host offer only prepares a draft');
  const submit = async () => { await act(async () => planner().findByType('form').props.onSubmit({ preventDefault() {} })); };
  await submit();
  assert.equal(writes.length, 1, 'A meet requires a visiting club');
  await act(async () => input(planner(), 'Visiting club').props.onChange({ target: { checked: true } }));
  await click('Club registrations');
  assert.equal(button(tree, planAt(partial.date)).props.disabled, true);
  await click('Teams & Players');
  assert.equal(draftClub().props.value, 'Unsaved club');
  await click('Schedule');
  assert.equal(input(planner(), 'Visiting club').props.checked, true);
  assert.ok(button(tree, 'Confirm and save meet'));
  await submit();
  assert.equal(writes.length, 2);
  assert.equal(writes[1].body.details.starts_at, '2027-01-09T09:30:00.000Z');
  assert.deepEqual(writes[1].body.details.club_ids, ['host', 'visitor']);
  assert.equal(writes[1].body.details.duration_minutes, 150);
  await act(async () => tree.unmount());
}

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
 await clubRegistrationPlanning();
 console.log('Club registration links, offer review, draft retention and explicit meet confirmation passed.');
 console.log('Managed interclub roster identity, concurrent typing, conflicts, revoked links and outage recovery passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
