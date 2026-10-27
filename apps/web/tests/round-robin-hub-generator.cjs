const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map(), h = React.createElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
const navigation = [];
const router = { push: url => navigation.push(url), replace: url => navigation.push(url), refresh() {} };
const Link = ({ children, ...props }) => h("a", props, children);
const stubs = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel, description, onConfirm, disabled }) => h("button", { onClick: onConfirm, disabled, "data-description": description }, triggerLabel) },
  "@/components/GeneratorSharing": () => null,
  "@/components/GeneratorSubmission": { __esModule: true, default: () => null, generatorResultLabel: () => "Unrated" },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: () => null },
};
function load(name, parent = root) {
  if (name === "@/components/PublicControls") return require("./helpers/public-controls-module.cjs")(name);
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts", path.join(base, "index.ts")].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}

const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const clone = value => JSON.parse(JSON.stringify(value));
const Runner = load('@/app/admin/play-generators/GeneratorRoundRunner').default;
const Standings = load('@/app/admin/play-generators/GeneratorStandings').default;
const PublicRunner = load('@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner').default;
const PublicStandings = load('@/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings').default;
const fixture = {
  session_key: 'rr', title: 'Hub group', status: 'active', version: 'v1', round_robin_hub_id: 'hub',
  generator_kind: 'round_robin', play_format: 'doubles', scoring_mode: 'scored', rating_mode: 'unrated',
  current_round_number: 1, total_rounds: 2, standings: [],
  event: { name: 'Hub group', generatorKind: 'round_robin', playFormat: 'doubles', scoringMode: 'scored',
    status: 'active', currentRoundNumber: 1, totalRounds: 2, courtCount: 2,
    participants: Array.from({length:8}, (_, i) => ({id:`p${i+1}`, name:`Player ${i+1}`, roster_order:i+1})),
    rounds: [
      {number:1,status:'active',matches:[
        {id:'r1a',court:1,courtLabel:'Stadium',sideA:['p1','p2'],sideB:['p3','p4'],scoreA:11,scoreB:8},
        {id:'r1b',court:2,courtLabel:'Palapa',sideA:['p5','p6'],sideB:['p7','p8'],scoreA:null,scoreB:null},
      ]},
      {number:2,status:'preview',matches:[{id:'r2a',court:1,sideA:['p1','p3'],sideB:['p2','p4']}]}
    ]
  }
};
const props = {apiBase:'https://api.invalid',clubId:'club',generatorKind:'round_robin',sessionKey:'rr',roundNumber:1,embedded:true,refreshKey:1};
let state, reads, writes, changes, routed, tree;
function setup() {
  state=clone(fixture); reads=0; writes=[]; changes=0; routed=[]; navigation.length=0;
  global.fetch = async (url, options={}) => {
    if (!options.method || options.method==='GET') reads++;
    else {
      writes.push({url,body:JSON.parse(options.body)});
      if(url.endsWith('/scores')) {
        for (const result of writes.at(-1).body.scores) {
          const match=state.event.rounds[0].matches.find(row=>row.id===result.match_id);
          match.scoreA=result.score_a; match.scoreB=result.score_b;
        }
        state.event.rounds[0].status='saved'; state.version='v2';
        state.event.currentRoundNumber=2; state.current_round_number=2; state.event.rounds[1].status='active';
        state.event.hubAutoAdvancedFromRound=1;
      } else if(url.endsWith('/advance')) {
        state.event.currentRoundNumber=2; state.current_round_number=2; state.event.rounds[1].status='active';
      } else if(url.endsWith('/complete')) {
        state.status=state.event.status='completed';state.event.rounds[1].status='skipped';
        delete state.event.hubAutoAdvancedFromRound;
      }
    }
    return {ok:true,status:200,json:async()=>({ok:true,session:clone(state)})};
  };
}
const withCallbacks = extra => ({...props,...extra,onNavigate:target=>routed.push(target),onSessionChanged:()=>changes++});
function embeddedClick(href) {
  let prevented=false;
  const event={button:0,target:{closest:()=>({getAttribute:()=>href})},preventDefault(){prevented=true;},stopPropagation(){}};
  tree.root.findAllByType('div').find(node=>node.props.onClickCapture).props.onClickCapture(event);
  assert.ok(prevented);
}
(async()=>{
  setup();
  await act(async()=>{tree=create(h(Runner,withCallbacks()));});
  assert.equal(reads,1,'Refresh key must not duplicate initial session load');
  assert.ok(text(tree.toJSON()).includes('1 of 2 matches complete — scores saved, standings pending.'));
  assert.ok(text(tree.toJSON()).includes('Stadium'));
  assert.equal(button(tree,'Refresh court results').props.disabled,false,'Persisted partial scores are not unsaved drafts');
  assert.equal(tree.root.findAllByProps({'aria-label':'Open round 2'}).length,0,'Legacy linked sessions expose only the current round until the hub prepares more');
  assert.ok(text(tree.toJSON()).includes('Add player'),'Full roster functionality remains in embedded runner');
  await act(async()=>embeddedClick('/admin/round-robin-generator/sessions/rr/standings'));
  assert.deepEqual(routed,[{view:'standings'}]); assert.deepEqual(navigation,[]);
  const score=()=>tree.root.findByProps({'aria-label':'r1b side A score'});
  await act(async()=>score().props.onChange({target:{value:'11'}}));
  await act(async()=>tree.update(h(Runner,withCallbacks({refreshKey:2}))));
  assert.equal(reads,1,'Hub polling must not overwrite a score being entered');
  assert.equal(score().props.value,'11');
  await act(async()=>tree.root.findByProps({'aria-label':'r1b side B score'}).props.onChange({target:{value:'5'}}));
  await act(async()=>button(tree,'Save round scores').props.onClick());
  assert.equal(writes.length,1); assert.equal(changes,1);
  assert.equal(state.event.rounds[0].status,'saved');
  assert.ok(!text(tree.toJSON()).includes('standings pending.'));
  assert.equal(state.current_round_number,2,'Manager follows the hub response that automatically queues the next round');
  await act(async()=>tree.unmount());

  setup(); state.event.rounds[0].status='saved';
  state.event.rounds[0].matches[1].scoreA=11;state.event.rounds[0].matches[1].scoreB=5;
  await act(async()=>{tree=create(h(Standings,withCallbacks()));});
  assert.ok(text(tree.toJSON()).includes('Standings include completed rounds only.'));
  await act(async()=>button(tree,'Next scheduled round').props.onClick());
  assert.deepEqual(routed,[{view:'round',roundNumber:2}]);
  assert.deepEqual(navigation,[],'Embedded advance keeps the hub open');
  assert.ok(changes>=1);
  await act(async()=>tree.unmount());

  global.sessionStorage={getItem:()=> 'organizer-token',setItem(){},removeItem(){}};
  global.window={location:{hash:'',pathname:'/standings',search:''},history:{replaceState(){}}};
  for (const Component of [Runner,PublicRunner]) {
    setup();
    if(Component===PublicRunner) state.version=1;
    state.event.hubPreparedThroughRound=2;
    state.event.rounds[1].status='active';
    state.event.rounds.push({...clone(state.event.rounds[1]),number:3,status:'preview',matches:[]});
    await act(async()=>{tree=create(h(Component,withCallbacks()));});
    assert.equal(tree.root.findAllByProps({'aria-label':'Open round 2'}).length,1,'Prepared future rounds are visible');
    assert.equal(tree.root.findAllByProps({'aria-label':'Open round 3'}).length,0,'Unprepared templates remain outside the visible horizon');
    assert.ok(text(tree.toJSON()).includes('Standings update in round order'));
    await act(async()=>tree.root.findByProps({'aria-label':'Open round 2'}).props.onClick());
    assert.equal(writes.length,0,'Viewing a prepared future round never selects or advances it');
    if(Component===Runner) assert.deepEqual(routed,[{view:'round',roundNumber:2}]);
    else assert.ok(navigation.at(-1).endsWith('/rounds/2'));
    await act(async()=>tree.unmount());

    await act(async()=>{tree=create(h(Component,withCallbacks({roundNumber:2})));});
    assert.ok(text(tree.toJSON()).includes('View only here.'));
    assert.equal(tree.root.findAllByProps({'aria-label':'r2a side A score'}).length,0,'Future rounds cannot bypass hub player dependencies with a whole-round score save');
    assert.equal(button(tree,'Save round scores'),undefined);
    assert.equal(button(tree,'Next scheduled round'),undefined);
    assert.equal(writes.length,0);
    await act(async()=>tree.unmount());

    setup();
    if(Component===PublicRunner) state.version=1;
    await act(async()=>{tree=create(h(Component,withCallbacks({roundNumber:2})));});
    if(Component===Runner) assert.deepEqual(routed,[{view:'round',roundNumber:1}]);
    else assert.ok(navigation.at(-1).endsWith('/rounds/1'));
    assert.equal(writes.length,0,'A direct unprepared-round URL only returns to the current round');
    await act(async()=>tree.unmount());

    for(const linked of [true,false]) {
      setup();
      if(Component===PublicRunner) state.version=1;
      state.status=state.event.status='completed';state.event.rounds[0].status='saved';
      if(!linked) delete state.round_robin_hub_id;
      await act(async()=>{tree=create(h(Component,withCallbacks()));});
      assert.equal(Boolean(button(tree,'Keep playing')),!linked,'Only standalone completed sessions offer Keep playing');
      assert.equal(reads,1,'Completed linked status needs no extra hub request');
      assert.equal(writes.length,0);
      await act(async()=>tree.unmount());
    }
  }
  for (const Component of [Standings,PublicStandings]) {
    for(const linked of [true,false]) {
      setup();
      if(Component===PublicStandings) state.version=1;
      state.status=state.event.status='completed';state.event.rounds[0].status='saved';
      if(!linked) delete state.round_robin_hub_id;
      await act(async()=>{tree=create(h(Component,withCallbacks()));});
      assert.equal(Boolean(button(tree,'Keep playing · Round 2')),!linked,'Completed hub standings cannot restart a closed group');
      assert.equal(reads,1,'Completed standings need no extra hub request');
      assert.equal(writes.length,0);
      await act(async()=>tree.unmount());
    }
  }
  for (const Component of [Runner,PublicRunner,Standings,PublicStandings]) {
    for (const started of [null,'assigned','reserved','partial','played','dispatched','round_dispatched','later_round_dispatched']) {
      setup();
      const expectedVersion=Component===PublicRunner||Component===PublicStandings?1:'v1';
      state.version=expectedVersion;
      state.current_round_number=state.event.currentRoundNumber=2;
      state.event.roundRobinHubId='hub';state.event.hubAutoAdvancedFromRound=1;
      state.event.rounds[0].status='saved';
      state.event.rounds[0].matches[1].scoreA=11;state.event.rounds[0].matches[1].scoreB=7;
      state.event.rounds[1].status='active';
      const match=state.event.rounds[1].matches[0];match.status='scheduled';
      if(started==='assigned') state.event.hubAssignedMatchIds=[match.id];
      if(started==='reserved') match.reservedCourtLabel='Main';
      if(started==='partial') match.scoreA=0;
      if(started==='played') match.status='played';
      if(started==='dispatched') match.hubStartedAt='2026-10-20T12:00:00Z';
      if(started==='round_dispatched') state.event.rounds[1].hubStartedAt='2026-10-20T12:00:00Z';
      if(started==='later_round_dispatched') state.event.rounds.push({
        number:3,status:'preview',hubStartedAt:'2026-10-20T12:00:00Z',
        matches:[{id:'replacement-r3',status:'scheduled',sideA:['p1','p3'],sideB:['p2','p4']}],
      });
      await act(async()=>{tree=create(h(Component,withCallbacks({roundNumber:2})));});
      assert.equal(Boolean(button(tree,'Finish session')),!started,`${Component.name}: Finish only while automatic next round is untouched (${started})`);
      assert.equal(button(tree,'Next scheduled round'),undefined,'Already queued next round needs no additional advance');
      if(!started) {
        await act(async()=>button(tree,'Finish session').props.onClick());
        assert.equal(writes.length,1);
        assert.ok(writes[0].url.endsWith('/complete'));
        assert.equal(writes[0].body.expected_version,expectedVersion);
      }
      await act(async()=>tree.unmount());
    }
  }
  setup();state.scoring_mode=state.event.scoringMode='unscored';
  state.current_round_number=state.event.currentRoundNumber=2;
  state.event.roundRobinHubId='hub';state.event.hubAutoAdvancedFromRound=1;
  state.event.rounds[0].status='played';state.event.rounds[1].status='active';
  await act(async()=>{tree=create(h(Runner,withCallbacks({roundNumber:2})));});
  assert.ok(button(tree,'Finish session'),'Unscored sessions can finish at the untouched automatic boundary');
  await act(async()=>tree.unmount());
  const Arrivals = load('@/components/GeneratorLateArrivals').default;
  let arrivalRequest;
  await act(async()=>{tree=create(h(Arrivals,{participants:Array.from({length:4},(_,i)=>({id:`a${i}`,name:`Arrival ${i}`,active_from_round:2})),roundNumber:1,totalRounds:3,generatorKind:'round_robin',playFormat:'doubles',roundStatus:'active',courtCount:2,usedCourts:[1,2],busy:false,queueOnly:true,onStart:async(ids,court)=>{arrivalRequest={ids,court};}}));});
  assert.equal(tree.root.findAllByProps({'aria-label':'Spare court'}).length,0,'Hub arrivals do not bypass court assignment');
  await act(async()=>button(tree,'Queue late-arrival game').props.onClick());
  assert.deepEqual(arrivalRequest,{ids:['a0','a1','a2','a3'],court:3});
  await act(async()=>tree.unmount());

  const rosterModule = load('@/components/GeneratorRosterSetup');
  stubs['@/components/GeneratorRosterSetup'] = {...rosterModule,__esModule:true,default:()=>null};
  const recoveryStorage=new Map(),drafts=new Map();
  global.window={sessionStorage:{getItem:key=>recoveryStorage.get(key)||null,setItem:(key,value)=>recoveryStorage.set(key,value),removeItem:key=>recoveryStorage.delete(key)}};
  global.sessionStorage = global.window.sessionStorage;
  const baseDraft='admin-play-generator-draft:club:round_robin';
  const savedDraft=()=>({title:'New group',playFormat:'doubles',ratingMode:'unrated',targetCount:4,
    participantText:'One\nTwo\nThree\nFour',linkedPlayerIds:{two:22,four:44},preview:{...clone(fixture.event),previewFingerprint:'preview'}});
  stubs['@/lib/playGeneratorDraft'] = {
    readPlayGeneratorDraft:key=>drafts.get(key)||null,
    writePlayGeneratorDraft:(key,value)=>drafts.set(key,clone(value)),clearPlayGeneratorDraft:key=>drafts.delete(key)
  };
  const Workspace = load('@/app/admin/play-generators/GeneratorWorkspace').default;
  const workspaceProps=(hub,draft)=>({apiBase:'https://api.invalid',clubId:'club',generatorKind:'round_robin',status:{enabled:true,writes_enabled:true},returnHubId:hub,draftToken:draft});
  const scopedKey=(hub,draft)=>hub||draft?`${baseDraft}:hub:${encodeURIComponent(hub||'')}:draft:${encodeURIComponent(draft||'legacy')}`:baseDraft;
  const resetDraft=(hub,draft)=>{recoveryStorage.clear();drafts.clear();drafts.set(scopedKey(hub,draft),savedDraft());navigation.length=0;};
  const response=(body,status=200)=>({ok:status<400,status,json:async()=>clone(body)});
  for (const hub of ['hub/2',undefined]) {
    resetDraft(hub);const creates=[];
    global.fetch=async(url,options={})=>{
      if(options.method==='POST') creates.push({url,body:JSON.parse(options.body)});
      return response(options.method==='POST'?{session:{session_key:'new/rr',current_round_number:1}}:{sessions:[{session_key:'old',title:'Existing group',round_robin_hub_id:'known-hub',current_round_number:1,status:'active'}]});
    };
    const props=workspaceProps(hub);
    await act(async()=>{tree=create(h(Workspace,props));});
    assert.ok(tree.root.findAllByType('a').some(link=>link.props.href==='/admin/round-robin-hub?hub=known-hub'),'Previously created linked sessions have a direct hub link');
    await act(async()=>button(tree,hub?'Create round robin':'Start unrated session').props.onClick());
    assert.equal(creates.length,1,'Starting creates exactly one round robin');
    assert.equal(creates[0].body.rating_mode,'unrated');
    assert.equal(creates[0].body.preview_fingerprint,hub?null:'preview');
    if(hub) { assert.deepEqual(creates[0].body.participant_names,[]); assert.equal(creates[0].body.registration_hub_id,hub); }
    assert.deepEqual(creates[0].body.participant_player_ids,hub?undefined:{Two:22,Four:44},'Guest entries preserve the selected profiles in mixed rosters');
    assert.equal(Object.hasOwn(creates[0].body,'player_ids'),false,'New starts send explicit sparse links rather than an all-or-nothing ID list');
    assert.deepEqual(navigation,[hub?'/admin/round-robin-hub?hub=hub%2F2&add_session=new%2Frr':'/admin/round-robin-generator/sessions/new%2Frr/rounds/1']);
    const runLabel=hub?'Run this round robin in the hub':'Run this session';
    await act(async()=>button(tree,runLabel).props.onClick());
    assert.equal(creates.length,1,'Cached success navigates without creating again');
    assert.equal(button(tree,'Preview matchups'),undefined,'Known success cannot be silently recreated by editing');
    await act(async()=>tree.unmount());
    await act(async()=>{tree=create(h(Workspace,props));});
    await act(async()=>button(tree,runLabel).props.onClick());
    assert.equal(creates.length,1,'Reloaded success receipt navigates without POST');
    await act(async()=>button(tree,'Create another round robin').props.onClick());
    const nextToken=new URL(navigation.at(-1),'https://test.local').searchParams.get('draft');
    assert.ok(nextToken);
    await act(async()=>tree.update(h(Workspace,workspaceProps(hub,nextToken))));
    assert.equal(button(tree,runLabel),undefined);
    assert.equal(drafts.get(scopedKey(hub,nextToken)).participantText,'','Explicit new token starts a fresh roster');
    assert.equal(drafts.get(scopedKey(hub,nextToken)).preview,null);
    await act(async()=>tree.unmount());
  }

  for (const failure of ['intent503','lost','uncertain409','rejected422']) {
    const hub='retry-hub',token=failure;resetDraft(hub,token);const creates=[];
    global.fetch=async(url,options={})=>{
      if(options.method!=='POST') return response({sessions:[]});
      creates.push(JSON.parse(options.body));
      if(creates.length===1) {
        if(failure==='lost') throw new TypeError('Response lost');
        return response({detail:failure==='intent503'?'The durable operation intent could not be persisted. No domain mutation was attempted.':failure==='uncertain409'?'The outcome is uncertain.':'Review the roster.'},failure==='intent503'?503:failure==='uncertain409'?409:422);
      }
      return response({session:{session_key:'confirmed',current_round_number:1}});
    };
    const props=workspaceProps(hub,token),key=scopedKey(hub,token);
    await act(async()=>{tree=create(h(Workspace,props));});
    await act(async()=>button(tree,'Create round robin').props.onClick());
    const alert=tree.root.findByProps({role:'alert'});
    assert.equal(alert.props.style.color,'#b91c1c','Failed starts show an explicit error regardless of wording');
    assert.ok(text(alert).includes('draft'));
    assert.equal(drafts.get(key).participantText,'One\nTwo\nThree\nFour');
    assert.ok(drafts.get(key).preview,'Failed starts preserve the preview');
    if(failure==='rejected422') {
      assert.equal(recoveryStorage.has(`${key}:start`),false,'An initially rejected request can be corrected');
      assert.ok(button(tree,'Create round robin'));
    } else {
      assert.equal(button(tree,'Create another round robin'),undefined);
      assert.equal(tree.root.findByType('fieldset').props.disabled,true);
      await act(async()=>tree.unmount());
      await act(async()=>{tree=create(h(Workspace,props));});
      await act(async()=>button(tree,'Retry this start').props.onClick());
      assert.deepEqual(creates[1],creates[0],'Reloaded retry retains exact body and key');
      assert.ok(button(tree,'Run this round robin in the hub'));
    }
    await act(async()=>tree.unmount());
  }

  resetDraft('auth-hub','auth-retry');const authRequests=[];
  global.fetch=async(url,options={})=>{
    if(options.method!=='POST') return response({sessions:[]});
    authRequests.push(JSON.parse(options.body));
    if(authRequests.length===1) throw new TypeError('Response lost after commit');
    if(authRequests.length===2) return response({detail:'Sign in again.'},401);
    return response({session:{session_key:'original-created',current_round_number:1}});
  };
  await act(async()=>{tree=create(h(Workspace,workspaceProps('auth-hub','auth-retry')));});
  await act(async()=>button(tree,'Create round robin').props.onClick());
  await act(async()=>button(tree,'Retry this start').props.onClick());
  assert.ok(button(tree,'Retry this start'),'Rejected retry cannot discard earlier uncertainty');
  assert.equal(button(tree,'Create round robin').props.disabled,true);
  await act(async()=>button(tree,'Retry this start').props.onClick());
  assert.deepEqual(authRequests[1],authRequests[0]);assert.deepEqual(authRequests[2],authRequests[0]);
  assert.equal(navigation.at(-1),'/admin/round-robin-hub?hub=auth-hub&add_session=original-created');
  await act(async()=>tree.unmount());

  resetDraft(undefined,'old-preview');let resolvePreview;
  global.fetch=async(url)=>url.endsWith('/preview')?new Promise(resolve=>{resolvePreview=()=>resolve(response({preview:savedDraft().preview}));}):response({sessions:[]});
  await act(async()=>{tree=create(h(Workspace,workspaceProps(undefined,'old-preview')));});
  await act(async()=>button(tree,'Preview matchups').props.onClick());
  await act(async()=>tree.update(h(Workspace,workspaceProps(undefined,'new-preview'))));
  await act(async()=>resolvePreview());
  assert.equal(button(tree,'Start unrated session'),undefined,'Old preview response cannot populate a new draft');
  assert.equal(drafts.get(scopedKey(undefined,'new-preview')).preview,null);
  await act(async()=>tree.unmount());

  resetDraft(undefined,'mixed-preview');const previewBodies=[];
  global.fetch=async(url,options={})=>{
    if(url.endsWith('/preview')) {previewBodies.push(JSON.parse(options.body));return response({preview:savedDraft().preview});}
    return response({sessions:[]});
  };
  await act(async()=>{tree=create(h(Workspace,workspaceProps(undefined,'mixed-preview')));});
  await act(async()=>button(tree,'Preview matchups').props.onClick());
  assert.deepEqual(previewBodies[0].participant_names,['One','Two','Three','Four']);
  assert.deepEqual(previewBodies[0].participant_player_ids,{Two:22,Four:44},'Preview preserves the same sparse profiles as Start');
  await act(async()=>tree.unmount());
  console.log('PASS hub generator integration: embedded controls, sequential standings, safe refresh, roster controls, and hub/standalone creation destinations.');
})().catch(error=>{console.error(error);process.exitCode=1;});
