const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map(), h = React.createElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
const navigation = [];
const router = { push: url => navigation.push(url), replace() {}, refresh() {} };
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
      } else if(url.endsWith('/advance')) {
        state.event.currentRoundNumber=2; state.current_round_number=2; state.event.rounds[1].status='active';
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
  assert.equal(tree.root.findAllByProps({'aria-label':'Open round 2'}).length,0,'Future rounds cannot be selected while linked');
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
  assert.equal(state.current_round_number,1,'Saving complete scores does not start another group round implicitly');
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
  const Arrivals = load('@/components/GeneratorLateArrivals').default;
  let arrivalRequest;
  await act(async()=>{tree=create(h(Arrivals,{participants:Array.from({length:4},(_,i)=>({id:`a${i}`,name:`Arrival ${i}`,active_from_round:2})),roundNumber:1,totalRounds:3,generatorKind:'round_robin',playFormat:'doubles',roundStatus:'active',courtCount:2,usedCourts:[1,2],busy:false,queueOnly:true,onStart:async(ids,court)=>{arrivalRequest={ids,court};}}));});
  assert.equal(tree.root.findAllByProps({'aria-label':'Spare court'}).length,0,'Hub arrivals do not bypass court assignment');
  await act(async()=>button(tree,'Queue late-arrival game').props.onClick());
  assert.deepEqual(arrivalRequest,{ids:['a0','a1','a2','a3'],court:3});
  await act(async()=>tree.unmount());

  const rosterModule = load('@/components/GeneratorRosterSetup');
  stubs['@/components/GeneratorRosterSetup'] = {...rosterModule,__esModule:true,default:()=>null};
  let clearedDrafts=0;
  stubs['@/lib/playGeneratorDraft'] = {
    readPlayGeneratorDraft:()=>({title:'New group',playFormat:'doubles',ratingMode:'unrated',targetCount:4,
      participantText:'One\nTwo\nThree\nFour',linkedPlayerIds:{},preview:{...clone(fixture.event),previewFingerprint:'preview'}}),
    writePlayGeneratorDraft(){},clearPlayGeneratorDraft(){clearedDrafts++;}
  };
  const Workspace = load('@/app/admin/play-generators/GeneratorWorkspace').default;
  for (const hub of ['hub/2',undefined]) {
    navigation.length=0;const creates=[];
    global.fetch=async(url,options={})=>{
      if(options.method==='POST') creates.push({url,body:JSON.parse(options.body)});
      return {ok:true,status:200,json:async()=>options.method==='POST'?{session:{session_key:'new/rr',current_round_number:1}}:{sessions:[]}};
    };
    await act(async()=>{tree=create(h(Workspace,{apiBase:'https://api.invalid',clubId:'club',generatorKind:'round_robin',status:{enabled:true,writes_enabled:true},returnHubId:hub}));});
    await act(async()=>button(tree,hub?'Start and add to hub':'Start unrated session').props.onClick());
    assert.equal(creates.length,1,'Starting creates exactly one round robin');
    assert.equal(creates[0].body.rating_mode,'unrated');
    assert.equal(creates[0].body.preview_fingerprint,'preview');
    assert.deepEqual(navigation,[hub?'/admin/round-robin-hub?hub=hub%2F2&add_session=new%2Frr':'/admin/round-robin-generator/sessions/new%2Frr/rounds/1']);
    await act(async()=>tree.unmount());
  }
  assert.equal(clearedDrafts,2);
  console.log('PASS hub generator integration: embedded controls, sequential standings, safe refresh, roster controls, and hub/standalone creation destinations.');
})().catch(error=>{console.error(error);process.exitCode=1;});
