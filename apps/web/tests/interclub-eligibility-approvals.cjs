const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const React=require('react'),ts=require('typescript'),{create,act}=require('react-test-renderer');
const base='app/admin/interclub/registrations/';
function load(file,mocks={}){const code=ts.transpileModule(fs.readFileSync(path.join(__dirname,'..',file),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;const module={exports:{}};new Function('require','module','exports',code)(name=>Object.hasOwn(mocks,name)?mocks[name]:require(name),module,module.exports);return module.exports;}
const resource=load(base+'usePoolResource.ts',{'@/lib/interclubRegistration':load('lib/interclubRegistration.ts')});
const common=load(base+'PoolPanelCommon.tsx',{'./playerPool.module.css':{}});
const Panel=load(base+'SeasonEligibilityApprovals.tsx',{'./usePoolResource':resource,'./PoolPanelCommon':common,'./playerPool.module.css':{}}).default;
const nodeText=node=>typeof node==='string'?node:(node.children||[]).map(nodeText).join('');
const button=(tree,label)=>tree.root.findAllByType('button').find(n=>nodeText(n)===label);
const text=tree=>JSON.stringify(tree.toJSON()),reply=(body,status=200)=>({ok:status<400,status,json:async()=>body});
const root='https://api.test/admin/clubs/organizer/interclub/registrations/season-1';
const props={root,accessToken:'token-one',clubs:[{id:'away',name:'Away Club'}]};
const member={id:'member-1',club_id:'away',name:'Late Traveler',player_id:'12',revision:4,approval_status:'pending',late_join:true,late_request_reason:'Club requests a visiting player arriving in February'};
async function reviews(){
 let requests=[],finish,members=[member,{...member,id:'timely',name:'Timely Unlinked',late_join:false,player_id:null}];
 global.fetch=async(url,options)=>{requests.push({url,options});if(options.method==='POST')return new Promise(resolve=>{finish=resolve});return reply({members});};
 let tree;await act(async()=>{tree=create(React.createElement(Panel,props));});
 assert.equal(requests[0].url,root+'/pool/approvals');assert.equal(requests[0].options.headers.Authorization,'Bearer token-one');assert.equal(requests[0].options.cache,'no-store');
 assert.ok(text(tree).includes('Late Traveler')&&text(tree).includes('Away Club'));assert.ok(!text(tree).includes('Timely Unlinked'));
 assert.ok(text(tree).includes('commissioner')&&text(tree).includes('contact details'));
 assert.ok(text(tree).includes('Club requests a visiting player arriving in February'));assert.ok(text(tree).includes('Cannot play'));assert.equal(button(tree,'Approve season eligibility').props.disabled,true);
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'  Visitor arriving in February  '}}));
 await act(async()=>{void button(tree,'Approve season eligibility').props.onClick();void button(tree,'Approve season eligibility').props.onClick();});
 assert.equal(requests.filter(r=>r.options.method==='POST').length,1);
 assert.deepEqual(JSON.parse(requests.at(-1).options.body),{member_id:'member-1',expected_revision:4,approve:true,reason:'Visitor arriving in February'});assert.ok(!requests.at(-1).options.body.includes('email'));
 members=[{...member,revision:5,approval_status:'approved',approval_reason:'Visitor arriving in February'}];await act(async()=>finish(reply({member:members[0]})));
 assert.ok(text(tree).includes('approved for the season pool'));assert.ok(text(tree).includes('Reviewed late players'));assert.ok(text(tree).includes('Visitor arriving in February'));assert.ok(text(tree).includes('No late player requests are waiting'));
 members=[{...member,revision:6}];await act(async()=>button(tree,'Refresh eligibility requests').props.onClick());
 await act(async()=>tree.update(React.createElement(Panel,{...props,accessToken:'token-two'})));
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Wrong club request'}}));await act(async()=>button(tree,'Do not approve').props.onClick());
 assert.equal(requests.at(-1).options.headers.Authorization,'Bearer token-two');assert.deepEqual(JSON.parse(requests.at(-1).options.body),{member_id:'member-1',expected_revision:6,approve:false,reason:'Wrong club request'});
 await act(async()=>finish(reply({detail:'This signup changed. Reload before continuing.'},409)));
 assert.equal(button(tree,'Do not approve').props.disabled,true);assert.equal(button(tree,'Approve season eligibility').props.disabled,true);assert.ok(text(tree).includes('signup changed'));
 members=[{...member,revision:7}];await act(async()=>button(tree,'Refresh eligibility requests').props.onClick());
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Organizer declined'}}));await act(async()=>button(tree,'Do not approve').props.onClick());
 members=[{...member,revision:8,approval_status:'rejected',approval_reason:'Organizer declined'}];await act(async()=>finish(reply({member:members[0]})));
 assert.ok(text(tree).includes('not approved'));assert.ok(text(tree).includes('Rejected · cannot play'));assert.ok(text(tree).includes('Organizer declined'));assert.ok(text(tree).includes('No late player requests are waiting'));await act(async()=>tree.unmount());
}
async function unlinkedAndStale(){
 let finish,lastSignal;global.fetch=async(url,options)=>{if(options.method){lastSignal=options.signal;return new Promise(resolve=>{finish=resolve})}return reply({members:[{...member,player_id:null}]});};
 let tree;await act(async()=>{tree=create(React.createElement(Panel,props));});assert.ok(text(tree).includes('must link this signup'));
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Not eligible'}}));assert.equal(button(tree,'Approve season eligibility').props.disabled,true);
 await act(async()=>button(tree,'Do not approve').props.onClick());await act(async()=>tree.update(React.createElement(Panel,{...props,root:root.replace('season-1','season-2')})));
 assert.equal(lastSignal.aborted,true);await act(async()=>finish(reply({detail:'OLD SEASON ERROR'},422)));
 assert.ok(!text(tree).includes('OLD SEASON ERROR'));assert.ok(!text(tree).includes('not approved'));await act(async()=>tree.unmount());
}
async function approvalAnchor(){
 let resolve,scrolls=0,focuses=0;
 global.window={location:{hash:'#season-eligibility-approvals'}};
 global.fetch=()=>new Promise(finish=>{resolve=finish;});
 let tree;await act(async()=>{tree=create(React.createElement(Panel,props),{createNodeMock:element=>element.props.id==='season-eligibility-approvals'?{scrollIntoView:()=>scrolls++,focus:options=>{assert.equal(options.preventScroll,true);focuses++;}}:null});});
 assert.equal(scrolls,0,'The deep link waits until its asynchronous queue loads');
 await act(async()=>resolve(reply({members:[]})));
 assert.equal(scrolls,1);assert.equal(focuses,1,'The destination receives keyboard focus');
 await act(async()=>button(tree,'Refresh eligibility requests').props.onClick());
 await act(async()=>resolve(reply({members:[member]})));
 assert.equal(scrolls,1,'Refreshing the queue does not pull the reader back to the anchor');
 await act(async()=>tree.update(React.createElement(Panel,{...props,clubs:[...props.clubs]})));
 assert.equal(scrolls,1,'Ordinary rerenders do not repeat the scroll');
 await act(async()=>tree.unmount());delete global.window;
}
async function refreshedQueuePreservesDecision(){
 let tree,finish,reads=0,decisions=0;
 global.fetch=async(url,options)=>{
  if(options.method)return reply({member:{...member,revision:5,approval_status:'approved'}});
  if(reads++)return new Promise(resolve=>{finish=resolve;});
  return reply({members:[member]});
 };
 const current={...props,refreshKey:0,onDecision:()=>{decisions++;}};
 await act(async()=>{tree=create(React.createElement(Panel,current));});
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Keep this decision reason'}}));
 await act(async()=>tree.update(React.createElement(Panel,{...current,refreshKey:1})));
 assert.equal(tree.root.findByType('input').props.value,'Keep this decision reason','A newly submitted request does not clear another commissioner decision draft');
 await act(async()=>finish(reply({members:[member,{...member,id:'member-2',name:'Second Traveler'}]})));
 assert.equal(tree.root.findAllByType('input')[0].props.value,'Keep this decision reason');
 await act(async()=>button(tree,'Approve season eligibility').props.onClick());
 assert.equal(decisions,1,'A completed decision refreshes the represented club pool');
 await act(async()=>finish(reply({members:[]})));
 await act(async()=>tree.unmount());
}
(async()=>{await reviews();await unlinkedAndStale();await approvalAnchor();await refreshedQueuePreservesDecision();console.log('PASS interclub eligibility approvals: revisions, request reasons and decisions, privacy, late-only queue, background draft preservation, stale context and async deep link');})().catch(error=>{console.error(error);process.exit(1)});
