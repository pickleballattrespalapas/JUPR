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
const member={id:'member-1',club_id:'away',name:'Late Traveler',player_id:'12',revision:4,approval_status:'pending',late_join:true};
async function reviews(){
 let requests=[],finish,members=[member,{...member,id:'timely',name:'Timely Unlinked',late_join:false,player_id:null}];
 global.fetch=async(url,options)=>{requests.push({url,options});if(options.method==='POST')return new Promise(resolve=>{finish=resolve});return reply({members});};
 let tree;await act(async()=>{tree=create(React.createElement(Panel,props));});
 assert.equal(requests[0].url,root+'/pool/approvals');assert.equal(requests[0].options.headers.Authorization,'Bearer token-one');assert.equal(requests[0].options.cache,'no-store');
 assert.ok(text(tree).includes('Late Traveler')&&text(tree).includes('Away Club'));assert.ok(!text(tree).includes('Timely Unlinked'));
 assert.ok(text(tree).includes('league organizer')&&text(tree).includes('contact details'));
 assert.equal(button(tree,'Approve season eligibility').props.disabled,true);
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'  Visitor arriving in February  '}}));
 await act(async()=>{void button(tree,'Approve season eligibility').props.onClick();void button(tree,'Approve season eligibility').props.onClick();});
 assert.equal(requests.filter(r=>r.options.method==='POST').length,1);
 assert.deepEqual(JSON.parse(requests.at(-1).options.body),{member_id:'member-1',expected_revision:4,approve:true,reason:'Visitor arriving in February'});assert.ok(!requests.at(-1).options.body.includes('email'));
 members=[{...member,revision:5,approval_status:'approved'}];await act(async()=>finish(reply({member:members[0]})));
 assert.ok(text(tree).includes('approved for the season pool'));assert.ok(text(tree).includes('No late signups are waiting'));
 members=[{...member,revision:6}];await act(async()=>button(tree,'Refresh eligibility requests').props.onClick());
 await act(async()=>tree.update(React.createElement(Panel,{...props,accessToken:'token-two'})));
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Wrong club request'}}));await act(async()=>button(tree,'Do not approve').props.onClick());
 assert.equal(requests.at(-1).options.headers.Authorization,'Bearer token-two');assert.deepEqual(JSON.parse(requests.at(-1).options.body),{member_id:'member-1',expected_revision:6,approve:false,reason:'Wrong club request'});
 await act(async()=>finish(reply({detail:'This signup changed. Reload before continuing.'},409)));
 assert.equal(button(tree,'Do not approve').props.disabled,true);assert.equal(button(tree,'Approve season eligibility').props.disabled,true);assert.ok(text(tree).includes('signup changed'));
 members=[{...member,revision:7}];await act(async()=>button(tree,'Refresh eligibility requests').props.onClick());
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Organizer declined'}}));await act(async()=>button(tree,'Do not approve').props.onClick());
 members=[{...member,revision:8,approval_status:'rejected'}];await act(async()=>finish(reply({member:members[0]})));
 assert.ok(text(tree).includes('not approved'));assert.ok(text(tree).includes('No late signups are waiting'));await act(async()=>tree.unmount());
}
async function unlinkedAndStale(){
 let finish,lastSignal;global.fetch=async(url,options)=>{if(options.method){lastSignal=options.signal;return new Promise(resolve=>{finish=resolve})}return reply({members:[{...member,player_id:null}]});};
 let tree;await act(async()=>{tree=create(React.createElement(Panel,props));});assert.ok(text(tree).includes('must link this signup'));
 await act(async()=>tree.root.findByType('input').props.onChange({target:{value:'Not eligible'}}));assert.equal(button(tree,'Approve season eligibility').props.disabled,true);
 await act(async()=>button(tree,'Do not approve').props.onClick());await act(async()=>tree.update(React.createElement(Panel,{...props,root:root.replace('season-1','season-2')})));
 assert.equal(lastSignal.aborted,true);await act(async()=>finish(reply({detail:'OLD SEASON ERROR'},422)));
 assert.ok(!text(tree).includes('OLD SEASON ERROR'));assert.ok(!text(tree).includes('not approved'));await act(async()=>tree.unmount());
}
(async()=>{await reviews();await unlinkedAndStale();console.log('PASS interclub eligibility approvals: revisions, review, privacy, late-only queue and stale context');})().catch(error=>{console.error(error);process.exit(1)});
