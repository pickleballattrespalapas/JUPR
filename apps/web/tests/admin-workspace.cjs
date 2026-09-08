const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const ts = require('typescript'), React = require('react'), {create,act} = require('react-test-renderer');
function load(file, mocks={}) {
 const output=ts.transpileModule(fs.readFileSync(path.join(__dirname,'..',file),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
 const mod={exports:{}};new Function('require','module','exports',output)(name=>Object.hasOwn(mocks,name)?mocks[name]:require(name),mod,mod.exports);return mod.exports;
}
const events=new Map(),docEvents=new Map();let cookie='',destination='',pathname='/admin';
global.crypto=require('node:crypto').webcrypto;global.localStorage={setItem(){}};global.location={protocol:'https:'};
global.document={get cookie(){return cookie;},set cookie(value){cookie=value.split(';')[0];},addEventListener:(n,f)=>docEvents.set(n,f),removeEventListener:n=>docEvents.delete(n)};
global.window={location:{assign:v=>destination=v},addEventListener:(n,f)=>events.set(n,f),removeEventListener:n=>events.delete(n),dispatchEvent:e=>events.get(e.type)?.(e),setInterval:()=>1,clearInterval(){}};
global.Element=class{closest(){return null;}};
const workspace=load('lib/adminWorkspace.ts'), context=load('lib/useAdminWorkspace.tsx');
const alpha={club_id:'alpha',club_slug:'alpha-club',club_name:'Alpha Club',roles:['administrator']},beta={club_id:'beta',club_slug:'beta-club',club_name:'Beta Club',roles:['operator']};
const selected={clubId:'beta',clubSlug:'beta-club'},link=({children,...props})=>React.createElement('a',props,children);
let session={user:{id:'user-a'},capabilities:{assignments:[{club_id:'alpha',role:'administrator'},{club_id:'beta',role:'operator'}]}},tree;
let availableClubs=[alpha,beta], workspaceError='', workspaceLoaded=true;
const availableMock={useAvailableWorkspaces:()=>({workspaces:availableClubs,loaded:workspaceLoaded,error:workspaceError,retry(){}})};
const redirects=[];
const auth={getAdminApiBaseUrl:()=> 'https://test.invalid',signOutAdminSession:async()=>{}};
(async()=>{
 for(const value of [undefined,'bad','%ZZ',encodeURIComponent(JSON.stringify({clubId:'../a',clubSlug:'a'})),encodeURIComponent(JSON.stringify({clubId:'a',clubSlug:'a/x'}))])assert.equal(workspace.parseAdminWorkspace(value),null);
 workspace.selectAdminWorkspace(beta);assert.deepEqual(workspace.readBrowserWorkspace(),selected);assert.equal(destination,'/admin','Switch clears old record paths and query parameters');
 const server=load('lib/adminWorkspaceServer.ts',{'server-only':{},'next/headers':{cookies:()=>({get:()=>({value:cookie.split('=')[1]})})},'next/navigation':{redirect:d=>{throw Error(d)}},'./adminWorkspace':workspace});
 assert.deepEqual(server.requireAdminWorkspace(),selected);cookie='';assert.throws(()=>server.requireAdminWorkspace(),/select-club/);workspace.selectAdminWorkspace(beta);
 const Shell=load('components/AdminShell.tsx',{'next/link':link,'next/navigation':{usePathname:()=>pathname,useRouter:()=>({replace:v=>redirects.push(v),refresh(){}})},'@/lib/adminAuthClient':auth,'@/lib/useAdminSession':{useAdminSession:()=>({accessToken:'fixture',session,loading:false})},'@/lib/useAvailableWorkspaces':availableMock,'@/lib/useAdminWorkspace':context,'@/lib/adminWorkspace':workspace,'./AdminShell.module.css':{}}).default;
 let mounted=0;function Child(){const {clubId}=context.useAdminWorkspace();mounted++;return React.createElement('input',{'data-club':clubId,defaultValue:'unsaved draft'});}
 await act(async()=>tree=create(React.createElement(Shell,{workspace:selected},React.createElement(Child))));
 assert.equal(tree.root.findByType('input').props['data-club'],'beta');
 const links=tree.root.findAllByType('a').map(n=>n.props.href).filter(h=>h.startsWith('/clubs/'));assert.ok(links.length>=4);assert.ok(links.every(h=>h.startsWith('/clubs/beta-club')));
 cookie=`${workspace.ADMIN_WORKSPACE_COOKIE}=${encodeURIComponent(JSON.stringify({clubId:'alpha',clubSlug:'alpha-club'}))}`;
 let prevented=false,stopped=false;await act(async()=>docEvents.get('click')({target:new Element(),preventDefault(){prevented=true},stopPropagation(){stopped=true}}));
 assert.ok(prevented&&stopped,'A cross-tab change blocks the next click');assert.equal(tree.root.findAllByType('input').length,0);await act(async()=>tree.unmount());
 workspace.selectAdminWorkspace(beta);session.capabilities.assignments=[{club_id:'alpha',role:'administrator'}];
 await act(async()=>tree=create(React.createElement(Shell,{workspace:selected},React.createElement(Child))));
 assert.equal(tree.root.findAllByType('input').length,0,'Revoked assignment cannot mount protected controls');assert.equal(mounted,1);await act(async()=>tree.unmount());
 const Picker=load('app/admin/select-club/page.tsx',{'next/link':link,'@/lib/useAdminSession':{useAdminSession:()=>({accessToken:'fixture',session,loading:false})},'@/lib/useAvailableWorkspaces':availableMock,'@/lib/adminWorkspace':workspace}).default;
 availableClubs=[alpha];destination='';
 await act(async()=>tree=create(React.createElement(Picker)));
 assert.equal(destination,'/admin','A single-club account opens automatically');assert.equal(workspace.readBrowserWorkspace().clubId,'alpha');await act(async()=>tree.unmount());
 await act(async()=>tree=create(React.createElement(Shell,{workspace:{clubId:'alpha',clubSlug:'alpha-club'}},React.createElement(Child))));
 assert.equal(tree.root.findAllByType('a').some(n=>n.children.includes('Switch club')),false,'Single-club navigation has no switcher');await act(async()=>tree.unmount());
 await act(async()=>tree=create(React.createElement(Shell,{workspace:selected},React.createElement(Child))));
 assert.equal(redirects.at(-1),'/admin/select-club','Previous account workspace is replaced via automatic selection');assert.equal(tree.root.findAllByType('input').length,0);await act(async()=>tree.unmount());
 for (const clubs of [[alpha,beta],[{...alpha,roles:['super_admin']}],[]]) {
   availableClubs=clubs;destination='';await act(async()=>tree=create(React.createElement(Picker)));
   assert.equal(destination,'','Multiple clubs, Super Admin, or no access never auto-select');await act(async()=>tree.unmount());
 }
 availableClubs=[alpha];workspaceError='Unable to verify clubs';destination='';
 await act(async()=>tree=create(React.createElement(Picker)));assert.equal(destination,'','Failed access check does not auto-select');await act(async()=>tree.unmount());workspaceError='';
 workspaceLoaded=false;await act(async()=>tree=create(React.createElement(Picker)));assert.equal(destination,'','Pending access check does not auto-select');await act(async()=>tree.unmount());workspaceLoaded=true;
 availableClubs=[alpha,beta];session.capabilities.assignments=[{club_id:'alpha',role:'administrator'},{club_id:'beta',role:'operator'}];workspace.selectAdminWorkspace(beta);
 const requests=[],Panel=()=>null;
 const find=(node,type)=>node?.type===type?node:[].concat(node?.props?.children||[]).map(child=>find(child,type)).find(Boolean);
 const Tournaments=load('app/admin/tournaments/page.tsx',{'@/lib/adminWorkspaceServer':server,'next/link':link,'@/lib/adminTournamentApi':{getAdminTournamentApiBaseUrl:()=>'/api',getAdminTournamentStatus:async id=>(requests.push(id),{data:{status:'enabled'}})},'./TournamentAdminPanel':Panel}).default;
 assert.equal(find(await Tournaments(),Panel).props.clubId,'beta');assert.deepEqual(requests,['beta']);
 const Uploader=load('app/admin/match-uploader/page.tsx',{'@/lib/adminWorkspaceServer':server,'next/link':link,'@/lib/api':{getClubPlayers:async slug=>(requests.push(slug),{data:{players:[]}})},'@/lib/adminMatchUploaderApi':{getAdminMatchUploaderApiBaseUrl:()=>'/api',getAdminMatchUploaderStatus:async id=>(requests.push(id),{data:{status:'enabled'}})},'./MatchUploaderForm':Panel}).default;
 assert.equal(find(await Uploader(),Panel).props.clubId,'beta');assert.deepEqual(requests.slice(1),['beta-club','beta']);
 const authClient=load('lib/adminAuthClient.ts');process.env.NEXT_PUBLIC_JUPR_API_BASE_URL='https://test.invalid';const urls=[];
 global.fetch=async url=>(urls.push(url),{ok:true,status:200,json:async()=>({authorized:true,user:{},assignments:[{club_id:'beta',role:'operator'}]})});
 await authClient.authorizeAdminSession({access_token:'fixture'});await authClient.authorizeAdminSession({access_token:'fixture'},'beta');assert.deepEqual(urls,['https://test.invalid/admin/auth/capabilities','https://test.invalid/admin/auth/capabilities?club_id=beta']);
 const available=load('lib/useAvailableWorkspaces.ts',{'./adminAuthClient':auth});let oldResponse,refreshResponse;
 global.fetch=(url,r)=>r.headers.Authorization==='Bearer old'?new Promise(resolve=>oldResponse=resolve):r.headers.Authorization==='Bearer refreshed'?new Promise(resolve=>refreshResponse=resolve):Promise.resolve({ok:true,json:async()=>({workspaces:[beta]})});
 function Read({token,identity}){const v=available.useAvailableWorkspaces(token,identity);return React.createElement('span',null,v.workspaces.map(w=>w.club_id).join(','));}
 await act(async()=>tree=create(React.createElement(Read,{token:'old',identity:'old-user'})));await act(async()=>tree.update(React.createElement(Read,{token:'new',identity:'new-user'})));
 await act(async()=>oldResponse({ok:true,json:async()=>({workspaces:[alpha]})}));assert.deepEqual(tree.toJSON().children,['beta'],'Late response cannot overwrite another identity');
 await act(async()=>tree.update(React.createElement(Read,{token:'refreshed',identity:'new-user'})));assert.deepEqual(tree.toJSON().children,['beta'],'Token refresh preserves the same workspace');
 await act(async()=>refreshResponse({ok:true,json:async()=>({workspaces:[beta]})}));await act(async()=>tree.unmount());
 console.log('Multi-club login, selection, authorization, cross-tab protection, scoped route data and token refresh passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
