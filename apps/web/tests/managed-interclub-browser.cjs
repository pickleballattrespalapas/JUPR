/* Real browser -> Next -> FastAPI -> actual migration and transactional PostgreSQL.
   Synthetic local identity/data only; all external browser requests are blocked. */
const { spawn } = require('node:child_process');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('playwright');
const web = path.resolve(__dirname, '..'), root = path.resolve(web, '../..');
const apiPort = process.env.MANAGED_BROWSER_API_PORT || '8044';
const nextMode = process.env.MANAGED_BROWSER_NEXT_MODE || 'dev';
assert.ok(['dev','start'].includes(nextMode));
const origin = 'http://127.0.0.1:3044', apiOrigin = 'http://127.0.0.1:'+apiPort;
const out = process.env.MANAGED_BROWSER_ARTIFACTS || '/tmp/managed-interclub-browser';
fs.mkdirSync(out, { recursive:true });
const logs = [], processes = [];
function start(command, args, cwd, env) {
 const p = spawn(command,args,{cwd,env,stdio:['ignore','pipe','pipe']}); processes.push(p);
 for (const stream of [p.stdout,p.stderr]) stream.on('data',chunk=>logs.push(chunk.toString()));
 return p;
}
async function ready(url) {
 for (let i=0;i<120;i++) { try { if((await fetch(url,{signal:AbortSignal.timeout(2000)})).ok) return; } catch {}
  await new Promise(r=>setTimeout(r,500)); }
 throw Error('Server did not start: '+url);
}
const state = async()=> (await fetch(apiOrigin+'/fixture-result')).json();
(async()=>{
 let browser, page;
 const errors=[];
 try {
  const env={...process.env,JUPR_API_BASE_URL:apiOrigin,NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin,NEXT_PUBLIC_JUPR_ENV:'staging',VERCEL_ENV:'development',NEXT_TELEMETRY_DISABLED:'1'};
  start(process.execPath,['tests/e2e/managed_interclub_db.mjs'],root,env); await ready('http://127.0.0.1:8144');
  start(process.env.MANAGED_BROWSER_PYTHON || 'python',['-m','uvicorn','managed_interclub_api:app','--app-dir','tests/e2e','--host','127.0.0.1','--port',apiPort],root,env);
  start(process.execPath,[require.resolve('next/dist/bin/next'),nextMode,...(nextMode==='dev'?['--webpack']:[]),'--hostname','127.0.0.1','--port','3044'],web,env);
  await ready(apiOrigin+'/fixture-result'); await ready(origin+'/api/environment');
  // Warm all routes before typing, avoiding development compilation refreshes.
  await fetch(origin+'/interclub/host'); await fetch(origin+'/interclub/managed/20000000-0000-4000-8000-000000000001');
  browser=await chromium.launch({headless:true,executablePath:process.env.MANAGED_BROWSER_EXECUTABLE,args:JSON.parse(process.env.MANAGED_BROWSER_ARGS || '["--no-sandbox"]')});
  async function context() {
   const c=await browser.newContext({viewport:{width:1280,height:900}});
   await c.route('**/*',r=>['localhost','127.0.0.1'].includes(new URL(r.request().url()).hostname)?r.continue():r.abort());
   c.on('page',p=>p.on('pageerror',e=>errors.push(e.message))); return c;
  }
  const admin=await context();
  await admin.addCookies([{name:'jupr_admin_workspace_v1',value:encodeURIComponent(JSON.stringify({clubId:'tres_palapas',clubSlug:'test'})),url:origin}]);
  await admin.addInitScript(()=>{
   if(window!==window.top || window.location.hostname!=='127.0.0.1' || localStorage.getItem('jupr_admin_session_v1')) return;
   localStorage.setItem('jupr_admin_session_v1',JSON.stringify({access_token:'local-reviewer',expires_at:Date.now()+3600000,
    user:{id:'10000000-0000-4000-8000-000000000001',email:'joe@example.invalid'},
    capabilities:{authorized:true,assignments:[{club_id:'tres_palapas',role:'administrator',permissions:['enter_scores','manage_players']}]}}));
  });
  page=await admin.newPage(); page.setDefaultTimeout(30000);
  await page.goto(origin+'/admin/interclub');
  await page.getByRole('button',{name:'Create season',exact:true}).click();
  await page.getByRole('heading',{name:'Southern BCS Interclub',exact:true}).waitFor();
  const sid=(await state()).pcs_managed_interclub_seasons[0].id;
  console.log('Created organizer season.');
  await page.getByLabel('Participating club',{exact:true}).fill('La Ribera');
  await page.getByRole('button',{name:'Add club',exact:true}).click();
  for (const [club,prefix] of [['Tres Palapas','Tres'],['La Ribera','Ribera']]) {
   const panel=page.locator('details').filter({has:page.locator('summary').filter({hasText:club})});
   await panel.locator('summary').click();
   await panel.getByLabel('Captain’s email or phone').fill('private@example.invalid');
   await panel.getByLabel('Women',{exact:true}).fill(prefix+' Ana\n'+prefix+' Bea');
   await panel.getByLabel('Men',{exact:true}).fill(prefix+' Carlos\n'+prefix+' Diego');
   await panel.getByRole('button',{name:'Save club & players',exact:true}).click();
   await panel.locator('summary').filter({hasText:'4 players'}).waitFor();
  }
  await page.getByRole('button',{name:'Schedule',exact:true}).click();
  console.log('Saved named club rosters.');
  await page.getByRole('button',{name:'Add meet',exact:true}).click();
  await page.getByLabel('Date & time (America/Mazatlan)',{exact:true}).fill('2099-01-16T10:00');
  await page.getByLabel('Courts',{exact:true}).fill('2');
  await page.getByLabel('Host',{exact:true}).selectOption({label:'La Ribera'});
  for (const checkbox of await page.getByRole('checkbox',{name:'3.5',exact:true}).all()) await checkbox.check();
  await page.getByRole('button',{name:'Save meet & lineups',exact:true}).click();
  await page.getByRole('button',{name:'Generate court schedule',exact:true}).click();
  await page.getByRole('button',{name:'Create host link',exact:true}).click();
  const link=await page.getByLabel('Private host link',{exact:true}).inputValue();
  console.log('Generated schedule and private host link.');
  assert.ok(link.includes('/interclub/host#token='));
  const anonymous=await context(), host=await anonymous.newPage(), stale=await anonymous.newPage();
  const requests=[]; anonymous.on('request',r=>requests.push(r));
  for(const p of [host,stale]) { await p.goto(link); await p.getByRole('button',{name:'Print schedule & score sheets',exact:true}).waitFor(); }
  assert.equal(await host.evaluate(()=>localStorage.getItem('jupr_admin_session_v1')),null);
  assert.ok(!(await host.textContent('body')).includes('private@example.invalid'));
  const print=await host.pdf({path:path.join(out,'host-score-sheets.pdf'),format:'Letter',printBackground:true}); assert.ok(print.length>10000);
  await host.screenshot({path:path.join(out,'host-desktop.png'),fullPage:true});
  await host.setViewportSize({width:390,height:844});
  assert.ok(await host.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'Host page fits mobile');
  await host.screenshot({path:path.join(out,'host-mobile.png'),fullPage:true});
  const scores=host.locator('[data-interclub-score]'); assert.equal(await scores.count(),12);
  for(let i=0;i<12;i++) await scores.nth(i).fill(i%2?'7':'11');
  await host.getByRole('status').filter({hasText:'All scores saved.'}).waitFor();
  assert.ok(await host.getByRole('button',{name:'Submit completed meet',exact:true}).isEnabled());
  const saved=(await state()).pcs_managed_interclub_meets[0];
  assert.equal(saved.state,'draft'); assert.equal(saved.approved_document,null);
  // A stale second scorekeeper cannot silently overwrite the saved sheet.
  await stale.locator('[data-interclub-score]').first().fill('8');
  await stale.getByRole('alert').filter({hasText:'changed'}).waitFor();
  assert.equal((await state()).pcs_managed_interclub_meets[0].revision,saved.revision);
  await stale.close({runBeforeUnload:false});
  await host.reload(); await host.getByRole('button',{name:'Submit completed meet',exact:true}).waitFor();
  assert.equal(await host.locator('[data-interclub-score]').first().inputValue(),'11');
  await host.getByRole('button',{name:'Submit completed meet',exact:true}).click();
  await host.getByText('Submitted for organizer review.',{exact:false}).waitFor();
  await page.goto(origin+'/admin/interclub?season='+sid);
  await page.getByRole('button',{name:'Enter Results',exact:true}).click();
  await page.getByRole('button',{name:'Approve official results',exact:true}).click();
  await page.getByText('These results are official.',{exact:false}).waitFor();
  await page.getByRole('button',{name:'Standings',exact:true}).click();
  await page.getByRole('button',{name:'Publish league page',exact:true}).click();
  const publicPage=await anonymous.newPage(); await publicPage.goto(origin+'/interclub/managed/'+sid);
  await publicPage.getByRole('heading',{name:'Southern BCS Interclub',exact:true}).waitFor();
  assert.ok(!(await publicPage.textContent('body')).includes('private@example.invalid'));
  await publicPage.screenshot({path:path.join(out,'public-league.png'),fullPage:true});
  const final=await state(); assert.equal(final.clubs.length,1,'One club account for the whole league');
  assert.equal(final.pcs_managed_interclub_meets[0].state,'approved');
  assert.deepEqual(final.pcs_managed_interclub_meets[0].approved_document,final.pcs_managed_interclub_meets[0].document);
  assert.ok(final.pcs_managed_interclub_audit.some(a=>a.actor_kind==='host'));
  assert.ok(requests.every(r=>!r.url().includes('token=')),'Private token never enters request URL');
  await page.getByRole('button',{name:'Enter Results',exact:true}).click();
  await page.getByRole('button',{name:'Disable host link',exact:true}).click();
  await page.getByRole('button',{name:'Create host link',exact:true}).waitFor();
  await host.reload(); await host.getByRole('alert').filter({hasText:'disabled'}).waitFor();
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({status:'passed',browser:'chromium',database:'actual migration and RPC in PGlite',games:6,club_accounts:1,host_login:false,checks:['organizer setup','rosters','schedule','print PDF','mobile','autosave','reload persistence','concurrent scorer conflict','submission','approval','public standings','revocation']},null,2));
  console.log('Managed interclub browser → API → PostgreSQL journey passed.');
 } catch(error) { if(page) await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{}); throw error; }
 finally { fs.writeFileSync(path.join(out,'server.log'),logs.join('')); await browser?.close(); for(const p of processes.reverse()) p.kill('SIGTERM'); }
})().catch(e=>{console.error(e);process.exitCode=1;});
