/*
 * Loopback-only E2E: actual Next UI -> FastAPI routes -> generator/hub services
 * -> transactional RPC double. Two seeded round robins; check-in and review
 * changes use the UI. Does not test real authentication, RLS or PostgreSQL locks.
 * Run from apps/web: node tests/hub-registration-browser.cjs
 * GENERATOR_BROWSER_{PYTHON,EXECUTABLE,ARGS,ARTIFACTS,WEB_PORT,API_PORT} supported.
 */
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const webRoot = path.resolve(__dirname, '..');
const root = path.resolve(webRoot, '../..');
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || '3072';
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || '8072';
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(), 'round-robin-hub-browser-'));
fs.mkdirSync(output, {recursive:true});
const logs = [], errors = [], evidence = [];
function start(command, args, env, cwd) {
  const child = spawn(command, args, {env, cwd, stdio:['ignore', 'pipe', 'pipe']});
  child.stdout.on('data', data => logs.push(data.toString()));
  child.stderr.on('data', data => logs.push(data.toString()));
  child.on('error', error => { child.startupError = error; logs.push(error.message); });
  return child;
}
async function ready(url, children) {
  for (let i=0; i<120; i++) {
    for (const child of children) {
      if (child.startupError) throw child.startupError;
      if (child.exitCode !== null) throw new Error(`Local server exited (${child.exitCode})`);
    }
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error(`Local server unavailable: ${url}`);
}
async function fixture() {
  const response = await fetch(`${apiOrigin}/hub-fixture-result`);
  assert(response.ok, `fixture state: ${response.status}`);
  return response.json();
}
async function clickRequest(page, button, suffix, action) {
  const pending = page.waitForResponse(response => response.url().startsWith(apiOrigin)
    && response.url().endsWith(suffix) && response.request().method() === 'POST'
    && (!action || response.request().postDataJSON()?.action === action));
  await button.click();
  const response = await pending;
  const body = await response.json();
  assert(response.ok(), `${action || suffix}: ${response.status()} ${JSON.stringify(body)}`);
  return body;
}
async function command(page, button, action) {
  return clickRequest(page, button, '/commands', action);
}
function record(label) { evidence.push(label); console.log(`PASS ${label}`); }
async function snapshot(page, name) { await page.screenshot({path:path.join(output, name), fullPage:true}); }
async function main() {
  const env = {...process.env, VERCEL_ENV:'development', JUPR_API_BASE_URL:apiOrigin,
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging', GENERATOR_BROWSER_WEB_PORT:webPort};
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m', 'uvicorn', 'hub_registration_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort], env, root);
  const next = start(process.execPath, [require.resolve('next/dist/bin/next'), 'dev', '--webpack', '--hostname', '127.0.0.1', '--port', webPort], env, webRoot);
  let browser, page;
  try {
    await ready(`${apiOrigin}/hub-fixture-result`, [api,next]);
    assert.equal((await fixture()).sessions.length, 2);
    await ready(`${webOrigin}/clubs/test/round-robin-generator`, [api,next]);
    browser = await chromium.launch({headless:true,
      ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE} : {}),
      args:['--no-sandbox','--disable-dev-shm-usage',...JSON.parse(process.env.GENERATOR_BROWSER_ARGS || '[]')]});
    const context = await browser.newContext({viewport:{width:1440,height:1000}});
    await context.route('**/*', route => ['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    context.on('page', p => {
      p.on('pageerror', error => errors.push({type:'pageerror',message:error.message}));
      p.on('console', message => { if (message.type() === 'error') errors.push({type:'console',message:message.text()}); });
    });
    await context.addCookies([{name:'jupr_admin_workspace_v1',value:encodeURIComponent(JSON.stringify({clubId:'club',clubSlug:'test'})),url:webOrigin}]);
    await context.addInitScript(origin => {
      if (window.location.origin !== origin) return;
      localStorage.setItem('jupr_admin_session_v1',JSON.stringify({access_token:'local-reviewer',expires_at:Date.now()+3600000,
        user:{id:'local-reviewer',email:'reviewer@example.invalid'},capabilities:{authorized:true,user:{email:'reviewer@example.invalid'},
          assignments:[{club_id:'club',role:'administrator',permissions:['enter_scores','manage_players']}]}}));
    }, webOrigin);
    page = await context.newPage();
    const state = await fixture();
    const hubId = state.hubs[0].hub.id;
    await page.goto(`${webOrigin}/admin/round-robin-hub?hub=${hubId}`);
    await page.getByRole('tab',{name:'Registration',exact:true}).click();
    await command(page,page.getByRole('button',{name:'Open check-in',exact:true}),'set_registration');
    const url = await page.getByLabel('Public check-in link').inputValue();
    assert.equal(url,`${webOrigin}/round-robin-hub/${hubId}/check-in`);
    const guestContext = await browser.newContext({viewport:{width:390,height:844}});
    const guest = await guestContext.newPage();
    await guest.goto(url);
    await guest.getByLabel('Full name',{exact:true}).fill('New Arrival');
    await guest.getByLabel('Round robin / level',{exact:true}).selectOption('rr-a');
    await clickRequest(guest,guest.getByRole('button',{name:'Check in',exact:true}),'/checkins');
    await guest.getByRole('heading',{name:'You’re checked in, New Arrival.'}).waitFor();
    await snapshot(guest,'mobile-checkin-confirmed.png');
    assert.equal(await guest.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
    await page.bringToFront();
    await page.getByRole('tab',{name:'Registration (1)',exact:true}).waitFor({timeout:25000});
    const arrival = page.locator('article').filter({hasText:'New Arrival'});
    await arrival.getByRole('button',{name:'Add new arrival',exact:true}).click();
    const dialog = page.getByRole('dialog',{name:'Add player',exact:true});
    assert.equal(await dialog.getByLabel('Player name or club profile',{exact:true}).inputValue(),'New Arrival');
    assert.equal(await dialog.getByLabel('Round robin',{exact:true}).inputValue(),'rr-a');
    await command(page,dialog.getByRole('button',{name:'Add player to round robin',exact:true}),'review_checkin');
    await dialog.waitFor({state:'hidden'});
    let result = (await fixture()).hubs[0];
    assert.equal(result.workspace.groups.find(g=>g.session_key==='rr-a').players.length,5);
    assert.equal(result.workspace.groups.find(g=>g.session_key==='rr-b').players.length,4);
    assert.equal(result.registration.pending_count,0);
    record('Unsigned mobile check-in arrives automatically; staff adds prefilled arrival only to the selected round robin');
    await guest.getByRole('button',{name:'Check in another player',exact:true}).click();
    assert.equal(await guest.getByLabel('Full name',{exact:true}).inputValue(),'');
    await guest.getByLabel('Full name',{exact:true}).fill('new arrival');
    await guest.getByLabel('Round robin / level',{exact:true}).selectOption('rr-a');
    await clickRequest(guest,guest.getByRole('button',{name:'Check in',exact:true}),'/checkins');
    assert.equal((await fixture()).hubs[0].registration.checkins.length,1);
    await guest.getByRole('button',{name:'Check in another player',exact:true}).click();
    await guest.getByLabel('Full name',{exact:true}).fill('Player 11');
    await guest.getByLabel('Round robin / level',{exact:true}).selectOption('rr-b');
    await clickRequest(guest,guest.getByRole('button',{name:'Check in',exact:true}),'/checkins');
    await page.bringToFront();
    await page.getByRole('tab',{name:'Registration (1)',exact:true}).waitFor({timeout:25000});
    await command(page,page.locator('article').filter({hasText:'Player 11'}).getByRole('button',{name:'Confirm on roster',exact:true}),'review_checkin');
    result = (await fixture()).hubs[0];
    assert.equal(result.workspace.groups.find(g=>g.session_key==='rr-b').players.length,4);
    record('Shared-tablet reset, normalized duplicate check-in, and existing-roster confirmation without duplicate players');
    await page.getByRole('combobox',{name:'Show check-ins',exact:true}).selectOption('all');
    await snapshot(page,'admin-checkin-reviewed.png');
    await command(page,page.getByRole('button',{name:'Close check-in',exact:true}),'set_registration');
    await guest.reload();
    await guest.getByText('Check-in is currently closed. Please ask the event organizer.',{exact:true}).waitFor();
    assert.equal(await guest.getByRole('button',{name:'Check in',exact:true}).count(),0);
    record('Closing check-in removes the public form');
    assert.deepEqual(errors,[],'No browser console errors or uncaught exceptions');
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:true,evidence,errors,hubId},null,2));
    console.log(`Check-in browser rehearsal passed. Artifacts: ${output}`);
  } catch (error) {
    if (page) await snapshot(page,'failure.png').catch(()=>{});
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:false,evidence,errors,error:error.stack},null,2));
    throw error;
  } finally {
    if (browser) await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 3000))]);
    api.kill('SIGTERM'); next.kill('SIGTERM');
    fs.writeFileSync(path.join(output,'servers.log'),logs.join(''));
  }
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
