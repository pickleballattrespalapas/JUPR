/*
 * Loopback-only E2E: actual Next UI -> FastAPI routes -> generator/hub services
 * -> transactional RPC double. Starts without sessions; all creation and hub
 * changes use the UI. Covers the hub-first Create round robin return flow.
 * Does not test real authentication, RLS or PostgreSQL locks.
 * Run from apps/web: node tests/round-robin-hub-start-browser.cjs
 * GENERATOR_BROWSER_{PYTHON,EXECUTABLE,ARGS,ARTIFACTS,WEB_PORT,API_PORT,NEXT_MODE}
 * supported. NEXT_MODE=start uses a previously built Next app; its baked-in
 * NEXT_PUBLIC_JUPR_API_BASE_URL must match API_PORT. Defaults to dev.
 */
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const webRoot = path.resolve(__dirname, '..');
const root = path.resolve(webRoot, '../..');
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || '3059';
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || '8059';
const nextMode = process.env.GENERATOR_BROWSER_NEXT_MODE || 'dev';
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || path.join(webRoot, 'test-results', 'round-robin-hub-start');
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
  const [response] = await Promise.all([pending, button.click()]);
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
    ['-m', 'uvicorn', 'round_robin_hub_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort], env, root);
  assert(['dev','start'].includes(nextMode),'GENERATOR_BROWSER_NEXT_MODE must be dev or start');
  const next = start(process.execPath, [require.resolve('next/dist/bin/next'), nextMode, ...(nextMode === 'dev' ? ['--webpack'] : []), '--hostname', '127.0.0.1', '--port', webPort], env, webRoot);
  let browser, page;
  try {
    await ready(`${apiOrigin}/hub-fixture-result`, [api,next]);
    assert.equal((await fixture()).sessions.length, 0);
    await ready(`${webOrigin}/admin/round-robin-hub`, [api,next]);
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
    await page.goto(`${webOrigin}/admin/round-robin-hub`);
    async function createHub(title) {
      if (await page.getByLabel('Event name',{exact:true}).count() === 0)
        await page.getByRole('button',{name:'New event',exact:true}).click();
      await page.getByLabel('Event name',{exact:true}).fill(title);
      await page.getByLabel('Number of courts',{exact:true}).fill('2');
      return clickRequest(page,page.getByRole('button',{name:'Create event',exact:true}),'/round-robin-hubs');
    }
    async function createFromHub(hub, title, expectedGroups) {
      await page.getByRole('tab',{name:'Round robins',exact:true}).click();
      const createLink = page.getByRole('link',{name:'Create round robin',exact:true});
      assert.equal(new URL(await createLink.getAttribute('href'),webOrigin).searchParams.get('hub'),hub.hub.id);
      await createLink.click();
      await page.waitForURL(url=>url.pathname === '/admin/round-robin-generator' && url.searchParams.get('hub') === hub.hub.id);
      await page.getByLabel('Session title',{exact:true}).fill(title);
      await page.getByLabel('Event format',{exact:true}).selectOption('standard');
      await page.getByRole('combobox',{name:/^Play format/}).selectOption('doubles');
      await page.getByRole('combobox',{name:/^Number of players/}).selectOption('4');
      await page.getByLabel('Match rating',{exact:true}).selectOption('unrated');
      await page.getByRole('textbox',{name:/^Players \(/}).fill(Array.from({length:4},(_,i)=>`${title} Player ${i+1}`).join('\n'));
      await clickRequest(page,page.getByRole('button',{name:'Preview matchups',exact:true}),'/preview');
      const started = await clickRequest(page,page.getByRole('button',{name:'Start and add to hub',exact:true}),'/sessions');
      assert.equal(started.session.event.rounds[0].matches.length,1);
      await page.waitForURL(url=>url.pathname === '/admin/round-robin-hub' && url.searchParams.get('hub') === hub.hub.id);
      let state, result;
      for (let attempt=0; attempt<40; attempt++) {
        state = await fixture();
        result = state.hubs.find(item=>item.hub.id === hub.hub.id);
        if (result.workspace.groups.some(group=>group.session_key === started.session.session_key)) break;
        await new Promise(resolve=>setTimeout(resolve,250));
      }
      const observed = {url:page.url(),title,sessionKey:started.session.session_key,
        createdSessions:state.sessions.length,hubId:hub.hub.id,groupCount:result.workspace.groups.length,
        availableOptions:await page.getByRole('combobox',{name:/^Existing active round robin/}).count()
          ? await page.getByRole('combobox',{name:/^Existing active round robin/}).locator('option').allTextContents() : [],
        selectedTab:await page.getByRole('tab',{selected:true}).textContent()};
      fs.writeFileSync(path.join(output,'last-return-state.json'),JSON.stringify(observed,null,2));
      assert.equal(result.workspace.groups.length,expectedGroups,'Start session must automatically attach to its originating hub');
      assert(result.workspace.groups.some(group=>group.session_key === started.session.session_key),'New session must belong to the chosen event');
      assert(result.workspace.eligible_queue.some(match=>match.session_key === started.session.session_key && match.round_number === 1),'First match must reach the court queue');
      await page.getByRole('tabpanel',{name:'Court board',exact:true}).waitFor();
      assert.equal(await page.getByRole('tab',{name:'Court board',exact:true}).getAttribute('aria-selected'),'true','Return flow must open the court board');
      await page.getByRole('listitem').filter({hasText:title}).getByRole('button',{name:'Next open court',exact:true}).waitFor();
      assert.equal(await page.getByLabel('Select hub',{exact:true}).inputValue(),hub.hub.id);
      record(`${title}: created from hub, attached exactly once, returned to correct court board with first queued match`);
      await snapshot(page,`${title.toLowerCase().replaceAll(' ','-')}-returned.png`);
      return started.session;
    }
    const first = await createHub('Friday Hub');
    assert.equal(first.workspace.groups.length,0);
    assert.equal((await fixture()).sessions.length,0,'The hub must exist before any sessions');
    await page.getByText('No other active round robins are available.',{exact:false}).waitFor();
    const firstSession = await createFromHub(first,'First group',1);
    const secondSession = await createFromHub(first,'Second group',2);
    const second = await createHub('Saturday Hub');
    await page.getByLabel('Select hub',{exact:true}).selectOption(first.hub.id);
    await page.getByRole('heading',{name:'Friday Hub',exact:true}).waitFor();
    await page.getByLabel('Select hub',{exact:true}).selectOption(second.hub.id);
    await page.getByRole('heading',{name:'Saturday Hub',exact:true}).waitFor();
    await createFromHub(second,'Saturday group',1);
    await page.reload();
    await page.getByRole('heading',{name:'Saturday Hub',exact:true}).waitFor();
    let state = await fixture();
    assert.equal(state.sessions.length,3,'Starting from a cached generator must not duplicate a session');
    assert.deepEqual(state.hubs.find(hub=>hub.hub.id === first.hub.id).workspace.groups.map(group=>group.session_key).sort(),
      [firstSession.session_key,secondSession.session_key].sort());
    assert.equal(state.hubs.find(hub=>hub.hub.id === second.hub.id).workspace.groups.length,1);
    await page.setViewportSize({width:390,height:844});
    await snapshot(page,'mobile-created-group-return.png');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),'Mobile page must not overflow horizontally');
    record('Repeated creation and switching cached event pages preserves correct membership after refresh');
    assert.deepEqual(errors,[],'No browser console errors or uncaught exceptions');
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:true,evidence,errors,rpcCalls:state.rpc_calls},null,2));
    console.log(`Hub-first browser regression passed. Artifacts: ${output}`);
  } catch (error) {
    if (page) await snapshot(page,'failure.png').catch(()=>{});
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:false,evidence,errors,error:error.stack},null,2));
    throw error;
  } finally {
    if (browser) await browser.close();
    api.kill('SIGTERM'); next.kill('SIGTERM');
    fs.writeFileSync(path.join(output,'servers.log'),logs.join(''));
  }
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
