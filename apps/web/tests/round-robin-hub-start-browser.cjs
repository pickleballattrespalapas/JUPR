/*
 * Loopback-only E2E: actual Next UI -> FastAPI routes -> generator/hub services
 * -> transactional RPC double. Starts without sessions; all creation and hub
 * changes use the UI. Covers hub-first creation, cached-back navigation without
 * duplicate creation, and an injected local 503 followed by an exact retry.
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
const logs = [], errors = [], evidence = [], creationRequests = [], creationDrafts = [];
const sessionCreateUrl = `${apiOrigin}/admin/clubs/club/play-generators/sessions`;
let injectedFailures = 0;
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
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging', GENERATOR_BROWSER_WEB_PORT:webPort, HUB_EMPTY_REGISTRATION_FIXTURE:'1'};
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m', 'uvicorn', 'hub_registration_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort], env, root);
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
      p.on('console', message => { if (message.type() === 'error') errors.push({type:'console',message:message.text(),url:message.location().url}); });
      p.on('request', request => {
        if (request.method() === 'POST' && request.url() === sessionCreateUrl)
          creationRequests.push(request.postDataJSON());
      });
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
    async function createFromHub(hub, title, expectedGroups, failFirstAttempt = false) {
      await page.getByRole('tab',{name:'Round robins',exact:true}).click();
      const createLink = page.getByRole('link',{name:'Create round robin',exact:true});
      assert.equal(new URL(await createLink.getAttribute('href'),webOrigin).searchParams.get('hub'),hub.hub.id);
      await createLink.click();
      await page.waitForURL(url=>url.pathname === '/admin/round-robin-generator' && url.searchParams.get('hub') === hub.hub.id);
      const builderUrl = page.url();
      const draftToken = new URL(builderUrl).searchParams.get('draft');
      assert(draftToken && !creationDrafts.includes(draftToken),'Create round robin must open a fresh draft, separate from previous success receipts');
      creationDrafts.push(draftToken);
      await page.getByLabel('Round robin name / level',{exact:true}).fill(title);
      await page.getByRole('combobox',{name:/^Play format/}).selectOption('doubles');
      assert.equal(await page.getByRole('combobox',{name:/^Number of players/}).count(),0);
      assert.equal(await page.getByRole('textbox',{name:/^Players \(/}).count(),0);
      await page.getByLabel('Match rating',{exact:true}).selectOption('unrated');
      let failedPayload;
      if (failFirstAttempt) {
        const before = await fixture();
        const failStart = async route => {
          if (route.request().method() !== 'POST') return route.continue();
          failedPayload = route.request().postDataJSON();
          injectedFailures += 1;
          await route.fulfill({status:503,contentType:'application/json',headers:{'access-control-allow-origin':webOrigin},
            body:JSON.stringify({detail:'The durable operation intent could not be persisted.'})});
        };
        await page.route(sessionCreateUrl,failStart);
        const failure = page.waitForResponse(response=>response.url() === sessionCreateUrl && response.request().method() === 'POST');
        const [response] = await Promise.all([failure,page.getByRole('button',{name:'Create round robin',exact:true}).click()]);
        assert.equal(response.status(),503);
        await page.unroute(sessionCreateUrl,failStart);
        const alert = page.getByRole('alert').filter({hasText:/start|persist/i});
        await alert.waitFor();
        const color = await alert.evaluate(element=>getComputedStyle(element).color);
        const channels = color.match(/[\d.]+/g)?.map(Number) || [];
        assert(channels[0] > channels[1] && channels[0] > channels[2],`Creation failure must use error styling, received ${color}`);
        assert.equal(await page.getByLabel('Round robin name / level',{exact:true}).inputValue(),title);
        assert.equal((await fixture()).sessions.length,before.sessions.length,'Rejected local request must not create a session');
        await snapshot(page,'creation-error-draft-retained.png');
        await page.reload();
        await page.getByRole('button',{name:'Retry this start',exact:true}).waitFor();
        assert.equal(await page.getByLabel('Round robin name / level',{exact:true}).inputValue(),title);
      }
      const started = await clickRequest(page,page.getByRole('button',{name:failFirstAttempt ? 'Retry this start' : 'Create round robin',exact:true}),'/sessions');
      if (failFirstAttempt) {
        assert.deepEqual(creationRequests.at(-1),failedPayload,'Retry after reload must preserve the exact creation payload and idempotency key');
        record('A failed start shows an error, keeps the draft across reload, and retries the identical creation request');
      }
      assert.equal(started.session.event.rounds[0].matches.length,0);
      assert.equal(started.session.event.participants.length,0);
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
      assert(!result.workspace.eligible_queue.some(match=>match.session_key === started.session.session_key),'Empty groups wait for check-in');
      await page.getByRole('tabpanel',{name:'Court board',exact:true}).waitFor();
      assert.equal(await page.getByRole('tab',{name:'Court board',exact:true}).getAttribute('aria-selected'),'true','Return flow must open the court board');
      assert.equal(await page.getByLabel('Select hub',{exact:true}).inputValue(),hub.hub.id);
      record(`${title}: created from hub, attached exactly once, returned to correct court board ready for arrivals`);
      await snapshot(page,`${title.toLowerCase().replaceAll(' ','-')}-returned.png`);
      return {...started.session,builderUrl};
    }
    async function runExistingAfterBack(hub, session, reload = false) {
      const before = await fixture();
      const requestsBefore = creationRequests.length;
      await page.goBack();
      await page.waitForURL(session.builderUrl);
      const run = page.getByRole('button',{name:'Run this round robin in the hub',exact:true});
      await run.waitFor();
      if (reload) { await page.reload(); await run.waitFor(); }
      await snapshot(page,reload ? 'created-session-receipt-reloaded.png' : 'cached-created-session-receipt.png');
      await run.click();
      await page.waitForURL(url=>url.pathname === '/admin/round-robin-hub' && url.searchParams.get('hub') === hub.hub.id);
      await page.getByRole('tabpanel',{name:'Court board',exact:true}).waitFor();
      const after = await fixture();
      assert.equal(creationRequests.length,requestsBefore,'Run an existing round robin must navigate without POSTing another session');
      assert.equal(after.sessions.length,before.sessions.length,'Returning to a created session must not duplicate it');
      assert(after.hubs.find(item=>item.hub.id === hub.hub.id).workspace.groups.some(group=>group.session_key === session.session_key));
      record(`${reload ? 'Reloaded' : 'Cached-back'} success receipt runs the exact existing round robin without another creation POST`);
    }
    const first = await createHub('Friday Hub');
    assert.equal(first.workspace.groups.length,0);
    assert.equal((await fixture()).sessions.length,0,'The hub must exist before any sessions');
    await page.getByText('No other active round robins are available.',{exact:false}).waitFor();
    const firstSession = await createFromHub(first,'First group',1);
    await runExistingAfterBack(first,firstSession);
    const secondSession = await createFromHub(first,'Second group',2,true);
    await runExistingAfterBack(first,secondSession,true);
    await page.getByRole('tab',{name:/^Registration/}).click();
    await command(page,page.getByRole('button',{name:'Open check-in',exact:true}),'set_registration');
    const guestContext = await browser.newContext({viewport:{width:390,height:844}});
    await guestContext.route('**/*', route => ['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    const frontDesk = await guestContext.newPage();
    frontDesk.on('pageerror',error=>errors.push({type:'pageerror',message:error.message}));
    await frontDesk.goto(`${webOrigin}/round-robin-hub/${first.hub.id}/check-in`);
    async function checkIn(name) {
      if (await frontDesk.getByRole('button',{name:'Check in another player'}).count())
        await frontDesk.getByRole('button',{name:'Check in another player'}).click();
      await frontDesk.getByLabel('Full name',{exact:true}).fill(name);
      await frontDesk.getByLabel('Round robin / level',{exact:true}).selectOption(firstSession.session_key);
      await clickRequest(frontDesk,frontDesk.getByRole('button',{name:'Check in',exact:true}),'/checkins');
      await frontDesk.getByRole('heading',{name:`You’re checked in, ${name}.`,exact:true}).waitFor();
    }
    for (let n=1;n<=4;n++) {
      await checkIn(`Arrival ${n}`);
      const detail=(await fixture()).hubs.find(h=>h.hub.id===first.hub.id);
      assert.equal(detail.workspace.groups.find(g=>g.session_key===firstSession.session_key).players.length,n);
      assert.equal(detail.workspace.eligible_queue.length>0,n>=4);
      assert.equal(detail.registration.pending_count,0);
    }
    await checkIn('Arrival 4');
    assert.equal((await fixture()).hubs.find(h=>h.hub.id===first.hub.id).registration.checkins.length,4);
    await snapshot(frontDesk,'front-desk-checkin.png');
    await page.getByRole('button',{name:'Refresh',exact:true}).click();
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    await page.getByRole('listitem').filter({hasText:firstSession.title}).getByRole('button',{name:'Next open court',exact:true}).waitFor();
    await command(page,page.getByRole('button',{name:'Fill open courts',exact:true}),'auto_fill_courts');
    const onCourt=(await fixture()).hubs.find(h=>h.hub.id===first.hub.id).workspace.courts.find(c=>c.current).current;
    for(let n=5;n<=8;n++) await checkIn(`Arrival ${n}`);
    const afterArrivals=(await fixture()).hubs.find(h=>h.hub.id===first.hub.id);
    const retained=afterArrivals.workspace.courts.find(c=>c.current?.match_id===onCourt.match_id).current;
    assert.deepEqual(retained.side_a,onCourt.side_a);
    assert.deepEqual(retained.side_b,onCourt.side_b);
    assert(afterArrivals.workspace.eligible_queue.length>0);
    await page.getByRole('listitem').filter({hasText:'Arrival 5'}).getByRole('button',{name:'Next open court',exact:true}).waitFor({timeout:25000});
    await snapshot(page,'live-match-and-new-arrivals.png');
    record('Front-desk check-ins fill an empty roster, the fourth player creates a match, duplicate retries are safe, and later arrivals preserve the live court');
    const waitingPlayer = afterArrivals.workspace.eligible_queue.find(m=>m.round_number===1).side_a[0];
    await page.getByRole('tab',{name:'Players',exact:true}).click();
    await page.getByLabel('Find player',{exact:true}).fill(waitingPlayer.name);
    let edited = await command(page,page.getByRole('button',{name:'Pause player',exact:true}),'pause_player');
    assert.equal(edited.workspace.groups.find(g=>g.session_key===firstSession.session_key).players.filter(p=>p.on_bye).length,3);
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    await page.getByRole('article').filter({has:page.getByRole('button',{name:'Enter score',exact:true})}).getByRole('button',{name:'Enter score',exact:true}).click();
    await page.getByLabel('Side A score',{exact:true}).fill('11');
    await page.getByLabel('Side B score',{exact:true}).fill('7');
    edited = await command(page,page.getByRole('button',{name:'Save score & release court',exact:true}),'score_and_release');
    assert.equal(edited.workspace.groups.find(g=>g.session_key===firstSession.session_key).current_round_number,2);
    edited = await command(page,page.getByRole('button',{name:'Fill open courts',exact:true}),'auto_fill_courts');
    const liveBeforeResume = structuredClone(edited.workspace.courts.filter(c=>c.current));
    await page.getByRole('tab',{name:'Players',exact:true}).click();
    edited = await command(page,page.getByRole('button',{name:'Ready again',exact:true}),'resume_player');
    assert(edited.workspace.eligible_queue.some(m=>m.round_number===2 && [...m.side_a,...m.side_b].some(p=>p.id===waitingPlayer.id)));
    const assignments = courts => courts.filter(c=>c.current).map(c=>({id:c.id,match_id:c.current.match_id,
      side_a:c.current.side_a,side_b:c.current.side_b,assigned_at:c.current.assigned_at}));
    assert.deepEqual(assignments(edited.workspace.courts),assignments(liveBeforeResume));
    assert.equal(edited.workspace.groups.find(g=>g.session_key===firstSession.session_key).players.filter(p=>p.on_bye).length,0);
    record('Resume after round advancement forms a ready game from four free players and preserves the occupied court');
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    const liveCourt = edited.workspace.courts.find(c=>c.current);
    const courtCard = page.getByRole('article',{name:liveCourt.label,exact:true});
    const donor = edited.workspace.eligible_queue.find(m=>m.round_number===2).side_a[0];
    async function openPlayers(target) {
      await courtCard.getByRole('button',{name:'Edit players',exact:true}).click();
      const dialog = page.getByRole('dialog',{name:'Edit players',exact:true});
      await dialog.getByLabel('Player to replace or swap',{exact:true}).selectOption(target);
      return dialog;
    }
    let editor = await openPlayers(liveCourt.current.side_a[0].id);
    await editor.getByLabel('Replacement',{exact:true}).selectOption(donor.id);
    edited = await command(page,editor.getByRole('button',{name:'Save players',exact:true}),'edit_match_players');
    assert.equal(edited.workspace.courts.find(c=>c.id===liveCourt.id).current.side_a[0].id,donor.id);
    assert(edited.workspace.eligible_queue.some(m=>[...m.side_a,...m.side_b].some(p=>p.id===liveCourt.current.side_a[0].id)));
    record('Edit players swaps a waiting player into the live court and puts the outgoing player in their waiting game');
    editor = await openPlayers(donor.id);
    await editor.getByLabel('Replacement',{exact:true}).selectOption('new');
    await editor.getByRole('combobox',{name:'New player name or club profile',exact:true}).fill('Walk-in Guest');
    await page.setViewportSize({width:390,height:844});
    await snapshot(page,'mobile-edit-players.png');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),'Player editor fits mobile');
    // Lose the response AFTER the real service commits, then retry the exact operation.
    const commandUrl = `${apiOrigin}/admin/clubs/club/round-robin-hubs/${first.hub.id}/commands`;
    let failedEditPayload;
    const loseEditResponse = async route => {
      if (route.request().postDataJSON()?.action !== 'edit_match_players') return route.continue();
      failedEditPayload = route.request().postDataJSON();
      const response = await route.fetch();
      assert.equal(response.status(),200);
      await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Local lost response after commit'})});
    };
    await page.route(commandUrl,loseEditResponse);
    const lost = page.waitForResponse(r=>r.url()===commandUrl && r.status()===503);
    await editor.getByRole('button',{name:'Save players',exact:true}).click();
    await lost;
    await page.unroute(commandUrl,loseEditResponse);
    await editor.getByRole('button',{name:'Retry saved action',exact:true}).waitFor();
    assert(await editor.getByRole('button',{name:'Cancel',exact:true}).isDisabled());
    const retry = page.waitForRequest(r=>r.url()===commandUrl && r.method()==='POST');
    edited = await command(page,editor.getByRole('button',{name:'Retry saved action',exact:true}),'edit_match_players');
    assert.deepEqual((await retry).postDataJSON(),failedEditPayload);
    const roster = edited.workspace.groups.find(g=>g.session_key===firstSession.session_key).players;
    assert.equal(roster.filter(p=>p.name==='Walk-in Guest').length,1);
    const guest = roster.find(p=>p.name==='Walk-in Guest');
    assert(edited.workspace.courts.find(c=>c.id===liveCourt.id).current.side_a.some(p=>p.id===guest.participant_id));
    assert(roster.find(p=>p.participant_id===donor.id).on_bye);
    editor = await openPlayers(guest.participant_id);
    await editor.getByLabel('Replacement',{exact:true}).selectOption(donor.id);
    edited = await command(page,editor.getByRole('button',{name:'Save players',exact:true}),'edit_match_players');
    assert(edited.workspace.groups.find(g=>g.session_key===firstSession.session_key).players.find(p=>p.participant_id===guest.participant_id).on_bye);
    await page.reload();
    await courtCard.getByText(donor.name,{exact:false}).waitFor();
    await page.setViewportSize({width:1440,height:1000});
    record('A new guest joins the live lineup, a lost-response retry adds them once, and a bye substitution survives reload');
    await guestContext.close();
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
    const expectedNetworkErrors = errors.filter(error=>error.type === 'console' && [sessionCreateUrl,commandUrl].includes(error.url) && /503/.test(error.message));
    const unexpectedErrors = errors.filter(error=>!expectedNetworkErrors.includes(error));
    assert.equal(injectedFailures,1,'Exactly one local creation failure should be injected');
    assert(expectedNetworkErrors.length <= 2,'Only the two injected 503s may produce console network errors');
    assert.deepEqual(unexpectedErrors,[],'No unexpected browser console errors or uncaught exceptions');
    assert.equal(creationRequests.length,4,'Three sessions plus one failed attempt must be the only creation requests');
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:true,evidence,errors:unexpectedErrors,expectedNetworkErrors,
      injectedFailures,creationRequests:creationRequests.length,rpcCalls:state.rpc_calls},null,2));
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
