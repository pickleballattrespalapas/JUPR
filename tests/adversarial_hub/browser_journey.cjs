/*
 * Loopback-only E2E: actual Next UI -> FastAPI routes -> generator/hub services
 * -> transactional RPC double. Starts without sessions; all creation and hub
 * changes use the UI. Does not test real authentication, RLS or PostgreSQL locks.
 * Run from apps/web: node tests/round-robin-hub-browser.cjs
 * GENERATOR_BROWSER_{PYTHON,EXECUTABLE,ARGS,ARTIFACTS,WEB_PORT,API_PORT,NEXT_MODE} supported.
 * NEXT_MODE=start uses a build with NEXT_PUBLIC_JUPR_API_BASE_URL matching API_PORT.
 */
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const {createRequire} = require('node:module');
const webRoot = path.resolve(__dirname, '../../apps/web');
const webRequire = createRequire(path.join(webRoot,'package.json'));
const {chromium} = webRequire('playwright');
const root = path.resolve(webRoot, '../..');
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || '3059';
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || '8059';
const nextMode = process.env.GENERATOR_BROWSER_NEXT_MODE || 'dev';
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(), 'round-robin-hub-browser-'));
fs.mkdirSync(output, {recursive:true});
const logs = [], errors = [], evidence = [];
let expectedLostResponse=false;
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
    ['-m', 'uvicorn', 'round_robin_hub_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort], env, root);
  assert(['dev','start'].includes(nextMode),'GENERATOR_BROWSER_NEXT_MODE must be dev or start');
  const next = start(process.execPath, [webRequire.resolve('next/dist/bin/next'), nextMode, ...(nextMode === 'dev' ? ['--webpack'] : []), '--hostname', '127.0.0.1', '--port', webPort], env, webRoot);
  let browser, page;
  try {
    await ready(`${apiOrigin}/hub-fixture-result`, [api,next]);
    assert.equal((await fixture()).sessions.length, 0);
    await ready(`${webOrigin}/clubs/test/round-robin-generator`, [api,next]);
    browser = await chromium.launch({headless:true,
      ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE} : {}),
      args:['--no-sandbox','--disable-dev-shm-usage',...JSON.parse(process.env.GENERATOR_BROWSER_ARGS || '[]')]});
    const context = await browser.newContext({viewport:{width:1440,height:1000}});
    await context.route('**/*', route => ['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    context.on('page', p => {
      p.on('pageerror', error => errors.push({type:'pageerror',message:error.message}));
      p.on('console', message => { if (message.type() === 'error') errors.push({type:'console',message:message.text(),expected:expectedLostResponse && /net::ERR_FAILED/.test(message.text())}); });
    });
    await context.addCookies([{name:'jupr_admin_workspace_v1',value:encodeURIComponent(JSON.stringify({clubId:'club',clubSlug:'test'})),url:webOrigin}]);
    await context.addInitScript(origin => {
      if (window.location.origin !== origin) return;
      localStorage.setItem('jupr_admin_session_v1',JSON.stringify({access_token:'local-reviewer',expires_at:Date.now()+3600000,
        user:{id:'local-reviewer',email:'reviewer@example.invalid'},capabilities:{authorized:true,user:{email:'reviewer@example.invalid'},
          assignments:[{club_id:'club',role:'administrator',permissions:['enter_scores','manage_players']}]}}));
    }, webOrigin);
    page = await context.newPage();
    const sessions = [];
    for (const prefix of ['Alpha','Beta']) {
      await page.goto(`${webOrigin}/clubs/test/round-robin-generator`);
      await page.getByLabel('Session title',{exact:true}).fill(`${prefix} round robin`);
      await page.getByRole('combobox',{name:/^Play format/}).selectOption('doubles');
      await page.getByRole('combobox',{name:/^Number of players/}).selectOption('8');
      await page.getByLabel('Match rating',{exact:true}).selectOption('unrated');
      await page.getByRole('textbox',{name:/^Players \(/}).fill(Array.from({length:8},(_,i)=>`${prefix} Player ${i+1}`).join('\n'));
      await clickRequest(page,page.getByRole('button',{name:'Preview matchups',exact:true}),'/preview');
      const started = await clickRequest(page,page.getByRole('button',{name:'Start unrated session',exact:true}),'/sessions');
      await page.waitForURL(/\/rounds\/1(?:#.*)?$/);
      assert.equal(started.session.event.rounds[0].matches.length,2);
      sessions.push({...started.session,roundUrl:page.url()});
    }
    const [alpha,beta] = sessions;
    record('Two independent 8-player scored/unrated round robins created through generator UI');
    await page.goto(`${webOrigin}/admin/round-robin-hub`);
    await page.getByLabel('Event name',{exact:true}).fill('Friday Night RR Hub');
    await page.getByLabel('Number of courts',{exact:true}).fill('2');
    await page.getByLabel('Court 1 name',{exact:true}).fill('Garden Court');
    await page.getByLabel('Court 2 name',{exact:true}).fill('Terrace Court');
    let detail = await clickRequest(page,page.getByRole('button',{name:'Create event',exact:true}),'/round-robin-hubs');
    for (const session of sessions) {
      await page.getByRole('combobox',{name:/^Existing active round robin/}).selectOption(session.session_key);
      detail = await command(page,page.getByRole('button',{name:'Add to hub',exact:true}),'add_group');
    }
    assert.equal(detail.workspace.eligible_queue.length,4);
    await page.getByRole('tab',{name:'Next round plan',exact:true}).click();
    const nextPlan = page.getByRole('region',{name:'Alpha round robin next round plan',exact:true});
    const beforePlan = detail.workspace.groups.find(group=>group.session_key===alpha.session_key);
    assert.equal(beforePlan.next_round_number,2);
    assert.equal(beforePlan.next_round_matches.length,2);
    assert.equal(await nextPlan.getByRole('listitem').count(),2);
    await nextPlan.getByText('Byes:',{exact:false}).waitFor();
    const plannedMatch = beforePlan.next_round_matches[0], plannedDonor = beforePlan.next_round_matches[1];
    await nextPlan.getByRole('button',{name:'Edit players',exact:true}).first().click();
    const teamDialog = page.getByRole('dialog',{name:'Edit players',exact:true});
    await teamDialog.getByLabel('Change players',{exact:true}).selectOption('exact');
    await teamDialog.getByLabel('Team A player 1',{exact:true}).selectOption(plannedDonor.side_a[0].id);
    detail = await command(page,teamDialog.getByRole('button',{name:'Save both teams',exact:true}),'set_match_lineup');
    const afterPlan = detail.workspace.groups.find(group=>group.session_key===alpha.session_key);
    assert.equal(afterPlan.current_round_number,1);
    assert(afterPlan.next_round_matches.every(match=>match.plan_locked),'Both future games affected by the staff swap are protected');
    assert.equal(afterPlan.next_round_matches.find(match=>match.match_id===plannedMatch.match_id).side_a[0].id,plannedDonor.side_a[0].id);
    assert(afterPlan.next_round_matches.find(match=>match.match_id===plannedDonor.match_id).side_a.some(player=>player.id===plannedMatch.side_a[0].id));
    await snapshot(page,'desktop-next-round-plan.png');
    record('Full next-round plan and byes visible before scores; exact team edit swaps a future donor and locks both games');
    await page.getByRole('tab',{name:'Court setup',exact:true}).click();
    await page.getByLabel('Court 1 use',{exact:true}).selectOption(alpha.session_key);
    detail = await command(page,page.getByRole('button',{name:'Save court setup',exact:true}),'configure_courts');
    for (const match of detail.workspace.eligible_queue)
      assert.deepEqual(match.allowed_court_ids,match.session_key === alpha.session_key ? ['court-1','court-2'] : ['court-2']);
    record('Dedicated court excludes other groups while its owner can also use the shared court');

    const pausedMatch = detail.workspace.eligible_queue.find(match => match.session_key === alpha.session_key);
    const pausedPlayer = pausedMatch.side_a[0];
    await page.getByRole('tab',{name:'Players',exact:true}).click();
    await page.getByLabel('Find player',{exact:true}).fill(pausedPlayer.name);
    detail = await command(page,page.getByRole('button',{name:'Pause player',exact:true}),'pause_player');
    assert(!detail.workspace.eligible_queue.some(match=>[...match.side_a,...match.side_b].some(player=>player.identity === pausedPlayer.identity)));
    assert(detail.workspace.eligible_queue.some(match=>match.session_key === alpha.session_key));
    assert.equal(detail.workspace.eligible_queue.filter(match=>match.session_key === beta.session_key).length,2);
    assert(detail.workspace.players.find(player=>player.identity === pausedPlayer.identity).paused);
    detail = await command(page,page.getByRole('button',{name:'Ready again',exact:true}),'resume_player');
    assert.equal(detail.workspace.eligible_queue.filter(match=>match.session_key === alpha.session_key).length,2);
    record('Pause excludes the player while available players refill waiting games; resume restores eligibility');

    await page.getByRole('tab',{name:'Match queue',exact:true}).click();
    await page.getByRole('listitem').filter({hasText:'Alpha round robin'}).first().getByRole('button',{name:'Choose court',exact:true}).click();
    const sharedPicker = page.getByRole('dialog',{name:'Choose court',exact:true});
    assert.deepEqual(await sharedPicker.getByRole('combobox',{name:/^Court/}).locator('option').evaluateAll(options=>options.map(option=>option.value)),['','court-1','court-2']);
    await sharedPicker.getByRole('combobox',{name:/^Court/}).selectOption('court-2');
    detail = await command(page,sharedPicker.getByRole('button',{name:'Assign to court',exact:true}),'assign_game');
    const overflowMatch = detail.workspace.courts[1].current.match_id;
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    detail = await command(page,page.getByRole('button',{name:'Fill open courts',exact:true}),'auto_fill_courts');
    assert.equal(detail.workspace.courts[0].current.session_key,alpha.session_key);
    assert.equal(detail.workspace.courts[1].current.session_key,alpha.session_key);
    await page.reload();
    await page.getByRole('article',{name:'Terrace Court',exact:true}).getByText('Alpha round robin · Round 1',{exact:true}).waitFor();
    const overflowState = await fixture();
    assert.equal(overflowState.hubs[0].workspace.courts[1].current.match_id,overflowMatch);
    await snapshot(page,'desktop-shared-court-overflow.png');
    record('Dedicated group uses both its dedicated and shared courts; shared assignment survives refresh');
    await page.getByRole('article',{name:'Terrace Court',exact:true}).getByRole('button',{name:'Move or remove',exact:true}).click();
    detail = await command(page,page.getByRole('dialog',{name:'Move or remove match',exact:true}).getByRole('button',{name:'Return to queue',exact:true}),'release_game');
    await page.getByRole('tab',{name:'Match queue',exact:true}).click();
    detail = await command(page,page.getByRole('listitem').filter({hasText:'Beta round robin'}).first().getByRole('button',{name:'Next open court',exact:true}),'assign_next_court');
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    assert.equal(detail.workspace.courts[1].current.session_key,beta.session_key);
    assert.equal(detail.workspace.eligible_queue.length,2);
    const identities = detail.workspace.courts.flatMap(court=>[...court.current.side_a,...court.current.side_b].map(player=>player.identity));
    assert.equal(new Set(identities).size,identities.length);
    await snapshot(page,'desktop-playing.png');
    async function score(courtLabel, scoreB=6) {
      await page.getByRole('tab',{name:'Court board',exact:true}).click();
      await page.getByRole('article',{name:courtLabel,exact:true}).getByRole('button',{name:'Enter score',exact:true}).click();
      await page.getByLabel('Side A score',{exact:true}).fill('11');
      await page.getByLabel('Side B score',{exact:true}).fill(String(scoreB));
      return command(page,page.getByRole('button',{name:'Save score & release court',exact:true}),'score_and_release');
    }
    detail = await score('Garden Court');
    assert.equal(detail.workspace.courts[0].current,null);
    let state = await fixture();
    let a = state.sessions.find(session=>session.session_key === alpha.session_key);
    assert.equal(a.event.rounds[0].status,'active');
    assert(a.standings.every(row=>row.matches === 0));
    assert.equal(detail.workspace.groups.find(group=>group.session_key === alpha.session_key).completed_matches,1);
    await page.reload();
    await page.getByRole('article',{name:'Garden Court',exact:true}).getByText('Available',{exact:true}).waitFor();
    state = await fixture();
    a = state.sessions.find(session=>session.session_key === alpha.session_key);
    assert(a.standings.every(row=>row.matches === 0));
    assert.equal(state.hubs[0].workspace.groups.find(group=>group.session_key === alpha.session_key).completed_matches,1);
    record('One saved match releases court, survives refresh, and remains excluded from incomplete-round standings');

    await page.getByRole('tab',{name:'Match queue',exact:true}).click();
    const betaQueue = page.getByRole('listitem').filter({hasText:'Beta round robin'});
    await betaQueue.getByRole('button',{name:'Choose court',exact:true}).click();
    const picker = page.getByRole('dialog',{name:'Choose court',exact:true});
    const courtSelect = picker.getByRole('combobox',{name:/^Court/});
    await courtSelect.waitFor();
    assert.deepEqual(await courtSelect.locator('option').evaluateAll(options=>options.map(option=>option.value)),['','court-2'],'Only placeholder and eligible shared court offered');
    await courtSelect.selectOption('court-2');
    detail = await command(page,picker.getByRole('button',{name:'Reserve next',exact:true}),'reserve_game_for_court');
    assert.equal(detail.workspace.courts[1].next.session_key,beta.session_key);
    const reservationId = detail.workspace.courts[1].next.match_id;
    const alphaQueue = page.getByRole('listitem').filter({hasText:'Alpha round robin'});
    detail = await command(page,alphaQueue.getByRole('button',{name:'Next open court',exact:true}),'assign_next_court');
    assert.equal(detail.workspace.courts[0].current.session_key,alpha.session_key);
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    await page.setViewportSize({width:390,height:844});
    await snapshot(page,'mobile-reserved-next.png');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),'Mobile page must not overflow horizontally');
    await page.setViewportSize({width:1440,height:1000});
    detail = await score('Garden Court',7);
    state = await fixture();
    a = state.sessions.find(session=>session.session_key === alpha.session_key);
    assert.equal(a.event.rounds[0].status,'saved');
    assert(a.standings.every(row=>row.matches === 1));
    await page.getByRole('tab',{name:'Round robins',exact:true}).click();
    const alphaGroup = page.getByRole('article').filter({has:page.getByRole('heading',{name:'Alpha round robin',exact:true})});
    assert.equal(detail.workspace.groups.find(group=>group.session_key === alpha.session_key).current_round_number,2);
    assert.equal(detail.workspace.groups.find(group=>group.session_key === beta.session_key).current_round_number,1);
    assert(detail.workspace.eligible_queue.filter(match=>match.session_key === alpha.session_key).every(match=>match.round_number === 2));
    record('Completing whole round publishes standings; Alpha advances independently while Beta stays in round 1');
    detail = await score('Terrace Court',8);
    assert.equal(detail.workspace.courts[1].current.match_id,reservationId);
    assert.equal(detail.workspace.courts[1].next,null);
    state = await fixture();
    assert(state.sessions.find(session=>session.session_key === beta.session_key).standings.every(row=>row.matches === 0));
    record('Reserve-next promotes the waiting match when its court is released');
    detail = await score('Terrace Court',9);
    assert.equal(detail.workspace.courts[1].current,null);
    state = await fixture();
    assert(state.sessions.find(session=>session.session_key === beta.session_key).standings.every(row=>row.matches === 1));
    await page.getByRole('tab',{name:'Round robins',exact:true}).click();
    await alphaGroup.getByRole('button',{name:'Standings',exact:true}).click();
    await page.getByRole('heading',{name:'Alpha round robin standings',exact:true}).waitFor();
    await snapshot(page,'desktop-independent-rounds-standings.png');
    await page.getByRole('tab',{name:'Round robins',exact:true}).click();
    await alphaGroup.getByRole('button',{name:'Manage round robin',exact:true}).click();
    await page.getByRole('region',{name:'Manage selected round robin',exact:true}).getByRole('heading',{name:/Round 2/}).first().waitFor();
    record('Original standings and round controls are accessible within the hub');
    const runner=page.getByRole('region',{name:'Manage selected round robin',exact:true});
    const scoreInputs=runner.getByRole('spinbutton',{name:/side [AB] score$/});
    assert.equal(await scoreInputs.count(),4);
    for(let i=0;i<4;i++) await scoreInputs.nth(i).fill(i%2?'7':'11');
    const beforeDraft=await scoreInputs.evaluateAll(inputs=>inputs.map(input=>input.value));
    const normalize=route=>{const headers=route.request().headers();if(headers.authorization==='Bearer local-reviewer-renewed')headers.authorization='Bearer local-reviewer';return headers;};
    await page.route(`${apiOrigin}/**`,route=>route.continue({headers:normalize(route)}));
    const refreshed=page.waitForResponse(response=>response.request().method()==='GET' && response.url().endsWith(`/play-generators/sessions/${alpha.session_key}`));
    await page.evaluate(()=>{const session=JSON.parse(localStorage.getItem('jupr_admin_session_v1'));session.access_token='local-reviewer-renewed';localStorage.setItem('jupr_admin_session_v1',JSON.stringify(session));window.dispatchEvent(new CustomEvent('jupr-admin-session-change',{detail:{source:'reliability-auth-refresh'}}));});
    const credentialReload=await refreshed;assert(credentialReload.ok());await credentialReload.finished();await page.waitForLoadState('networkidle');
    assert.deepEqual(await scoreInputs.evaluateAll(inputs=>inputs.map(input=>input.value)),beforeDraft);
    record('Four unsaved scores survive credential renewal');
    await scoreInputs.nth(2).fill('');await scoreInputs.nth(3).fill('');
    const beforePartial=(await fixture()).sessions.find(session=>session.session_key===alpha.session_key);
    const scorePath=`${apiOrigin}/admin/clubs/club/play-generators/sessions/${alpha.session_key}/rounds/2/scores`;
    const scoreRequests=[];
    await page.route(scorePath,async route=>{scoreRequests.push(route.request().postData());if(scoreRequests.length===1){const response=await route.fetch({headers:normalize(route)});assert(response.ok(),await response.text());await route.abort('failed');}else await route.continue({headers:normalize(route)});});
    expectedLostResponse=true;
    await runner.getByRole('button',{name:'Save round scores',exact:true}).click();
    const retrySave=runner.getByRole('button',{name:'Retry score save',exact:true});await retrySave.waitFor();
    const committed=(await fixture()).sessions.find(session=>session.session_key===alpha.session_key);
    assert.equal(committed.event.rounds.find(round=>round.number===2).status,'saved');
    assert(await scoreInputs.first().isDisabled());
    const retryResponse=page.waitForResponse(response=>response.request().method()==='PATCH' && response.url()===scorePath);
    await retrySave.click();const saved=await retryResponse;assert(saved.ok(),await saved.text());
    const afterPartial=(await fixture()).sessions.find(session=>session.session_key===alpha.session_key);
    assert.equal(scoreRequests.length,2);assert.equal(scoreRequests[0],scoreRequests[1]);assert.equal(afterPartial.version,committed.version);
    const resolvedRound=afterPartial.event.rounds.find(round=>round.number===2);
    assert.equal(resolvedRound.status,'saved');assert.equal(resolvedRound.matches.filter(match=>match.status==='scored').length,1);
    const unplayed=resolvedRound.matches.filter(match=>match.status==='not_played');assert.equal(unplayed.length,1);assert.equal(unplayed[0].scoreA,null);assert.equal(unplayed[0].scoreB,null);
    assert.deepEqual(afterPartial.event.rounds[0],beforePartial.event.rounds[0]);assert.equal(afterPartial.standings.reduce((total,row)=>total+row.matches,0),12);
    await page.unroute(scorePath);expectedLostResponse=false;
    record('One real score plus a blank game saves; a lost-response retry keeps the exact request and commits once');
    await runner.getByRole('button',{name:'Close round robin controls',exact:true}).click();
    await page.getByRole('tab',{name:'Players',exact:true}).click();await page.getByLabel('Find player',{exact:true}).fill('');
    const beforeTransfer=await fixture();const sourceBefore=beforeTransfer.sessions.find(session=>session.session_key===beta.session_key);const movedPlayer=sourceBefore.event.participants[0];
    await page.getByRole('button',{name:'Move player',exact:true}).click();
    const moveDialog=page.getByRole('dialog',{name:'Move player',exact:true});
    await moveDialog.getByLabel('From round robin',{exact:true}).selectOption(beta.session_key);await moveDialog.getByLabel('Player to move',{exact:true}).selectOption(movedPlayer.id);await moveDialog.getByLabel('To round robin',{exact:true}).selectOption(alpha.session_key);
    detail=await command(page,moveDialog.getByRole('button',{name:'Move player to round robin',exact:true}),'transfer_player');
    assert.equal(detail.workspace.groups.find(group=>group.session_key===beta.session_key).players.find(player=>player.participant_id===movedPlayer.id).membership_status,'departed');
    assert.equal(detail.workspace.groups.find(group=>group.session_key===alpha.session_key).players.filter(player=>player.name===movedPlayer.name&&player.active).length,1);
    assert.deepEqual((await fixture()).sessions.find(session=>session.session_key===beta.session_key).event.rounds[0],sourceBefore.event.rounds[0]);
    await page.getByLabel('Show departed and absent roster history',{exact:true}).check();
    const historyRow=page.getByRole('region',{name:'Players in Beta round robin',exact:true}).locator('[data-participant-id]').filter({hasText:movedPlayer.name});
    await historyRow.getByText('Left round robin · Historical entry',{exact:false}).waitFor();
    detail=await command(page,historyRow.getByRole('button',{name:'Rejoin round robin',exact:true}),'rejoin_player');
    assert.equal(detail.workspace.groups.find(group=>group.session_key===beta.session_key).players.filter(player=>player.name===movedPlayer.name&&player.active).length,1);
    record('Atomic transfer preserves earlier results; explicit rejoin restores one active membership');
    await page.getByRole('tab',{name:'Match queue',exact:true}).click();
    const queued=detail.workspace.eligible_queue.find(match=>match.session_key===alpha.session_key);assert(queued);
    await page.getByRole('listitem').filter({hasText:'Alpha round robin'}).filter({hasText:queued.side_a[0].name}).first().getByRole('button',{name:'Not played',exact:true}).click();
    detail=await command(page,page.getByRole('dialog',{name:'Mark match not played?',exact:true}).getByRole('button',{name:'Confirm not played',exact:true}),'resolve_game');
    state=await fixture();const settled=state.sessions.find(session=>session.session_key===alpha.session_key).event.rounds.flatMap(round=>round.matches||[]).find(match=>match.id===queued.match_id);
    assert.equal(settled.status,'not_played');assert.equal(settled.scoreA,null);assert.equal(settled.scoreB,null);
    assert(!detail.workspace.eligible_queue.some(match=>match.session_key===alpha.session_key&&match.match_id===queued.match_id));
    record('Hub closes a queued game as not played without a fabricated result');
    await snapshot(page,'hub-reliability-complete.png');
    assert.deepEqual(errors.filter(error=>!error.expected),[],'No unexpected browser errors');
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:true,evidence,errors,hubId:detail.hub.id,rpcCalls:state.rpc_calls},null,2));
    console.log(`Browser rehearsal passed. Artifacts: ${output}`);
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
