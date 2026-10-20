/*
 * Loopback-only E2E: actual Next UI -> FastAPI routes -> generator/hub services
 * -> transactional RPC double. Starts without sessions. Sessions are created
 * using real generator APIs; identity review, pause and court assignment
 * use the UI. Covers guest/profile identity, review cleanup, automatic round
 * progression, grouped roster additions and exact response-loss retries.
 * Does not test real authentication, RLS or PostgreSQL locks.
 * Run from apps/web: node tests/round-robin-hub-identities-browser.cjs
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
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || path.join(webRoot, 'test-results', 'round-robin-hub-identities');
fs.mkdirSync(output, {recursive:true});
const logs = [], errors = [], evidence = [];
const expectedConsoleErrors = [];
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
async function requestApi(suffix, body) {
  const response = await fetch(`${apiOrigin}${suffix}`, {method:body === undefined ? 'GET' : 'POST',
    headers:{Authorization:'Bearer local-reviewer',...(body === undefined ? {} : {'Content-Type':'application/json'})},
    ...(body === undefined ? {} : {body:JSON.stringify(body)})});
  const result = await response.json();
  assert(response.ok, `${suffix}: ${response.status} ${JSON.stringify(result)}`);
  return result;
}
async function createSession(title, names, profiles = {}) {
  const body = {generator_kind:'round_robin',event_format:'standard',play_format:'doubles',title,
    participant_names:names,participant_player_ids:profiles,total_rounds:3,court_count:0,
    scoring_mode:'scored',rating_mode:'unrated',standings_sort:'wins'};
  const preview = await requestApi('/admin/clubs/club/play-generators/preview',body);
  return (await requestApi('/admin/clubs/club/play-generators/sessions',{...body,
    preview_fingerprint:preview.preview.previewFingerprint,expected_version:'new',idempotency_key:`identity-${crypto.randomUUID()}`})).session;
}
function people(match) { return [...match.side_a,...match.side_b]; }
function occupied(detail) { return detail.workspace.courts.filter(court=>court.current); }

function record(label) { evidence.push(label); console.log(`PASS ${label}`); }
async function snapshot(page, name) { await page.screenshot({path:path.join(output, name), fullPage:true}); }
async function main() {
  const env = {...process.env, VERCEL_ENV:'development', JUPR_API_BASE_URL:apiOrigin,
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging', GENERATOR_BROWSER_WEB_PORT:webPort};
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m', 'uvicorn', 'round_robin_hub_identity_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort], env, root);
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
    });
    await context.addCookies([{name:'jupr_admin_workspace_v1',value:encodeURIComponent(JSON.stringify({clubId:'club',clubSlug:'test'})),url:webOrigin}]);
    await context.addInitScript(origin => {
      if (window.location.origin !== origin) return;
      localStorage.setItem('jupr_admin_session_v1',JSON.stringify({access_token:'local-reviewer',expires_at:Date.now()+3600000,
        user:{id:'local-reviewer',email:'reviewer@example.invalid'},capabilities:{authorized:true,user:{email:'reviewer@example.invalid'},
          assignments:[{club_id:'club',role:'administrator',permissions:['enter_scores','manage_players']}]}}));
    }, webOrigin);
    page = await context.newPage();
    async function createHub(title, sessions) {
      await page.goto(`${webOrigin}/admin/round-robin-hub`);
      await page.getByRole('button',{name:'New event',exact:true}).click();
      await page.getByLabel('Event name',{exact:true}).fill(title);
      await page.getByLabel('Number of courts',{exact:true}).fill('2');
      let detail = await clickRequest(page,page.getByRole('button',{name:'Create event',exact:true}),'/round-robin-hubs');
      for (const session of sessions) {
        await page.getByRole('combobox',{name:/^Existing active round robin/}).selectOption(session.session_key);
        detail = await command(page,page.getByRole('button',{name:'Add to hub',exact:true}),'add_group');
      }
      return detail;
    }
    const readHub = hubId => requestApi(`/admin/clubs/club/round-robin-hubs/${hubId}`);
    async function openReview() {
      await page.getByRole('button',{name:'Review matching names',exact:true}).first().click();
      await page.getByRole('heading',{name:'Matching player names',exact:true}).waitFor();
    }
    async function assignFromGroup(title) {
      await page.getByRole('tab',{name:'Match queue',exact:true}).click();
      return command(page,page.getByRole('listitem').filter({hasText:title}).getByRole('button',{name:'Next open court',exact:true}).first(),'assign_next_court');
    }
    async function fillCourts() {
      await page.getByRole('tab',{name:'Court board',exact:true}).click();
      return command(page,page.getByRole('button',{name:'Fill open courts',exact:true}),'auto_fill_courts');
    }
    async function scoreCourt(label) {
      await page.getByRole('tab',{name:'Court board',exact:true}).click();
      await page.getByRole('article',{name:label,exact:true}).getByRole('button',{name:'Enter score',exact:true}).click();
      await page.getByLabel('Side A score',{exact:true}).fill('11');
      await page.getByLabel('Side B score',{exact:true}).fill('6');
      return command(page,page.getByRole('button',{name:'Save score & release court',exact:true}),'score_and_release');
    }
    async function openAddPlayer(groupTitle, name, profileId) {
      await page.getByRole('tab',{name:'Players',exact:true}).click();
      await page.getByRole('region',{name:`Players in ${groupTitle}`,exact:true}).getByRole('button',{name:'Add player',exact:true}).click();
      const dialog = page.getByRole('dialog',{name:'Add player',exact:true});
      await dialog.getByRole('combobox',{name:'Player name or club profile',exact:true}).fill(name);
      if (profileId !== undefined) await dialog.locator(`[data-player-value="${profileId}"]`).click();
      return dialog;
    }

    const numbers = Array.from({length:8},(_,i)=>String(i+1));
    const numberedA = await createSession('Numbers A',numbers);
    const numberedB = await createSession('Numbers B',numbers);
    let detail = await createHub('Numbered guest review',[numberedA,numberedB]);
    const numberedHubId = detail.hub.id;
    assert.equal(detail.workspace.identity_pending_count,8);
    assert.equal(detail.workspace.eligible_queue.length,0,'Unreviewed duplicate names cannot silently claim courts');
    assert.equal(detail.workspace.blocked_queue.length,4);
    await openReview();
    assert.equal(await page.locator('[data-review-id]').count(),8);
    await snapshot(page,'matching-numbered-guests.png');
    const firstReviewId = await page.locator('[data-review-id]').first().getAttribute('data-review-id');
    detail = await command(page,page.locator('[data-review-id]').first().getByRole('button',{name:/^Different people:/}),'resolve_identities');
    assert.equal(detail.workspace.identity_pending_count,7);
    await page.locator(`[data-review-id="${firstReviewId}"]`).waitFor({state:'detached'});
    assert.equal(await page.locator('[data-review-id]').count(),7,'Resolved name must disappear immediately without reloading');
    assert.equal(await page.getByRole('button',{name:'Saved decisions (1)',exact:true}).getAttribute('aria-expanded'),'false');
    await snapshot(page,'one-resolved-seven-pending.png');
    record('An individually resolved name disappears immediately; only seven pending names remain');

    // Commit the identity decision, then lose only the response. The UI must
    // retry the exact saved operation, and the service must replay its receipt.
    const commandUrl = `${apiOrigin}/admin/clubs/club/round-robin-hubs/${numberedHubId}/commands`;
    let lostRequest, committedResponse;
    const loseResponse = async route => {
      if (route.request().method() !== 'POST' || route.request().postDataJSON()?.action !== 'resolve_identities') return route.continue();
      lostRequest = route.request().postDataJSON();
      const result = await route.fetch();
      committedResponse = {status:result.status(),body:await result.json()};
      return route.fulfill({status:503,contentType:'application/json',headers:{'access-control-allow-origin':webOrigin},
        body:JSON.stringify({detail:'Local rehearsal: the response was lost after commit.'})});
    };
    await page.route(commandUrl,loseResponse);
    const failedResponse = page.waitForResponse(response=>response.url() === commandUrl && response.request().postDataJSON()?.action === 'resolve_identities');
    const [lost] = await Promise.all([failedResponse,page.getByRole('button',{name:'These are all different people',exact:true}).click()]);
    assert.equal(lost.status(),503);
    await page.unroute(commandUrl,loseResponse);
    assert.equal(committedResponse.status,200,JSON.stringify(committedResponse.body));
    assert.equal((await readHub(numberedHubId)).workspace.identity_pending_count,0,'Decision committed before response was lost');
    const replayedResponse = page.waitForResponse(response=>response.url() === commandUrl && response.request().postDataJSON()?.action === 'resolve_identities');
    const [replayed] = await Promise.all([replayedResponse,page.getByRole('button',{name:'Retry saved action',exact:true}).first().click()]);
    detail = await replayed.json();
    assert(replayed.ok(),JSON.stringify(detail));
    assert.deepEqual(replayed.request().postDataJSON(),lostRequest,'Identity retry must preserve payload, version and idempotency key');
    assert.equal(detail.hub.version,committedResponse.body.hub.version,'Replay must not write a second identity decision');
    assert.equal(detail.workspace.identity_pending_count,0);
    assert.equal(detail.workspace.players.length,16,'Same labels confirmed different must remain distinct people');
    assert.equal(detail.workspace.eligible_queue.length,4);
    await page.locator('[data-review-id]').first().waitFor({state:'detached'});
    assert.equal(await page.locator('[data-review-id]').count(),0);
    const savedDecisions = page.getByRole('button',{name:'Saved decisions (8)',exact:true});
    assert.equal(await savedDecisions.getAttribute('aria-expanded'),'false');
    await snapshot(page,'all-resolved-saved-collapsed.png');
    await savedDecisions.click();
    assert.equal(await page.getByRole('region',{name:'Saved identity decisions',exact:true}).locator('[data-review-id]').count(),8);
    await savedDecisions.click();
    assert.equal(await page.locator('[data-review-id]').count(),0);
    record('Two numbered 1–8 groups require review; all-different decisions survive an exact response-loss retry');
    await page.reload();
    await page.getByRole('heading',{name:'Numbered guest review',exact:true}).waitFor();
    detail = await readHub(numberedHubId);
    assert.equal(detail.workspace.identity_pending_count,0);
    assert(detail.workspace.identity_review.every(review=>review.status === 'different'));
    const guestA = detail.workspace.players.find(player=>player.name === '1' && player.session_keys.includes(numberedA.session_key));
    const guestB = detail.workspace.players.find(player=>player.name === '1' && player.session_keys.includes(numberedB.session_key));
    assert(guestA && guestB && guestA.identity !== guestB.identity);
    await page.getByRole('tab',{name:'Players',exact:true}).click();
    for (const title of ['Numbers A','Numbers B']) {
      const roster = page.getByRole('region',{name:`Players in ${title}`,exact:true});
      assert.equal(await roster.locator('[data-player-identity]').count(),8);
      assert.deepEqual(await roster.locator('[data-player-identity] strong').allTextContents(),numbers,'Each group uses plain player names beneath its own heading');
    }
    await snapshot(page,'grouped-numbered-rosters.png');
    record('Players are grouped by round robin with plain numbered names beneath each group heading');
    const playerRow = page.locator(`[data-player-identity="${guestA.identity}"]`);
    detail = await command(page,playerRow.getByRole('button',{name:'Pause player',exact:true}),'pause_player');
    assert(detail.workspace.players.find(player=>player.identity === guestA.identity).paused);
    assert(!detail.workspace.players.find(player=>player.identity === guestB.identity).paused,'Pausing guest 1 in A must not pause guest 1 in B');
    assert(detail.workspace.blocked_queue.every(match=>match.session_key === numberedA.session_key));
    assert.equal(detail.workspace.eligible_queue.filter(match=>match.session_key === numberedB.session_key).length,2);
    await command(page,playerRow.getByRole('button',{name:'Ready again',exact:true}),'resume_player');
    await assignFromGroup('Numbers A');
    detail = await assignFromGroup('Numbers B');
    assert.equal(occupied(detail).length,2);
    assert.deepEqual(new Set(occupied(detail).map(court=>court.current.session_key)),new Set([numberedA.session_key,numberedB.session_key]));
    const occupants = occupied(detail).flatMap(court=>people(court.current).map(player=>player.identity));
    assert.equal(new Set(occupants).size,8);
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    for (const court of occupied(detail)) {
      const match = court.current;
      const label = `${match.side_a.map(player=>player.name).join(' / ')} vs ${match.side_b.map(player=>player.name).join(' / ')}`;
      assert.equal(await page.getByRole('article',{name:court.label,exact:true}).locator('strong').first().textContent(),label,'Court match names must not repeat group titles beside each player');
    }
    await snapshot(page,'different-numbered-guests-playing.png');
    await page.setViewportSize({width:390,height:844});
    await snapshot(page,'mobile-different-numbered-guests.png');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),'Mobile court board must not overflow');
    await page.setViewportSize({width:1440,height:1000});
    record('Different guests retain independent pause state after reload and matches from both groups use courts');

    const aCourt = occupied(detail).find(court=>court.current.session_key === numberedA.session_key);
    const bCourt = occupied(detail).find(court=>court.current.session_key === numberedB.session_key);
    detail = await scoreCourt(aCourt.label);
    let sessionState = (await fixture()).sessions.find(session=>session.session_key === numberedA.session_key);
    assert.equal(sessionState.event.currentRoundNumber,1);
    assert.equal(sessionState.event.rounds[0].status,'active');
    assert(sessionState.standings.every(row=>row.matches === 0),'Partial round scores must not count yet');
    assert.equal(detail.workspace.groups.find(group=>group.session_key === numberedA.session_key).completed_matches,1);
    assert.equal(detail.workspace.courts.find(court=>court.id === aCourt.id).current,null);
    await assignFromGroup('Numbers A');
    detail = await scoreCourt(aCourt.label);
    const nextA = detail.workspace.groups.find(group=>group.session_key === numberedA.session_key);
    const stillB = detail.workspace.groups.find(group=>group.session_key === numberedB.session_key);
    assert.equal(nextA.current_round_number,2,'Final score must advance the same group automatically');
    assert.equal(nextA.completed_matches,0);
    assert.equal(stillB.current_round_number,1,'Other group must stay on its own round');
    assert.equal(stillB.completed_matches,0);
    assert.equal(detail.workspace.courts.find(court=>court.id === bCourt.id).current.match_id,bCourt.current.match_id);
    assert.equal(detail.workspace.courts.find(court=>court.id === aCourt.id).current,null,'New matches queue; they must not auto-assign');
    assert.equal(detail.workspace.eligible_queue.filter(match=>match.session_key === numberedA.session_key && match.round_number === 2).length,2);
    sessionState = (await fixture()).sessions.find(session=>session.session_key === numberedA.session_key);
    assert.equal(sessionState.event.rounds[0].status,'saved');
    assert(sessionState.standings.every(row=>row.matches === 1),'Finished round scores must now count');
    await page.getByRole('tab',{name:'Round robins',exact:true}).click();
    await snapshot(page,'automatic-round-two-other-group-round-one.png');
    assert.equal(await page.getByRole('button',{name:'Next round',exact:true}).count(),0,'Automatic advancement must not require another Next round click');
    await page.reload();
    await page.getByRole('heading',{name:'Numbered guest review',exact:true}).waitFor();
    detail = await readHub(numberedHubId);
    assert.equal(detail.workspace.groups.find(group=>group.session_key === numberedA.session_key).current_round_number,2);
    assert.equal(detail.workspace.groups.find(group=>group.session_key === numberedB.session_key).current_round_number,1);
    await page.getByRole('tab',{name:'Match queue',exact:true}).click();
    await snapshot(page,'automatic-next-round-queue.png');
    record('Final score saves standings and queues the next round automatically; the other group and court stay unchanged after reload');

    const hubUrl = page.url();
    const bCurrentBefore = (await fixture()).sessions.find(session=>session.session_key === numberedB.session_key).event.rounds[0];
    const bMatchBefore = bCurrentBefore.matches.find(match=>match.id === bCourt.current.match_id);
    let addRequest, addCommitted;
    const loseAddResponse = async route => {
      if (route.request().method() !== 'POST' || route.request().postDataJSON()?.action !== 'add_player') return route.continue();
      addRequest = route.request().postDataJSON();
      const response = await route.fetch();
      addCommitted = {status:response.status(),body:await response.json()};
      return route.fulfill({status:503,contentType:'application/json',headers:{'access-control-allow-origin':webOrigin},
        body:JSON.stringify({detail:'Local rehearsal: player was added but the response was lost.'})});
    };
    let addDialog = await openAddPlayer('Numbers B','Late Guest');
    await page.route(commandUrl,loseAddResponse);
    const failedAdd = page.waitForResponse(response=>response.url() === commandUrl && response.request().postDataJSON()?.action === 'add_player');
    const [lostAdd] = await Promise.all([failedAdd,addDialog.getByRole('button',{name:'Add player to round robin',exact:true}).click()]);
    assert.equal(lostAdd.status(),503);
    await page.unroute(commandUrl,loseAddResponse);
    assert.equal(addCommitted.status,200,JSON.stringify(addCommitted.body));
    assert.equal(await addDialog.getByRole('combobox',{name:'Player name or club profile',exact:true}).inputValue(),'Late Guest');
    await addDialog.getByRole('alert').first().waitFor();
    await page.screenshot({path:path.join(output,'guest-add-retry-preserved.png'),fullPage:false});
    const retriedAdd = page.waitForResponse(response=>response.url() === commandUrl && response.request().postDataJSON()?.action === 'add_player');
    const [addReplay] = await Promise.all([retriedAdd,addDialog.getByRole('button',{name:'Retry saved action',exact:true}).click()]);
    detail = await addReplay.json();
    assert(addReplay.ok(),JSON.stringify(detail));
    assert.deepEqual(addReplay.request().postDataJSON(),addRequest);
    assert.equal(detail.hub.version,addCommitted.body.hub.version,'Exact add-player replay must not add twice');
    await addDialog.waitFor({state:'hidden'});
    assert.equal(page.url(),hubUrl,'Adding a guest must keep the organizer in the hub');
    let rosterState = await fixture();
    let bSession = rosterState.sessions.find(session=>session.session_key === numberedB.session_key);
    let aSession = rosterState.sessions.find(session=>session.session_key === numberedA.session_key);
    assert.equal(bSession.event.participants.length,9);
    assert.equal(aSession.event.participants.length,8);
    assert.equal(bSession.event.participants.filter(player=>player.name === 'Late Guest').length,1);
    assert.equal(bSession.event.participants.find(player=>player.name === 'Late Guest').player_id ?? null,null);
    assert.deepEqual(bSession.event.rounds[0].matches.find(match=>match.id === bCourt.current.match_id),bMatchBefore,'Adding a player must preserve the game already on court');
    assert.equal(detail.workspace.courts.find(court=>court.id === bCourt.id).current.match_id,bCourt.current.match_id);
    await snapshot(page,'guest-added-to-selected-group.png');
    record('Guest added inside its group without navigation; response-loss retry adds exactly one participant and preserves the active match');

    for (const title of ['Numbers A','Numbers B']) {
      addDialog = await openAddPlayer(title,'Player 1',1);
      detail = await command(page,addDialog.getByRole('button',{name:'Add player to round robin',exact:true}),'add_player');
      await addDialog.waitFor({state:'hidden'});
      assert.equal(page.url(),hubUrl);
    }
    rosterState = await fixture();
    for (const key of [numberedA.session_key,numberedB.session_key]) {
      const session = rosterState.sessions.find(session=>session.session_key === key);
      assert.equal(session.event.participants.find(player=>player.name === 'Player 1').player_id,1);
    }
    assert.deepEqual(new Set(detail.workspace.players.find(player=>player.identity === 'player:1').session_keys),new Set([numberedA.session_key,numberedB.session_key]));
    for (const title of ['Numbers A','Numbers B']) assert.equal(await page.getByRole('region',{name:`Players in ${title}`,exact:true}).locator('[data-player-identity="player:1"]').count(),1);
    addDialog = await openAddPlayer('Numbers A','Late Guest');
    detail = await command(page,addDialog.getByRole('button',{name:'Add player to round robin',exact:true}),'add_player');
    await addDialog.waitFor({state:'hidden'});
    assert.equal(detail.workspace.identity_pending_count,1,'A newly added matching guest name must request review');
    assert.equal(page.url(),hubUrl);
    await page.getByRole('region',{name:'Matching player names',exact:true}).getByRole('article',{name:'Identity review for Late Guest',exact:true}).waitFor();
    await snapshot(page,'added-guest-needs-identity-review.png');
    detail = await command(page,page.getByRole('article',{name:'Identity review for Late Guest',exact:true}).getByRole('button',{name:/^Different people:/}),'resolve_identities');
    assert.equal(detail.workspace.identity_pending_count,0);
    assert.equal(detail.workspace.players.filter(player=>player.name === 'Late Guest').length,2);
    assert.equal(detail.workspace.courts.find(court=>court.id === bCourt.id).current.match_id,bCourt.current.match_id);
    await page.setViewportSize({width:390,height:844});
    await snapshot(page,'mobile-grouped-rosters-after-additions.png');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth),'Grouped rosters must not overflow on mobile');
    await page.setViewportSize({width:1440,height:1000});
    record('Linked additions retain one shared profile identity in both groups; matching added guests prompt review within the same hub');

    const sameA = await createSession('Same Guest A',['Shared Guest','A Guest 2','A Guest 3','A Guest 4']);
    const sameB = await createSession('Same Guest B',['Shared Guest','B Guest 2','B Guest 3','B Guest 4']);
    detail = await createHub('Same guest across groups',[sameA,sameB]);
    assert.equal(detail.workspace.identity_pending_count,1);
    await openReview();
    detail = await command(page,page.locator('[data-review-id]').getByRole('button',{name:/^Same person:/}),'resolve_identities');
    assert.equal(detail.workspace.identity_pending_count,0);
    detail = await fillCourts();
    assert.equal(occupied(detail).length,1,'Confirmed same person cannot play two courts simultaneously');
    assert.equal(detail.workspace.blocked_queue.length,1);
    assert(detail.workspace.blocked_queue[0].blockers.some(blocker=>/already assigned/.test(blocker)));
    const sharedGuest = detail.workspace.players.find(player=>player.name === 'Shared Guest');
    assert.equal(sharedGuest.session_keys.length,2);
    await page.reload();
    await page.getByRole('heading',{name:'Same guest across groups',exact:true}).waitFor();
    detail = await readHub(detail.hub.id);
    assert.equal(detail.workspace.identity_pending_count,0);
    assert.equal(occupied(detail).length,1);
    record('Same-person confirmation persists and prevents assignment to two courts across groups');

    // Old persisted hubs identified guests by name. Preserve their active
    // courts while the real read/command paths upgrade identities safely.
    const legacyHubId = detail.hub.id;
    const legacyCourt = occupied(detail)[0];
    const legacyState = await requestApi(`/identity-fixture/legacy-hubs/${legacyHubId}`,{});
    assert.equal(legacyState.identity_version,null);
    assert(legacyState.player_keys.includes('name:shared guest'));
    await page.reload();
    await page.getByRole('heading',{name:'Same guest across groups',exact:true}).waitFor();
    detail = await readHub(legacyHubId);
    assert.equal(detail.workspace.identity_pending_count,1);
    assert.equal(occupied(detail)[0].current.match_id,legacyCourt.current.match_id);
    assert.equal((await requestApi(`/identity-fixture/hubs/${legacyHubId}`)).identity_version,null,'Read-only legacy upgrade must not write');
    await page.getByRole('tab',{name:'Court board',exact:true}).click();
    await snapshot(page,'legacy-active-court-review.png');
    await page.getByRole('article',{name:legacyCourt.label,exact:true}).getByRole('button',{name:'Enter score',exact:true}).click();
    await page.getByLabel('Side A score',{exact:true}).fill('11');
    await page.getByLabel('Side B score',{exact:true}).fill('6');
    detail = await command(page,page.getByRole('button',{name:'Save score & release court',exact:true}),'score_and_release');
    assert.equal(occupied(detail).length,0);
    assert.equal(detail.workspace.groups.find(group=>group.session_key === legacyCourt.current.session_key).current_round_number,2);
    assert.equal(detail.workspace.groups.find(group=>group.session_key === legacyCourt.current.session_key).completed_matches,0);
    const upgradedState = await requestApi(`/identity-fixture/hubs/${legacyHubId}`);
    assert.equal(upgradedState.identity_version,2);
    assert(upgradedState.player_keys.every(key=>!key.startsWith('name:')));
    await openReview();
    detail = await command(page,page.locator('[data-review-id]').getByRole('button',{name:/^Different people:/}),'resolve_identities');
    assert.equal(detail.workspace.eligible_queue.length,2);
    record('Legacy active courts survive reload and can save scores before resolving matching names');

    // Pick a real profile in the builder, then mix in manual guests. This is
    // intentionally UI-driven to catch sparse profile selections being lost.
    await page.goto(`${webOrigin}/admin/round-robin-generator?draft=identity-mixed-profile`);
    await page.getByLabel('Session title',{exact:true}).fill('Mixed profile roster');
    await page.getByRole('combobox',{name:/^Number of players/}).selectOption('4');
    await page.getByLabel('Search club players',{exact:true}).fill('Player 1');
    await page.getByRole('button',{name:'Add',exact:true}).click();
    await page.getByRole('textbox',{name:/^Players \(/}).fill('Player 1\nMixed Guest 2\nMixed Guest 3\nMixed Guest 4');
    const previewResponse = page.waitForResponse(response=>response.url().endsWith('/preview') && response.request().method() === 'POST');
    const [preview] = await Promise.all([previewResponse,page.getByRole('button',{name:'Preview matchups',exact:true}).click()]);
    assert(preview.ok());
    assert.deepEqual(preview.request().postDataJSON().participant_player_ids,{'Player 1':1});
    const mixed = (await clickRequest(page,page.getByRole('button',{name:'Start unrated session',exact:true}),'/sessions')).session;
    await page.waitForURL(/\/rounds\/1(?:#.*)?$/);
    assert.equal(mixed.event.participants.find(player=>player.name === 'Player 1').player_id,1);
    assert.equal(mixed.event.participants.filter(player=>player.player_id != null).length,1);
    const sameProfile = await createSession('Same linked profile',['Player 1','Linked Guest 2','Linked Guest 3','Linked Guest 4'],{'Player 1':1});
    detail = await createHub('Linked profile safety',[mixed,sameProfile]);
    assert.equal(detail.workspace.identity_pending_count,0,'Same linked profile is automatically one person');
    detail = await fillCourts();
    assert.equal(occupied(detail).length,1);
    assert.equal(detail.workspace.blocked_queue.length,1);
    assert(detail.workspace.players.some(player=>player.identity === 'player:1' && player.session_keys.length === 2));
    record('Mixed roster keeps its selected profile; the same profile remains a hard cross-group court constraint');

    const profileA = await createSession('Profile Alex A',['Alex Same','Alex A2','Alex A3','Alex A4'],{'Alex Same':101});
    const profileB = await createSession('Profile Alex B',['Alex Same','Alex B2','Alex B3','Alex B4'],{'Alex Same':102});
    detail = await createHub('Distinct profiles same name',[profileA,profileB]);
    assert.equal(detail.workspace.identity_pending_count,0,'Distinct linked profiles must not be merged by display name');
    detail = await fillCourts();
    assert.equal(occupied(detail).length,2);
    const alex = occupied(detail).flatMap(court=>people(court.current)).filter(player=>player.name === 'Alex Same');
    assert.deepEqual(new Set(alex.map(player=>player.identity)),new Set(['player:101','player:102']));
    await snapshot(page,'distinct-profiles-same-name.png');
    record('Two different club profiles with the same display name play independently');

    expectedConsoleErrors.push(...errors.filter(error=>error.type === 'console' && error.url === commandUrl && /503/.test(error.message)));
    const unexpectedErrors = errors.filter(error=>!expectedConsoleErrors.includes(error));
    assert(expectedConsoleErrors.length <= 2,'Only the two intentionally lost responses may emit network errors');
    assert.deepEqual(unexpectedErrors,[],'No unexpected browser errors');
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify({passed:true,evidence,errors:unexpectedErrors,expectedConsoleErrors,
      sessions:(await fixture()).sessions.length},null,2));
    console.log(`Identity browser regression passed. Artifacts: ${output}`);
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
