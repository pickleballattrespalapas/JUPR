/*
 * Local end-to-end regression: real Next UI -> FastAPI routes -> services ->
 * in-memory database. Starts with NO sessions; never bypasses creation/completion.
 *
 * From apps/web: node tests/generator-submission-browser.cjs
 * Requires npm dependencies, Python test dependencies plus uvicorn, and a
 * Playwright browser (or GENERATOR_BROWSER_EXECUTABLE pointing to Chromium).
 * Optional GENERATOR_BROWSER_PYTHON, GENERATOR_BROWSER_ARTIFACTS and
 * GENERATOR_BROWSER_WEB_PORT / GENERATOR_BROWSER_API_PORT configure execution.
 * Post-commit badge/email side effects and real authentication are not tested.
 */
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

const webRoot = path.resolve(__dirname, '..');
const root = path.resolve(webRoot, '../..');
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || '3038';
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || '8038';
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(), 'generator-browser-'));
const nextMode = process.env.GENERATOR_BROWSER_NEXT_MODE || 'dev';
assert(['dev', 'start'].includes(nextMode), 'Use dev or a prebuilt start server');
fs.mkdirSync(output, {recursive:true});
const logs = [];
function start(command, args, options) {
  const child = spawn(command, args, {...options, stdio:['ignore', 'pipe', 'pipe']});
  child.stdout.on('data', data => logs.push(data.toString()));
  child.stderr.on('data', data => logs.push(data.toString()));
  child.on('error', error => { child.startupError = error; logs.push(error.message); });
  return child;
}
async function ready(url, children) {
  for (let i=0; i<120; i++) {
    for (const child of children) {
      if (child.startupError) throw child.startupError;
      if (child.exitCode !== null) throw new Error(`Local test server exited before startup (${child.exitCode})`);
    }
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error(`Local test server unavailable: ${url}`);
}
async function fixture() {
  const response = await fetch(`${apiOrigin}/fixture-result`);
  assert(response.ok, `fixture state: ${response.status}`);
  return response.json();
}
async function clickRequest(page, button, suffix, method = 'POST') {
  const pending = page.waitForResponse(response => response.url().startsWith(apiOrigin)
    && response.url().endsWith(suffix) && response.request().method() === method);
  await button.click();
  const response = await pending;
  const body = await response.json();
  assert(response.ok(), `${suffix}: ${response.status()} ${JSON.stringify(body)}`);
  return {body, request:response.request().postDataJSON()};
}
async function screenshot(page, name) {
  await page.screenshot({path:path.join(output, name), fullPage:true});
}

async function main() {
  const env = {...process.env, VERCEL_ENV:'development', JUPR_API_BASE_URL:apiOrigin,
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging',
    GENERATOR_BROWSER_WEB_PORT:webPort};
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m', 'uvicorn', 'round_robin_hub_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort],
    {cwd:root, env});
  const next = start(process.execPath, [require.resolve('next/dist/bin/next'), nextMode, ...(nextMode === 'dev' ? ['--webpack'] : []), '--hostname', '127.0.0.1', '--port', webPort], {cwd:webRoot, env});
  let browser;
  try {
    await ready(`${apiOrigin}/fixture-result`, [api, next]);
    assert.equal((await fixture()).sessions.length, 0, 'The test must start with an empty session database');
    await ready(`${webOrigin}/clubs/test/round-robin-generator`, [api, next]);
    browser = await chromium.launch({headless:true,
      ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE} : {}),
      args:['--no-sandbox', '--disable-dev-shm-usage', ...JSON.parse(process.env.GENERATOR_BROWSER_ARGS || '[]')]});
    const errors = [];
    const contexts = [];
    async function makeContext() {
      const context = await browser.newContext({viewport:{width:1280, height:900}});
      contexts.push(context);
      // Rehearsal is isolated to local services even if navigation regresses.
      await context.route('**/*', route => {
        const url = new URL(route.request().url());
        return ['127.0.0.1', 'localhost'].includes(url.hostname) ? route.continue() : route.abort();
      });
      context.on('page', page => page.on('pageerror', error => errors.push(error.message)));
      return context;
    }
    const adminContext = await makeContext();
    await adminContext.addCookies([{name:'jupr_admin_workspace_v1', value:encodeURIComponent(JSON.stringify({clubId:'club', clubSlug:'test'})), url:webOrigin}]);
    await adminContext.addInitScript(origin => {
      // A fresh page starts at about:blank, where localStorage is unavailable.
      if (window.location.origin !== origin) return;
      localStorage.setItem('jupr_admin_session_v1', JSON.stringify({
        access_token:'local-reviewer', expires_at:Date.now()+3600000,
        user:{id:'local-reviewer', email:'reviewer@example.invalid'},
        capabilities:{authorized:true, user:{email:'reviewer@example.invalid'}, assignments:[{club_id:'club', role:'administrator', permissions:['enter_scores', 'manage_players']}]}
      }));
    }, webOrigin);
    const adminPage = await adminContext.newPage();
    const spectatorContext = await makeContext();
    const spectator = await spectatorContext.newPage();
    const evidence = [];

    for (const [mode, format] of [['rated', 'doubles'], ['unrated', 'singles']]) {
      let createdPlayerId;
      console.log(`Checking ${mode} ${format} from an empty organizer browser`);
      const context = await makeContext();
      const page = await context.newPage();
      const title = `${mode} RR browser rehearsal`;
      const before = await fixture();
      await page.goto(`${webOrigin}/clubs/test/round-robin-generator`);
      await page.getByLabel('Session title', {exact:true}).fill(title);
      await page.getByRole('combobox', {name:/^Play format/}).selectOption(format);
      await page.getByRole('combobox', {name:/^Number of players/}).selectOption('4');
      await page.getByLabel('Match rating', {exact:true}).selectOption(mode);
      assert.equal(await page.getByLabel('Round scoring', {exact:false}).inputValue(), 'scored');
      assert.equal(await page.getByLabel('Round scoring', {exact:false}).isDisabled(), mode === 'rated');
      await page.getByRole('textbox', {name:/^Players \(/}).fill(mode === 'rated'
        ? 'Player 1\nPlayer 2\nPlayer 3\nNew Browser Player'
        : 'Player 1\nPlayer 2\nPlayer 3\nPlayer 4');
      const preview = await clickRequest(page, page.getByRole('button', {name:'Preview matchups', exact:true}), '/preview');
      assert.equal(preview.request.rating_mode, mode);
      assert.equal(preview.body.preview.ratingMode, mode);
      const started = await clickRequest(page, page.getByRole('button', {name:`Start ${mode} session`, exact:true}), '/sessions');
      assert.equal(started.request.rating_mode, mode);
      assert.equal(started.body.session.rating_mode, mode);
      let session = started.body.session;
      assert.equal(session.status, 'active');
      const rounds = session.event.rounds.length;
      assert(rounds > 1, 'Rehearsal must exercise intermediate standings and every round');
      await page.waitForURL(/\/rounds\/1(?:#.*)?$/);
      await page.getByRole('heading', {name:title, exact:true}).waitFor();
      await page.getByText(new RegExp(`Scores on.*${mode === 'rated' ? 'Rated' : 'Unrated'}.*Not added to club records`)).waitFor();
      assert.equal(await page.getByRole('button', {name:'Submit for approval', exact:true}).count(), 0, 'Cannot submit unfinished games');
      await screenshot(page, `${mode}-started.png`);
      console.log(`PASS ${mode}: preview/start preserves rating choice; ${rounds} rounds scheduled`);

      for (let round=1; round<=rounds; round++) {
        const scores = page.getByRole('spinbutton');
        await scores.first().waitFor();
        const a = page.locator('input[aria-label$="side A score"]');
        const b = page.locator('input[aria-label$="side B score"]');
        const matchCount = await a.count();
        assert(matchCount > 0);
        for (let i=0; i<matchCount; i++) {
          await a.nth(i).fill('11');
          await b.nth(i).fill(String(6 + round));
        }
        const saved = await clickRequest(page, page.getByRole('button', {name:'Save round scores', exact:true}), `/rounds/${round}/scores`, 'PATCH');
        assert.equal(saved.body.session.rating_mode, mode);
        assert.equal(saved.body.session.event.rounds[round-1].status, 'saved');
        const last = round === rounds;
        if (last) {
        await page.getByRole('link', {name:'Standings', exact:true}).click();
        await page.waitForURL(/\/standings$/);
        await page.getByRole('heading', {name:`${title} standings`, exact:true}).waitFor();
        assert.equal(await page.getByRole('button', {name:'Submit for approval', exact:true}).count(), 0);
        if (mode === 'unrated') {
          // Keep coverage of hub-linked sessions alongside standalone sessions.
          const headers = {Authorization:'Bearer local-reviewer'};
          const hubResponse = await adminContext.request.post(`${apiOrigin}/admin/clubs/club/round-robin-hubs`, {
            headers, data:{title:'Submission rehearsal hub', courts:[{id:'court-1',label:'Court 1'}], idempotency_key:'submission-hub-create'}
          });
          assert(hubResponse.ok(), await hubResponse.text());
          const hub = (await hubResponse.json()).hub;
          const linked = await adminContext.request.post(`${apiOrigin}/admin/clubs/club/round-robin-hubs/${hub.id}/commands`, {
            headers, data:{action:'add_group', payload:{session_key:session.session_key}, expected_version:hub.version, idempotency_key:'submission-hub-link'}
          });
          assert(linked.ok(), await linked.text());
          await page.reload();
        }
        }
        const advanced = await clickRequest(page, page.getByRole('button', {name:last ? 'Finish session' : 'Next scheduled round', exact:true}), last ? '/complete' : '/advance');
        session = advanced.body.session;
        assert.equal(session.rating_mode, mode);
        if (!last) await page.waitForURL(new RegExp(`/rounds/${round+1}(?:#.*)?$`));
        console.log(`PASS ${mode}: scored round ${round}, viewed standings, ${last ? 'finished' : 'advanced'}`);
      }
      assert.equal(session.status, 'completed');
      const originalRounds = structuredClone(session.event.rounds);
      for (const ratingChoice of [mode === 'rated' ? 'unrated' : 'rated', mode]) {
        await page.getByRole('button', {name:'Edit session details', exact:true}).click();
        await page.getByLabel('Session rating', {exact:true}).selectOption(ratingChoice);
        const edited = await clickRequest(page, page.getByRole('button', {name:'Save session details', exact:true}), '/details', 'PATCH');
        session = edited.body.session;
        assert.equal(session.rating_mode, ratingChoice);
        assert.deepEqual(session.event.rounds, originalRounds, 'Editing rating intent preserves every saved score');
      }

      await page.getByRole('heading', {name:'Final standings', exact:true}).waitFor();
      // User finishes on this page: the submission must be available here.
      const submit = page.getByRole('button', {name:'Submit for approval', exact:true});
      await submit.waitFor();
      const standingsUrl = page.url();
      await screenshot(page, `${mode}-completed-standings.png`);
      const completed = await fixture();
      assert.deepEqual(completed.matches, before.matches, 'Finished but unapproved scores must not create official matches');
      assert.deepEqual(completed.players, before.players, 'Finished but unapproved scores must not update ratings');

      // Start at the screen staff actually use: standalone sessions must be
      // discoverable here just like sessions created through the RR Hub.
      await adminPage.goto(`${webOrigin}/admin/round-robin-generator`);
      const sessionCard = adminPage.locator('article').filter({has:adminPage.getByText(title, {exact:true})}).last();
      const openSession = sessionCard.getByRole('link', {name:'Open session', exact:true});
      await openSession.waitFor();
      const sessionHref = await openSession.getAttribute('href');
      assert.equal(sessionHref, `/admin/round-robin-generator/sessions/${session.session_key}/rounds/${session.current_round_number}`);
      await adminPage.goto(new URL(sessionHref, webOrigin).href);
      await adminPage.getByRole('heading', {name:title, exact:true}).waitFor();
      const standingsHref = await adminPage.getByRole('link', {name:'Standings', exact:true}).getAttribute('href');
      await adminPage.goto(new URL(standingsHref, webOrigin).href);
      await adminPage.getByRole('button', {name:'Submit for approval', exact:true}).waitFor();
      console.log(`PASS ${mode}: staff found and opened the ${mode === 'rated' ? 'standalone' : 'hub-linked'} completed session from the session list`);

      await spectator.goto(standingsUrl);
      await spectator.getByRole('heading', {name:'Final standings', exact:true}).waitFor();
      assert.equal(await spectator.getByRole('button', {name:'Submit for approval', exact:true}).count(), 0, 'Spectators cannot submit');
      const denied = await fetch(`${apiOrigin}/clubs/test/play-generators/sessions/${session.session_key}/submit`, {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({edit_token:'invalid-spectator-token', expected_version:session.version,
          idempotency_key:`spectator-denied-${mode}`, organizer_name:'Spectator', match_date:'2026-09-20'})
      });
      assert.equal(denied.status, 403, 'The API also denies a spectator submission');
      // Reproduce a club admin submitting a public-origin session without its
      // organizer token; the other scenario still exercises organizer submission.
      const submitPage = mode === 'unrated' ? adminPage : page;
      if (submitPage === adminPage) await adminPage.goto(`${webOrigin}/admin/round-robin-generator/sessions/${session.session_key}/standings`);
      await submitPage.getByLabel('Organizer’s name').fill('Guest organizer');
      await submitPage.getByLabel('Date played', {exact:true}).fill('2026-09-20');
      const submitted = await clickRequest(submitPage, submitPage.getByRole('button', {name:'Submit for approval', exact:true}), '/submit');
      if (submitPage === adminPage) {
        assert.equal(submitted.request.edit_token, undefined, 'Admin submission uses club authorization, not an organizer token');
        await page.reload();
      }
      assert.equal(submitted.body.session.submission.rating_mode, mode);
      assert.equal(submitted.body.session.submission.status, 'pending');
      await page.getByRole('status').filter({hasText:'Awaiting admin approval'}).waitFor();
      const pending = await fixture();
      const submittedVersion = Number(pending.sessions.find(item => item.session_key === session.session_key).version);
      assert.deepEqual(pending.matches, before.matches);
      assert.deepEqual(pending.players, before.players);
      await spectator.reload();
      await spectator.getByRole('status').filter({hasText:'Awaiting admin approval'}).waitFor();
      await screenshot(page, `${mode}-pending.png`);

      await adminPage.goto(`${webOrigin}/admin/play-generators/submissions`);
      await adminPage.getByRole('button', {name:new RegExp(title)}).click();
      const capitalized = mode === 'rated' ? 'Rated' : 'Unrated';
      await adminPage.getByRole('heading', {name:`${title} · ${capitalized}`, exact:true}).waitFor();
      const approveLabel = `Approve ${mode} results`;
      assert.equal(await adminPage.getByRole('button', {name:`Approve ${mode === 'rated' ? 'unrated' : 'rated'} results`, exact:true}).count(), 0);
      if (mode === 'rated') await adminPage.getByRole('spinbutton', {name:'Player 1 doubles starting level', exact:true}).fill('4.25');
      const correctedTitle = `${mode} RR corrected title`;
      const beforeTitleEdit = await fixture();
      await adminPage.getByRole('button', {name:'Edit session details', exact:true}).click();
      await adminPage.getByLabel('Session title', {exact:true}).fill('   ');
      assert.equal(await adminPage.getByRole('button', {name:'Save session details', exact:true}).isDisabled(), true, 'Blank titles cannot be saved');
      await adminPage.getByLabel('Session title', {exact:true}).fill(correctedTitle);
      assert.equal(await adminPage.getByRole('button', {name:approveLabel, exact:true}).isDisabled(), true, 'Finish the title edit before reviewing');
      const renamed = await clickRequest(adminPage, adminPage.getByRole('button', {name:'Save session details', exact:true}), '/details', 'PATCH');
      assert.deepEqual(renamed.request, {expected_version:submittedVersion, title:correctedTitle, rating_mode:mode});
      assert.equal(renamed.body.title, correctedTitle);
      assert.equal(renamed.body.version, submittedVersion + 1);
      await adminPage.getByRole('heading', {name:`${correctedTitle} · ${capitalized}`, exact:true}).waitFor();
      await adminPage.getByRole('button', {name:new RegExp(correctedTitle)}).waitFor();
      assert.equal(await adminPage.getByLabel('Date played', {exact:true}).inputValue(), '2026-09-20', 'Title save preserves the review date');
      const afterTitleEdit = await fixture();
      assert.deepEqual(afterTitleEdit.matches, beforeTitleEdit.matches, 'Renaming cannot create or change official matches');
      assert.deepEqual(afterTitleEdit.players, beforeTitleEdit.players, 'Renaming cannot change ratings or profiles');
      const renamedSession = afterTitleEdit.sessions.find(item => item.session_key === session.session_key);
      const priorSession = beforeTitleEdit.sessions.find(item => item.session_key === session.session_key);
      assert.equal(renamedSession.title, correctedTitle);
      assert.deepEqual(renamedSession.event, {...priorSession.event, name:correctedTitle}, 'Renaming preserves all scores, rounds, and roster');
      assert.deepEqual(renamedSession.submission, priorSession.submission, 'Renaming preserves the pending submission');
      await spectator.reload();
      await spectator.getByRole('heading', {name:`${correctedTitle} standings`, exact:true}).waitFor();
      await spectator.getByRole('status').filter({hasText:'Awaiting admin approval'}).waitFor();
      await screenshot(adminPage, `${mode}-title-corrected.png`);
      console.log(`PASS ${mode}: admin title edit -> versioned API save -> preserved scores/ratings -> queue and public standings updated`);
      const beforeCorrection = await fixture();
      const savedSession = beforeCorrection.sessions.find(item => item.session_key === session.session_key);
      const firstRound = savedSession.event.rounds.find(round => round.status === 'saved');
      const firstMatch = firstRound.matches[0];
      const prefix = `Round ${firstRound.number}, game ${firstMatch.id}, `;
      await adminPage.getByRole('button', {name:'Edit scores', exact:true}).click();
      await adminPage.getByLabel('Filter scores by round').selectOption(String(firstRound.number));
      const firstName = savedSession.event.participants.find(player => player.id === firstMatch.sideA[0]).name;
      await adminPage.getByLabel('Find a player in saved scores').fill(firstName);
      const scoreInputs = adminPage.getByRole('spinbutton', {name:new RegExp(`^${prefix}`)});
      assert.equal(await scoreInputs.count(), 2);
      await scoreInputs.nth(0).fill('5');
      await scoreInputs.nth(1).fill('11');
      assert(await adminPage.getByRole('button', {name:approveLabel, exact:true}).isDisabled());
      await adminPage.setViewportSize({width:390, height:844});
      await screenshot(adminPage, `${mode}-score-edit-mobile.png`);
      const editDimensions = await adminPage.evaluate(() => ({width:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth}));
      assert(editDimensions.scroll <= editDimensions.width+1, 'Score editor fits mobile width');
      const correction = await clickRequest(adminPage, adminPage.getByRole('button', {name:'Save score corrections', exact:true}), '/submission/scores', 'PATCH');
      assert.deepEqual(correction.request, {expected_version:renamed.body.version, scores:[{
        round_number:firstRound.number, match_id:firstMatch.id, score_a:5, score_b:11}]});
      assert.equal(correction.body.submission.status, 'pending');
      const afterCorrection = await fixture();
      assert.deepEqual(afterCorrection.players, beforeCorrection.players, 'Correcting pending scores does not change ratings');
      assert.deepEqual(afterCorrection.matches, beforeCorrection.matches, 'Correcting pending scores does not publish games');
      const expectedEvent = structuredClone(savedSession.event);
      Object.assign(expectedEvent.rounds.find(round => round.number === firstRound.number).matches[0], {scoreA:5, scoreB:11});
      assert.deepEqual(afterCorrection.sessions.find(item => item.session_key === session.session_key).event, expectedEvent, 'Only the chosen game scores change');
      await adminPage.setViewportSize({width:1280, height:900});
      // A separate reviewer page must also receive the durable corrected score.
      const reloaded = await adminContext.newPage();
      await reloaded.goto(`${webOrigin}/admin/play-generators/submissions`);
      await reloaded.getByRole('button', {name:new RegExp(correctedTitle)}).click();
      assert(await reloaded.getByRole('cell', {name:'5 – 11', exact:true}).count());
      await reloaded.close();
      console.log(`PASS ${mode}: filtered score edit -> API persistence -> reload -> pending; other games and player ratings preserved`);
      if (mode === 'rated') {
        assert.equal(await adminPage.getByRole('spinbutton', {name:'Player 1 doubles starting level', exact:true}).inputValue(), '4.25', 'Title save preserves an unsaved starting level');
        assert.equal(await adminPage.getByRole('button', {name:approveLabel, exact:true}).isDisabled(), true);
        const savedLevel = await clickRequest(adminPage, adminPage.getByRole('button', {name:'Save doubles starting level', exact:true}), '/starting-ratings/1', 'PATCH');
        assert.equal(savedLevel.body.player.ratings.doubles.rating_jupr, 4.25);
        const seeded = await fixture();
        const player = seeded.players.find(p => p.id === 1);
        assert.equal(player.rating, 1700);
        assert.equal(player.starting_rating, 1700);
        assert.equal(player.singles_rating, before.players.find(p => p.id === 1).singles_rating);
        assert.deepEqual(seeded.matches, before.matches, 'Saving a starting level does not approve scores');
        assert.equal(await adminPage.getByRole('button', {name:approveLabel, exact:true}).isDisabled(), true, 'Missing profiles block approval');
        await adminPage.getByRole('button', {name:'Create new player for New Browser Player', exact:true}).click();
        assert.equal(await adminPage.getByLabel('New player name for New Browser Player', {exact:true}).inputValue(), 'New Browser Player');
        assert.equal(await adminPage.getByRole('button', {name:'Create and select player', exact:true}).isDisabled(), true);
        await adminPage.getByLabel('New player starting level for New Browser Player', {exact:true}).fill('3.75');
        await adminPage.setViewportSize({width:390, height:844});
        await screenshot(adminPage, 'create-player-mobile.png');
        const dimensions = await adminPage.evaluate(() => ({width:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth}));
        assert(dimensions.scroll <= dimensions.width+1, 'Starting levels fit mobile width');
        const createButton = adminPage.getByRole('button', {name:'Create and select player', exact:true});
        // Settle focus scrolling after changing viewport with a number field focused.
        // Otherwise Chromium can move the target between mouse-down and mouse-up.
        await createButton.focus();
        const created = await clickRequest(adminPage, createButton, '/players/editor/players');
        createdPlayerId = created.body.player.id;
        await adminPage.getByRole('spinbutton', {name:'New Browser Player doubles starting level', exact:true}).waitFor();
        assert.equal(await adminPage.getByRole('spinbutton', {name:'New Browser Player doubles starting level', exact:true}).inputValue(), '3.75');
        const withNewPlayer = await fixture();
        assert.equal(withNewPlayer.players.length, seeded.players.length + 1);
        const newPlayer = withNewPlayer.players.find(p => p.id === createdPlayerId);
        assert.equal(newPlayer.rating, 1500);
        assert.equal(newPlayer.starting_rating, 1500);
        assert.equal(newPlayer.matches_played, 0);
        assert.deepEqual(withNewPlayer.matches, before.matches, 'Creating a player does not approve scores');
        const retry = await fetch(`${apiOrigin}/admin/clubs/club/players/editor/players`, {
          method:'POST', headers:{Authorization:'Bearer local-reviewer', 'Content-Type':'application/json'}, body:JSON.stringify(created.request)});
        assert(retry.ok);
        assert.equal((await retry.json()).player.id, createdPlayerId);
        assert.equal((await fixture()).players.length, withNewPlayer.players.length, 'Exact retry cannot duplicate the new player');
        assert.equal(await adminPage.getByRole('button', {name:approveLabel, exact:true}).isDisabled(), false);
        await adminPage.setViewportSize({width:1280, height:900});
        await screenshot(adminPage, 'created-player-desktop.png');
        console.log('PASS starting level: browser edit -> API -> current and baseline saved; approval blocked until save; singles preserved');
        console.log('PASS missing player: inline creation -> existing API -> one profile with explicit starting level -> automatically selected; exact retry creates no duplicate');
      }
      await adminPage.getByRole('button', {name:approveLabel, exact:true}).click();
      const approved = await clickRequest(adminPage, adminPage.getByRole('dialog').getByRole('button', {name:approveLabel, exact:true}), '/review');
      assert.equal(approved.request.expected_version, correction.body.submission.version, 'Approval uses the version returned by the score correction');
      assert.equal(approved.body.session.submission.approved_mode, mode);
      await adminPage.getByRole('heading', {name:'Review saved', exact:true}).waitFor();
      const after = await fixture();
      const records = after.matches.slice(before.matches.length);
      // The hub may prepare another unplayed round. Only the original rounds
      // actually scored in this rehearsal should become official matches.
      const expectedGames = started.body.session.event.rounds.reduce((sum, r) => sum+(r.matches || []).length, 0);
      assert.equal(records.length, expectedGames);
      assert.equal(records[0].score_t1, 5, 'Approval publishes the corrected Team 1 score');
      assert.equal(records[0].score_t2, 11, 'Approval publishes the corrected Team 2 score');
      assert(records.every(match => match.rating_scope === (mode === 'rated' ? 'overall_only' : 'unrated')));
      assert(records.every(match => match.week_tag.startsWith(`${correctedTitle} Round `)), 'Approved games use the corrected title');
      if (mode === 'rated') {
        const firstGame = records.find(m => ['t1_p1','t1_p2','t2_p1','t2_p2'].some(k => m[k] === 1));
        const position = ['t1_p1','t1_p2','t2_p1','t2_p2'].find(k => firstGame[k] === 1);
        assert.equal(firstGame[position + '_r'], 1700, 'First approved game uses the saved starting level');
        const newPlayerGame = records.find(m => ['t1_p1','t1_p2','t2_p1','t2_p2'].some(k => m[k] === createdPlayerId));
        const newPlayerPosition = ['t1_p1','t1_p2','t2_p1','t2_p2'].find(k => newPlayerGame[k] === createdPlayerId);
        assert.equal(newPlayerGame[newPlayerPosition + '_r'], 1500, 'New player starts their first approved match at the entered level');
      }
      const ratings = players => players.map(({id, rating, starting_rating, singles_rating, singles_starting_rating}) =>
        ({id, rating, starting_rating, singles_rating, singles_starting_rating}));
      if (mode === 'rated') assert.notDeepEqual(ratings(after.players), ratings(before.players), 'Rated approval updates player ratings');
      else assert.deepEqual(ratings(after.players), ratings(before.players), 'Unrated approval leaves player ratings untouched while recording activity');
      console.log(`PASS ${mode}: ${mode === 'unrated' ? 'admin' : 'organizer'} submitted from final standings; admin approval created ${records.length} official matches`);
      await spectator.reload();
      await spectator.getByRole('status').filter({hasText:`Approved · ${capitalized}`}).waitFor();
      await spectator.setViewportSize({width:390, height:844});
      await screenshot(spectator, `${mode}-approved-mobile.png`);
      const dimensions = await spectator.evaluate(() => ({width:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth}));
      assert(dimensions.scroll <= dimensions.width+1, `Mobile overflow: ${JSON.stringify(dimensions)}`);
      evidence.push({mode, format, rounds, matches:records.length, status:'approved'});
      console.log(`PASS ${mode} ${format}: setup -> preview -> start -> ${rounds} scored rounds -> standings -> finish -> ${mode === 'unrated' ? 'admin' : 'organizer'} submit -> admin approve; ${records.length} official matches`);
    }
    assert.deepEqual(errors, [], 'No browser runtime errors');
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({passed:true, evidence, browserErrors:errors}, null, 2));
    console.log('PASS spectator restrictions, public pending/approved status, mobile width, and no runtime errors');
  } catch (error) {
    console.error(error.stack || String(error));
    if (browser) for (const context of browser.contexts()) for (const page of context.pages()) {
      console.error((await page.locator('body').innerText()).slice(0, 6500));
      await screenshot(page, `failure-${Date.now()}.png`).catch(() => {});
    }
    console.error(logs.slice(-35).join('').slice(-10000));
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(output, 'servers.log'), logs.join(''));
    if (browser) await browser.close();
    next.kill('SIGTERM');
    api.kill('SIGTERM');
  }
}
main();
