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
    ['-m', 'uvicorn', 'generator_submission_api:app', '--app-dir', path.join(root, 'tests/e2e'), '--host', '127.0.0.1', '--port', apiPort],
    {cwd:root, env});
  const next = start(process.execPath, [require.resolve('next/dist/bin/next'), 'dev', '--hostname', '127.0.0.1', '--port', webPort], {cwd:webRoot, env});
  let browser;
  try {
    await ready(`${apiOrigin}/fixture-result`, [api, next]);
    assert.equal((await fixture()).sessions.length, 0, 'The test must start with an empty session database');
    await ready(`${webOrigin}/clubs/test/round-robin-generator`, [api, next]);
    browser = await chromium.launch({headless:true,
      ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE} : {}),
      args:['--no-sandbox', '--disable-dev-shm-usage']});
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
      await page.getByRole('textbox', {name:/^Players \(/}).fill('Player 1\nPlayer 2\nPlayer 3\nPlayer 4');
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
      await page.getByRole('heading', {name:`Club results · ${mode === 'rated' ? 'Rated' : 'Unrated'}`, exact:true}).waitFor();
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
        await page.getByRole('link', {name:'View standings and continue', exact:true}).click();
        await page.waitForURL(/\/standings$/);
        await page.getByRole('heading', {name:`${title} standings`, exact:true}).waitFor();
        assert.equal(await page.getByRole('button', {name:'Submit for approval', exact:true}).count(), 0);
        const last = round === rounds;
        const advanced = await clickRequest(page, page.getByRole('button', {name:last ? 'Finish session' : `Continue to Round ${round+1}`, exact:true}), '/advance');
        session = advanced.body.session;
        assert.equal(session.rating_mode, mode);
        if (!last) await page.waitForURL(new RegExp(`/rounds/${round+1}(?:#.*)?$`));
        console.log(`PASS ${mode}: scored round ${round}, viewed standings, ${last ? 'finished' : 'advanced'}`);
      }
      assert.equal(session.status, 'completed');
      await page.getByRole('heading', {name:'Final standings', exact:true}).waitFor();
      // User finishes on this page: the submission must be available here.
      const submit = page.getByRole('button', {name:'Submit for approval', exact:true});
      await submit.waitFor();
      const standingsUrl = page.url();
      await screenshot(page, `${mode}-completed-standings.png`);
      const completed = await fixture();
      assert.deepEqual(completed.matches, before.matches, 'Finished but unapproved scores must not create official matches');
      assert.deepEqual(completed.players, before.players, 'Finished but unapproved scores must not update ratings');

      await spectator.goto(standingsUrl);
      await spectator.getByRole('heading', {name:'Final standings', exact:true}).waitFor();
      assert.equal(await spectator.getByRole('button', {name:'Submit for approval', exact:true}).count(), 0, 'Spectators cannot submit');
      const denied = await fetch(`${apiOrigin}/clubs/test/play-generators/sessions/${session.session_key}/submit`, {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({edit_token:'invalid-spectator-token', expected_version:session.version,
          idempotency_key:`spectator-denied-${mode}`, organizer_name:'Spectator', match_date:'2026-09-20'})
      });
      assert.equal(denied.status, 403, 'The API also denies a spectator submission');
      await page.getByLabel('Organizer’s name').fill('Guest organizer');
      await page.getByLabel('Date played', {exact:true}).fill('2026-09-20');
      const submitted = await clickRequest(page, submit, '/submit');
      assert.equal(submitted.body.session.submission.rating_mode, mode);
      assert.equal(submitted.body.session.submission.status, 'pending');
      await page.getByRole('status').filter({hasText:'Awaiting admin approval'}).waitFor();
      const pending = await fixture();
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
      await adminPage.getByRole('button', {name:approveLabel, exact:true}).click();
      const approved = await clickRequest(adminPage, adminPage.getByRole('dialog').getByRole('button', {name:approveLabel, exact:true}), '/review');
      assert.equal(approved.body.session.submission.approved_mode, mode);
      await adminPage.getByRole('heading', {name:'Review saved', exact:true}).waitFor();
      const after = await fixture();
      const records = after.matches.slice(before.matches.length);
      const expectedGames = session.event.rounds.reduce((sum, r) => sum+(r.matches || []).length, 0);
      assert.equal(records.length, expectedGames);
      assert(records.every(match => match.rating_scope === (mode === 'rated' ? 'overall_only' : 'unrated')));
      if (mode === 'rated') assert.notDeepEqual(after.players, before.players, 'Rated approval updates player ratings');
      else assert.deepEqual(after.players, before.players, 'Unrated approval leaves player ratings untouched');
      console.log(`PASS ${mode}: guest submitted from final standings; admin approval created ${records.length} official matches`);
      await spectator.reload();
      await spectator.getByRole('status').filter({hasText:`Approved · ${capitalized}`}).waitFor();
      await spectator.setViewportSize({width:390, height:844});
      await screenshot(spectator, `${mode}-approved-mobile.png`);
      const dimensions = await spectator.evaluate(() => ({width:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth}));
      assert(dimensions.scroll <= dimensions.width+1, `Mobile overflow: ${JSON.stringify(dimensions)}`);
      evidence.push({mode, format, rounds, matches:records.length, status:'approved'});
      console.log(`PASS ${mode} ${format}: setup -> preview -> start -> ${rounds} scored rounds -> standings -> finish -> guest submit -> admin approve; ${records.length} official matches`);
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
