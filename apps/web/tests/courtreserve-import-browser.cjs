/*
 * CourtReserve import regression: real Next UI -> FastAPI routes -> services ->
 * in-memory database. Starts with NO sessions; never bypasses creation/completion.
 *
 * From apps/web: node tests/courtreserve-import-browser.cjs
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
const {expect: expectBrowser} = require('@playwright/test');

const webRoot = path.resolve(__dirname, '..');
const root = path.resolve(webRoot, '../..');
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || '3048';
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || '8042';
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(), 'courtreserve-browser-'));
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
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging', GENERATOR_BROWSER_WEB_PORT:webPort};
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m','uvicorn','round_robin_hub_api:app','--app-dir',path.join(root,'tests/e2e'),'--host','127.0.0.1','--port',apiPort], {cwd:root,env});
  const next = start(process.execPath,[require.resolve('next/dist/bin/next'),nextMode,...(nextMode==='dev'?['--webpack']:[]),'--hostname','127.0.0.1','--port',webPort],{cwd:webRoot,env});
  let browser, page;
  try {
    await ready(`${apiOrigin}/fixture-result`,[api,next]);
    await ready(`${webOrigin}/clubs/test/round-robin-generator`,[api,next]);
    browser = await chromium.launch({headless:true,...(process.env.GENERATOR_BROWSER_EXECUTABLE?{executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE}:{}),args:['--no-sandbox','--disable-dev-shm-usage']});
    const context = await browser.newContext({viewport:{width:1280,height:900}});
    await context.route('**/*',route=>['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname)?route.continue():route.abort());
    await context.addCookies([{name:'jupr_admin_workspace_v1',value:encodeURIComponent(JSON.stringify({clubId:'club',clubSlug:'test'})),url:webOrigin}]);
    await context.addInitScript(origin=>{
      if(window.location.origin!==origin)return;
      localStorage.setItem('jupr_admin_session_v1',JSON.stringify({access_token:'local-reviewer',expires_at:Date.now()+3600000,
        user:{id:'local-reviewer',email:'reviewer@example.invalid'},capabilities:{authorized:true,user:{email:'reviewer@example.invalid'},assignments:[{club_id:'club',role:'administrator',permissions:['enter_scores','manage_players']}]}}));
    },webOrigin);
    page=await context.newPage(); const errors=[];page.on('pageerror',error=>errors.push(error.message));
    const before=await fixture();
    await page.goto(`${webOrigin}/clubs/test/round-robin-generator`);
    await page.getByLabel('Session title',{exact:true}).fill('CourtReserve inline creation');
    await page.getByLabel('Match rating',{exact:true}).selectOption('rated');
    await page.getByRole('button',{name:'Import CourtReserve roster',exact:true}).click();
    await page.getByLabel('CourtReserve report cells').fill('Full Name\tSkill Level\nPlayer 1\t4.0\nPlayer 2\t4.0\nPlayer 3\t4.0\nJavier Gonzalez-Bringas\t3.25');
    await page.getByRole('button',{name:'Preview report',exact:true}).click();
    await page.getByRole('button',{name:'Create new player for Javier Gonzalez-Bringas',exact:true}).click();
    assert.equal(await page.getByLabel('New player name for Javier Gonzalez-Bringas').inputValue(),'Javier Gonzalez-Bringas');
    assert.equal(await page.getByLabel('New player starting level for Javier Gonzalez-Bringas').inputValue(),'3.25');
    await page.setViewportSize({width:390,height:844});
    await screenshot(page,'courtreserve-create-mobile.png');
    const dimensions=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
    assert(dimensions.scroll<=dimensions.width+1,`Import stays within the mobile page width: ${JSON.stringify(dimensions)}`);
    const created=await clickRequest(page,page.getByRole('button',{name:'Create and select player',exact:true}),'/players/editor/players');
    assert.equal(created.request.source,'courtreserve_roster_import');
    const playerId=created.body.player.id;
    await expectBrowser(page.getByRole('combobox',{name:'PCS profile for report row 5: Javier Gonzalez-Bringas'})).toHaveValue(`Javier Gonzalez-Bringas · #${playerId}`);
    const review=page.getByRole('spinbutton',{name:'Javier Gonzalez-Bringas doubles starting rating',exact:true});
    await expectBrowser(review).toHaveValue('3.25');
    const withPlayer=await fixture();
    assert.equal(withPlayer.players.length,before.players.length+1);
    assert.equal(withPlayer.players.find(p=>p.id===playerId).rating,1300);
    assert.deepEqual(withPlayer.matches,before.matches,'Creating a profile does not publish matches');
    await clickRequest(page,page.getByRole('button',{name:/^Confirm starting rating/}),'/starting-rating-review/confirm');
    await page.getByRole('button',{name:'Use imported roster (4)',exact:true}).click();
    await expectBrowser(page.getByRole('textbox',{name:/^Players \(/})).toHaveValue('Player 1\nPlayer 2\nPlayer 3\nJavier Gonzalez-Bringas');
    const preview=await clickRequest(page,page.getByRole('button',{name:'Preview matchups',exact:true}),'/clubs/test/play-generators/preview');
    assert.equal(preview.request.participant_player_ids['Javier Gonzalez-Bringas'],playerId);
    assert.equal(preview.request.participant_names.length,4);
    await expectBrowser(page.getByRole('button',{name:'Start rated session',exact:true})).toBeVisible();
    await page.setViewportSize({width:1280,height:900});
    await screenshot(page,'courtreserve-imported-desktop.png');
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify({passed:true,playerId,players:preview.request.participant_names,browserErrors:errors},null,2));
    console.log('PASS CourtReserve browser flow: prefilled profile -> real creation API -> one saved player -> selection -> rating review -> imported lineup -> matchups; desktop/mobile and no browser errors.');
  } catch(error) {
    console.error(error.stack||String(error));
    if(page){console.error((await page.locator('body').innerText()).slice(-5000));await screenshot(page,'failure.png').catch(()=>{});}
    console.error(logs.slice(-20).join('').slice(-5000));process.exitCode=1;
  } finally {
    fs.writeFileSync(path.join(output,'servers.log'),logs.join(''));
    if(browser)await browser.close();next.kill('SIGTERM');api.kill('SIGTERM');
  }
}
main();
