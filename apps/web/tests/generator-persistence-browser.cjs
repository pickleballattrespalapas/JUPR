/* Local-only browser/API persistence regression. Uses the same environment controls as generator-submission-browser.cjs. */
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
    NEXT_PUBLIC_JUPR_API_BASE_URL:apiOrigin, NEXT_PUBLIC_JUPR_ENV:'staging', GENERATOR_BROWSER_WEB_PORT:webPort};
  const api = start(process.env.CODEX_PRIMARY_RUNTIME_PYTHON || 'python3',
    ['-m','uvicorn','generator_submission_api:app','--app-dir',path.join(root,'tests/e2e'),'--host','127.0.0.1','--port',apiPort], {cwd:root,env});
  const next = start(process.execPath, [require.resolve('next/dist/bin/next'),nextMode,...(nextMode==='dev'?['--webpack']:[]),'--hostname','127.0.0.1','--port',webPort], {cwd:webRoot,env});
  let browser;
  try {
    await ready(`${apiOrigin}/fixture-result`,[api,next]);
    await ready(`${webOrigin}/clubs/test/round-robin-generator`,[api,next]);
    browser = await chromium.launch({headless:true,executablePath:process.env.GENERATOR_BROWSER_EXECUTABLE,args:['--no-sandbox','--disable-dev-shm-usage',...JSON.parse(process.env.GENERATOR_BROWSER_ARGS||'[]')]});
    const errors=[];
    async function makeContext(options={}) {
      const context=await browser.newContext({viewport:{width:390,height:844},...options});
      await context.route('**/*',route=>['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname)?route.continue():route.abort());
      context.on('page',page=>page.on('pageerror',error=>errors.push(error.message)));
      return context;
    }
    let context=await makeContext();
    let page=await context.newPage();
    const workspace=`${webOrigin}/clubs/test/round-robin-generator`;
    await page.goto(workspace);
    await page.getByLabel('Session title',{exact:true}).fill('Persistent RR rehearsal');
    await page.getByRole('combobox',{name:/^Number of players/}).selectOption('4');
    await page.getByRole('textbox',{name:/^Players \(/}).fill('Player 1\nPlayer 2\nPlayer 3\nPlayer 4');
    await clickRequest(page,page.getByRole('button',{name:'Preview matchups',exact:true}),'/preview');
    const created=await clickRequest(page,page.getByRole('button',{name:'Start unrated session',exact:true}),'/sessions');
    let session=created.body.session;
    const key=session.session_key;
    const ownerKey=`public-generator-edit:test:${key}`;
    await page.waitForURL(/\/rounds\/1(?:#.*)?$/);
    await page.getByRole('heading',{name:'Persistent RR rehearsal',exact:true}).waitFor();
    assert(await page.evaluate(k=>Boolean(localStorage.getItem(k)),ownerKey),'Organizer token saved beyond this tab');
    const match=session.event.rounds[0].matches[0];
    await page.getByLabel(`${match.id} side A score`,{exact:true}).fill('11');
    await page.getByLabel(`${match.id} side B score`,{exact:true}).fill('7');
    const saved=await clickRequest(page,page.getByRole('button',{name:'Save round scores',exact:true}),'/scores','PATCH');
    assert.equal(saved.body.session.event.rounds[0].matches[0].scoreA,11);
    await page.close();
    page=await context.newPage();
    await page.goto(workspace);
    await page.getByRole('link',{name:'Resume session',exact:true}).waitFor();
    await screenshot(page,'resume-mobile.png');
    await page.getByRole('link',{name:'Resume session',exact:true}).click();
    await page.getByRole('heading',{name:'Persistent RR rehearsal',exact:true}).waitFor();
    await page.getByText("11–7",{exact:true}).waitFor();
    const nextButton=page.getByRole('button',{name:/^(Next round|Next scheduled round)$/});
    const advanced=await clickRequest(page,nextButton,'/advance');
    session=advanced.body.session;
    await page.waitForURL(/\/rounds\/2$/);
    const second=session.event.rounds.find(r=>r.number===2).matches[0];
    await page.getByLabel(`${second.id} side A score`,{exact:true}).fill('11');
    await page.getByLabel(`${second.id} side B score`,{exact:true}).fill('8');
    await clickRequest(page,page.getByRole('button',{name:'Save round scores',exact:true}),'/scores','PATCH');
    const browserState=await context.storageState();
    await context.close();
    context=await makeContext({storageState:browserState});
    page=await context.newPage();
    await page.goto(workspace);
    await page.getByRole('link',{name:'Resume session',exact:true}).click();
    await page.waitForURL(/\/rounds\/2$/);
    await page.getByText("11–8",{exact:true}).waitFor();
    assert.equal(await page.evaluate(k=>sessionStorage.getItem(k),ownerKey),null,'Restart restores from durable storage with an empty tab store');
    assert.equal((await fixture()).sessions.length,1,'Reopening never creates duplicate sessions');
    await screenshot(page,'resumed-round-mobile.png');
    const spectatorContext=await makeContext();
    const spectator=await spectatorContext.newPage();
    await spectator.goto(`${workspace}/sessions/${key}/rounds/2`);
    await spectator.getByText('Only the organizer can enter scores or change players.').waitFor();
    assert.equal(await spectator.getByRole('button',{name:/^(Next round|Next scheduled round)$/}).count(),0);
    await spectatorContext.close();
    assert.deepEqual(errors,[],'No browser errors during persistence journey');
    console.log('PASS real browser -> API -> data: create, save, close tab, resume, advance/save, browser restart, spectator isolation; one session and saved scores intact.');
    console.log(`Screenshots: ${output}`);
  } finally {
    if(browser) await browser.close();
    api.kill('SIGTERM'); next.kill('SIGTERM');
    fs.writeFileSync(path.join(output,'servers.log'),logs.join(''));
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
