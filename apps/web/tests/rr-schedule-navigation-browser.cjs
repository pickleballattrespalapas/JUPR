// Real Next UI -> FastAPI -> isolated in-memory persistence; no external writes.
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const { chromium } = require("@playwright/test");
const webRoot = path.resolve(__dirname, ".."), root = path.resolve(webRoot, "../..");
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || "8057", webPort = process.env.GENERATOR_BROWSER_WEB_PORT || "3038";
const api = `http://127.0.0.1:${apiPort}`, web = `http://127.0.0.1:${webPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || path.join(webRoot, "test-results", "rr-navigation");
const children = [], logs = [];
function start(command, args, env, cwd) {
  const child = spawn(command, args, { env, cwd, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", data => logs.push(data.toString()));
  child.stderr.on("data", data => logs.push(data.toString()));
  child.on("error", error => { child.startupError = error; });
  children.push(child);
}
async function ready(url) {
  for (let i = 0; i < 120; i++) {
    for (const child of children) {
      if (child.startupError) throw child.startupError;
      if (child.exitCode !== null) throw new Error(`Fixture server exited: ${child.exitCode}`);
    }
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error(`Fixture server unavailable: ${url}`);
}
const shape = rounds => rounds.map(r => ({ games: r.matches.map(m => [m.id, m.court, m.sideA, m.sideB]), byes: r.byeParticipantIds }));
async function request(path, body) {
  const r = await fetch(api + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await r.json();
  assert.equal(r.ok, true, `${path}: ${r.status}`);
  return data;
}
(async () => {
  fs.mkdirSync(output, { recursive: true });
  let browser;
  const errors = [];
  try {
    if (process.env.GENERATOR_BROWSER_EXISTING_SERVERS !== "1") {
      const env = { ...process.env, VERCEL_ENV: "development", JUPR_API_BASE_URL: api, NEXT_PUBLIC_JUPR_API_BASE_URL: api,
        NEXT_PUBLIC_JUPR_ENV: "staging", GENERATOR_BROWSER_WEB_PORT: webPort };
      start(process.env.GENERATOR_BROWSER_PYTHON || "python3", ["-m", "uvicorn", "generator_submission_api:app",
        "--app-dir", path.join(root, "tests/e2e"), "--host", "127.0.0.1", "--port", apiPort], env, root);
      start(process.execPath, [require.resolve("next/dist/bin/next"), "dev", "--hostname", "127.0.0.1", "--port", webPort], env, webRoot);
    }
    await ready(`${api}/fixture-result`);
    await ready(`${web}/clubs/test/round-robin-generator`);
    browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"], ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? { executablePath: process.env.GENERATOR_BROWSER_EXECUTABLE } : {}) });
    for (const count of [4,5,8,9,12,16]) {
      const rounds = count % 4 === 0 ? count - 1 : count;
      const created = await request("/clubs/test/play-generators/sessions", {
        generator_kind: "round_robin", play_format: "doubles", title: `${count} player schedule`,
        participant_names: Array.from({length:count}, (_, i) => `Player ${i+1}`),
        total_rounds: rounds, court_count: 0, rating_mode: "unrated", scoring_mode: "scored",
        idempotency_key: crypto.randomUUID()
      });
      const key = created.session.session_key, original = shape(created.session.event.rounds);
      assert.equal(created.session.event.perfectRotation, true);
      const ctx = await browser.newContext({ viewport: {width:1280,height:900} });
      await ctx.route("**/*", route => ["127.0.0.1","localhost"].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
      const page = await ctx.newPage();
      page.on("pageerror", error => errors.push(error.message));
      await page.goto(`${web}/clubs/test/round-robin-generator/sessions/${key}/rounds/1#edit=${encodeURIComponent(created.edit_token)}`);
      await page.getByRole("button", {name:`Open round ${rounds}`,exact:true}).waitFor();
      assert.equal(await page.getByRole("button", {name:/^Open round \d+$/}).count(), rounds);
      const selecting = page.waitForResponse(r => r.url().endsWith(`/rounds/${rounds}/select`) && r.request().method() === "POST");
      await page.getByRole("button", {name:`Open round ${rounds}`,exact:true}).click();
      assert.equal((await selecting).ok(),true);
      await page.waitForURL(new RegExp(`/rounds/${rounds}$`));
      const a = page.locator('input[aria-label$="side A score"]'), b = page.locator('input[aria-label$="side B score"]');
      await a.first().waitFor();
      for (let i=0;i<await a.count();i++) { await a.nth(i).fill("11");await b.nth(i).fill("7"); }
      assert.equal(await page.getByRole("button",{name:"Open round 1",exact:true}).isDisabled(),true);
      const saving = page.waitForResponse(r => r.url().endsWith(`/rounds/${rounds}/scores`) && r.request().method() === "PATCH");
      await page.getByRole("button",{name:"Save round scores",exact:true}).click();
      assert.equal((await saving).ok(),true);
      await page.getByRole("button",{name:"Next scheduled round",exact:true}).click();
      await page.waitForURL(/\/rounds\/1$/);
      await page.getByRole("button",{name:"Skip round",exact:true}).click();
      await page.getByRole("dialog").getByRole("button",{name:"Yes, skip round",exact:true}).click();
      await page.getByRole("dialog").getByRole("button",{name:"OK",exact:true}).click();
      await page.getByRole("button",{name:"Next scheduled round",exact:true}).click();
      await page.waitForURL(/\/rounds\/2$/);
      const saved = (await request(`/clubs/test/play-generators/sessions/${key}`)).session;
      assert.deepEqual(shape(saved.event.rounds),original);
      assert.equal(saved.event.rounds[0].status,"skipped");
      assert.equal(saved.event.rounds[rounds-1].status,"saved");
      assert.equal(saved.event.adaptiveFromRound,null);
      if (count===9) await page.screenshot({path:path.join(output,"rr-schedule-navigation.png"),fullPage:true});
      await ctx.close();
      console.log(`PASS ${count}: full ${rounds}-round schedule, alternate start, scores, direct next, skip, and preserved pairings`);
    }
    assert.deepEqual(errors,[]);
    assert.equal((await request("/fixture-result")).matches.length,0);
  } finally {
    if (browser) await browser.close();
    children.forEach(child => child.kill("SIGTERM"));
    fs.writeFileSync(path.join(output, "servers.log"), logs.join(""));
  }
})().catch(error=>{console.error(error);process.exit(1);});
