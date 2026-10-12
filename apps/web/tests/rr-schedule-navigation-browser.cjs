// Real Next UI -> FastAPI -> isolated in-memory persistence; no external writes.
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const { chromium, expect } = require("@playwright/test");
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
function assertPerfect(event) {
  const partners = new Map(), opponents = new Map();
  const add = (counts, a, b) => { const key = [a,b].sort().join("|"); counts.set(key,(counts.get(key)||0)+1); };
  for (const row of event.rounds) {
    const playing = row.matches.flatMap(m => [...m.sideA,...m.sideB]);
    assert.equal(new Set(playing).size,playing.length);
    for (const game of row.matches) {
      add(partners,...game.sideA); add(partners,...game.sideB);
      for (const a of game.sideA) for (const b of game.sideB) add(opponents,a,b);
    }
  }
  for (let i=0;i<event.participants.length;i++) for (let j=i+1;j<event.participants.length;j++) {
    const key = [event.participants[i].id,event.participants[j].id].sort().join("|");
    assert.equal(partners.get(key),1,`Partner count ${key}`);
    assert.equal(opponents.get(key),2,`Opponent count ${key}`);
  }
}
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
    // Reproduce the recording: five players, then three arrivals during Round 2.
    const created = await request("/clubs/test/play-generators/sessions", {
      generator_kind: "round_robin", play_format: "doubles", title: "Five plus three arrivals",
      participant_names: ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"],
      total_rounds: 5, court_count: 1, rating_mode: "unrated", scoring_mode: "scored", idempotency_key: crypto.randomUUID()
    });
    const key = created.session.session_key;
    const ctx = await browser.newContext({ viewport: {width:1280,height:900} });
    await ctx.route("**/*", route => ["127.0.0.1","localhost"].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    const page = await ctx.newPage();
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(`${web}/clubs/test/round-robin-generator/sessions/${key}/rounds/1#edit=${encodeURIComponent(created.edit_token)}`);
    await page.getByLabel("r1-c1 side A score").fill("11");
    await page.getByLabel("r1-c1 side B score").fill("7");
    await page.getByRole("button",{name:"Save round scores",exact:true}).click();
    await page.getByRole("button",{name:"Next scheduled round",exact:true}).click();
    await page.waitForURL(/\/rounds\/2$/);
    const before = (await request(`/clubs/test/play-generators/sessions/${key}`)).session;
    const original = before.event.rounds[1].matches;
    const bye = before.event.rounds[1].byeParticipantIds[0];
    const byeName = before.event.participants.find(p => p.id === bye).name;
    await page.getByLabel("r2-c1 side A score").fill("5");
    for (let i=6;i<=8;i++) {
      await page.getByLabel("New player name").fill(`Player ${i}`);
      const adding = page.waitForResponse(r => r.url().endsWith("/roster") && r.request().postDataJSON()?.action === "add");
      await page.getByRole("button",{name:"Add player for next round",exact:true}).click();
      assert.equal((await adding).ok(),true);
      await expect(page.getByLabel("New player name")).toBeEnabled();
      await expect(page.getByLabel("New player name")).toHaveValue("");
      assert.equal(await page.getByLabel("r2-c1 side A score").inputValue(),"5");
    }
    const late = page.locator('section[aria-label="Late arrivals"]');
    assert.equal(await late.getByRole("checkbox").count(),4);
    assert.equal(await late.getByRole("checkbox",{name:`${byeName} (on a bye)`,exact:true}).isChecked(),true);
    const seating = page.waitForResponse(r => r.url().endsWith("/roster") && r.request().postDataJSON()?.action === "seat_arrivals");
    await late.getByRole("button",{name:"Start late-arrival game on Court 2",exact:true}).click();
    assert.equal((await seating).ok(),true);
    await page.getByLabel("r2-c2 side A score").waitFor();
    assert.equal(await page.getByLabel("r2-c1 side A score").inputValue(),"5");
    const seated = (await request(`/clubs/test/play-generators/sessions/${key}`)).session;
    assert.deepEqual(seated.event.rounds[1].matches[0],original[0]);
    assert.deepEqual(seated.event.rounds[1].byeParticipantIds,[]);
    assert.deepEqual(new Set(seated.event.rounds[1].matches[1].sideA.concat(seated.event.rounds[1].matches[1].sideB)),new Set([bye,"p-new-1","p-new-2","p-new-3"]));
    await page.screenshot({path:path.join(output,"late-arrival-bye-game.png"),fullPage:true});
    for (const [id,value] of [["r2-c1 side A score","11"],["r2-c1 side B score","7"],["r2-c2 side A score","11"],["r2-c2 side B score","6"]]) {
      await page.getByLabel(id).fill(value);
    }
    await page.getByRole("button",{name:"Save round scores",exact:true}).click();
    await page.getByRole("button",{name:"Next scheduled round",exact:true}).click();
    await page.waitForURL(/\/rounds\/3$/);
    const next = (await request(`/clubs/test/play-generators/sessions/${key}`)).session;
    assert.equal(next.event.rounds[1].status,"saved");
    assert.equal(next.event.rounds[1].matches.length,2);
    assert.deepEqual(next.event.rounds[2].byeParticipantIds,[]);
    assert.equal(next.event.rounds[2].matches.length,2);
    await ctx.close();
    console.log("PASS 5 + 3: bye player joins three arrivals on Court 2 during Round 2, original game and draft preserved, both games scored, eight-player next round");
    // Six starters + two arrivals: both opening games must anchor a perfect 8-player RR.
    const six = await request("/clubs/test/play-generators/sessions", {
      generator_kind:"round_robin",play_format:"doubles",title:"Six plus two perfect rotation",
      participant_names:Array.from({length:6},(_,i)=>`Player ${i+1}`),
      total_rounds:8,court_count:1,rating_mode:"unrated",scoring_mode:"scored",idempotency_key:crypto.randomUUID()
    });
    const sixKey = six.session.session_key, firstGame = six.session.event.rounds[0].matches[0];
    const eightCtx = await browser.newContext({viewport:{width:1280,height:900}});
    await eightCtx.route("**/*",route=>["127.0.0.1","localhost"].includes(new URL(route.request().url()).hostname)?route.continue():route.abort());
    const eightPage = await eightCtx.newPage();
    eightPage.on("pageerror",error=>errors.push(error.message));
    await eightPage.goto(`${web}/clubs/test/round-robin-generator/sessions/${sixKey}/rounds/1#edit=${encodeURIComponent(six.edit_token)}`);
    await eightPage.getByLabel("r1-c1 side A score").fill("5");
    for (let i=7;i<=8;i++) {
      await eightPage.getByLabel("New player name").fill(`Player ${i}`);
      const adding = eightPage.waitForResponse(r=>r.url().endsWith("/roster")&&r.request().postDataJSON()?.action==="add");
      await eightPage.getByRole("button",{name:"Add player for next round",exact:true}).click();
      assert.equal((await adding).ok(),true);
      await expect(eightPage.getByLabel("New player name")).toBeEnabled();
      await expect(eightPage.getByLabel("New player name")).toHaveValue("");
    }
    const openingArrivals = eightPage.locator('section[aria-label="Late arrivals"]');
    assert.equal(await openingArrivals.getByRole("checkbox").count(),4);
    const openingSeating = eightPage.waitForResponse(r=>r.url().endsWith("/roster")&&r.request().postDataJSON()?.action==="seat_arrivals");
    await openingArrivals.getByRole("button",{name:"Start late-arrival game on Court 2",exact:true}).click();
    assert.equal((await openingSeating).ok(),true);
    await eightPage.getByLabel("r1-c2 side A score").waitFor();
    assert.equal(await eightPage.getByLabel("r1-c1 side A score").inputValue(),"5");
    const perfect = (await request(`/clubs/test/play-generators/sessions/${sixKey}`)).session.event;
    assert.deepEqual(perfect.rounds[0].matches[0],firstGame);
    assert.equal(perfect.rounds[0].matches.length,2);
    assert.equal(perfect.totalRounds,7);
    assert.equal(perfect.perfectRotation,true);
    assert.equal(perfect.adaptiveFromRound,null);
    assertPerfect(perfect);
    await expect(eightPage.getByRole("button",{name:/^Open round \d+$/})).toHaveCount(7);
    await eightPage.screenshot({path:path.join(output,"six-plus-two-perfect-rotation.png"),fullPage:true});
    for (const id of ["r1-c1","r1-c2"]) {
      await eightPage.getByLabel(`${id} side A score`).fill("11");
      await eightPage.getByLabel(`${id} side B score`).fill("7");
    }
    await eightPage.getByRole("button",{name:"Save round scores",exact:true}).click();
    await eightPage.getByRole("button",{name:"Next scheduled round",exact:true}).click();
    await eightPage.waitForURL(/\/rounds\/2$/);
    await eightPage.getByRole("button",{name:"Open round 7",exact:true}).click();
    await eightPage.waitForURL(/\/rounds\/7$/);
    const navigated = (await request(`/clubs/test/play-generators/sessions/${sixKey}`)).session.event;
    assert.deepEqual(shape(navigated.rounds),shape(perfect.rounds));
    assertPerfect(navigated);
    await eightCtx.close();
    console.log("PASS 6 + 2: both opening games and score draft preserved, exactly seven visible rounds, every partnership once and opponent pair twice, save and alternate-round navigation preserve the perfect schedule");
    assert.deepEqual(errors,[]);
    assert.equal((await request("/fixture-result")).matches.length,0);
  } finally {
    if (browser) await browser.close();
    children.forEach(child => child.kill("SIGTERM"));
    fs.writeFileSync(path.join(output, "servers.log"), logs.join(""));
  }
})().catch(error=>{console.error(error);process.exit(1);});
