/* Real Next UI -> FastAPI -> persisted test session -> bracket and final scores.
 * Uses the existing loopback-only API fixture, with no pre-created sessions.
 * GENERATOR_BROWSER_PYTHON must include the Python test dependencies + uvicorn.
 * Real authentication, remote databases, and external side effects are excluded.
 */
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

const webRoot = path.resolve(__dirname, "..");
const root = path.resolve(webRoot, "../..");
const webPort = process.env.GENERATOR_BROWSER_WEB_PORT || "3049";
const apiPort = process.env.GENERATOR_BROWSER_API_PORT || "8049";
const webOrigin = `http://127.0.0.1:${webPort}`;
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const output = process.env.GENERATOR_BROWSER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(), "generator-playoff-browser-"));
fs.mkdirSync(output, { recursive: true });
const logs = [];
const env = { ...process.env, VERCEL_ENV: "development", JUPR_API_BASE_URL: apiOrigin,
  NEXT_PUBLIC_JUPR_API_BASE_URL: apiOrigin, NEXT_PUBLIC_JUPR_ENV: "staging", GENERATOR_BROWSER_WEB_PORT: webPort };
function start(cmd, args, cwd) {
  const child = spawn(cmd, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", data => logs.push(String(data)));
  child.stderr.on("data", data => logs.push(String(data)));
  child.on("error", error => { child.startupError = error; });
  return child;
}
async function ready(url, children) {
  for (let i = 0; i < 120; i++) {
    for (const child of children) if (child.startupError || child.exitCode !== null) throw child.startupError || Error(`Server exited: ${child.exitCode}`);
    try { if ((await fetch(url)).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw Error(`Test server did not start: ${url}`);
}
async function clickRequest(page, label, suffix, method = "POST") {
  const response = page.waitForResponse(r => r.url().startsWith(apiOrigin) && r.url().endsWith(suffix) && r.request().method() === method);
  await page.getByRole("button", { name: label, exact: true }).click();
  const result = await response;
  const body = await result.json();
  assert(result.ok(), `${suffix}: ${result.status()} ${JSON.stringify(body)}`);
  return body;
}
async function scoreRound(page, number) {
  const a = page.locator('input[aria-label$="side A score"]');
  const b = page.locator('input[aria-label$="side B score"]');
  await a.first().waitFor();
  for (let i = 0; i < await a.count(); i++) { await a.nth(i).fill("11"); await b.nth(i).fill(String(4 + i)); }
  const saved = await clickRequest(page, "Save round scores", `/rounds/${number}/scores`, "PATCH");
  await page.getByRole("link", { name: "View standings and continue", exact: true }).click();
  await page.waitForURL(/\/standings$/);
  return saved.session;
}

(async () => {
  const api = start(process.env.GENERATOR_BROWSER_PYTHON || "python3",
    ["-m", "uvicorn", "generator_submission_api:app", "--app-dir", path.join(root, "tests/e2e"), "--host", "127.0.0.1", "--port", apiPort], root);
  const next = start(process.execPath, [require.resolve("next/dist/bin/next"), "dev", "--hostname", "127.0.0.1", "--port", webPort], webRoot);
  let browser;
  try {
    await ready(`${apiOrigin}/fixture-result`, [api, next]);
    assert.equal((await (await fetch(`${apiOrigin}/fixture-result`)).json()).sessions.length, 0);
    await ready(`${webOrigin}/clubs/test/round-robin-generator`, [api, next]);
    browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage", ...JSON.parse(process.env.GENERATOR_BROWSER_ARGS || "[]")],
      ...(process.env.GENERATOR_BROWSER_EXECUTABLE ? { executablePath: process.env.GENERATOR_BROWSER_EXECUTABLE } : {}) });
    const errors = [];
    const evidence = [];
    async function context() {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
      await ctx.route("**/*", route => ["127.0.0.1", "localhost"].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
      ctx.on("page", page => page.on("pageerror", error => errors.push(error.message)));
      return ctx;
    }
    for (const [format, count] of [["groups_of_four", 13], ["top_eight", 8]]) {
      const ctx = await context();
      const page = await ctx.newPage();
      await page.goto(`${webOrigin}/clubs/test/round-robin-generator`);
      await page.getByLabel("Session title", { exact: true }).fill(`${format} playoff`);
      await page.getByRole("combobox", { name: /^Number of players/ }).selectOption(String(count));
      await page.getByLabel("Match rating", { exact: true }).selectOption("unrated");
      await page.getByRole("textbox", { name: /^Players \(/ }).fill(Array.from({ length: count }, (_, i) => `Player ${i + 1}`).join("\n"));
      await clickRequest(page, "Preview matchups", "/preview");
      await clickRequest(page, "Start unrated session", "/sessions");
      await page.waitForURL(/\/rounds\/1(?:#.*)?$/);
      for (const number of [1, 2]) {
        await scoreRound(page, number);
        if (number === 1) { await clickRequest(page, "Continue to Round 2", "/advance"); await page.waitForURL(/\/rounds\/2$/); }
      }
      const standingsUrl = page.url();
      const viewCtx = await context();
      const viewer = await viewCtx.newPage();
      await viewer.goto(standingsUrl);
      await viewer.getByRole("heading", { name: `${format} playoff standings`, exact: true }).waitFor();
      assert.equal(await viewer.getByRole("button", { name: "Playoff", exact: true }).count(), 0);
      await page.getByRole("button", { name: "Playoff", exact: true }).click();
      await page.getByLabel("Playoff format", { exact: true }).selectOption(format);
      await page.setViewportSize({ width: 390, height: 844 });
      await page.screenshot({ path: path.join(output, `${format}-preview-mobile.png`), fullPage: true });
      const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
      assert(dimensions.scroll <= dimensions.width + 1, "Playoff format picker fits on mobile");
      let session = (await clickRequest(page, "Start playoff", "/playoff")).session;
      assert.equal(session.event.playoff.format, format);
      const seeded = structuredClone(session.standings);
      if (format === "groups_of_four") { assert.equal(session.event.playoff.seeds.length, 13); assert.equal(session.event.playoff.sitOutParticipantIds.length, 1); }
      await page.waitForURL(/\/rounds\/3$/);
      await page.reload();
      await page.getByRole("heading", { name: format === "top_eight" ? "Playoff semifinals" : "Playoff games", exact: true }).waitFor();
      assert.equal(await page.getByRole("button", { name: "Skip round", exact: true }).count(), 0);
      assert.equal(await page.getByRole("heading", { name: "Change players", exact: true }).count(), 0);
      let semifinalTeams;
      if (format === "top_eight") semifinalTeams = session.event.rounds.find(r => r.number === 3).matches.map(m => m.sideA);
      while (session.status !== "completed") {
        const number = session.current_round_number;
        session = await scoreRound(page, number);
        assert.deepEqual(session.standings, seeded);
        const last = number === session.total_rounds;
        session = (await clickRequest(page, last ? "Finish session" : "Continue to Playoff final", "/advance")).session;
        if (!last) {
          assert.deepEqual(session.event.rounds.at(-1).matches[0].sideA, semifinalTeams[0]);
          assert.deepEqual(session.event.rounds.at(-1).matches[0].sideB, semifinalTeams[1]);
          await page.waitForURL(/\/rounds\/4$/);
          await page.getByRole("heading", { name: "Playoff final", exact: true }).waitFor();
        }
      }
      await page.locator('article[aria-label="Playoff bracket"]').filter({ hasText: "Winner:" }).waitFor();
      assert.equal(await page.getByRole("button", { name: "Submit for approval", exact: true }).count(), 1);
      await page.screenshot({ path: path.join(output, `${format}-completed-mobile.png`), fullPage: true });
      await viewer.reload();
      await viewer.locator('article[aria-label="Playoff bracket"]').filter({ hasText: "Winner:" }).waitFor();
      evidence.push({ format, players: count, status: session.status, playoffGames: session.event.rounds.filter(r => r.stage === "playoff").reduce((n, r) => n + r.matches.length, 0) });
      console.log(`PASS ${format}: create -> 2 scored rounds -> standings choice -> persisted playoff -> score -> ${format === "top_eight" ? "winning teams in final -> " : ""}complete; spectator and mobile checks passed.`);
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "results.json"), JSON.stringify({ passed: true, evidence, browserErrors: errors }, null, 2));
  } catch (error) {
    console.error(error.stack || error);
    if (browser) for (const ctx of browser.contexts()) for (const page of ctx.pages()) {
      console.error((await page.locator("body").innerText()).slice(0, 4000));
      await page.screenshot({ path: path.join(output, `failure-${Date.now()}.png`), fullPage: true }).catch(() => {});
    }
    console.error(logs.slice(-30).join("").slice(-8000));
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(output, "servers.log"), logs.join(""));
    if (browser) await browser.close();
    next.kill("SIGTERM"); api.kill("SIGTERM");
  }
})();
