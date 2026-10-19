/* Real Next editor -> real FastAPI recap services -> isolated in-memory rows.
   Source results and identity are synthetic. Never connects to external services. */
const { spawn } = require("node:child_process");
const fs = require("node:fs"),
  path = require("node:path"),
  assert = require("node:assert/strict");
const { chromium } = require("playwright");
const web = path.resolve(__dirname, ".."),
  root = path.resolve(web, "../..");
const apiOrigin = "http://127.0.0.1:8042",
  origin = "http://127.0.0.1:3042";
const output = process.env.RECAP_BROWSER_ARTIFACTS || "/tmp/recap-browser";
fs.mkdirSync(output, { recursive: true });
const logs = [];
function start(cmd, args, opts) {
  const child = spawn(cmd, args, {
    ...opts,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", (d) => logs.push(d.toString()));
  child.stderr.on("data", (d) => logs.push(d.toString()));
  return child;
}
async function ready(url) {
  for (let i = 0; i < 120; i++) {
    try {
      if ((await fetch(url, { signal: AbortSignal.timeout(5000) })).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw Error(`Not ready: ${url}`);
}
async function state() {
  return (await fetch(apiOrigin + "/fixture-result")).json();
}
async function confirm(page, label, text) {
  await page.getByRole("button", { name: label, exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog.waitFor();
  const input = dialog.getByRole("textbox");
  if (await input.count()) await input.fill(text);
  await dialog
    .getByRole("button", {
      name: /^(Generate draft|Publish recap|Delete draft|Unpublish recap)$/,
    })
    .click();
  await dialog.getByRole("button", { name: "OK", exact: true }).click();
  await dialog.waitFor({ state: "hidden" });
}
(async () => {
  const env = {
    ...process.env,
    VERCEL_ENV: "development",
    JUPR_API_BASE_URL: apiOrigin,
    NEXT_PUBLIC_JUPR_API_BASE_URL: apiOrigin,
    NEXT_PUBLIC_JUPR_ENV: "staging",
  };
  const api = start(
    process.env.RECAP_PYTHON ||
      (fs.existsSync(path.join(root, ".venv/bin/python"))
        ? path.join(root, ".venv/bin/python")
        : "python"),
    [
      "-m",
      "uvicorn",
      "recap_builder_api:app",
      "--app-dir",
      path.join(root, "tests/e2e"),
      "--host",
      "127.0.0.1",
      "--port",
      "8042",
    ],
    { cwd: root, env },
  );
  const next = start(
    process.execPath,
    [
      require.resolve("next/dist/bin/next"),
      "dev",
      "--hostname",
      "127.0.0.1",
      "--port",
      "3042",
    ],
    { cwd: web, env },
  );
  let browser, page;
  try {
    await ready(apiOrigin + "/fixture-result");
    await ready(origin + "/api/environment");
    browser = await chromium.launch({
      headless: true,
      executablePath: process.env.RECAP_BROWSER_EXECUTABLE,
      args: JSON.parse(process.env.RECAP_BROWSER_ARGS || '["--no-sandbox"]'),
    });
    const context = await browser.newContext({
      viewport: { width: 1280, height: 900 },
    });
    await context.route("**/*", (route) =>
      ["127.0.0.1", "localhost"].includes(
        new URL(route.request().url()).hostname,
      )
        ? route.continue()
        : route.abort(),
    );
    await context.addCookies([
      {
        name: "jupr_admin_workspace_v1",
        value: encodeURIComponent(
          JSON.stringify({ clubId: "club", clubSlug: "test" }),
        ),
        url: origin,
      },
    ]);
    await context.addInitScript(() => {
      // Email preview frames must not rewrite the test sign-in or fire a
      // cross-tab storage event that deliberately clears protected editors.
      if (
        window !== window.top ||
        localStorage.getItem("jupr_admin_session_v1")
      )
        return;
      localStorage.setItem(
        "jupr_admin_session_v1",
        JSON.stringify({
          access_token: "local-reviewer",
          expires_at: Date.now() + 3600000,
          user: { id: "local-reviewer", email: "reviewer@example.invalid" },
          capabilities: {
            authorized: true,
            assignments: [
              {
                club_id: "club",
                role: "administrator",
                permissions: ["enter_scores", "manage_players"],
              },
            ],
          },
        }),
      );
    });
    page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => {
      errors.push(e.message);
      console.error("Browser error:", e.message);
    });
    await page.goto(origin + "/admin/weekly-recap");
    await page.getByRole("heading", { name: "Set up your edition" }).waitFor();
    await confirm(page, "Generate draft", "GENERATE RECAP");
    await page
      .getByRole("heading", { name: "Choose and arrange content" })
      .waitFor();
    const numbers = page
      .locator("section")
      .filter({
        has: page.locator("strong", { hasText: "The week in numbers" }),
      })
      .last();
    await numbers.getByRole("checkbox").first().uncheck();
    const events = page
      .locator("section")
      .filter({ has: page.locator("strong", { hasText: "Event results" }) })
      .last();
    await events.locator("summary").click();
    const exclude = events
      .locator("label")
      .filter({ hasText: "Tuesday Moneyball" })
      .first();
    await exclude.getByRole("checkbox").uncheck();
    await events
      .getByLabel("Feature Thursday Social Round Robin in email", {
        exact: true,
      })
      .uncheck();
    await events
      .getByLabel("Feature Example Tournament · Mixed 3.5 in email", {
        exact: true,
      })
      .check();
    await page
      .getByRole("button", { name: "3. Personalize", exact: true })
      .click();
    await page
      .getByLabel("Opening message", { exact: true })
      .fill("A great week at the club.");
    await page
      .getByRole("button", { name: "+ Coach’s tip", exact: true })
      .click();
    await page
      .getByLabel("Story", { exact: true })
      .fill("Stay grounded through your attack.");
    await page.getByRole("button", { name: "4. Preview", exact: true }).click();
    const preview = page.getByTestId("recap-edition");
    await preview.waitFor();
    assert(!(await preview.innerText()).includes("Tuesday Moneyball"));
    assert(!(await preview.innerText()).includes("The week in numbers"));
    assert(
      (await preview.innerText()).includes(
        "Stay grounded through your attack.",
      ),
    );
    await page.waitForFunction(
      () =>
        document.body.innerText.includes("Draft saved") &&
        !document.body.innerText.includes("Saving draft…"),
    );
    await page.screenshot({
      path: path.join(output, "desktop-preview.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "5. Publish & share", exact: true })
      .click();
    await page.getByLabel("Layout name").fill("Member update");
    await page
      .getByRole("button", { name: "Save as club default", exact: true })
      .click();
    await page
      .getByText("Club default layout saved.", { exact: true })
      .waitFor();
    await confirm(page, "Publish recap", "PUBLISH RECAP");
    let live = await (
      await fetch(apiOrigin + "/clubs/test/weekly-recaps")
    ).json();
    assert.equal(
      live.selected_recap.recap.report.headline,
      "This week on the courts",
    );
    assert(!JSON.stringify(live).includes("Tuesday Moneyball"));
    const compactTitles =
      live.selected_recap.recap.report.email_edition.flatMap((b) =>
        b.items.map((i) => i.title),
      );
    assert(!compactTitles.includes("Thursday Social Round Robin"));
    assert(compactTitles.includes("Example Tournament · Mixed 3.5"));
    await page
      .getByRole("button", { name: "Edit draft revision", exact: true })
      .click();
    await page
      .getByRole("button", { name: "3. Personalize", exact: true })
      .click();
    await page.getByLabel("Report headline").fill("Private revision headline");
    await page.waitForFunction(
      () =>
        document.body.innerText.includes("Draft saved") &&
        !document.body.innerText.includes("Saving draft…"),
    );
    live = await (await fetch(apiOrigin + "/clubs/test/weekly-recaps")).json();
    assert.equal(
      live.selected_recap.recap.report.headline,
      "This week on the courts",
    );
    await page.reload();
    await page.getByRole("button", { name: "1. Setup", exact: true }).click();
    const saved = (await state()).weekly_recaps[0];
    await page.getByLabel("Existing recaps").selectOption(saved.week_start);
    await page
      .getByRole("button", { name: "Edit draft revision", exact: true })
      .click();
    await page
      .getByRole("button", { name: "3. Personalize", exact: true })
      .click();
    assert.equal(
      await page.getByLabel("Report headline").inputValue(),
      "Private revision headline",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "2. Content", exact: true }).click();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      "No horizontal overflow on mobile",
    );
    await page.screenshot({
      path: path.join(output, "mobile-content.png"),
      fullPage: true,
    });
    await page.getByRole("button", { name: "1. Setup", exact: true }).click();
    await page.waitForFunction(
      () =>
        !document.body.innerText.includes("Unsaved changes") &&
        !document.body.innerText.includes("Saving draft…"),
    );
    await confirm(page, "Refresh results", "GENERATE RECAP");
    await page
      .getByRole("button", { name: "3. Personalize", exact: true })
      .click();
    assert.equal(
      await page.getByLabel("Report headline").inputValue(),
      "Private revision headline",
    );
    assert.equal(
      await page.getByLabel("Story", { exact: true }).inputValue(),
      "Stay grounded through your attack.",
    );
    live = await (await fetch(apiOrigin + "/clubs/test/weekly-recaps")).json();
    assert.equal(
      live.selected_recap.recap.report.headline,
      "This week on the courts",
    );
    await page.getByRole("button", { name: "1. Setup", exact: true }).click();
    await page.waitForFunction(
      () =>
        !document.body.innerText.includes("Unsaved changes") &&
        !document.body.innerText.includes("Saving draft…"),
    );
    await page.getByLabel("Existing recaps").selectOption("");
    await page.getByLabel("Start date").fill("2026-09-28");
    await page.getByLabel("End date").fill("2026-10-04");
    await confirm(page, "Generate draft", "GENERATE RECAP");
    assert.equal(
      await numbers.getByRole("checkbox").first().isChecked(),
      false,
      "Saved club default applies next week",
    );
    await page
      .getByRole("button", { name: "3. Personalize", exact: true })
      .click();
    assert.equal(
      await page.getByLabel("Story", { exact: true }).count(),
      0,
      "Dated stories are not copied into a new edition",
    );
    assert.deepEqual(errors, []);
    assert.equal((await state()).club_recap_preferences.length, 1);
    console.log(
      "PASS: generate, exact email selections, autosave, live preview, reusable default, publish, private revision, refresh, reload, desktop/mobile.",
    );
  } catch (e) {
    if (page) {
      await page
        .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
        .catch(() => {});
      fs.writeFileSync(
        path.join(output, "failure.txt"),
        await page
          .locator("body")
          .innerText()
          .catch(() => ""),
      );
    }
    throw e;
  } finally {
    fs.writeFileSync(path.join(output, "servers.log"), logs.join(""));
    await browser?.close();
    api.kill("SIGTERM");
    next.kill("SIGTERM");
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
