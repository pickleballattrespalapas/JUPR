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
const nextMode = process.env.RECAP_BROWSER_NEXT_MODE || "dev";
assert(["dev", "start"].includes(nextMode), "Use dev or a prebuilt start server");
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
      name: /^(Generate draft|Publish recap|Delete draft|Unpublish recap|Delete layout)$/,
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
      nextMode,
      ...(nextMode === "dev" ? ["--webpack"] : []),
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
    if (nextMode === "dev") {
      // Compile the public preview before editing. A cold Next 16 development
      // compilation can refresh the shared module graph and reload the editor.
      // Release acceptance uses a prebuilt start server instead.
      await ready(origin + "/clubs/test/weekly-recap");
    }
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
    // Local synthetic photo; all other external requests remain blocked.
    await context.route("https://photos.example.invalid/week.png", (route) =>
      route.fulfill({
        status: 200,
        contentType: "image/png",
        headers: { "access-control-allow-origin": "*" },
        body: Buffer.from(
          require("./fixtures/recap-photo.json").base64,
          "base64",
        ),
      }),
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
    const evidence = page.getByTestId("recap-evidence-picker");
    await evidence
      .getByRole("checkbox", {
        name: "Include The week in numbers",
        exact: true,
      })
      .uncheck();
    assert.equal(await evidence.getByTestId("recap-evidence-numbers").count(), 0);
    const achievements = evidence.getByLabel("Include New achievements", { exact: true });
    await achievements.uncheck();
    assert.equal(await evidence.getByTestId("recap-evidence-achievements").count(), 0);
    await achievements.check();
    await evidence.getByTestId("recap-evidence-achievements").waitFor();
    await achievements.uncheck();
    const events = evidence;
    await events
      .getByTestId("recap-evidence-rr")
      .locator("label")
      .filter({ hasText: "Tuesday Moneyball" })
      .getByRole("checkbox")
      .click();
    await events.getByTestId("recap-evidence-rr").waitFor({ state: "detached" });
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
    await events
      .getByLabel("Feature Ellis ×3 in email", {
        exact: true,
      })
      .check();
    const courtList = evidence.getByTestId("recap-evidence-courts");
    assert(!(await courtList.innerText()).includes("point difference"));
    assert.equal(await courtList.getByLabel("Include Ellis ×3", { exact: true }).count(), 1);
    await courtList.screenshot({ path: path.join(output, "compact-courts-desktop.png") });
    await page.setViewportSize({ width: 390, height: 844 });
    await courtList.screenshot({ path: path.join(output, "compact-courts-mobile.png") });
    assert(await courtList.evaluate(el => el.scrollWidth <= el.clientWidth + 1));
    await page.setViewportSize({ width: 1280, height: 900 });
    await events
      .getByLabel("Feature Summer Social in email", { exact: true })
      .check();
    const ratings = evidence.getByTestId("recap-evidence-BIGGEST_JUMP_WEEK");
    await ratings.getByRole("checkbox", { name: /Arden/ }).uncheck();
    await ratings.getByRole("checkbox", { name: /Blair/ }).check();
    await ratings.getByRole("checkbox", { name: /Casey/ }).check();
    await ratings
      .getByLabel("Include Rating increases in email", { exact: true })
      .check();
    await evidence.getByLabel("Include Most wins", { exact: true }).check();
    await evidence
      .getByLabel("Include Most wins in email", { exact: true })
      .check();
    await page.screenshot({
      path: path.join(output, "content-evidence.png"),
      fullPage: true,
    });
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
    await page
      .getByRole("button", { name: "+ Photo of the week", exact: true })
      .click();
    const photo = page.locator("fieldset").filter({
      has: page.locator("legend", { hasText: "Photo of the week" }),
    });
    await photo.getByLabel("Heading", { exact: true }).fill("Saturday smiles");
    await photo
      .getByLabel("Photo URL", { exact: true })
      .fill("https://photos.example.invalid/week.png");
    await photo
      .getByLabel("Photo description", { exact: true })
      .fill("A colorful local test photo");
    await photo
      .getByLabel("Photo caption", { exact: true })
      .fill("Saturday at the club");
    await photo
      .getByLabel("Photo credit", { exact: true })
      .fill("Club volunteer");
    await photo
      .getByLabel("Photo presentation", { exact: true })
      .selectOption("cover");
    await page
      .getByRole("button", { name: "+ Featured event", exact: true })
      .click();
    const promotion = page
      .locator("fieldset")
      .filter({ has: page.locator("legend", { hasText: "Featured event" }) });
    await promotion
      .getByLabel("Heading", { exact: true })
      .fill("Autumn social");
    await promotion
      .getByLabel("Story", { exact: true })
      .fill("Join us next Saturday.");
    await promotion
      .getByLabel("Link URL", { exact: true })
      .fill("https://events.example.invalid/register");
    await page.getByRole("button", { name: "2. Content", exact: true }).click();
    await page
      .getByTestId("recap-layout-options")
      .locator("summary")
      .first()
      .click();
    const emailOrder = page.getByTestId("recap-email-order");
    await emailOrder.locator("summary").click();
    await emailOrder
      .getByLabel("Move Autumn social earlier in email", { exact: true })
      .click();
    await emailOrder
      .getByLabel("Move Club stories up in email", { exact: true })
      .click();
    await page.getByRole("button", { name: "4. Preview", exact: true }).click();
    const preview = page.getByTestId("recap-edition").filter({ visible: true });
    await preview.waitFor();
    await page.getByRole("button", { name: "Arrange content", exact: true }).click();
    const previewEvents = preview.locator('section[data-section="events"]');
    const eventIds = () => previewEvents.locator("article").evaluateAll(nodes => nodes.map(node => node.dataset.recapItem));
    const originalEventIds = await eventIds();
    await previewEvents.locator("article").first().getByRole("button", { name: /down in website$/ }).click();
    assert.equal((await eventIds())[1], originalEventIds[0], "Preview moves the selected result without changing its content");
    await previewEvents.locator("article").nth(1).getByRole("button", { name: /up in website$/ }).click();
    assert.deepEqual(await eventIds(), originalEventIds);
    await preview.getByRole("button", { name: "Group events by type", exact: true }).click();
    await page.screenshot({ path: path.join(output, "preview-arrange-desktop.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await preview.evaluate(el => el.scrollWidth <= el.clientWidth + 1));
    await page.screenshot({ path: path.join(output, "preview-arrange-mobile.png"), fullPage: true });
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.getByRole("button", { name: "Done arranging", exact: true }).click();
    assert(!(await preview.innerText()).includes("Tuesday Moneyball"));
    assert(!(await preview.innerText()).includes("The week in numbers"));
    assert(
      (await preview.innerText()).includes(
        "Stay grounded through your attack.",
      ),
    );
    await page.getByRole("button", { name: "Edit text", exact: true }).click();
    const textEditor = page.getByTestId("recap-text-editor");
    const tournamentText = textEditor.locator("[data-edit-item]").filter({
      has: page.locator("legend", {
        hasText: "Example Tournament · Mixed 3.5",
      }),
    });
    await tournamentText
      .getByLabel("Title", { exact: true })
      .fill("Temporary title");
    await tournamentText
      .getByRole("button", { name: "Reset Title", exact: true })
      .click();
    assert.equal(
      await tournamentText.getByLabel("Title", { exact: true }).inputValue(),
      "Example Tournament · Mixed 3.5",
    );
    await tournamentText
      .getByLabel("Title", { exact: true })
      .fill("Club tournament champions");
    await tournamentText
      .getByLabel("Summary", { exact: true })
      .fill("Celebrating our division winners");
    await tournamentText
      .getByLabel("Detail 1", { exact: true })
      .fill("Gold: Alex and Bea — well played!");
    const leagueText = textEditor
      .locator("[data-edit-item]")
      .filter({ has: page.locator("legend", { hasText: "Summer Social" }) });
    await leagueText
      .getByLabel("Results link label", { exact: true })
      .fill("Explore the league results");
    await textEditor
      .getByLabel("Recap headline", { exact: true })
      .fill("Club weekly highlights");
    await textEditor
      .getByLabel("Date wording", { exact: true })
      .fill("October · Our week on court");
    const photoText = textEditor
      .locator("[data-edit-item]")
      .filter({ has: page.locator("legend", { hasText: "Saturday smiles" }) });
    await photoText
      .getByLabel("Photo caption", { exact: true })
      .fill("Friends enjoying Saturday play");
    await page.screenshot({
      path: path.join(output, "text-editor-desktop.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Done editing text", exact: true })
      .click();
    assert((await preview.innerText()).includes("Club tournament champions"));
    assert(
      (await preview.innerText()).includes("Gold: Alex and Bea — well played!"),
    );
    assert((await preview.innerText()).includes("Explore the league results"));
    await page
      .getByRole("button", { name: "Member email", exact: true })
      .click();
    const emailFrame = page.frameLocator(
      'iframe[title="Live member email preview"]',
    );
    await emailFrame
      .getByText("Club tournament champions", { exact: true })
      .waitFor();
    await page.getByRole("button", { name: "Edit text", exact: true }).click();
    assert.equal(
      await tournamentText.getByLabel("Title", { exact: true }).inputValue(),
      "Club tournament champions",
    );
    await textEditor
      .getByLabel("Full recap link label", { exact: true })
      .fill("Open this week’s recap");
    await page
      .getByRole("button", { name: "Done editing text", exact: true })
      .click();
    await page.getByRole("button", { name: "Website", exact: true }).click();
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
    const exported = page.getByTestId("weekly-report-export");
    await exported.getByRole("combobox").selectOption("full");
    const pdfDownload = page.waitForEvent("download");
    await exported
      .getByRole("button", { name: "Download styled PDF", exact: true })
      .click();
    await (await pdfDownload).saveAs(path.join(output, "styled-recap.pdf"));
    const pngDownload = page.waitForEvent("download");
    await exported
      .getByRole("button", { name: "Download PNG", exact: true })
      .click();
    await (await pngDownload).saveAs(path.join(output, "styled-recap.png"));
    await confirm(page, "Publish recap", "PUBLISH RECAP");
    let live = await (
      await fetch(apiOrigin + "/clubs/test/weekly-recaps")
    ).json();
    assert.equal(
      live.selected_recap.recap.report.headline,
      "Club weekly highlights",
    );
    assert(!JSON.stringify(live).includes("Tuesday Moneyball"));
    for (const channel of ["edition", "email_edition"]) {
      const serialized = JSON.stringify(
        live.selected_recap.recap.report[channel],
      );
      assert(serialized.includes("Blair — +0.18 JUPR"));
      assert(serialized.includes("Casey — +0.17 JUPR"));
      assert(serialized.includes("Drew — 6 wins"));
      assert(serialized.includes("Ellis ×3"));
      const courts = live.selected_recap.recap.report[channel].flatMap(b => b.items).filter(i => i.id === "ladder-court-winners");
      assert.equal(courts.length, 1);
      assert.deepEqual(courts[0].lines, ["Ellis ×3"]);
      assert(!serialized.includes("Arden —"));
    }
    const compactTitles =
      live.selected_recap.recap.report.email_edition.flatMap((b) =>
        b.items.map((i) => i.title),
      );
    assert(!compactTitles.includes("Thursday Social Round Robin"));
    assert(compactTitles.includes("Club tournament champions"));
    assert.equal(
      live.selected_recap.recap.report.display_text.date,
      "October · Our week on court",
    );
    for (const channel of ["edition", "email_edition"]) {
      const content = JSON.stringify(live.selected_recap.recap.report[channel]);
      assert(content.includes("Gold: Alex and Bea — well played!"));
      assert(content.includes("Friends enjoying Saturday play"));
      assert(content.includes("Explore the league results"));
    }
    const publicStories = live.selected_recap.recap.report.edition.find(
      (b) => b.key === "custom",
    ).items;
    assert.deepEqual(
      publicStories.map((i) => i.title),
      ["Coach’s tip", "Saturday smiles", "Autumn social"],
    );
    const emailStories = live.selected_recap.recap.report.email_edition.find(
      (b) => b.key === "custom",
    ).items;
    assert.deepEqual(
      emailStories.map((i) => i.title),
      ["Coach’s tip", "Autumn social", "Saturday smiles"],
    );
    assert.equal(publicStories[1].credit, "Club volunteer");
    assert.equal(publicStories[2].layout, "promotion");
    const publicPage = await context.newPage();
    await publicPage.goto(origin + "/clubs/test/weekly-recap");
    await publicPage.getByTestId("recap-edition").waitFor();
    await publicPage
      .getByRole("heading", { name: "Club tournament champions", exact: true })
      .waitFor();
    assert(
      (await publicPage.getByTestId("recap-edition").innerText()).includes(
        "Photo: Club volunteer",
      ),
    );
    const publicPdf = publicPage.waitForEvent("download");
    await publicPage.getByTestId("weekly-recap-styled-pdf").click();
    await (await publicPdf).saveAs(path.join(output, "published-recap.pdf"));
    // Block the photo download while keeping the page readable: export must
    // show a recoverable error, not hand the reader a PDF with missing photos.
    const blockPhoto = (route) => route.abort();
    await context.route("https://photos.example.invalid/week.png", blockPhoto);
    await publicPage.getByTestId("weekly-recap-styled-pdf").click();
    await publicPage
      .getByRole("alert")
      .filter({ hasText: "A photo could not be included" })
      .waitFor();
    await context.unroute(
      "https://photos.example.invalid/week.png",
      blockPhoto,
    );
    await publicPage.close();
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
      "Club weekly highlights",
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
    assert(
      await page
        .getByTestId("recap-evidence-BIGGEST_JUMP_WEEK")
        .getByRole("checkbox", { name: /Blair/ })
        .isChecked(),
      "Saved player selection survives reload",
    );
    assert(
      !(await page
        .getByTestId("recap-evidence-BIGGEST_JUMP_WEEK")
        .getByRole("checkbox", { name: /Arden/ })
        .isChecked()),
    );
    await page.screenshot({
      path: path.join(output, "mobile-content.png"),
      fullPage: true,
    });
    await page.getByRole("button", { name: "4. Preview", exact: true }).click();
    await page.getByRole("button", { name: "Edit text", exact: true }).click();
    assert.equal(
      await tournamentText.getByLabel("Title", { exact: true }).inputValue(),
      "Club tournament champions",
      "Saved text survives reload",
    );
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      "Text editor fits mobile",
    );
    await page.screenshot({
      path: path.join(output, "text-editor-mobile.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Done editing text", exact: true })
      .click();
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
      await page.getByLabel("Story", { exact: true }).first().inputValue(),
      "Stay grounded through your attack.",
    );
    live = await (await fetch(apiOrigin + "/clubs/test/weekly-recaps")).json();
    assert.equal(
      live.selected_recap.recap.report.headline,
      "Club weekly highlights",
    );
    const refreshedDraft = (await state()).weekly_recaps[0];
    assert(
      JSON.stringify(refreshedDraft.final_json.report.edition).includes(
        "Club tournament champions",
      ),
      "Refresh preserves text for the same source result",
    );
    const savedLayout = (await state()).club_recap_preferences[0].settings
      .templates[0].builder;
    assert.equal(savedLayout.text_overrides, undefined);
    assert.equal(savedLayout.display_text, undefined);
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
      await evidence
        .getByRole("checkbox", {
          name: "Include The week in numbers",
          exact: true,
        })
        .isChecked(),
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
    await page.getByRole("button", { name: "1. Setup", exact: true }).click();
    await page.waitForFunction(
      () =>
        !document.body.innerText.includes("Saving draft…") &&
        !document.body.innerText.includes("Unsaved changes"),
    );
    let manager = page.getByTestId("recap-layout-manager");
    await manager.locator("summary").click();
    await manager
      .getByLabel("Rename layout", { exact: true })
      .fill("Club highlights");
    await manager
      .getByRole("button", { name: "Rename saved layout", exact: true })
      .click();
    await page.getByText("Layout renamed.", { exact: true }).waitFor();
    assert.equal(
      (await state()).club_recap_preferences[0].settings.templates[0].name,
      "Club highlights",
    );
    await manager.locator("summary").click();
    await manager
      .getByLabel("Default for new recaps", { exact: true })
      .selectOption("");
    await page
      .getByText("New recaps will use the standard layout.", { exact: true })
      .waitFor();
    await manager.locator("summary").click();
    await confirm(page, "Delete saved layout", "DELETE LAYOUT");
    assert.equal(
      (await state()).club_recap_preferences[0].settings.templates.length,
      0,
    );
    assert.equal(
      (await state()).weekly_recaps.length,
      2,
      "Layout deletion preserves both recap editions",
    );
    console.log(
      "PASS: generate, independent email ordering, photo/promotion cards, PDF/PNG export, autosave, live preview, reusable default, publish, private revision, refresh, reload, layout rename/default/delete, desktop/mobile.",
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
