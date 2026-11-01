const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
function load(file) {
  const code = ts.transpileModule(
    fs.readFileSync(path.join(__dirname, "..", file), "utf8"),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.CommonJS,
      },
    },
  ).outputText;
  const module = { exports: {} };
  new Function("module", "exports", code)(module, module.exports);
  return module.exports;
}
const { projectEditions, normalizeBuilder, layoutTemplate, previewRecap, resolveBuilderSelections, sourceOnlyRecapChange } =
  load("lib/recapBuilder.ts");
const { buildWeeklyReportEmail } = load("lib/weeklyReportEmail.ts");
// The Python suite independently verifies this shared projection fixture.
const fixture = require("./fixtures/weekly-recap-builder.json");
const groups = { ...fixture.catalog, custom: fixture.config.custom_blocks };
const draws = require("./fixtures/weekly-recap-draws.json");
for (const catalog of [draws.catalog, draws.refreshed_catalog]) {
  const config = normalizeBuilder(draws.config);
  const resolved = resolveBuilderSelections(config, catalog);
  assert.equal(new Set(resolved.sections.events.item_ids).size, 4);
  const editions = projectEditions(catalog, config);
  const emailHtml = buildWeeklyReportEmail({ report: editions }, { origin: "", clubSlug: "test", compact: true });
  for (const winner of draws.winners) assert(emailHtml.includes(winner), "Every explicitly selected podium appears in the actual email HTML, even beyond the default three");
  for (const channel of ["edition", "email_edition"])
    assert.deepEqual(editions[channel][0].items.map(item => item.lines[0]), draws.winners);
  const selected = resolveBuilderSelections(normalizeBuilder({ sections: { events: {
    item_ids: [draws.catalog.events[0].id, draws.catalog.events[3].id],
    email_item_ids: [draws.catalog.events[3].id],
  } } }), catalog);
  const selective = projectEditions(catalog, selected);
  assert.deepEqual(selective.edition[0].items.map(item => item.lines[0]), [draws.winners[0], draws.winners[3]]);
  assert.deepEqual(selective.email_edition[0].items.map(item => item.lines[0]), [draws.winners[3]]);
  assert.deepEqual(config, normalizeBuilder(draws.config), "Resolving old IDs leaves authored config untouched");
}
const sharedId = draws.config.sections.events.item_ids[0];
const ambiguous = projectEditions(draws.catalog, normalizeBuilder({ ...draws.config, text_overrides: { [sharedId]: { title: "Wrong shared title" } } }));
assert(!ambiguous.edition[0].items.some(item => item.title === "Wrong shared title"));
const beforeRename = { id: "recap", club_id: "club", week_start: "2026-10-26", week_end: "2026-11-01", status: "draft", row_version: 2, edits_json: { editorial: { headline: "Our week", opening: "Welcome" } } };
const renamed = { ...beforeRename, row_version: 3, generated_json: { title: "4.0 Round Robin" }, edits_json: { editorial: { opening: "Welcome", headline: "Our week" } } };
assert(sourceOnlyRecapChange(beforeRename, renamed), "Source-only updates allow preserving local edits");
for (const changed of [{ edits_json: { editorial: { headline: "Someone else's edits" } } }, { id: "replacement" }, { club_id: "other" }, { status: "published" }, { row_version: 2 }, { week_end: "2026-11-02" }])
  assert(!sourceOnlyRecapChange(beforeRename, { ...renamed, ...changed }), "A conflicting editor, publication, or identity is never overwritten");
assert.deepEqual(
  projectEditions(groups, normalizeBuilder(fixture.config)),
  fixture.expected,
);
const report = { headline: "Club recap", ...fixture.expected };
const html = buildWeeklyReportEmail(
  { report },
  {
    origin: "https://staging.example.invalid",
    clubSlug: "test",
    published: true,
  },
);
assert(!html.includes("Tuesday Moneyball"));
assert(html.includes("Example Tournament") && html.includes("Coach’s tip"));
assert(!html.includes("Fall Ladder") && !html.includes("The week in numbers"));
assert(html.indexOf("Coach’s tip") < html.indexOf("Event results"));
const template = layoutTemplate(fixture.config);
assert.deepEqual(template.custom_blocks, []);
assert.equal(template.sections.events.item_ids, undefined);
assert.equal(template.sections.numbers.enabled, false);
const preview = previewRecap(
  { builder_catalog: fixture.catalog },
  {
    editorial: { headline: "Test", opening: "", event_notes: "", upcoming: [] },
    looking_ahead: [],
    scope: { league_names: [] },
    builder: normalizeBuilder(),
    spotlight_overrides: {
      new_category: {
        include: true,
        order: 1,
        description: "",
        players: ["player", "player"],
      },
    },
  },
  {
    new_category: [
      { candidate_id: "player", label: "New category", display: "Alex" },
    ],
  },
);
const spotlight = preview.report.edition.find((b) => b.key === "spotlight");
assert.equal(spotlight.items[0].id, "spotlight-new_category");
assert.deepEqual(spotlight.items[0].lines, ["Alex"]);
console.log(
  "Recap builder: server/client projection, exact email selections, templates and spotlight identity passed",
);
const rich = normalizeBuilder(
  require("./fixtures/weekly-recap-stories.json").config,
);
const richEditions = projectEditions(
  { ...fixture.catalog, custom: rich.custom_blocks },
  rich,
);
assert.deepEqual(
  richEditions.edition.slice(0, 2).map((b) => b.key),
  ["custom", "events"],
);
assert.deepEqual(
  richEditions.email_edition.slice(0, 2).map((b) => b.key),
  ["events", "custom"],
);
assert.deepEqual(
  richEditions.edition[0].items.map((i) => i.id),
  ["photo", "register"],
);
assert.deepEqual(
  richEditions.email_edition[1].items.map((i) => i.id),
  ["register", "photo"],
);
const richHtml = buildWeeklyReportEmail(
  { report: { headline: "Club recap", ...richEditions } },
  {
    origin: "https://staging.example.invalid",
    clubSlug: "test",
    published: true,
  },
);
assert(richHtml.indexOf("Autumn social") < richHtml.indexOf("Saturday smiles"));
assert(!richHtml.includes("Next week draft"));
assert(
  richHtml.includes("Photo: Club volunteer") &&
    richHtml.includes("Saturday at the club"),
);
assert(
  richHtml.includes('data-card-style="promotion"') &&
    richHtml.includes("Register now"),
);
const injectionHtml = buildWeeklyReportEmail(
  {
    report: {
      edition: [
        {
          key: "custom",
          title: "Stories",
          items: [
            {
              id: "x",
              title: "Safe",
              text: "",
              lines: [],
              caption: "<script>alert(1)</script>",
              credit: "<b>Joe</b>",
              image_url: "https://images.example.invalid/x.png",
              layout: "photo",
            },
          ],
        },
      ],
    },
  },
  { origin: "", clubSlug: "test" },
);
assert(
  !injectionHtml.includes("<script>") &&
    injectionHtml.includes("&lt;script&gt;"),
);
assert(!injectionHtml.includes("<b>Joe</b>"));
assert.deepEqual(layoutTemplate(rich).email_order.slice(0, 2), [
  "events",
  "custom",
]);
const { recapPageBreaks } = load("lib/weeklyRecapExport.ts");
const pages = recapPageBreaks(2400, 900, [
  { top: 700, bottom: 1100 },
  { top: 1550, bottom: 1900 },
]);
assert.deepEqual(pages, [
  { top: 0, bottom: 700 },
  { top: 700, bottom: 1550 },
  { top: 1550, bottom: 2400 },
]);
assert.deepEqual(recapPageBreaks(1800, 900, [{ top: 0, bottom: 1800 }]), [
  { top: 0, bottom: 900 },
  { top: 900, bottom: 1800 },
]);
assert.deepEqual(recapPageBreaks(100, 0, []), []);
console.log(
  "Rich stories: independent email order, selection, captions, safe HTML, reusable layouts and PDF pagination passed",
);

const textConfig = normalizeBuilder(
  require("./fixtures/weekly-recap-text.json"),
);
const sourceGroups = { ...fixture.catalog, custom: textConfig.custom_blocks };
const originalGroups = JSON.stringify(sourceGroups);
const editedEditions = projectEditions(sourceGroups, textConfig);
assert.equal(
  JSON.stringify(sourceGroups),
  originalGroups,
  "Presentation edits never mutate source results",
);
for (const channel of ["edition", "email_edition"]) {
  const blocks = Object.fromEntries(
    editedEditions[channel].map((b) => [b.key, b]),
  );
  const event = blocks.events.items.find(
    (i) => i.title === "Thursday champions",
  );
  assert.equal(blocks.events.title, "This week’s results");
  assert.equal(event.text, "Congratulations, Cam!");
  assert.deepEqual(event.lines.slice(0, 2), [
    "A lively day on the courts",
    "1. Cam — first place",
  ]);
  assert.equal(event.results_label, "See the final standings");
  assert.equal(blocks.numbers.items[0].text, "Ninety-six");
  assert.equal(blocks.upcoming.items[0].text, "Next week’s Moneyball");
}
const editedHtml = buildWeeklyReportEmail(
  {
    week_start: "2026-10-05",
    week_end: "2026-10-11",
    report: { ...editedEditions, display_text: textConfig.display_text },
  },
  {
    origin: "https://staging.example.invalid",
    clubSlug: "test",
    published: true,
  },
);
for (const copy of [
  "Thursday champions",
  "Social play",
  "Congratulations, Cam!",
  "See our recorded league week",
  "Courtesy of Morgan",
  "Friends on court",
  "Members celebrating",
  "Reserve your place",
  "Sign up for next week",
  "October 5–11 · A week to remember",
  "Read the entire recap",
])
  assert(editedHtml.includes(copy), copy);
assert(editedHtml.includes("data-number-cards") && editedHtml.includes("font-size:26px"), "Email preserves the website's number-card presentation");
assert(!editedHtml.includes("letter-spacing:1.7px"), "Edited section headings retain normal heading formatting");
const draftEditedReport = structuredClone(editedEditions);
for (const channel of ["edition", "email_edition"]) draftEditedReport[channel].find(block => block.key === "events").items.find(item => item.title === "Thursday champions").href = "round-robin-generator/shared/test-session/standings";
const draftEditedHtml = buildWeeklyReportEmail({ report: draftEditedReport }, { origin: "https://staging.example.invalid", clubSlug: "test", published: false });
assert(draftEditedHtml.includes("See the final standings") && draftEditedHtml.includes("See our recorded league week"), "Draft email previews retain edited results-link wording");
assert(draftEditedHtml.includes("DRAFT PREVIEW") && !draftEditedHtml.includes("View the full weekly report"), "An unpublished recap still has no live recap link");
assert(!editedHtml.includes("Must never appear"));
assert.equal(layoutTemplate(textConfig).text_overrides, undefined);
assert.equal(layoutTemplate(textConfig).display_text, undefined);
const changedSources = structuredClone(sourceGroups);
changedSources.events.reverse();
changedSources.spotlight[0].lines = ["New winner — new result"];
const refreshed = projectEditions(changedSources, textConfig);
assert(JSON.stringify(refreshed).includes("Thursday champions"));
assert(!JSON.stringify(refreshed).includes("Alex — a standout performance"));
changedSources.events = [];
assert(
  !JSON.stringify(projectEditions(changedSources, textConfig)).includes(
    "Thursday champions",
  ),
);
const reset = { ...textConfig, text_overrides: undefined };
assert(
  JSON.stringify(projectEditions(sourceGroups, reset)).includes(
    "Thursday Social Round Robin",
  ),
);
const movedUpcoming = structuredClone(sourceGroups);
movedUpcoming.upcoming.unshift({ id: "new", title: "", text: "A different event", url: "https://example.invalid/new", lines: [] });
assert.deepEqual(projectEditions(movedUpcoming, textConfig).edition.find(b => b.key === "upcoming").items.slice(0, 2).map(i => i.text), ["A different event", "Next week’s Moneyball"]);
movedUpcoming.upcoming.pop();
assert(!JSON.stringify(projectEditions(movedUpcoming, textConfig)).includes("Next week’s Moneyball"));
console.log(
  "Preview text: shared source-bound edits, captions, links, email wording, reset, refresh and template isolation passed",
);

const courts = require("./fixtures/weekly-recap-courts.json");
const courtConfig = normalizeBuilder({ sections: { events: { detail: "brief", email_item_ids: courts.email_ids } } });
const courtEditions = projectEditions({ events: courts.events }, courtConfig);
assert.deepEqual(courtEditions.edition[0].items.map(i => i.id), ["ladder-court-winners", "rr"]);
assert.deepEqual(courtEditions.edition[0].items[0].lines, courts.expected_lines);
assert.deepEqual(courtEditions.email_edition[0].items[0].lines, courts.expected_email_lines);
const courtEmail = buildWeeklyReportEmail({ report: courtEditions }, { origin: "", clubSlug: "test" });
assert(courtEmail.includes("Alex ×2"));
assert(!/point difference|recorded games|Round 1/.test(courtEmail));
const editedCourt = projectEditions({ events: courts.events }, {
  ...courtConfig, text_overrides: { "ladder-court-winners": { title: "Our court champions", lines: { "Alex ×3": "Alex — three court wins!" } } },
});
assert.equal(editedCourt.edition[0].items[0].title, "Our court champions");
assert.equal(editedCourt.edition[0].items[0].lines[0], "Alex — three court wins!");
courtConfig.sections.events.item_ids = ["a1", "a3"];
const selectedCourts = projectEditions({ events: courts.events }, courtConfig);
assert.deepEqual(selectedCourts.edition[0].items[0].lines, ["Alex ×2"]);
assert.deepEqual(selectedCourts.email_edition[0].items[0].lines, ["Alex"]);
console.log("Ladder recognition: compact list, repeat counts, ties, incomplete courts, saved drafts and exact email selection passed");
