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
const { projectEditions, normalizeBuilder, layoutTemplate, previewRecap } =
  load("lib/recapBuilder.ts");
const { buildWeeklyReportEmail } = load("lib/weeklyReportEmail.ts");
// The Python suite independently verifies this shared projection fixture.
const fixture = require("./fixtures/weekly-recap-builder.json");
const groups = { ...fixture.catalog, custom: fixture.config.custom_blocks };
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
