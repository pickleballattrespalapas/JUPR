const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { create, act } = require("react-test-renderer");
global.IS_REACT_ACT_ENVIRONMENT = true;
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const lib = load("lib/recapBuilder.ts");
const email = load("lib/weeklyReportEmail.ts");
const Edition = load("components/RecapEdition.tsx", {
  "@/lib/weeklyReportEmail": email,
  "@/components/PublicClubLink": props => React.createElement("a", props),
}).default;
const Preview = load("app/admin/weekly-recap/AdminWeeklyRecapPreview.tsx", {
  "@/components/RecapEdition": Edition,
  "@/components/WeeklyReportSections": {},
}).default;
const catalog = structuredClone(require("./fixtures/weekly-recap-draws.json").refreshed_catalog);
const item = (id, kind, title) => ({ id, kind, title, text: "Winner: Alex", lines: ["Alex — 3–0 · point difference +9"] });
catalog.events = [catalog.events[0], item("rr", "Round robin", "Monday Round Robin"), catalog.events[2], item("court1", "Ladder court", "Court 1"), item("money", "Moneyball", "Moneyball"), item("court2", "Ladder court", "Court 2")];
catalog.custom = [item("story", "", "Club story")];
const initial = lib.normalizeBuilder();
const initialEmail = lib.projectEditions(catalog, initial).email_edition;
const eventItems = config => lib.projectEditions(catalog, config).edition.find(block => block.key === "events").items;
let moved = lib.reorderRecapPreview(initial, catalog, "edition", { type: "item", section: "events", itemId: "rr", delta: -1 });
assert.equal(eventItems(moved)[0].id, "rr");
assert.deepEqual(lib.projectEditions(catalog, moved).email_edition, initialEmail, "Website movement preserves implicit email selection and order");
const before = new Set(moved.sections.events.item_ids);
moved = lib.reorderRecapPreview(moved, catalog, "edition", { type: "item", section: "events", itemId: "ladder-court-winners", delta: 1 });
assert.equal(eventItems(moved).at(-1).id, "ladder-court-winners");
assert.deepEqual(eventItems(moved).at(-1).lines, ["Alex ×2"]);
assert.deepEqual(new Set(moved.sections.events.item_ids), before, "Moving a compact ladder result retains every selected source court");
const grouped = lib.reorderRecapPreview(moved, catalog, "edition", { type: "group_events" });
assert.deepEqual(eventItems(grouped).map(item => item.kind || item.id), ["Round robin", "Tournament", "Tournament", "Moneyball", "ladder-court-winners"]);
const movedSection = lib.reorderRecapPreview(initial, catalog, "edition", { type: "section", section: "custom", delta: -1 });
assert.deepEqual(lib.projectEditions(catalog, movedSection).edition.map(block => block.key), ["custom", "events"]);
assert.deepEqual(lib.projectEditions(catalog, movedSection).email_edition, initialEmail);
const emailMove = lib.reorderRecapPreview(movedSection, catalog, "email_edition", { type: "item", section: "events", itemId: "rr", delta: -1 });
assert.deepEqual(lib.projectEditions(catalog, emailMove).edition, lib.projectEditions(catalog, movedSection).edition, "Email rearrangement does not change the website");
assert.strictEqual(lib.reorderRecapPreview(initial, catalog, "edition", { type: "section", section: "events", delta: -1 }), initial, "Top boundary is a no-op");

(async () => {
  let builder = lib.normalizeBuilder(), tree;
  const render = (overrides = {}) => React.createElement(Preview, {
    recap: { id: "recap", week_start: "2026-10-26", week_end: "2026-11-01", status: "draft", row_version: 1,
      final_json: { report: { headline: "Our week", ...lib.projectEditions(catalog, builder) } } },
    onReorder: action => { builder = lib.reorderRecapPreview(builder, catalog, "edition", action); tree.update(render()); },
    ...overrides,
  });
  await act(async () => { tree = create(render()); });
  const button = label => tree.root.findByProps({ "aria-label": label });
  assert(button("Move section Event results up in website").props.disabled);
  assert(button("Move section Club stories down in website").props.disabled);
  await act(async () => button("Move Monday Round Robin up in website").props.onClick());
  assert.equal(tree.root.findAllByType("article")[0].props["data-recap-item"], "rr");
  await act(async () => tree.root.findAllByType("button").find(node => node.children.includes("Group events by type")).props.onClick());
  assert.deepEqual(eventItems(builder).slice(1, 3).map(item => item.kind), ["Tournament", "Tournament"]);
  for (const override of [{ printMode: true }, { onReorder: undefined }]) {
    await act(async () => tree.update(render(override)));
    assert(!tree.root.findAllByType("button").some(node => String(node.props["aria-label"] || "").startsWith("Move ")), "Print and ordinary rendering contain no arranging controls");
  }
  await act(async () => tree.unmount());
  let publicTree;
  await act(async () => { publicTree = create(React.createElement(Edition, { blocks: lib.projectEditions(catalog, builder).edition, clubSlug: "test" })); });
  assert.equal(publicTree.root.findAllByType("button").length, 0, "Public recap has no admin controls");
  const html = email.buildWeeklyReportEmail({ report: lib.projectEditions(catalog, builder) }, { origin: "", clubSlug: "test" });
  assert(!html.includes("Move up") && !html.includes("Group events"), "Exports contain only report content");
  await act(async () => publicTree.unmount());
  console.log("Preview arranging: sections, items, grouping, compact courts, independent email order and clean public/print output passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
