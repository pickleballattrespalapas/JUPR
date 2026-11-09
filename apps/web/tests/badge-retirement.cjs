const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
function load(file, imports) {
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(name => {
    if (Object.hasOwn(imports, name)) return imports[name];
    assert.ok(!name.startsWith("@/"), `Retired page must not import a service or badge panel: ${name}`);
    return require(name);
  }, module, module.exports);
  return module.exports.default;
}
(async () => {
  const page = load("app/clubs/[clubSlug]/badge-codex/page.tsx", {
    "next/navigation": { notFound: () => { throw Error("NOT_FOUND"); } },
  });
  for (const slug of ["tres-palapas", "other-club"]) {
    assert.throws(() => page({ params: Promise.resolve({ clubSlug: slug }), searchParams: Promise.resolve({ badge: "old" }) }), /NOT_FOUND/);
  }
  let authChecks = 0;
  const admin = load("app/admin/badges/page.tsx", {
    "@/lib/adminWorkspaceServer": { requireAdminWorkspace: async () => { authChecks++; return {}; } },
    "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  });
  const html = renderToStaticMarkup(await admin());
  assert.equal(authChecks, 1);
  assert.match(html, /Badges are disabled/);
  assert.doesNotMatch(html, /<button|<form|badge-codex/);
  const Edition = load("components/RecapEdition.tsx", {
    "@/lib/weeklyReportEmail": { reportPath: () => "", emailUrl: () => "" },
    "@/components/PublicClubLink": ({ children, ...props }) => React.createElement("a", props, children),
  });
  const blocks = [
    { key: "achievements", title: "Retired badge list", items: [{ id: "badge", title: "Old badge", lines: [] }] },
    { key: "events", title: "Event results", items: [{ id: "result", title: "Tournament podium", lines: [] }] },
  ];
  const recap = renderToStaticMarkup(React.createElement(Edition, { blocks, clubSlug: "tres-palapas" }));
  assert.doesNotMatch(recap, /Retired badge|Old badge/);
  assert.match(recap, /Tournament podium/);
  assert.equal(blocks.length, 2, "Historical data is preserved while retired badge content is hidden");
  console.log("PASS retired badge pages perform no badge reads and offer no processing controls.");
})().catch(error => { console.error(error); process.exitCode = 1; });
