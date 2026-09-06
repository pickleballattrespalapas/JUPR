const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const ts = require("typescript");

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] || originalRequire(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), { compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText, filename);
  return compiled.exports;
}
const model = load("lib/tournamentSponsors.ts");
const Display = load("components/TournamentSponsorDisplay.tsx", {
  "@/lib/tournamentSponsors": model,
  "./TournamentSponsorDisplay.module.css": { presenting: "presenting", sponsor: "sponsor", logo: "logo", footer: "footer" },
  "next/image": ({ unoptimized, onError, ...props }) => React.createElement("img", props)
}).default;
const sponsor = { id: "1", name: "Coastal Homes", tier: "presenting", level: "", website: "https://example.com", logo_url: "https://example.com/logo.webp" };
const render = (sponsors, placement) => renderToStaticMarkup(React.createElement(Display, { sponsors, placement }));
const header = render([sponsor], "header");
assert.match(header, /Presented by/);
assert.match(header, /<strong>Coastal Homes<\/strong>/);
assert.ok(header.indexOf("<strong>") < header.indexOf("<img"), "Name precedes the uploaded logo in the presenting line");
assert.match(header, /rel="sponsored noopener noreferrer"/);
assert.match(header, /referrerPolicy="no-referrer"/i);
const footer = render([{ ...sponsor, tier: "premier" }], "footer");
assert.match(footer, /<strong>Coastal Homes<\/strong>/, "Uploading a footer logo never replaces the sponsor name");
assert.ok(footer.indexOf("<img") < footer.indexOf("<strong>"));
assert.match(render([{ ...sponsor, logo_url: "", website: "" }], "header"), /Coastal Homes/);
assert.doesNotMatch(render([{ ...sponsor, website: "javascript:alert(1)" }], "header"), /href=/);
assert.equal(render([sponsor], "footer"), "");
assert.equal(render([], "header"), "");
assert.equal(model.normalizeSponsorWebsite("example.com"), "https://example.com/");
console.log("Sponsor render checks passed: names with logos, tier placement, empty tiers, and safe links.");

const titled = (sponsors, headingLevel) => renderToStaticMarkup(React.createElement(Display, {
  sponsors, placement: "header", title: "Baja Classic 2026", headingLevel
}));
assert.match(titled([sponsor]), /<h1[^>]*>Baja Classic 2026<\/h1>/);
assert.ok(titled([sponsor]).indexOf("</h1>") < titled([sponsor]).indexOf("Presented by"));
assert.match(titled([]), /Baja Classic 2026/, "The tournament title remains when sponsors are absent or unavailable");
assert.doesNotMatch(titled([]), /Presented by/);
assert.match(titled([sponsor], "h2"), /<h2[^>]*>Baja Classic 2026<\/h2>/);
assert.match(titled([sponsor, { ...sponsor, id: "2", name: "Second sponsor" }]), / and /);
