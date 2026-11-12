const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { renderToStaticMarkup } = require("react-dom/server");
const root = path.resolve(__dirname, ".."), cache = new Map();
let response, failure = false, requests = [], display = {};
const mocks = {
  "next/navigation": { notFound() { throw Error("NOT_FOUND"); } },
  "@/components/PublicClubLink": ({ children, prefetch, ...props }) => React.createElement("a", props, children),
  "@/components/SectionLoadError": ({ title }) => React.createElement("p", { role: "alert" }, title),
  "@/components/ClubDisplay": { Display: ({ field, children }) => display[field] === false ? null : children },
  "@/lib/clubSiteServer": { publicSiteFetch: async url => { requests.push(url); if (failure) throw Error("service unavailable"); return response; } },
};
function load(name, parent = root) {
  if (Object.hasOwn(mocks, name)) return mocks[name];
  if (name.endsWith(".css")) return {};
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts"].find(f => fs.existsSync(f) && fs.statSync(f).isFile());
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} }, code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  new Function("require", "module", "exports", code)(dep => load(dep, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports); return module.exports;
}
const Page = load("@/app/clubs/[clubSlug]/round-robins/page").default;
const Detail = load("@/app/clubs/[clubSlug]/round-robins/[sessionKey]/page").default;
const Results = load("@/app/clubs/[clubSlug]/round-robins/RoundRobinResults").default;
const summary = { session_key: "rr/one", title: "Tuesday Moneyball", created_at: "2026-11-10T02:00:00Z", participant_count: 8, kind: "moneyball", play_format: "doubles", scoring_mode: "scored", rating_mode: "rated" };
const page = async search => renderToStaticMarkup(await Page({ params: Promise.resolve({ clubSlug: "alpha" }), searchParams: Promise.resolve(search || {}) }));

(async () => {
  response = { sessions: [summary], has_more: true, offset: 24, limit: 24 };
  let html = await page({ kind: "moneyball", page: "1", from: "2026-11-01", to: "2026-11-30" });
  const query = new URL(requests.at(-1), "https://test.invalid").searchParams;
  assert.equal(query.get("kind"), "moneyball"); assert.equal(query.get("offset"), "24");
  assert.equal(query.get("from_date"), "2026-11-01"); assert.equal(query.get("to_date"), "2026-11-30");
  assert.match(html, /Tuesday Moneyball/); assert.match(html, /Nov 9, 2026/);
  assert.match(html, /8 players/); assert.match(html, /View final standings/);
  assert.match(html, /href="\/clubs\/alpha\/round-robins\/rr%2Fone"/);
  assert.match(html, /page=2.*from=2026-11-01.*to=2026-11-30/); assert.match(html, /Newer sessions/);
  assert.match(html, /<form[^>]*method="get"/);
  response = { sessions: [], has_more: false, offset: 0, limit: 24 };
  assert.match(await page(), /No past sessions found/);
  failure = true; html = await page(); assert.match(html, /temporarily unavailable/); assert.doesNotMatch(html, /No past sessions found/);
  failure = false; const count = requests.length;
  assert.match(await page({ from: "2026-11-30", to: "2026-11-01" }), /end date on or after/);
  assert.equal(requests.length, count, "Invalid date ranges do not make an API call");
  await page({ page: "-999", kind: "unknown" });
  assert.match(requests.at(-1), /kind=all&offset=0/);

  const session = { ...summary, kind: "round_robin", standings_sort: "wins",
    standings: [{ rank: 1, participantId: "a", name: "Carrie Blair", matches: 1, wins: 1, losses: 0, pointsFor: 11, pointsAgainst: 0, differential: 11 }],
    rounds: [{ number: 1, status: "saved", matches: [{ court: 1, side_a: ["Carrie Blair", "Joe"], side_b: ["Linda", "Manuel"], score_a: 11, score_b: 0, status: "saved" }] }] };
  html = renderToStaticMarkup(React.createElement(Results, { session }));
  assert.match(html, /Final standings/); assert.match(html, /Carrie Blair/); assert.match(html, /11–0/);
  assert.match(html, /Round 1.*Complete/); assert.doesNotMatch(html, /<button|<form|<input|Publish|Submit results|Finish session/);
  display = { result_scores: false };
  html = renderToStaticMarkup(React.createElement(Results, { session }));
  assert.match(html, /Final standings/); assert.doesNotMatch(html, /Round results|11–0/);
  display = { result_summary: false };
  html = renderToStaticMarkup(React.createElement(Results, { session }));
  assert.doesNotMatch(html, /Final standings/); assert.match(html, /11–0/);
  display = {};
  html = renderToStaticMarkup(React.createElement(Results, { session: { ...session, scoring_mode: "unscored" } }));
  assert.match(html, /without recorded scores/); assert.doesNotMatch(html, /Final standings/);
  response = null;
  await assert.rejects(() => Detail({ params: Promise.resolve({ clubSlug: "alpha", sessionKey: "missing" }) }), /NOT_FOUND/);
  failure = true;
  html = renderToStaticMarkup(await Detail({ params: Promise.resolve({ clubSlug: "alpha", sessionKey: "rr" }) }));
  assert.match(html, /temporarily unavailable/);
  console.log("Round robin history: filters, pagination, read-only scores, display settings and errors passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
