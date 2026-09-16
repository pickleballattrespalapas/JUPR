const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react");
const { act, create } = require("react-test-renderer");
const { renderToStaticMarkup } = require("react-dom/server");
let pathname = "/clubs/alpha";
const initial = {
  schema_version: 1, name: "Alpha", description: "Welcome", location: "Baja",
  visitor_info: "", logo_url: "", accent: "#1d4ed8", visibility: "listed", display: {},
  pages: [
    { slug: "home", title: "Home", in_navigation: true, blocks: [] },
    { slug: "visiting", title: "Visiting", in_navigation: true, blocks: [] },
    { slug: "members", title: "Members", in_navigation: false, blocks: [] },
  ],
};
let site = { club_id: "alpha", slug: "alpha", club_active: true, revision: 1,
  draft: structuredClone(initial), published: structuredClone(initial), published_at: "2026-09-16" };
const mocks = {
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "next/navigation": { usePathname: () => pathname, notFound: () => { throw Error("NOT_FOUND"); } },
  "next/headers": { headers: () => new Headers({ "x-pcs-club-path": pathname }) },
  "@/lib/clubSiteServer": {
    getPublicSite: async slug => slug === "alpha" ? { slug, document: structuredClone(site.published) } : null,
    getClubDirectory: async () => ({ clubs: [{ slug: "alpha" }], total: 1, limit: 100 }),
  },
  "@/lib/useAdminWorkspace": { useAdminWorkspace: () => ({ clubId: "alpha" }) },
  "@/lib/useAdminSession": { useAdminSession: () => ({ loading: false, accessToken: "fixture-token",
    session: { user: { id: "one" }, capabilities: { assignments: [{ club_id: "alpha", role: "administrator" }] } } }) },
  "@/lib/adminAuthClient": { getAdminApiBaseUrl: () => "https://api.test" },
  "@/lib/adminWorkspace": { readBrowserWorkspace: () => ({ clubId: "alpha" }) },
};
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const m = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  new Function("require", "module", "exports", code)(name => {
    if (name.endsWith(".css")) return {};
    if (Object.hasOwn(mocks, name)) return mocks[name];
    if (name.startsWith("@/") || name.startsWith(".")) {
      const f = name.startsWith("@/") ? name.slice(2) : path.join(path.dirname(file), name);
      return load(fs.existsSync(path.join(__dirname, "..", f + ".tsx")) ? f + ".tsx" : f + ".ts");
    }
    return require(name);
  }, m, m.exports);
  cache.set(file, m.exports);
  return m.exports;
}
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const h = React.createElement;
const html = (Component, props) => renderToStaticMarkup(h(Component, props));
const mutations = [];
global.window = { location: { origin: "https://staging.test" }, addEventListener() {}, removeEventListener() {} };
global.fetch = async (url, options = {}) => {
  assert.ok(url.startsWith("https://api.test/admin/clubs/alpha/site"));
  if (options.method) {
    const body = JSON.parse(options.body);
    assert.equal(body.revision, site.revision);
    assert.equal(options.headers.Authorization, "Bearer fixture-token");
    mutations.push({ url, body });
    if (options.method === "PUT") site.draft = structuredClone(body.document);
    else if (url.endsWith("/publish")) site.published = structuredClone(site.draft);
    site.revision++;
  }
  return { ok: true, status: 200, json: async () => structuredClone(site) };
};

(async () => {
  const { CLUB_LINKS, clubPageSection, publicClubPage, publicClubLinks } = load("lib/clubSite.ts");
  assert.equal(publicClubLinks(initial).length, 10, "Existing documents retain their public sections");
  const doc = { ...initial, page_visibility: { players: "private", tournaments: "private", matches: "private", leagues: "private", play: "private" } };
  for (const [route, section] of [
    ["players/12?source=results", "players"], ["matches/20", "matches"],
    ["tournament-registration/confirmation?id=3", "tournaments"], ["tournaments/past", "tournaments"],
    ["team-leagues/fall", "leagues"], ["league-results", "leagues"], ["challenge-ladder", "leagues"],
    ["round-robin-generator/sessions/abc/rounds/2", "play"], ["live/session", "play"],
    ["pages/members", "pages/members"],
  ]) {
    assert.equal(clubPageSection(`/clubs/alpha/${route}`, "alpha"), section);
    assert.equal(publicClubPage(doc, "alpha", `/clubs/alpha/${route}`), false, route);
  }
  assert.equal(publicClubPage(doc, "alpha", "/clubs/beta/players"), true, "Other clubs keep their own settings");
  assert.equal(publicClubPage(doc, "alpha", "/clubs/alpha/%70layers/12"), false);
  const Header = load("components/ClubSiteHeader.tsx").default;
  const Content = load("components/ClubSiteContent.tsx").default;
  for (const Component of [Header, Content]) {
    const markup = html(Component, { document: doc, slug: "alpha" });
    assert.doesNotMatch(markup, /href="\/clubs\/alpha\/(players|tournaments|matches|leagues|play)"/);
    assert.match(markup, /href="\/clubs\/alpha\/leaderboards"/);
    assert.doesNotMatch(markup, /href="\/clubs\/alpha\/pages\/members"/);
  }
  const Footer = load("components/PublicFooterNav.tsx").default;
  const footer = html(Footer, {});
  assert.doesNotMatch(footer, /href="\/clubs\/alpha\//, "Global footer outside the provider must not advertise private club pages");
  assert.match(footer, /href="\/clubs"/);
  const block = { id: "links", kind: "links", heading: "", text: "", url: "", alt: "", span: 12, align: "left", tone: "plain", padding: "small" };
  const withBlocks = { ...doc, pages: [{ ...doc.pages[0], blocks: [block, { ...block, id: "button", kind: "button", text: "Hidden target", url: "/clubs/alpha/players" }] }] };
  assert.doesNotMatch(html(Content, { document: withBlocks, slug: "alpha" }), /Hidden target|href="\/clubs\/alpha\/players"/);
  assert.doesNotMatch(html(Content, { document: { ...initial, page_visibility: Object.fromEntries(CLUB_LINKS.map(([, key]) => [key, "private"])) }, slug: "alpha" }), /Around the club/);

  const { default: Link, ClubPageNavigationProvider: Provider } = load("components/PublicClubLink.tsx");
  const links = () => h(Provider, { slug: "alpha", settings: doc },
    h(Link, { href: "/clubs/alpha/players/12" }, "Avery"),
    h(Link, { href: "/clubs/alpha/players" }, "All players"),
    h(Link, { href: "/clubs/alpha/matches/20" }, "11–8"),
    h(Link, { href: "/clubs/alpha/tournament-registration?id=3" }, "Register"));
  pathname = "/clubs/alpha/leaderboards";
  let markup = renderToStaticMarkup(links());
  assert.match(markup, /<span>Avery<\/span>.*<span>11–8<\/span>/);
  assert.doesNotMatch(markup, /href=|All players|Register/);
  pathname = "/clubs/alpha/players/12";
  markup = renderToStaticMarkup(links());
  assert.match(markup, /href="\/clubs\/alpha\/players\/12"/);
  assert.match(markup, /All players/);
  assert.doesNotMatch(markup, /href="\/clubs\/alpha\/matches\/20"|Register/);
  pathname = "/clubs/alpha/tournaments";
  assert.match(renderToStaticMarkup(links()), /href="\/clubs\/alpha\/tournament-registration\?id=3"/);

  site.published = structuredClone(doc);
  const { generateMetadata } = load("app/clubs/[clubSlug]/layout.tsx");
  for (const route of ["players/12", "tournament-registration", "pages/members", "live/session"]) {
    pathname = `/clubs/alpha/${route}`;
    assert.deepEqual((await generateMetadata({ params: { clubSlug: "alpha" } })).robots, { index: false, follow: false });
  }
  pathname = "/clubs/alpha/leaderboards";
  assert.equal((await generateMetadata({ params: { clubSlug: "alpha" } })).robots.index, true);
  site.published.visibility = "unlisted";
  assert.equal((await generateMetadata({ params: { clubSlug: "alpha" } })).robots.index, false);
  site.published.visibility = "listed";
  const sitemap = await load("app/sitemap.ts").default();
  assert.ok(sitemap.some(r => r.url.endsWith("/clubs/alpha/leaderboards")));
  assert.ok(sitemap.some(r => r.url.endsWith("/clubs/alpha/pages/visiting")));
  assert.ok(!sitemap.some(r => /\/alpha\/(players|matches|tournaments|pages\/members)$/.test(r.url)));
  const Custom = load("app/clubs/[clubSlug]/pages/[pageSlug]/page.tsx").default;
  assert.match(renderToStaticMarkup(await Custom({ params: { clubSlug: "alpha", pageSlug: "members" } })), /Members/);
  const { NextRequest } = require("next/server");
  const forwarded = load("middleware.ts").middleware(new NextRequest("https://staging.test/clubs/alpha/players/12", { headers: { "x-pcs-club-path": "/clubs/alpha/leaderboards" } }));
  assert.equal(forwarded.headers.get("x-middleware-request-x-pcs-club-path"), "/clubs/alpha/players/12");

  site.published = structuredClone(initial);
  const Website = load("app/admin/website/page.tsx").default;
  let tree;
  await act(async () => { tree = create(h(Website)); });
  const button = label => tree.root.findAllByType("button").find(b => text(b) === label);
  const field = label => tree.root.findByProps({ "aria-label": label });
  await act(async () => button("Page visibility").props.onClick());
  assert.equal(field("Visibility for Players").props.value, "public");
  await act(async () => field("Visibility for Players").props.onChange({ target: { value: "private" } }));
  await act(async () => field("Visibility for Visiting").props.onChange({ target: { value: "private" } }));
  assert.equal(mutations.length, 0, "Editing does not publish or save automatically");
  assert.equal(button("Publish website").props.disabled, true);
  await act(async () => button("Save draft").props.onClick());
  assert.equal(mutations[0].body.document.page_visibility.players, "private");
  assert.equal(mutations[0].body.document.pages[1].in_navigation, false);
  assert.equal(site.published.page_visibility, undefined, "Saving leaves published visibility unchanged");
  await act(async () => button("Preview draft").props.onClick());
  assert.ok(!tree.root.findAllByType("a").some(a => a.props.href === "/clubs/alpha/players"));
  await act(async () => button("Edit page visibility").props.onClick());
  assert.equal(field("Link to Players").props.value, "https://staging.test/clubs/alpha/players");
  let copied;
  Object.defineProperty(global, "navigator", { configurable: true, value: { clipboard: { writeText: async value => { copied = value; } } } });
  await act(async () => field("Copy link to Players").props.onClick());
  assert.equal(copied, "https://staging.test/clubs/alpha/players");
  assert.match(text(tree.toJSON()), /Link copied for Players/);
  global.navigator.clipboard.writeText = async () => { throw Error("Unavailable"); };
  await act(async () => field("Copy link to Players").props.onClick());
  assert.match(text(tree.toJSON()), /Select and copy the link shown for Players/);
  await act(async () => button("Publish website").props.onClick());
  assert.equal(site.published.page_visibility.players, "private");
  assert.match(text(tree.toJSON()), /Live: Private · link only/);
  assert.deepEqual(mutations[1].body, { revision: 2 });
  await act(async () => tree.unmount());
  site.published = null;
  await act(async () => { tree = create(h(Website)); });
  await act(async () => button("Page visibility").props.onClick());
  assert.equal(field("Copy link to Players").props.disabled, true, "Unpublished websites cannot share a live link");
  await act(async () => tree.unmount());
  console.log("Club page visibility: private discovery, shared routes, scoped navigation, noindex, sitemap, draft/preview/publish and copy links passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
