const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  ts = require("typescript"),
  React = require("react");
const { act, create } = require("react-test-renderer");
let clubId = "alpha",
  identity = "one",
  token = "token-one",
  workspace = "alpha",
  restored = null,
  redirects = [],
  cookie = "";
let network = [],
  pending = null,
  replyStatus = 200;
const docs = (name) => ({
  schema_version: 1,
  name,
  description: "Welcome",
  location: "Baja",
  visitor_info: "Visitors welcome",
  logo_url: "",
  accent: "#1d4ed8",
  visibility: "unlisted",
  display: {},
  pages: [{ slug: "home", title: "Home", in_navigation: true, blocks: [] }],
});
let current = {
  club_id: "alpha",
  slug: "alpha",
  club_active: true,
  revision: 1,
  draft: docs("Alpha"),
  published: docs("Live Alpha"),
  published_at: "2026-09-16",
};
const cache = new Map();
const mocks = {
  "next/link": ({ children, ...props }) =>
    React.createElement("a", props, children),
  "next/navigation": {
    redirect: (url) => {
      throw new Error(`REDIRECT:${url}`);
    },
  },
  "next/headers": { cookies: () => ({ get: () => ({ value: cookie }) }) },
  "@/lib/clubSiteServer": {
    getPublicSite: async (slug) =>
      slug === "alpha" ? { document: docs("Alpha") } : null,
  },
  "@/lib/useAdminWorkspace": { useAdminWorkspace: () => ({ clubId }) },
  "@/lib/useAdminSession": {
    useAdminSession: () => ({
      loading: false,
      accessToken: token,
      session: {
        user: { id: identity },
        capabilities: {
          assignments: [{ club_id: clubId, role: "administrator" }],
        },
      },
    }),
  },
  "@/lib/adminAuthClient": {
    getAdminApiBaseUrl: () => "https://api.test",
    consumeStaffInvitationSession: async () => restored,
    refreshAdminSession: async (s) => s,
    authorizeAndSaveAdminSession: async () => {},
    getAdminAuthConfig: () => null,
  },
  "@/lib/adminWorkspace": {
    readBrowserWorkspace: () => ({ clubId: workspace }),
    selectAdminWorkspace: (w) => redirects.push(w.club_id),
  },
};
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const m = { exports: {} };
  const code = ts.transpileModule(
    fs.readFileSync(path.join(__dirname, "..", file), "utf8"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        jsx: ts.JsxEmit.ReactJSX,
        esModuleInterop: true,
      },
    },
  ).outputText;
  new Function("require", "module", "exports", code)(
    (n) => {
      if (n.endsWith(".css")) return {};
      if (Object.hasOwn(mocks, n)) return mocks[n];
      if (n.startsWith("@/") || n.startsWith(".")) {
        let f = n.startsWith("@/") ? n.slice(2) : path.join(path.dirname(file), n);
        return load(
          fs.existsSync(path.join(__dirname, "..", f + ".tsx"))
            ? f + ".tsx"
            : f + ".ts",
        );
      }
      return require(n);
    },
    m,
    m.exports,
  );
  cache.set(file, m.exports);
  return m.exports;
}
const response = (body, status = 200) => ({
  ok: status < 400,
  status,
  json: async () => structuredClone(body),
});
global.window = {
  addEventListener() {},
  removeEventListener() {},
  location: { origin: "https://staging.test" },
};
global.fetch = async (url, options = {}) => {
  network.push({ url, options });
  if (options.method)
    return new Promise((resolve) => {
      pending = resolve;
    });
  return response(
    url.endsWith("signup-options") ? { email_enabled: false } : current,
  );
};
const Website = load("app/admin/website/page.tsx").default;
let tree;
function text(node) {
  return typeof node === "string"
    ? node
    : (node.children || []).map(text).join("");
}
const button = (label) =>
  tree.root.findAllByType("button").find((b) => text(b) === label);
(async () => {
  const { safeSiteUrl, validClubSlug } = load("lib/clubSite.ts");
  assert.equal(safeSiteUrl("javascript:alert(1)"), undefined);
  assert.equal(safeSiteUrl("//evil.test"), undefined);
  assert.equal(validClubSlug("../../admin"), false);
  const { ClubDisplayProvider, Display } = load("components/ClubDisplay.tsx");
  let t;
  await act(async () => {
    t = create(
      React.createElement(
        ClubDisplayProvider,
        { display: { ratings: false } },
        React.createElement(
          Display,
          { field: "ratings" },
          React.createElement("span", null, "secret draft statistic"),
        ),
      ),
    );
  });
  assert.equal(t.toJSON(), null);
  await act(async () => t.unmount());
  await act(async () => {
    tree = create(React.createElement(Website));
  });
  let input = tree.root.findAllByType("input")[0];
  assert.equal(input.props.value, "Alpha");
  await act(async () =>
    input.props.onChange({ target: { value: "Draft Alpha" } }),
  );
  assert.equal(button("Publish website").props.disabled, true);
  await act(async () => button("Preview draft").props.onClick());
  assert.equal(network.length, 1, "Preview makes no public request or write");
  assert.ok(text(tree.toJSON()).includes("Draft Alpha"));
  await act(async () => {
    void button("Save draft").props.onClick();
    void button("Save draft").props.onClick();
  });
  assert.equal(network.length, 2, "Duplicate saves prevented");
  const payload = JSON.parse(network.at(-1).options.body);
  assert.equal(payload.document.name, "Draft Alpha");
  assert.equal(payload.revision, 1);
  assert.ok(!payload.actor_id);
  current = { ...current, revision: 2, draft: docs("Draft Alpha") };
  await act(async () => pending(response(current)));
  assert.equal(button("Publish website").props.disabled, false);
  await act(async () => button("Publish website").props.onClick());
  assert.ok(network.at(-1).url.endsWith("/site/publish"));
  assert.deepEqual(JSON.parse(network.at(-1).options.body), { revision: 2 });
  await act(async () => pending(response({ detail: "Draft changed" }, 409)));
  assert.equal(button("Publish website").props.disabled, true);
  assert.ok(text(tree.toJSON()).includes("Draft changed"));
  await act(async () => button("Reload saved website").props.onClick());
  await act(async () => button("Club introduction").props.onClick());
  input = tree.root.findAllByType("input")[0];
  await act(async () =>
    input.props.onChange({ target: { value: "Keep editing" } }),
  );
  token = "refreshed";
  await act(async () => tree.update(React.createElement(Website)));
  assert.equal(
    tree.root.findAllByType("input")[0].props.value,
    "Keep editing",
    "Token refresh preserves edits",
  );
  await act(async () => button("Save draft").props.onClick());
  const oldPending = pending,
    oldSignal = network.at(-1).options.signal;
  assert.equal(
    network.at(-1).options.headers.Authorization,
    "Bearer refreshed",
  );
  clubId = "beta";
  workspace = "beta";
  identity = "two";
  current = {
    ...current,
    club_id: "beta",
    slug: "beta",
    draft: docs("Beta"),
    published: null,
  };
  await act(async () => tree.update(React.createElement(Website)));
  assert.equal(oldSignal.aborted, true);
  await act(async () =>
    oldPending(response({ ...current, draft: docs("Wrong club") })),
  );
  assert.equal(tree.root.findAllByType("input")[0].props.value, "Beta");
  workspace = "alpha";
  await act(async () =>
    tree.root
      .findAllByType("input")[0]
      .props.onChange({ target: { value: "Beta edit" } }),
  );
  const before = network.length;
  await act(async () => button("Save draft").props.onClick());
  assert.equal(
    network.length,
    before,
    "Other-tab workspace changes block writes",
  );
  await act(async () => tree.unmount());
  console.log("Club website editor: draft/save/publish, conflicts, duplicate saves, safe content and account isolation passed.");
})().catch((e) => { console.error(e); process.exitCode = 1; });
