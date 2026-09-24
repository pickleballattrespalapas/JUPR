const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(file, overrides = {}) {
  const source = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", source)(name => overrides[name] || require(name), module, module.exports);
  return module.exports;
}

const savedGlobals = { fetch, setTimeout, clearTimeout, warn: console.warn };
const savedBase = process.env.JUPR_API_BASE_URL;
const warnings = [];
let calls = [], timers = new Set(), expire = false;
global.setTimeout = (callback, delay) => {
  assert.equal(delay, 8000);
  const timer = { callback };
  timers.add(timer);
  if (expire) queueMicrotask(callback);
  return timer;
};
global.clearTimeout = timer => timers.delete(timer);
console.warn = (...args) => warnings.push(args);
process.env.JUPR_API_BASE_URL = "https://api.example.invalid/";

function responses(...sequence) {
  calls = [];
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.equal(options.method, "GET");
    assert.equal(options.cache, "no-store");
    assert.equal(options.headers, undefined);
    const next = sequence.shift();
    assert.notEqual(next, undefined, "Read retries must be bounded");
    if (next instanceof Error) throw next;
    return next;
  };
}

const api = load("lib/adminLeagueManagerApi.ts");
const ready = { enabled: true, league_count: 9, warnings: [] };
const ok = () => Response.json(ready);
const reset = () => new TypeError("sensitive transport context must not be logged", { cause: { code: "ECONNRESET" } });

(async () => {
  for (const failure of [reset(), new DOMException("Timeout", "TimeoutError"), new DOMException("Aborted", "AbortError"), ...[502, 503, 504].map(status => Response.json({}, { status }))]) {
    responses(failure, ok());
    assert.deepEqual(await api.getAdminLeagueManagerStatus("club a"), { data: ready, error: null });
    assert.equal(calls.length, 2);
    assert.equal(calls[1].url, "https://api.example.invalid/admin/clubs/club%20a/league-manager/status");
    assert.equal(timers.size, 0);
    assert.notEqual(calls[0].options.signal, calls[1].options.signal);
  }

  // A reset while consuming the body also recovers.
  responses({ ok: true, status: 200, json: async () => { throw reset(); } }, ok());
  assert.equal((await api.getAdminLeagueLiveStatus("club-b")).data.enabled, true);
  assert.match(calls[1].url, /\/club-b\/league-manager\/live\/status$/);

  responses(reset(), reset());
  assert.match((await api.getAdminLeagueManagerStatus("club-a")).error, /connection.*interrupted/);
  assert.equal(calls.length, 2);

  responses(Response.json({}, { status: 503 }), Response.json({}, { status: 503 }));
  assert.equal((await api.getAdminLeagueManagerStatus("club-a")).data, null);
  assert.equal(calls.length, 2);

  // Authorization/validation failures are terminal; never treat them as an outage.
  for (const status of [400, 401, 403, 404, 409, 422, 429]) {
    responses(Response.json({ detail: "Access denied" }, { status }));
    assert.equal((await api.getAdminLeagueManagerStatus("club-a")).error, `API error (${status}). Access denied`);
    assert.equal(calls.length, 1);
  }

  responses(Response.json({ enabled: false, warnings: ["Disabled"] }));
  assert.equal((await api.getAdminLeagueManagerStatus("club-a")).data.enabled, false);
  assert.equal(calls.length, 1);
  responses(new Response("invalid JSON"));
  assert.match((await api.getAdminLeagueManagerStatus("club-a")).error, /unexpected response/);
  assert.equal(calls.length, 1);

  // Both a hanging connection and a hanging response body are aborted.
  for (const hangingBody of [false, true]) {
    expire = true;
    calls = [];
    global.fetch = async (_url, options) => {
      calls.push(options);
      const aborted = () => new Promise((_resolve, reject) => {
        if (options.signal.aborted) reject(new DOMException("Aborted", "AbortError"));
        else options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
      });
      return hangingBody ? { ok: true, status: 200, json: aborted } : aborted();
    };
    assert.match((await api.getAdminLeagueManagerStatus("club-a")).error, /connection.*interrupted/);
    assert.equal(calls.length, 2);
    assert.equal(timers.size, 0);
    expire = false;
  }
  assert.doesNotMatch(JSON.stringify(warnings), /sensitive transport context/);
  assert.match(JSON.stringify(warnings), /ECONNRESET/);

  // Restore real timers for React; the recovery button only refreshes the route.
  global.setTimeout = savedGlobals.setTimeout;
  global.clearTimeout = savedGlobals.clearTimeout;
  let refreshes = 0;
  global.fetch = async () => { throw new Error("Retry UI must not issue a mutation"); };
  const navigation = { useRouter: () => ({ refresh: () => refreshes++ }), redirect: () => { throw new Error("Unexpected redirect"); } };
  const errorModule = load("app/admin/league-manager/LeagueManagerLoadError.tsx", { "next/navigation": navigation });
  const ErrorPanel = errorModule.default;
  let tree;
  await act(async () => { tree = create(React.createElement(ErrorPanel, { error: "Connection interrupted" })); });
  assert.equal(tree.root.findAllByProps({ role: "alert" }).length, 1);
  assert.equal(tree.root.findByType("button").props.type, "button");
  await act(async () => tree.root.findByType("button").props.onClick());
  assert.equal(refreshes, 1);
  await act(async () => tree.unmount());

  const PendingError = load("app/admin/league-manager/LeagueManagerLoadError.tsx", {
    "next/navigation": navigation, react: { ...React, useTransition: () => [true, () => {}] }
  }).default;
  await act(async () => { tree = create(React.createElement(PendingError, { error: "Connection interrupted" })); });
  assert.equal(tree.root.findByType("button").props.disabled, true);
  assert.match(JSON.stringify(tree.toJSON()), /Trying again/);
  await act(async () => tree.unmount());

  // Exercise the actual server pages: failed status reads must expose recovery,
  // while never rendering a settings/roster editor with assumed capabilities.
  for (const section of ["", "create", "league", "settings", "teams", "roster", "results", "awards", "print", "live"]) {
    const relative = `app/admin/league-manager/${section ? section + "/" : ""}page.tsx`;
    let editors = 0;
    const source = fs.readFileSync(path.join(__dirname, "..", relative), "utf8");
    const overrides = {
      "next/navigation": navigation,
      "next/link": { __esModule: true, default: props => React.createElement("a", props) },
      "@/lib/adminWorkspaceServer": { requireAdminWorkspace: () => ({ clubId: "club-a", clubSlug: "club-a" }) },
      "@/lib/adminLeagueManagerApi": {
        getAdminLeagueManagerApiBaseUrl: () => "https://api.example.invalid",
        getAdminLeagueManagerStatus: async () => ({ data: null, error: "Connection interrupted" }),
        getAdminLeagueLiveStatus: async () => ({ data: ready, error: null })
      },
      "@/lib/adminMatchUploaderApi": { getAdminMatchUploaderStatus: async () => ({ data: ready, error: null }) },
      "@/lib/api": { getClubPlayerOptions: async () => ({ data: { players: [] }, error: null }) },
      "@/lib/leagueRouteContext": {
        readLeagueRouteContext: () => ({ leagueId: "id-a", leagueName: "Season teams", leagueType: section === "teams" ? "Team" : "Individual" }),
        isTeamLeagueType: type => type === "Team"
      }
    };
    for (const [, name] of source.matchAll(/from "(\.[^"]+)"/g)) {
      overrides[name] = name.endsWith("LeagueManagerLoadError") ? errorModule : {
        __esModule: true, default: () => { if (!name.endsWith("LeagueManagerNav")) editors++; return null; }
      };
    }
    const Page = load(relative, overrides).default;
    await act(async () => { tree = create(await Page({ searchParams: {} })); });
    assert.equal(tree.root.findAllByType(ErrorPanel).length, 1, section);
    assert.equal(editors, 0, `${section} must not enable editors during an outage`);
    assert.equal(tree.root.findByType("button").children.join(""), "Retry loading");
    await act(async () => tree.unmount());
  }
  console.log("League Manager recovery: transient retries, bounded connection/body timeouts, authorization, diagnostic privacy, and all 10 route recovery buttons passed.");
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => {
  Object.assign(global, { fetch: savedGlobals.fetch, setTimeout: savedGlobals.setTimeout, clearTimeout: savedGlobals.clearTimeout });
  console.warn = savedGlobals.warn;
  if (savedBase === undefined) delete process.env.JUPR_API_BASE_URL;
  else process.env.JUPR_API_BASE_URL = savedBase;
});
