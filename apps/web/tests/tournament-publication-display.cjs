const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] || originalRequire(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}

const tournament = { id: "tournament-fixture", name: "Summer Classic", status: "ACTIVE" };
const draw = { id: "women-fixture", name: "Women's 3.5+", status: "DRAFT" };
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");
const status = {
  enabled: true, writes_enabled: true, official_publish_writes_enabled: true,
  official_publish_write_flag: { enabled: true }, service_role_ready: true,
  operation_store_ready: true, audit_store_ready: true
};

function snapshot({ games = 27, eligible = 30, published = 30, complete = true, ready = false, blocker = "already published", code = "DRAW_ALREADY_PUBLISHED" } = {}) {
  const readiness = { ready, complete, blockers: blocker ? [{ code, message: blocker }] : [] };
  const lifecycleDraw = {
    draw_id: draw.id, name: draw.name, counts: {
      games, finalized_games: games, open_games: 0,
      rating_publish_eligible_games: eligible, published_games: published
    },
    states: { live_operations: "complete", official_publish: complete ? "complete" : "blocked" },
    readiness: { official_publish: readiness }, podium: [], operations: []
  };
  return {
    tournament, draws: [draw], draw_id: draw.id, scope: "draw",
    games: [], teams: [], podium: [], operations: [],
    lifecycle: {
      tournament, draws: [lifecycleDraw], counts: lifecycleDraw.counts,
      domain_readiness: { official_publish: readiness }
    },
    readiness: { publish_official_matches: { ready, blockers: blocker ? [blocker] : [] } }
  };
}

(async () => {
  const Panel = load("app/admin/tournament-live/TournamentLivePanel.tsx", {
    "next/link": { __esModule: true, default: ({ children, ...props }) => React.createElement("a", props, children) },
    "next/navigation": { usePathname: () => "/admin/tournaments/ops/publish", useRouter: () => ({ replace() {} }) },
    "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel, disabled }) => React.createElement("button", { disabled }, triggerLabel) },
    "@/components/interaction": {},
    "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "fixture-token", loading: false }) },
    "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
    "@/lib/tournamentRouteContext": load("lib/tournamentRouteContext.ts"),
    "@/lib/tournamentDayWorkspaceState.mjs": await import("../lib/tournamentDayWorkspaceState.mjs"),
    "@/lib/tournamentDrawOperationalStatus.mjs": await import("../lib/tournamentDrawOperationalStatus.mjs"),
    "./TournamentLivePanel.module.css": { __esModule: true, default: new Proxy({}, { get: (_, key) => key }) }
  }).default;
  global.window = { localStorage: { getItem: () => null } };
  let renderer;
  async function render(payload, runtime = status) {
    if (renderer) act(() => renderer.unmount());
    global.fetch = async (_url, options) => {
      assert.ok(!options.method, "Publication display only performs reads");
      return { ok: true, json: async () => payload };
    };
    await act(async () => {
      renderer = create(React.createElement(Panel, {
        apiBase: "https://fixture.invalid", clubId: "fixture-club", status: runtime,
        initialTournamentId: tournament.id, initialDrawId: draw.id, view: "publish"
      }));
    });
    return content(renderer.root);
  }
  const publishButton = () => renderer.root.findAllByType("button").find(node => content(node) === "Publish official matches");

  const published = await render(snapshot());
  assert.match(published, /Published · 30 official matches/);
  assert.match(published, /27 of 27 matchups finalized.*30 of 30 played games published/);
  assert.match(published, /All 30 played games have verified official Match Log records/);
  assert.match(published, /best-of-three matchup is counted separately/);
  assert.doesNotMatch(published, /already published|Status unavailable/);
  assert.equal(publishButton(), undefined, "A published draw must not offer another publication");
  assert.equal(renderer.root.findAllByProps({ id: "winner-bonus" }).length, 0);
  const closeout = renderer.root.findAllByType("a").find(node => content(node) === "Review tournament closeout");
  assert.equal(new URL(closeout.props.href, "https://fixture.invalid").searchParams.get("tournament"), tournament.id);

  await render(snapshot({ complete: false, published: 27, blocker: "Publication requires reconciliation", code: "OFFICIAL_LINKS_INCOMPLETE" }));
  assert.match(content(renderer.root), /Publish recovery needed · 27 of 30 official/);
  assert.match(content(renderer.root), /Publication requires reconciliation/);
  assert.equal(renderer.root.findAllByProps({ "aria-label": "Publication complete" }).length, 0);
  assert.equal(publishButton().props.disabled, true);

  await render(snapshot({ complete: false, published: 0, ready: true, blocker: "" }));
  assert.equal(publishButton().props.disabled, false, "An eligible unpublished draw remains publishable");
  await render(snapshot({ complete: false, published: 0, ready: true, blocker: "" }), { ...status, official_publish_writes_enabled: false });
  assert.equal(publishButton().props.disabled, true, "Runtime write guards remain enforced");

  const unavailable = snapshot();
  delete unavailable.lifecycle.draws[0].readiness.official_publish.complete;
  await render(unavailable);
  assert.equal(renderer.root.findAllByProps({ "aria-label": "Publication complete" }).length, 0, "Completion requires explicit server evidence");
  assert.equal(publishButton().props.disabled, true);

  const noRatedGames = await render(snapshot({ games: 1, eligible: 0, published: 0 }));
  assert.match(noRatedGames, /No rated games to publish/);
  assert.equal(publishButton(), undefined);
  act(() => renderer.unmount());
  console.log("Tournament publication display: series counts, completion, recovery, and write guards passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
