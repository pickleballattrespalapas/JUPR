const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");
global.IS_REACT_ACT_ENVIRONMENT = true;

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
const interaction = load("components/interaction/types.ts");
const ConfirmAction = () => null;
const SearchablePlayerSelect = () => null;
const shared = {
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: SearchablePlayerSelect },
  "@/components/interaction": interaction,
};
const Review = load("app/admin/tournaments/ops/TournamentPlayerLinkReview.tsx", shared).default;
const Panel = load("app/admin/tournaments/ops/TournamentOpsPanel.tsx", {
  ...shared,
  "./TournamentPlayerLinkReview": { __esModule: true, default: Review },
  "./TournamentDrawPoolsPanel": { __esModule: true, default: () => null },
  "next/link": { __esModule: true, default: ({ children }) => children },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "fixture-token", loading: false }) },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/lib/tournamentRouteContext": { tournamentRouteHref: () => "/admin/tournaments" },
  "@/lib/tournamentRoundRobinIdempotency.mjs": { clearRoundRobinGenerationIdempotencyKey: () => {}, stableRoundRobinGenerationIdempotencyKey: () => "fixture" },
}).default;
const registration = { id: "reg-1", display_name: "Alex Example", updated_at: "2026-10-31T12:00:00Z", draw_ids: ["draw-1"] };
const players = [{ id: 1, name: "Alex Example", active: true }, { id: 2, name: "Blair Partner", active: true }];
const props = { apiBase: "http://fixture.local", clubId: "club", initialTournamentId: "tour-1", initialDrawId: "draw-1", workflow: "import",
  status: { enabled: true, mutation_runtime: { service_role_ready: true, surface_flags: { operations: { enabled: true } } }, operations_runtime: { operations_mutations_enabled: true } } };
function snapshot(linked = false) {
  return { ok: true, tournament: { id: "tour-1", name: "Fixture" }, summary: { draws: 1, teams: 0, games: 0, podium: 0 },
    draws: [{ id: "draw-1", name: "3.5", updated_at: "draw-version" }], teams: [], games: [], podium: [], players,
    state_fingerprint: linked ? "after-link" : "before-link", unlinked_registrations: linked ? [] : [registration] };
}
const reply = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
const button = (renderer, label) => renderer.root.findAllByType(ConfirmAction).find(row => row.props.triggerLabel === label);

async function run() {
  let renderer;
  let linked = false;
  let importStatus = 400;
  const writes = [];
  const storage = new Map();
  global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  global.fetch = async (url, options = {}) => {
    if (!options.method) return reply(snapshot(linked));
    const body = JSON.parse(options.body);
    writes.push({ url, body });
    if (url.endsWith("/link-player")) {
      linked = true;
      return reply({ ok: true });
    }
    return reply({ detail: "Resolve the duplicate registration before importing." }, importStatus);
  };
  await act(async () => { renderer = create(React.createElement(Panel, props)); });
  assert.equal(button(renderer, "Import confirmed registrations").props.disabled, true);
  assert.equal(button(renderer, "Link Alex Example").props.disabled, false);
  assert.ok(button(renderer, "Link Alex Example").props.description.includes("Alex Example (profile #1)"));
  let result;
  await act(async () => { result = await button(renderer, "Link Alex Example").props.onConfirm("LINK PLAYER"); });
  assert.equal(result.status, "success");
  assert.deepEqual(writes[0].body, { player_id: 1, expected_updated_at: registration.updated_at, confirmation_text: "LINK PLAYER" });
  assert.equal(button(renderer, "Link Alex Example"), undefined);
  assert.equal(button(renderer, "Import confirmed registrations").props.disabled, false);
  let failure;
  await act(async () => {
    try { await button(renderer, "Import confirmed registrations").props.onConfirm("IMPORT REGISTRATIONS"); }
    catch (error) { failure = interaction.normalizeInteractionActionError(error); }
  });
  assert.equal(failure.kind, "validation");
  assert.equal(failure.message, "Resolve the duplicate registration before importing.");
  assert.equal(writes[1].body.expected_state_fingerprint, "after-link", "Import must use the refreshed state");
  assert.equal(storage.size, 0, "Conclusive rejection must not strand recovery state");
  importStatus = 500;
  await act(async () => { result = await button(renderer, "Import confirmed registrations").props.onConfirm("IMPORT REGISTRATIONS"); });
  assert.equal(result.status, "uncertain");
  assert.equal(storage.size, 1, "Uncertain imports retain the exact request");
  assert.equal(button(renderer, "Import confirmed registrations").props.disabled, true);
  await act(async () => renderer.unmount());

  // Ambiguous names require an explicit choice; exact matches are suggestions,
  // never a write until the admin confirms.
  let linkCalls = 0;
  await act(async () => { renderer = create(React.createElement(Review, {
    registrations: [registration], players: [...players, { id: 3, name: "Alex Example", active: false }], disabled: false,
    onLink: async () => { linkCalls++; return interaction.actionSuccess("Linked", "Done"); },
  })); });
  assert.equal(linkCalls, 0);
  assert.equal(button(renderer, "Link Alex Example").props.disabled, true);
  await act(async () => renderer.root.findByType(SearchablePlayerSelect).props.onValueChange("3"));
  assert.equal(button(renderer, "Link Alex Example").props.disabled, false);
  await act(async () => { await button(renderer, "Link Alex Example").props.onConfirm("LINK PLAYER"); });
  assert.equal(linkCalls, 1);
  await act(async () => renderer.unmount());

  // A lost link response can be checked using the identical, idempotent link.
  storage.clear(); linked = false;
  let linkAttempts = 0;
  const linkBodies = [];
  global.fetch = async (_url, options = {}) => {
    if (!options.method) return reply(snapshot(linked));
    linkBodies.push(options.body); linkAttempts++;
    if (linkAttempts === 1) throw new TypeError("Network interrupted");
    linked = true; return reply({ ok: true, idempotent_replay: true });
  };
  await act(async () => { renderer = create(React.createElement(Panel, props)); });
  await act(async () => { result = await button(renderer, "Link Alex Example").props.onConfirm("LINK PLAYER"); });
  assert.equal(result.status, "uncertain");
  await act(async () => { result = await result.onRecover(); });
  assert.equal(result.status, "success");
  assert.equal(linkBodies[0], linkBodies[1]);
  assert.equal(button(renderer, "Import confirmed registrations").props.disabled, false);
  await act(async () => renderer.unmount());
  console.log("Tournament player linking and import error/recovery checks passed.");
}
run().catch(error => { console.error(error); process.exitCode = 1; });
