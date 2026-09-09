const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(file, overrides = {}) {
  const output = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const mod = { exports: {} };
  new Function("require", "module", "exports", output)(name => overrides[name] || require(name), mod, mod.exports);
  return mod.exports;
}

const overrides = {
  "@/lib/useAdminSession": {
    useAdminSession: () => ({ accessToken: "test-token", session: { user: { id: "admin" } } }),
    adminSessionLabel: () => "Signed in"
  },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/components/ConfirmAction": { ConfirmAction: () => null },
  "@/components/interaction": { actionSuccess: value => value }
};
const Setup = load("app/admin/league-manager/settings/LeagueAwardsSetupPanel.tsx", overrides).default;
const Awards = load("app/admin/league-manager/awards/LeagueAwardsPanel.tsx", overrides).default;
const props = { apiBase: "https://api.example.invalid", clubId: "club", leagueName: "Fall Ladder League 2026", leagueStatus: "draft" };
let state = {
  league: { status: "draft", min_games: 32, awards_config: {} },
  award_catalog: [{ key: "most_wins", label: "Most Wins", recipient_type: "player", metric: "wins", minimum_metric: "games", default_enabled: false }],
  awards: [],
  awards_config_version: 0,
  writes_enabled: true,
  writes_unavailable_reason: null,
  service_role_ready: true,
  badge_definitions_ready: true,
  wizard: { status: "not_started", revision: 0 }
};
const writes = [];
global.fetch = async (url, request) => {
  assert.equal(request.headers.get("Authorization"), "Bearer test-token");
  if (request.method === "PUT") {
    assert.ok(url.endsWith("/Fall%20Ladder%20League%202026/awards/config"));
    const body = JSON.parse(request.body);
    writes.push(body);
    state = { ...state, league: { ...state.league, awards_config: body.awards_config }, awards_config_version: 1 };
  }
  const payload = url.endsWith("/leagues") ? { leagues: [{ league_name: props.leagueName }] } : structuredClone(state);
  return { ok: true, status: 200, json: async () => payload };
};
let tree;
async function mountSetup(leagueStatus = "draft") {
  await act(async () => { tree = create(React.createElement(Setup, { ...props, leagueStatus })); });
}
async function unmount() {
  await act(async () => tree.unmount());
}
function text() {
  return JSON.stringify(tree.toJSON());
}

(async () => {
  await mountSetup();
  assert.equal(tree.root.findByProps({ type: "checkbox" }).props.checked, false);
  await act(async () => {
    tree.root.findByProps({ type: "checkbox" }).props.onChange({ target: { checked: true } });
  });
  await act(async () => {
    tree.root.findByType("select").props.onChange({ target: { value: "3" } });
  });
  const save = tree.root.findAllByType("button").find(node => node.children.includes("Save award setup"));
  assert.equal(save.props.disabled, false);
  await act(async () => save.props.onClick());
  assert.equal(writes.length, 1);
  assert.equal(writes[0].expected_config_version, 0);
  assert.deepEqual(writes[0].awards_config.categories.most_wins, { enabled: true, depth: 3, minimum: 32 });
  await unmount();

  await mountSetup();
  assert.equal(tree.root.findByProps({ type: "checkbox" }).props.checked, true);
  assert.equal(tree.root.findByType("select").props.value, "3");
  assert.equal(tree.root.findByProps({ type: "number" }).props.value, "32");
  await unmount();

  state = { ...state, writes_enabled: false, writes_unavailable_reason: "Awards editing is disabled for this site. Saved awards remain available to review." };
  await mountSetup();
  assert.equal(tree.root.findAllByProps({ type: "checkbox" }).length, 0);
  assert.match(text(), /disabled for this site/);
  assert.match(text(), /Most Wins/);
  await unmount();

  await act(async () => { tree = create(React.createElement(Awards, { ...props, status: { enabled: true, awards_write_enabled: false } })); });
  await act(async () => tree.root.findAllByType("select")[0].props.onChange({ target: { value: props.leagueName } }));
  assert.match(text(), /disabled for this site/);
  assert.doesNotMatch(text(), /staging test/);
  await unmount();

  state = { ...state, writes_enabled: true, writes_unavailable_reason: null, service_role_ready: false };
  await mountSetup();
  assert.equal(tree.root.findAllByProps({ type: "checkbox" }).length, 0);
  assert.match(text(), /unavailable/);
  await unmount();

  state = { ...state, service_role_ready: true };
  await mountSetup("active");
  assert.equal(tree.root.findAllByProps({ type: "checkbox" }).length, 0);
  assert.match(text(), /locked after the league starts/);
  await unmount();

  state = { ...state, wizard: { status: "frozen" } };
  await mountSetup();
  assert.equal(tree.root.findAllByProps({ type: "checkbox" }).length, 0);
  assert.match(text(), /final award review has started/);
  assert.equal(writes.length, 1);
  await unmount();
  console.log("League awards setup save/reload, availability, and locked states passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
