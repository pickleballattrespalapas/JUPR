const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(file, overrides = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const mod = { exports: {} };
  new Function("require", "module", "exports", code)(name => overrides[name] || require(name), mod, mod.exports);
  return mod.exports;
}

const interactions = load("components/interaction/types.ts");
const ConfirmAction = () => null;
const Setup = load("app/admin/league-manager/settings/TeamLeagueSetupPanel.tsx", {
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-token" }) },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": interactions
}).default;
const props = { apiBase: "https://api.example.invalid", clubId: "club-a", leagueName: "Club Team League", leagueStatus: "draft", status: { league_manager_writes_enabled: true } };
const response = (payload, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(payload) });
const settings = { league_name: props.leagueName, team_size: 2, registration_open: false, settings_version: 7 };
const pending = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
let tree;
const text = () => JSON.stringify(tree.toJSON());
const assertNoEditor = () => {
  assert.equal(tree.root.findAllByType("input").length, 0);
  assert.equal(tree.root.findAllByType("select").length, 0);
  assert.equal(tree.root.findAllByType(ConfirmAction).length, 0);
};

(async () => {
  const initial = pending();
  let mode = "pending";
  let stored = { ...settings };
  const writes = [];
  global.fetch = async (url, request) => {
    assert.equal(request.headers.get("Authorization"), "Bearer test-token");
    if (request.method === "PUT") {
      const body = JSON.parse(request.body);
      writes.push(body);
      assert.equal(body.expected_settings_version, 7);
      stored = { ...stored, ...body.settings, settings_version: 8 };
      return response({ saved: true });
    }
    if (mode === "pending") return initial.promise;
    return response({ leagues: [stored] });
  };
  await act(async () => { tree = create(React.createElement(Setup, props)); });
  assert.match(text(), /Loading team league setup/);
  assertNoEditor();
  await act(async () => { initial.resolve(response({ detail: "Team leagues are temporarily unavailable." }, 403)); });
  assert.match(text(), /Team league setup is not enabled on this site yet/);
  assert.equal(tree.root.findAllByProps({ role: "alert" }).length, 1);
  assertNoEditor();
  assert.equal(writes.length, 0);

  mode = "ready";
  await act(async () => { await tree.root.findByType("button").props.onClick(); });
  await act(async () => { tree.root.findAllByProps({ type: "checkbox" })[0].props.onChange({ target: { checked: true } }); });
  const save = tree.root.findByType(ConfirmAction);
  assert.equal(save.props.disabled, false);
  let completion;
  await act(async () => { completion = await save.props.onConfirm("SAVE TEAM LEAGUE"); });
  assert.equal(completion.status, "success");
  assert.equal(writes.length, 1);
  assert.equal(writes[0].settings.registration_open, true);
  assert.equal(tree.root.findAllByProps({ type: "checkbox" })[0].props.checked, true);
  assert.equal(tree.root.findByType(ConfirmAction).props.disabled, true);
  await act(async () => tree.unmount());

  global.fetch = async () => response({ detail: "Temporary load failure." }, 503);
  await act(async () => { tree = create(React.createElement(Setup, props)); });
  assert.match(text(), /Temporary load failure/);
  assertNoEditor();
  await act(async () => tree.unmount());

  const clubA = pending(), clubB = pending();
  const reads = [];
  global.fetch = async url => { reads.push(url); return url.includes("club-b") ? clubB.promise : clubA.promise; };
  await act(async () => { tree = create(React.createElement(Setup, props)); });
  await act(async () => { tree.update(React.createElement(Setup, { ...props, clubId: "club-b" })); });
  assert.equal(reads.length, 2);
  await act(async () => { clubA.resolve(response({ leagues: [settings] })); });
  assertNoEditor();
  await act(async () => { clubB.resolve(response({ leagues: [{ ...settings, team_size: 4 }] })); });
  assert.equal(tree.root.findAllByType("select")[0].props.value, "4");
  await act(async () => tree.unmount());

  global.fetch = async (_url, request) => request.method === "PUT"
    ? response({ detail: "Admin team-league writes are staging-only. Open only the approved league-manager wave." }, 403)
    : response({ leagues: [settings] });
  await act(async () => { tree = create(React.createElement(Setup, props)); });
  await act(async () => { tree.root.findAllByProps({ type: "checkbox" })[0].props.onChange({ target: { checked: true } }); });
  await act(async () => {
    await assert.rejects(tree.root.findByType(ConfirmAction).props.onConfirm("SAVE TEAM LEAGUE"), error => {
      assert.ok(error instanceof interactions.InteractionActionError);
      assert.equal(error.kind, "forbidden");
      assert.match(error.message, /Saving team league settings is not enabled/);
      return true;
    });
  });
  assert.equal(tree.root.findAllByProps({ type: "checkbox" })[0].props.checked, true);
  await act(async () => tree.unmount());
  console.log("Team league setup: loading, disabled/error states, safe read retry, save/reload, club isolation and clear save errors passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
