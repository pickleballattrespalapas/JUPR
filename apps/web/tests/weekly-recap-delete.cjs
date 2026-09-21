const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(relative, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", relative), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}

const ConfirmAction = props => React.createElement("confirm-action", props);
let canDelete = true;
let requests = [];
let recaps;
let deleteError = null;
let accessToken = "local-token";
const draft = { id: "recap-1", club_id: "alpha", week_start: "2026-07-06", week_end: "2026-07-12", status: "draft", row_version: 7, generated_json: {}, final_json: {}, edits_json: {} };
const other = { ...draft, id: "recap-2", week_start: "2026-06-29", week_end: "2026-07-05" };
const Panel = load("app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx", {
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": { actionSuccess: (title, message) => ({ title, message }) },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken, loading: false, session: { user: { email: "admin@example.com" } } }), adminSessionLabel: () => "Admin" },
  "@/lib/clubDate": { clubWeekStartIso: () => draft.week_start, clubTodayIso: () => draft.week_end },
  "./AdminWeeklyRecapPreview": () => React.createElement("div", null, "Recap preview")
}).default;
global.fetch = async (url, options = {}) => {
  requests.push({ url, ...options });
  if (options.method === "DELETE") {
    if (deleteError) return { ok: false, status: 409, json: async () => ({ detail: deleteError }) };
    const body = JSON.parse(options.body);
    recaps = recaps.filter(row => row.id !== body.expected_recap_id);
    return { ok: true, json: async () => ({ ok: true, deleted_recap_id: body.expected_recap_id, week_start: draft.week_start }) };
  }
  if (options.method === "PATCH" || options.method === "POST") {
    const body = JSON.parse(options.body);
    const current = recaps.find(row => row.id === body.expected_recap_id);
    assert.ok(current, "Mutation identifies the exact loaded recap");
    return { ok: true, json: async () => ({ ok: true, recap: current, candidates: {} }) };
  }
  if (url.includes("?limit=")) return { ok: true, json: async () => ({ ok: true, recaps, count: recaps.length, can_delete_drafts: canDelete }) };
  const week = new URL(url).pathname.split("/").at(-1);
  return { ok: true, json: async () => ({ ok: true, recap: recaps.find(row => row.week_start === week), candidates: {} }) };
};
const props = { apiBase: "https://example.local", clubId: "alpha", status: { enabled: true, mutations_enabled: true, warnings: [] }, initialWeekStart: draft.week_start };
const text = node => typeof node === "string" || typeof node === "number" ? String(node) : node?.children?.map(text).join(" ") || "";
const deletion = tree => tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Delete draft");
async function mount(overrides = {}) {
  let tree;
  await act(async () => { tree = create(React.createElement(Panel, { ...props, ...overrides })); });
  return tree;
}
function reset() { canDelete = true; accessToken = "local-token"; requests = []; recaps = [{ ...draft }, { ...other }]; deleteError = null; }

(async () => {
  reset();
  let tree = await mount();
  let action = deletion(tree);
  assert.ok(action, "Selected draft exposes Delete draft to an administrator");
  assert.match(action.props.description, /2026-07-06 through 2026-07-12/, "Confirmation identifies the selected date range");
  assert.equal(action.props.tone, "danger");
  assert.equal(action.props.confirmationText, "DELETE RECAP");
  assert.equal(requests.filter(request => request.method === "DELETE").length, 0, "Loading a draft does not delete it");
  await act(async () => action.props.onConfirm(action.props.confirmationText));
  const request = requests.find(request => request.method === "DELETE");
  assert.match(request.url, /\/admin\/clubs\/alpha\/weekly-recap\/recaps\/2026-07-06$/);
  assert.deepEqual(JSON.parse(request.body), { expected_recap_id: "recap-1", expected_row_version: 7, confirmation_text: "DELETE RECAP" });
  assert.equal(request.headers.get("Authorization"), "Bearer local-token");
  assert.equal(deletion(tree), undefined, "Deleted draft clears the active editor");
  assert.equal(tree.root.findAllByType("option").filter(node => node.props.value === draft.week_start).length, 0);
  assert.equal(tree.root.findAllByType("option").filter(node => node.props.value === other.week_start).length, 1, "Other recap remains selectable");
  assert.equal(requests.filter(request => request.url.includes("?limit=")).length, 2, "Deletion refreshes saved recap list");
  await act(async () => tree.unmount());

  for (const label of ["Regenerate draft", "Save draft edits", "Publish recap", "Unpublish recap"]) {
    reset();
    if (label === "Unpublish recap") recaps[0].status = "published";
    tree = await mount();
    const mutation = tree.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === label);
    assert.ok(mutation);
    await act(async () => mutation.props.onConfirm(mutation.props.confirmationText));
    const mutationRequest = requests.find(request => request.method === "PATCH" || request.method === "POST");
    assert.equal(JSON.parse(mutationRequest.body).expected_recap_id, draft.id, `${label} sends the loaded recap identity`);
    assert.equal(JSON.parse(mutationRequest.body).expected_row_version, draft.row_version, `${label} sends the loaded version`);
    await act(async () => tree.unmount());
  }

  reset(); deleteError = "Weekly recap changed. Reload before deleting.";
  tree = await mount(); action = deletion(tree);
  await act(async () => { await assert.rejects(action.props.onConfirm("DELETE RECAP"), /Reload/); });
  assert.match(text(tree.toJSON()), /Weekly recap changed/);
  assert.ok(deletion(tree), "Stale conflict preserves editor for reload rather than claiming deletion");
  assert.equal(recaps.length, 2);
  await act(async () => tree.unmount());

  reset(); canDelete = false;
  tree = await mount();
  assert.equal(deletion(tree), undefined, "Operators have no delete control");
  await act(async () => tree.unmount());

  reset(); recaps[0].status = "published";
  tree = await mount();
  assert.equal(deletion(tree), undefined, "Published recaps have no delete control");
  await act(async () => tree.unmount());

  reset();
  tree = await mount({ status: { ...props.status, mutations_enabled: false } });
  assert.equal(deletion(tree).props.disabled, true, "Read-only mode disables deletion");
  await act(async () => tree.unmount());
  console.log("weekly recap deletion component behavior: passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
