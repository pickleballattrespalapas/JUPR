const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");
global.crypto ||= require("node:crypto").webcrypto;
function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename; compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] || original(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}
const ConfirmAction = () => null;
const Component = load("app/admin/tournaments/registrations/TournamentEmailDelivery.tsx", {
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction/types": load("components/interaction/types.ts"),
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts")
}).default;
const text = node => typeof node === "string" ? node : (node.children || []).map(text).join("");
const confirmText = "SEND TO SELECTED PARTICIPANTS";
const recipients = [{ index: 0, name: "Alex", email: "alex@example.com", status: "pending", detail: "" },
  { index: 1, name: "Beth", email: "beth@example.com", status: "pending", detail: "" }];
const preview = { send_available: true, preview_fingerprint: "f".repeat(64), selected_registration_ids: ["a", "b"],
  recipient_count: 2, recipients, preview: { subject: "Baja: Update" }, sender: { from_email: "organizer@example.com" }, delivery_mode: "live" };

async function scenario() {
  const requests = [];
  let stored, failAfterSend = false, deferred, deferSend = false;
  const requestJson = async (url, options = {}) => {
    requests.push({ url, method: options.method || "GET", body: options.body && JSON.parse(options.body) });
    if (options.method === "POST" && url.endsWith("broadcasts")) {
      const payload = JSON.parse(options.body);
      stored ||= { operation_key: payload.operation_key, subject: "Baja: Update", message: payload.message,
        include_registration_events: payload.include_registration_events,
        include_registration_edit_links: payload.include_registration_edit_links,
        recipients: structuredClone(recipients), pending_count: 2, recipient_count: 2, delivery_mode: "live", sender: preview.sender };
      return structuredClone(stored);
    }
    if (url.endsWith("/send")) {
      const index = Number(url.match(/recipients\/(\d+)/)[1]);
      stored.recipients[index].status = "sent"; stored.pending_count--;
      if (deferSend) { deferSend = false; await new Promise(resolve => { deferred = resolve; }); }
      if (failAfterSend) { failAfterSend = false; throw new Error("Connection lost after acceptance"); }
      return { index, status: "sent", detail: "Accepted" };
    }
    if (url.endsWith("broadcasts")) return { broadcasts: stored ? [stored] : [] };
    return structuredClone(stored);
  };
  let props = { clubId: "club", tournamentId: "baja", accessToken: "fixture", apiBase: "http://fixture.local", preview,
    previewScope: "scope1", subject: "Update", message: "Hi Alex,\nHello.", includeCancelled: false, includeRegistrationEvents: true, includeRegistrationEditLinks: true, busy: false, onBusy() {}, requestJson };
  let renderer;
  await act(async () => { renderer = create(React.createElement(Component, props)); });
  const action = prefix => renderer.root.findAllByType(ConfirmAction).find(row => row.props.triggerLabel.startsWith(prefix));
  const sendRequests = () => requests.filter(r => r.url.endsWith("/send"));
  assert.equal(sendRequests().length, 0, "Loading and reviewing must not send");
  assert.match(action("Send email").props.description, /private Edit Registration button/);
  const staleConfirm = action("Send email").props.onConfirm;
  props = { ...props, previewScope: "changed", preview: null };
  await act(async () => renderer.update(React.createElement(Component, props)));
  await assert.rejects(() => staleConfirm(confirmText), /changed/);
  assert.equal(requests.filter(r => r.method === "POST").length, 0, "A stale dialog cannot start a send");
  props = { ...props, previewScope: "scope2", preview };
  await act(async () => renderer.update(React.createElement(Component, props)));
  failAfterSend = true;
  let completion;
  await act(async () => { completion = await action("Send email").props.onConfirm(confirmText); });
  assert.equal(completion.status, "uncertain");
  assert.equal(requests.find(r => r.method === "POST" && r.url.endsWith("broadcasts")).body.include_registration_events, true);
  assert.equal(requests.find(r => r.method === "POST" && r.url.endsWith("broadcasts")).body.include_registration_edit_links, true);
  assert.equal(sendRequests().length, 1, "Stop on lost response, rather than send the rest blindly");
  await act(async () => { await completion.onRecover(); });
  assert.match(text(renderer.root), /1 sent · 1 not sent/);
  assert.equal(action("Send email").props.disabled, true, "The original send stays linked to its saved operation");
  await act(async () => { completion = await action("Continue sending").props.onConfirm(confirmText); });
  assert.equal(completion.status, "success");
  assert.equal(sendRequests().length, 2);
  assert.match(sendRequests()[1].url, /recipients\/1\/send$/, "Resume sends only the remaining person");
  assert.match(text(renderer.root), /2 sent/);
  assert.match(text(renderer.root), /includes that recipient’s registration events/);
  assert.match(text(renderer.root), /private Edit Registration button/);
  assert.equal(action("Continue sending"), undefined);
  await act(async () => renderer.unmount());

  // A reload finds saved history without reconstructing an outbound message.
  props = { ...props, preview: null };
  await act(async () => { renderer = create(React.createElement(Component, props)); });
  const historyButton = renderer.root.findAllByType("button").find(row => text(row).startsWith("Baja: Update"));
  await act(async () => historyButton.props.onClick());
  assert.match(text(renderer.root), /2 sent/);
  assert.match(text(renderer.root), /includes that recipient’s registration events/);
  assert.match(text(renderer.root), /private Edit Registration button/);
  assert.equal(sendRequests().length, 2);
  await act(async () => renderer.unmount());

  // Concurrent clicks are rejected; a session change stops remaining requests.
  stored = undefined; requests.length = 0; deferSend = true;
  props = { ...props, preview, previewScope: "scope3" };
  await act(async () => { renderer = create(React.createElement(Component, props)); });
  let running;
  await act(async () => { running = action("Send email").props.onConfirm(confirmText); });
  await assert.rejects(() => action("Send email").props.onConfirm(confirmText), /already being processed/);
  props = { ...props, accessToken: "new-session" };
  await act(async () => renderer.update(React.createElement(Component, props)));
  await act(async () => { deferred(); await running; });
  assert.equal(sendRequests().length, 1, "Session changes stop the batch");
  assert.doesNotMatch(text(renderer.root), /1 sent/);
  await act(async () => renderer.unmount());
  console.log("Tournament email delivery: reviewed sending, stale confirmation, response loss, resume, reload, double click and session change passed.");
}
scenario().catch(error => { console.error(error); process.exitCode = 1; });
