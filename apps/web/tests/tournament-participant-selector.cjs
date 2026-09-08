const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
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

let token = "test-session";
const Panel = load("app/admin/tournaments/registrations/RegistrationManagementPanel.tsx", {
  "next/link": { __esModule: true, default: ({ children }) => children },
  "./TournamentEmailDelivery": { __esModule: true, default: () => null },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: token, session: {}, loading: false }), adminSessionLabel: () => "Test admin" },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/lib/tournamentRouteContext": { tournamentRouteHref: value => value }
}).default;

const registrations = [
  { id: "alex", display_name: "Alex", email: "alex@example.com", registration_status: "confirmed", payment_status: "paid" },
  { id: "beth", display_name: "Beth", email: "beth@example.com", registration_status: "waitlist", payment_status: "unpaid" },
  { id: "shared", display_name: "Sam", email: "ALEX@example.com", registration_status: "confirmed", payment_status: "paid" },
  { id: "partner", display_name: "Partner", email: "partner@example.com", registration_status: "confirmed", payment_status: "paid" },
  { id: "cancelled", display_name: "Cancelled", email: "cancelled@example.com", registration_status: "cancelled" },
  { id: "noemail", display_name: "No Email", email: null, registration_status: "confirmed" }
];
function text(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  return (node.children || []).map(text).join("");
}
const response = value => new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });

async function main() {
  const originalFetch = global.fetch;
  const requests = [];
  let deferNext = false;
  let resolvePreview;
  let legacyApi = false;
  let legacyEditApi = false;
  let renderer;
  global.fetch = async (url, options = {}) => {
    if (url.endsWith("broadcast-preview")) {
      const body = JSON.parse(options.body);
      requests.push(body);
      const emails = new Map();
      for (const row of registrations.filter(r => body.registration_ids.includes(r.id))) {
        if (row.email && (body.include_cancelled || row.registration_status !== "cancelled")) emails.set(row.email.toLowerCase(), row);
      }
      const payload = {
        selected_registration_ids: legacyApi ? undefined : body.registration_ids,
        include_registration_events: legacyApi ? undefined : body.include_registration_events,
        include_registration_edit_links: legacyEditApi ? undefined : body.include_registration_edit_links,
        recipient_count: emails.size,
        recipients: [...emails].map(([email, row]) => ({ name: row.display_name, email })),
        recipient_csv: [...emails.keys()].join("\n"),
        preview: { to_email: body.preview_recipient_email || [...emails.keys()][0], to_name: "Selected participant", text: `Preview: ${body.message}`, html: `<html><body><h1>Email</h1><p>Presented by Homes and Land</p><p>${body.message}</p>${body.include_registration_events ? `<h2>Your registration events</h2><p>Events for ${body.preview_recipient_email || [...emails.keys()][0]}</p>` : ""}${body.include_registration_edit_links ? `<span aria-disabled="true">Edit Registration for ${body.preview_recipient_email || [...emails.keys()][0]}</span>` : ""}<h2>Supporting sponsors</h2></body></html>` }
      };
      if (deferNext) { deferNext = false; return new Promise(resolve => { resolvePreview = () => resolve(response(payload)); }); }
      return response(payload);
    }
    if (url.endsWith("import-handoff")) return response({ write_count: 0 });
    if (url.endsWith("/admin/tournaments")) return response({ tournaments: [{ id: "baja", name: "Baja Classic" }, { id: "other", name: "Other Tournament" }] });
    return response({ tournament: { id: url.endsWith("/other") ? "other" : "baja" }, registrations, selections: [{ registration_id: "partner", partner_email: "alex@example.com" }], days: [], event_options: [] });
  };
  const props = { apiBase: "http://fixture.local", clubId: "club", status: { enabled: true }, initialTournamentId: "baja", initialTournamentName: "Baja Classic", initialDrawId: "" };
  try {
    await act(async () => { renderer = create(React.createElement(Panel, props)); });
    const root = renderer.root;
    const button = label => root.findAllByType("button").find(b => text(b).startsWith(label));
    const checkbox = name => root.findAllByType("input").find(i => i.props["aria-label"]?.startsWith(`Select ${name} (`));
    const changeSearch = async value => act(async () => root.findAllByType("input").find(i => i.props.type === "search").props.onChange({ target: { value } }));
    const select = async (name, checked = true) => act(async () => checkbox(name).props.onChange({ target: { checked } }));
    const setContent = async (label, value) => act(async () => root.findAllByType("label").find(l => text(l).startsWith(label)).findByType(label === "Message" ? "textarea" : "input").props.onChange({ target: { value } }));
    const toggleEvents = async checked => act(async () => root.findAllByType("label").find(l => text(l) === " Include registration events").findByType("input").props.onChange({ target: { checked } }));
    const toggleEditLinks = async checked => act(async () => root.findAllByType("label").find(l => text(l) === " Include Edit Registration button").findByType("input").props.onChange({ target: { checked } }));
    const toggleCancelled = async checked => act(async () => root.findAllByType("label").find(l => text(l) === " Include cancelled registrations").findByType("input").props.onChange({ target: { checked } }));

    assert.equal(button("Preview recipients").props.disabled, true);
    assert.match(text(root), /0 participants selected · 0 email recipients/);
    assert.equal(checkbox("Cancelled").props.disabled, true);
    assert.equal(checkbox("No Email").props.disabled, true);
    await setContent("Subject", "Court update");
    await setContent("Message", "See you at court 1.");
    await changeSearch("alex@example.com");
    assert.ok(checkbox("Partner"), "Search may match partner contacts, but selection must stay exact");
    await select("Alex");
    await act(async () => button("Preview recipients").props.onClick());
    assert.deepEqual(requests.at(-1).registration_ids, ["alex"]);
    assert.equal(requests.at(-1).search, undefined);
    assert.ok(button("Download recipient CSV"));
    const emailFrame = root.findByProps({ title: "Tournament email preview" });
    assert.equal(emailFrame.props.sandbox, "");
    assert.equal(emailFrame.props.referrerPolicy, "no-referrer");
    assert.match(emailFrame.props.srcDoc, /Presented by Homes and Land/);
    assert.match(emailFrame.props.srcDoc, /default-src 'none'/);
    assert.match(emailFrame.props.srcDoc, /img-src data:/);
    assert.equal(requests.at(-1).include_registration_events, false);
    assert.equal(requests.at(-1).include_registration_edit_links, false);
    await toggleEditLinks(true);
    assert.equal(button("Download recipient CSV"), undefined, "Edit buttons invalidate the previous review");
    await act(async () => button("Preview recipients").props.onClick());
    assert.equal(requests.at(-1).include_registration_edit_links, true);
    assert.equal(requests.at(-1).include_registration_events, false, "Edit buttons work without event details");
    assert.match(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Edit Registration for alex@example.com/);
    await toggleEvents(true);
    assert.equal(button("Download recipient CSV"), undefined, "Event option invalidates the previous review");
    await act(async () => button("Preview recipients").props.onClick());
    assert.equal(requests.at(-1).include_registration_events, true);
    assert.match(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Events for alex@example.com/);

    await changeSearch("beth");
    assert.match(text(root), /1 selected participant is outside the current filters/);
    await act(async () => button("Select all filtered").props.onClick());
    assert.equal(button("Download recipient CSV"), undefined, "Changing selection invalidates the prior preview");
    await act(async () => button("Preview recipients").props.onClick());
    assert.deepEqual(requests.at(-1).registration_ids, ["alex", "beth"], "Bulk select adds visible participants and preserves earlier selections");
    await act(async () => root.findAllByType("label").find(l => text(l).startsWith("Preview for")).findByType("select").props.onChange({ target: { value: "beth@example.com" } }));
    assert.equal(requests.at(-1).preview_recipient_email, "beth@example.com");
    assert.deepEqual(requests.at(-1).registration_ids, ["alex", "beth"], "Preview selection does not change the audience");
    assert.match(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Events for beth@example.com/);
    assert.doesNotMatch(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Events for alex@example.com/);
    assert.match(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Edit Registration for beth@example.com/);
    assert.doesNotMatch(root.findByProps({ title: "Tournament email preview" }).props.srcDoc, /Edit Registration for alex@example.com/);
    await act(async () => root.findByProps({ "aria-label": "Remove Alex from selection" }).props.onClick());
    assert.match(text(root), /1 participant selected · 1 email recipient/);
    await changeSearch("");
    await act(async () => button("Select all filtered").props.onClick());
    assert.match(text(root), /4 participants selected · 3 email recipients/, "Shared addresses are counted once");
    await toggleCancelled(true);
    await select("Cancelled");
    assert.match(text(root), /5 participants selected · 4 email recipients/);
    await toggleCancelled(false);
    assert.equal(checkbox("Cancelled").props.checked, false);
    await act(async () => button("Clear selection").props.onClick());
    assert.match(text(root), /0 participants selected · 0 email recipients/);
    assert.equal(button("Preview recipients").props.disabled, true);

    await select("Alex");
    deferNext = true;
    await act(async () => { void button("Preview recipients").props.onClick(); });
    await toggleEvents(false);
    await act(async () => resolvePreview());
    assert.equal(button("Download recipient CSV"), undefined, "A late response cannot restore events after the option changes");
    deferNext = true;
    await act(async () => { void button("Preview recipients").props.onClick(); });
    await toggleEditLinks(false);
    await act(async () => resolvePreview());
    assert.equal(button("Download recipient CSV"), undefined, "A late response cannot restore edit buttons after the option changes");
    await toggleEditLinks(true);
    legacyEditApi = true;
    await act(async () => button("Preview recipients").props.onClick());
    assert.match(text(root), /Edit Registration buttons could not be included/);
    assert.equal(button("Download recipient CSV"), undefined, "An older API cannot silently omit requested edit buttons");
    legacyEditApi = false;
    await toggleEditLinks(false);
    deferNext = true;
    await act(async () => { void button("Preview recipients").props.onClick(); });
    await select("Beth");
    await act(async () => resolvePreview());
    assert.equal(button("Download recipient CSV"), undefined, "A late preview for the previous selection stays hidden");
    await act(async () => button("Preview recipients").props.onClick());
    await setContent("Message", "Changed message");
    assert.equal(button("Download recipient CSV"), undefined, "Editing content invalidates the preview too");
    legacyApi = true;
    await act(async () => button("Preview recipients").props.onClick());
    assert.match(text(root), /Participant selection could not be verified/);
    assert.equal(button("Download recipient CSV"), undefined, "An older API cannot silently broaden the selection");
    legacyApi = false;
    await act(async () => button("Refresh tournaments").props.onClick());
    assert.match(text(root), /0 participants selected/);
    await select("Beth");
    await act(async () => renderer.update(React.createElement(Panel, { ...props, initialTournamentId: "other" })));
    assert.match(text(root), /0 participants selected/, "Changing tournaments clears selected recipients");
    await select("Alex");
    token = "";
    await act(async () => renderer.update(React.createElement(Panel, props)));
    assert.equal(checkbox("Alex"), undefined, "Signing out removes protected participant details");
  } finally {
    if (renderer) act(() => renderer.unmount());
    global.fetch = originalFetch;
  }
  console.log("Participant selector checks passed: exact and bulk selection, exclusions, shared emails, stale previews, refresh, tournament change, and sign-out.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
