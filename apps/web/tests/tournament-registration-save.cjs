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
const interaction = load("components/interaction/types.ts");
const errors = load("lib/tournamentRegistrationActionError.ts", { "@/components/interaction/types": interaction });
const ConfirmAction = () => null;
const Panel = load("app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx", {
  "next/link": { __esModule: true, default: ({ children }) => children },
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": { ...interaction, FormDialog: () => null },
  "@/lib/tournamentRegistrationActionError": errors,
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "fixture-session", loading: false }) },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts"),
  "@/lib/tournamentRouteContext": { tournamentRouteHref: value => value },
  "@/lib/tournamentCommerceApi": { getAdminTournamentCommerceDetail: async () => ({ data: null }) }
}).default;
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");

async function main() {
  const originalFetch = global.fetch;
  const requests = [];
  let rejection = { status: 400, detail: "Another registration already uses that email." };
  const registration = { id: "reg-fixture", display_name: "Fixture Player", email: "fixture@example.com", registration_status: "pending", payment_status: "unpaid", updated_at: "2026-09-07T00:00:00Z" };
  global.fetch = async (_url, options = {}) => {
    if (options.method === "PATCH") {
      requests.push(JSON.parse(options.body));
      if (rejection) return new Response(JSON.stringify({ detail: rejection.detail }), { status: rejection.status });
      registration.registration_status = "confirmed";
      return new Response(JSON.stringify({ ok: true }));
    }
    return new Response(JSON.stringify({ registrations: [registration], selections: [], event_options: [], days: [] }));
  };
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(Panel, { apiBase: "http://fixture.local", clubId: "fixture", status: { enabled: true }, tournamentId: "t-fixture", tournamentName: "Fixture", registrationId: registration.id, drawId: "" })); });
    const field = name => renderer.root.findAllByType("label").find(node => content(node).startsWith(name));
    assert.equal(field("Registration status").findByType("select").props.value, "confirmed");
    await act(async () => field("Phone").findByType("input").props.onChange({ target: { value: "555-9999" } }));
    const save = () => renderer.root.findAllByType(ConfirmAction).find(node => node.props.confirmLabel === "Yes, save registration").props.onConfirm("SAVE REGISTRATION");
    let failure;
    await act(async () => { try { await save(); } catch (error) { failure = interaction.normalizeInteractionActionError(error); } });
    assert.equal(failure.kind, "validation");
    assert.equal(failure.message, rejection.detail, "The dialog retains the actionable server rejection");
    assert.equal(field("Phone").findByType("input").props.value, "555-9999", "Failed saves preserve the draft");
    assert.equal(requests[0].registration_status, "confirmed", "The visible dropdown and submitted status agree");
    assert.equal(requests[0].expected_updated_at, registration.updated_at);

    rejection = { status: 409, detail: "Registration changed after it was loaded. Refresh and try again." };
    await act(async () => { try { await save(); } catch (error) { failure = interaction.normalizeInteractionActionError(error); } });
    assert.equal(failure.kind, "conflict");
    assert.equal(failure.message, rejection.detail);
    const internal = errors.tournamentRegistrationActionError({ status: 500 }, { detail: "private database failure" });
    assert.doesNotMatch(internal.message, /private database/);
    assert.doesNotMatch(errors.tournamentRegistrationActionError({ status: 422 }, { detail: [{ input: "private input" }] }).message, /private input/);
    rejection = null;
    let completion;
    await act(async () => { completion = await save(); });
    assert.equal(completion.status, "success");
    console.log("Registration saves: legacy status, error feedback, draft retention, version checks and success passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.fetch = originalFetch;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
