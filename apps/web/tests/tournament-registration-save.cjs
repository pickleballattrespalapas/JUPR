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
const eligibility = load("lib/tournamentRegistrationEligibility.ts", {
  "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts")
});
const ConfirmAction = () => null;
const Panel = load("app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx", {
  "next/link": { __esModule: true, default: ({ children }) => children },
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": { ...interaction, FormDialog: ({ open, children }) => open ? React.createElement("div", { role: "dialog" }, children) : null },
  "@/lib/tournamentRegistrationActionError": errors,
  "@/lib/tournamentRegistrationEligibility": eligibility,
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
  const registration = { id: "reg-fixture", display_name: "Fixture Player", email: "fixture@example.com", gender: "Men", age: 38, doubles_skill: 4.2, singles_skill: 4.2, registration_status: "pending", payment_status: "unpaid", updated_at: "2026-09-07T00:00:00Z" };
  const event = (id, patch = {}) => ({ id, label: id, event_family_label: id, registration_day_id: "day-1", event_type: "SINGLES", gender_restriction: "MEN", skill_mode: "OPEN", status: "open", enabled: true, ...patch });
  const eventOptions = [
    event("singles", { event_family_label: "Men's Singles" }),
    event("singles-alternative", { event_family_label: "Men's Singles" }),
    event("women", { gender_restriction: "WOMEN" }),
    event("doubles", { event_type: "GENDER_DOUBLES", partner_required: true }),
    event("mixed", { event_type: "MIXED_DOUBLES", partner_required: true, gender_restriction: "MIXED" }),
    event("closed", { status: "closed" }),
    event("disabled", { enabled: false }),
    event("disabled-day", { registration_day_id: "day-off" }),
    event("senior", { age_mode: "MIN_AGE", age_rules: { min_age: 50 } }),
    event("other-singles", { registration_day_id: "day-2" })
  ];
  const selections = [
    { id: "sel-single", registration_id: registration.id, event_option_id: "singles", event_label: "Singles entry", partner_mode: "NONE", updated_at: registration.updated_at },
    { id: "sel-double", registration_id: registration.id, event_option_id: "doubles", event_label: "Doubles entry", partner_mode: "NEEDS_PARTNER", show_on_partner_board: true, updated_at: registration.updated_at }
  ];
  global.fetch = async (_url, options = {}) => {
    if (options.method === "PATCH") {
      requests.push(JSON.parse(options.body));
      if (rejection) return new Response(JSON.stringify({ detail: rejection.detail }), { status: rejection.status });
      registration.registration_status = "confirmed";
      return new Response(JSON.stringify({ ok: true }));
    }
    return new Response(JSON.stringify({ registrations: [registration], selections, event_options: eventOptions, days: [{ id: "day-1" }, { id: "day-2" }, { id: "day-off", enabled: false }] }));
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

    const choices = () => field("Add another event entry").findByType("select");
    assert.equal(choices().props.value, "", "Loading a registration must not preselect an arbitrary event");
    assert.deepEqual(choices().findAllByType("option").map(node => node.props.value), ["", "mixed", "other-singles"], "Gender, age, closed days and duplicate event families are excluded");
    const entry = title => renderer.root.findAllByType("section").find(node => node.findAllByType("h3").some(heading => content(heading) === title));
    assert.equal(entry("Singles entry").findAllByType("button").some(node => content(node) === "Change partner"), false);
    assert.equal(entry("Doubles entry").findAllByType("button").some(node => content(node) === "Change partner"), true);
    await act(async () => entry("Singles entry").findAllByType("button").find(node => content(node) === "Edit entry").props.onClick());
    assert.equal(renderer.root.findAllByType("label").some(node => content(node).startsWith("Partner state")), false);
    assert.equal(field("Division").findByType("select").props.value, "singles");
    await act(async () => entry("Doubles entry").findAllByType("button").find(node => content(node) === "Edit entry").props.onClick());
    assert.equal(field("Partner state").findByType("select").props.value, "NEEDS_PARTNER");
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "other-singles" } }));
    assert.equal(renderer.root.findAllByType("label").some(node => content(node).startsWith("Partner state")), false, "Moving an entry to singles removes partner controls");
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "doubles" } }));
    assert.equal(field("Partner state").findByType("select").props.value, "NEEDS_PARTNER");
    assert.equal(renderer.root.findAllByType("input").find(node => node.props.type === "checkbox" && node.parent.type === "label" && content(node.parent).includes("Show this request")).props.checked, false, "Partner-board opt-in is not carried through a singles entry");
    console.log("Registration editor: saves, legacy status, eligible event choices and singles partner controls passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.fetch = originalFetch;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
