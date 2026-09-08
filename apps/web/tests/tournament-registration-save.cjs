const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
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

async function verifyPublicLegacyGenders() {
  const submissions = [];
  const PartnerDetails = load("components/tournaments/TournamentPartnerDetails.tsx", {
    "@/lib/tournamentRegistrationEligibility": eligibility,
    "@/lib/tournamentRegistrationApi": {}
  }).default;
  const PublicEditForm = load("app/clubs/[clubSlug]/tournament-registration/edit/EditTournamentRegistrationForm.tsx", {
    "next/link": { __esModule: true, default: ({ children }) => children },
    "@/lib/tournamentRegistrationEligibility": eligibility,
    "@/lib/tournamentRegistrationApi": {
      submitClubTournamentRegistrationEdit: async (_club, payload) => {
        submissions.push(payload);
        return { data: { registration_id: "reg-edit", confirmation_token: "fixture-confirmation" } };
      }
    },
    "@/components/tournaments/TournamentPartnerDetails": { __esModule: true, default: PartnerDetails },
    "@/components/interaction": { InteractionDialog: ({ children }) => React.createElement("div", { role: "dialog" }, children) },
    "../TournamentCommerceChooser": { __esModule: true, default: () => null }
  }).default;
  const originalFormData = global.FormData;
  try {
    for (const [stored, canonical] of [["Female", "Women"], ["Male", "Men"], ["Self-described", "Self-described"]]) {
      const registration = { id: "reg-edit", first_name: "Fixture", last_name: "Player", email: "fixture@example.com", gender: stored, age: 51, doubles_skill: 4.5, updated_at: "2026-09-07T00:00:00Z" };
      const props = {
        clubSlug: "fixture", tournamentId: "t-fixture", editToken: "fixture-edit", registration,
        players: [], days: [{ id: "day-1", label: "Day 1" }],
        events: [{ id: "doubles", event_family_label: "Doubles", division_name: "Doubles Open", registration_day_id: "day-1", event_type: "GENDER_DOUBLES", gender_restriction: "ANY", skill_mode: "OPEN", partner_required: true, selectable: true }],
        selections: [{ id: "sel-double", event_option_id: "doubles", registration_day_id: "day-1", partner_mode: "HAS_PARTNER", partner_name: "Fixture Partner", partner_email: "partner@example.com", partner_age: 50, partner_gender: stored, partner_skill: 4.5, updated_at: registration.updated_at }]
      };
      const html = renderToStaticMarkup(React.createElement(PublicEditForm, props));
      const genderMarkup = html.match(/<select name="gender"[^>]*>(.*?)<\/select>/s)?.[1];
      assert.ok(genderMarkup?.includes(`<option value="${canonical}" selected="">${canonical}</option>`), "The browser must receive a selected gender option for a legacy value");
      let renderer;
      try {
        await act(async () => { renderer = create(React.createElement(PublicEditForm, props)); });
        const gender = renderer.root.findAllByType("select").find(node => node.props.name === "gender");
        assert.equal(gender.props.value, canonical);
        await act(async () => renderer.root.findAllByType("button").find(node => content(node) === "Edit event").props.onClick());
        const partnerField = renderer.root.findAllByType("label").find(node => content(node).startsWith("Partner gender"));
        assert.equal(partnerField.findByType("select").props.value, canonical, "Partner aliases must also populate their selected option");
        const formValues = { first_name: "Fixture", last_name: "Player", age: "51", gender: gender.props.value, doubles_skill: "4.5", terms_accepted: "on" };
        global.FormData = class { get(name) { return formValues[name] ?? null; } };
        await act(async () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} }));
        const saved = submissions.at(-1);
        assert.equal(saved?.gender, canonical, "Saving an unrelated change must retain the selected player gender");
        assert.equal(saved.selections[0].partner_gender, canonical, "The submitted partner gender must match the visible field");
        assert.equal(saved.selections[0].id, "sel-double");
      } finally {
        if (renderer) await act(async () => renderer.unmount());
      }
    }
  } finally {
    global.FormData = originalFormData;
  }
}

async function main() {
  const originalFetch = global.fetch;
  const requests = [];
  let rejection = { status: 400, detail: "Another registration already uses that email." };
  const registration = { id: "reg-fixture", display_name: "Fixture Player", email: "fixture@example.com", gender: "Male", age: 38, doubles_skill: 4.2, singles_skill: 4.2, registration_status: "pending", payment_status: "unpaid", updated_at: "2026-09-07T00:00:00Z" };
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
    { id: "sel-double", registration_id: registration.id, event_option_id: "doubles", event_label: "Doubles entry", partner_mode: "NEEDS_PARTNER", partner_gender: "Male", show_on_partner_board: true, updated_at: registration.updated_at }
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
    assert.equal(field("Gender").findByType("select").props.value, "Men");
    await act(async () => field("Phone").findByType("input").props.onChange({ target: { value: "555-9999" } }));
    const save = () => renderer.root.findAllByType(ConfirmAction).find(node => node.props.confirmLabel === "Yes, save registration").props.onConfirm("SAVE REGISTRATION");
    let failure;
    await act(async () => { try { await save(); } catch (error) { failure = interaction.normalizeInteractionActionError(error); } });
    assert.equal(failure.kind, "validation");
    assert.equal(failure.message, rejection.detail, "The dialog retains the actionable server rejection");
    assert.equal(field("Phone").findByType("input").props.value, "555-9999", "Failed saves preserve the draft");
    assert.equal(requests[0].registration_status, "confirmed", "The visible dropdown and submitted status agree");
    assert.equal(requests[0].gender, "Men", "The admin save preserves the canonical gender");
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
    await act(async () => field("Partner state").findByType("select").props.onChange({ target: { value: "HAS_PARTNER" } }));
    assert.equal(field("Partner gender").findByType("select").props.value, "Men");
    await act(async () => field("Partner state").findByType("select").props.onChange({ target: { value: "NEEDS_PARTNER" } }));
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "other-singles" } }));
    assert.equal(renderer.root.findAllByType("label").some(node => content(node).startsWith("Partner state")), false, "Moving an entry to singles removes partner controls");
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "doubles" } }));
    assert.equal(field("Partner state").findByType("select").props.value, "NEEDS_PARTNER");
    assert.equal(renderer.root.findAllByType("input").find(node => node.props.type === "checkbox" && node.parent.type === "label" && content(node.parent).includes("Show this request")).props.checked, false, "Partner-board opt-in is not carried through a singles entry");
    await verifyPublicLegacyGenders();
    console.log("Registration editors: saves, legacy gender/status, eligible events and singles partner controls passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.fetch = originalFetch;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
