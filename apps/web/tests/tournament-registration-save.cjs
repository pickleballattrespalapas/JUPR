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
  compiled.require = name => name === "@/lib/useRegistrationAttempt" ? load("lib/useRegistrationAttempt.ts") : name === "@/components/tournaments/RegistrationAttemptHistory" ? { __esModule: true, default: () => null } : name === "@/components/tournaments/usePartnerInvitationRegistration" ? { usePartnerInvitationRegistration: () => ({ token: "", invitation: null, error: "" }), PartnerInvitationRegistrationNotice: () => null } : name === "@/lib/tournamentPartnerInvitations" ? { invitationReturnPath: () => "" } : name === "@/lib/tournamentRegistrationProfile" ? load("lib/tournamentRegistrationProfile.ts") : overrides[name] || originalRequire(name);
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
const FormDialog = ({ open, children }) => open ? React.createElement("div", { role: "dialog" }, children) : null;
const redirects = [];
const PartnerDetails = load("components/tournaments/TournamentPartnerDetails.tsx", {
  "@/lib/tournamentRegistrationEligibility": eligibility,
  "@/lib/tournamentRegistrationApi": { resolveClubTournamentPartnerProfile: () => assert.fail("Admin lookup must use its authenticated endpoint") }
}).default;
const Panel = load("app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx", {
  "next/link": { __esModule: true, default: ({ children }) => children },
  "next/navigation": { useRouter: () => ({ replace: href => redirects.push(href) }) },
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/tournaments/TournamentPartnerDetails": { __esModule: true, default: PartnerDetails },
  "@/components/interaction": { ...interaction, FormDialog },
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
    "@/lib/tournamentRegistrationApi": { resolveClubTournamentPartnerProfile: async () => ({ data: { profile_candidates: [] } }) }
  }).default;
  const PublicEditForm = load("app/clubs/[clubSlug]/tournament-registration/edit/EditTournamentRegistrationForm.tsx", {
    "@/components/PublicClubLink": { __esModule: true, default: ({ children }) => children },
    "@/lib/tournamentCommerceApi": {},
    "@/lib/tournamentRegistrationEligibility": eligibility,
    "@/lib/tournamentRegistrationApi": {
      resolveClubTournamentPartnerProfile: async () => ({ data: { profile_candidates: [] } }),
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
        await act(async () => { renderer = create(React.createElement(PublicEditForm, props), {
          createNodeMock: element => element.type === "form" ? {} : element.type === "fieldset" ? { querySelectorAll: () => [] } : null
        }); });
        const gender = renderer.root.findAllByType("select").find(node => node.props.name === "gender");
        assert.equal(gender.props.value, canonical);
        await act(async () => renderer.root.findAllByType("button").find(node => content(node) === "Edit event").props.onClick());
        const partnerField = renderer.root.findAllByType("label").find(node => content(node).startsWith("Partner gender"));
        assert.equal(partnerField.findByType("select").props.value, canonical, "Partner aliases must also populate their selected option");
        await act(async () => new Promise(resolve => setTimeout(resolve, 275)));
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
  registration.player_id = 200;
  registration.linked_profile_skills = { player_id: 200, doubles_skill: 5.11585375, singles_skill: null };
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
  selections[1].gender_review = { status: "PENDING", fingerprint: "a".repeat(64), review_version: "b".repeat(64), reason: "Mixed doubles requires one men's and one women's registrant.", player_gender: "Women", partner_gender: "Non-binary" };
  let eligibilityRequest;
  const partnerLookups = [];
  const partner = { id: "301", display_name: "Kirsten Fixture", doubles_skill: 4.123456789, dupr_id: "fixture-dupr" };
  let profileResponse = { profile_match_kind: "name_partial", profile_candidates: [partner, { ...partner, id: "302", display_name: "Kirsten Other" }] };
  global.fetch = async (_url, options = {}) => {
    if (String(_url).endsWith("/partner-profile-resolution")) {
      assert.equal(new Headers(options.headers).get("Authorization"), "Bearer fixture-session");
      assert.ok(String(_url).includes("/admin/clubs/fixture/tournaments/admin/tournaments/t-fixture/"));
      partnerLookups.push(JSON.parse(options.body));
      return new Response(JSON.stringify(profileResponse));
    }
    if (String(_url).endsWith("/gender-review")) {
      eligibilityRequest = JSON.parse(options.body);
      selections[1].gender_review.status = eligibilityRequest.decision;
      return new Response(JSON.stringify({ ok: true }));
    }
    if (options.method === "PATCH") {
      requests.push(JSON.parse(options.body));
      if (rejection) return new Response(JSON.stringify({ detail: rejection.detail }), { status: rejection.status });
      if (requests.at(-1).registration_status === "cancelled") return new Response(JSON.stringify({ ok: true, registration_removed: true }));
      registration.registration_status = "confirmed";
      return new Response(JSON.stringify({ ok: true }));
    }
    return new Response(JSON.stringify({ registrations: [registration], selections, event_options: eventOptions, days: [{ id: "day-1" }, { id: "day-2" }, { id: "day-off", enabled: false }] }));
  };
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(Panel, { apiBase: "http://fixture.local", clubId: "fixture", status: { enabled: true }, tournamentId: "t-fixture", tournamentName: "Fixture", registrationId: registration.id, drawId: "" })); });
    const field = name => renderer.root.findAllByType("label").find(node => content(node).startsWith(name));
    assert.equal(field("Doubles skill").findByType("input").props.value, "5.11585375");
    assert.equal(field("Doubles skill").findByType("input").props.readOnly, true);
    assert.equal(field("Doubles skill").findByType("input").props.step, "any");
    assert.equal(field("Singles skill").findByType("input").props.readOnly, false);
    assert.equal(field("Singles skill").findByType("input").props.value, "4.2");
    assert.match(content(renderer.root), /This value is self-rated/);
    await act(async () => field("Doubles skill").findByType("input").props.onChange({target: {value: "2"}}));
    assert.equal(field("Doubles skill").findByType("input").props.value, "5.11585375");
    assert.equal(field("Registration status").findByType("select").props.value, "confirmed");
    assert.equal(field("Gender").findByType("select").props.value, "Men");
    assert.match(content(renderer.root), /Needs admin approval/);
    await act(async () => renderer.root.findAllByType(ConfirmAction).find(node => node.props.triggerLabel === "Approve eligibility").props.onConfirm("REVIEW GENDER ELIGIBILITY"));
    assert.equal(eligibilityRequest.decision, "APPROVED");
    assert.equal(eligibilityRequest.expected_review_version, "b".repeat(64));
    assert.match(content(renderer.root), /Gender eligibility: Approved/);
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
    assert.deepEqual(choices().findAllByType("option").map(node => node.props.value), ["", "women", "mixed", "other-singles"], "Gender exceptions can be reviewed; age, closed days and duplicate event families stay excluded");
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
    const eventDialog = () => renderer.root.findAllByType(FormDialog).find(node => node.props.open);
    const partnerField = name => renderer.root.findByProps({ "aria-label": `Event entry partner ${name}` });
    const changePartner = (name, value) => act(async () => partnerField(name).props.onChange({ target: { value } }));
    await changePartner("name", "Kirsten");
    assert.equal(eventDialog().props.submitDisabled, true, "Saving waits for profile lookup");
    await act(async () => new Promise(resolve => setTimeout(resolve, 275)));
    assert.equal(partnerLookups.at(-1).name, "Kirsten");
    assert.equal(eventDialog().props.submitDisabled, true, "Partial name matches require a choice");
    await act(async () => renderer.root.findAllByType("input").find(node => node.props.type === "radio" && content(node.parent).includes("Kirsten Fixture")).props.onChange());
    assert.equal(partnerField("name").props.value, partner.display_name);
    assert.equal(partnerField("skill").props.value, String(partner.doubles_skill));
    assert.equal(partnerField("skill").props.readOnly, true);
    assert.equal(partnerField("skill").props.step, "any");
    assert.equal(renderer.root.findAllByProps({ "aria-label": "Event entry partner DUPR ID" }).length, 0);
    await changePartner("skill", "2.5");
    assert.equal(partnerField("skill").props.value, String(partner.doubles_skill));
    await changePartner("email", "partner@example.com");
    assert.equal(partnerField("skill").props.readOnly, true, "Entering email preserves the selection");
    assert.equal(partnerField("age").props.required, false, "Admin demographic requirements stay unchanged");
    assert.equal(eventDialog().props.submitDisabled, false);
    await act(async () => eventDialog().props.onSubmit());
    assert.equal(requests.at(-1).partner_name, partner.display_name);
    assert.equal(requests.at(-1).partner_skill, partner.doubles_skill);
    assert.equal(requests.at(-1).partner_dupr_id, partner.dupr_id);
    assert.equal(requests.at(-1).expected_updated_at, selections[1].updated_at);
    assert.equal(requests.at(-1).profileId, undefined, "Only validated selection fields are submitted");
    await act(async () => entry("Doubles entry").findAllByType("button").find(node => content(node) === "Edit entry").props.onClick());
    await act(async () => field("Partner state").findByType("select").props.onChange({ target: { value: "HAS_PARTNER" } }));
    assert.equal(partnerField("skill").props.readOnly, false, "Reopening discards an unsaved profile choice");
    profileResponse = { profile_match_kind: "name_exact", profile_candidates: [partner] };
    await changePartner("name", partner.display_name);
    await act(async () => new Promise(resolve => setTimeout(resolve, 275)));
    assert.equal(partnerField("skill").props.readOnly, true, "A unique full name selects its profile automatically");
    profileResponse = { profile_match_kind: "none", profile_candidates: [] };
    await changePartner("name", "New Partner");
    assert.equal(partnerField("skill").props.value, "", "Changing identity clears the previous profile rating");
    await act(async () => new Promise(resolve => setTimeout(resolve, 275)));
    assert.equal(partnerField("skill").props.readOnly, false);
    assert.match(content(renderer.root), /No matching profile found/);
    await act(async () => field("Partner state").findByType("select").props.onChange({ target: { value: "NEEDS_PARTNER" } }));
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "other-singles" } }));
    assert.equal(renderer.root.findAllByType("label").some(node => content(node).startsWith("Partner state")), false, "Moving an entry to singles removes partner controls");
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "doubles" } }));
    assert.equal(field("Partner state").findByType("select").props.value, "NEEDS_PARTNER");
    assert.equal(renderer.root.findAllByType("input").find(node => node.props.type === "checkbox" && node.parent.type === "label" && content(node.parent).includes("Show this request")).props.checked, false, "Partner-board opt-in is not carried through a singles entry");
    // A confirmed team changes division in one save, preserving its pairing and
    // sending the version snapshot so a stale partner update cannot be overwritten.
    eventOptions.push(event("doubles-4.5", { event_family_label: "doubles", division_name: "Men's Doubles 4.5", event_type: "GENDER_DOUBLES", partner_required: true, registration_day_id: "day-2" }));
    selections[1].partner_selection_id = "sel-partner";
    selections[1].partner_display_name = "Fixture Partner";
    selections[1].division_transfer_context = { team_link_id: "link", team_link_updated_at: "v1", partner_selection_id: "sel-partner", partner_selection_updated_at: "v2", partner_registration_id: "reg-partner", partner_registration_updated_at: "v3" };
    await act(async () => { await save(); });
    await act(async () => entry("Doubles entry").findAllByType("button").find(node => content(node) === "Change division").props.onClick());
    const divisionDialog = () => renderer.root.findAllByType(FormDialog).find(node => node.props.open);
    const newDivision = () => renderer.root.findByProps({ "aria-label": "New division" });
    assert.equal(newDivision().props.disabled, false, "A confirmed partnership must not lock the division selector");
    assert.deepEqual(newDivision().findAllByType("option").map(node => node.props.value), ["", "doubles", "doubles-4.5"], "Division choices stay within the event, including divisions on another day");
    assert.match(content(renderer.root), /Fixture Player and Fixture Partner move together/);
    assert.equal(divisionDialog().props.submitDisabled, true, "The current division is not a change");
    await act(async () => newDivision().props.onChange({ target: { value: "doubles-4.5" } }));
    assert.equal(divisionDialog().props.submitLabel, "Move both players");
    assert.equal(divisionDialog().props.submitDisabled, false);
    rejection = { status: 409, detail: "Partner entry changed. Reload before moving this team." };
    await act(async () => { try { await divisionDialog().props.onSubmit(); } catch {} });
    assert.equal(newDivision().props.value, "doubles-4.5", "A rejected transfer retains the chosen division");
    rejection = null;
    await act(async () => { completion = await divisionDialog().props.onSubmit(); });
    assert.equal(completion.status, "success");
    const transfer = requests.at(-1);
    assert.equal(transfer.event_option_id, "doubles-4.5");
    assert.deepEqual(transfer.division_transfer_context, selections[1].division_transfer_context);
    assert.equal(transfer.partner_mode, undefined, "Moving a division must not reset the partner state");
    assert.equal(transfer.expected_updated_at, selections[1].updated_at);
    // The Edit entry path also supports moving a confirmed team while clearing
    // an obsolete public note, as in the reported Mixed 2.5 -> Mixed 3.0 move.
    selections[1].partner_note = "2.0 level";
    await act(async () => { await save(); });
    await act(async () => entry("Doubles entry").findAllByType("button").find(node => content(node) === "Edit entry").props.onClick());
    assert.equal(field("Public partner note").findByType("textarea").props.value, "2.0 level");
    await act(async () => field("Public partner note").findByType("textarea").props.onChange({ target: { value: "" } }));
    await act(async () => field("Division").findByType("select").props.onChange({ target: { value: "doubles-4.5" } }));
    assert.equal(divisionDialog().props.submitLabel, "Move both players");
    assert.equal(divisionDialog().props.submitDisabled, false);
    await act(async () => { completion = await divisionDialog().props.onSubmit(); });
    assert.equal(completion.status, "success");
    assert.deepEqual(requests.at(-1), {
      event_option_id: "doubles-4.5", partner_note: "",
      division_transfer_context: selections[1].division_transfer_context,
      expected_updated_at: selections[1].updated_at,
      confirmation_text: "SAVE SELECTION", source: "next_tournament_registration_detail"
    }, "The move carries the requested note edit without replacing the confirmed partner");
    await verifyPublicLegacyGenders();
    await act(async () => field("Registration status").findByType("select").props.onChange({ target: { value: "cancelled" } }));
    const cancellation = () => renderer.root.findAllByType(ConfirmAction).find(node => node.props.confirmLabel === "Yes, cancel and remove");
    assert.equal(cancellation().props.tone, "danger");
    assert.match(cancellation().props.description, /all its event entries and partner connections/);
    rejection = { status: 409, detail: "Registration changed. Refresh and try again." };
    await act(async () => { try { await cancellation().props.onConfirm("SAVE REGISTRATION"); } catch {} });
    assert.deepEqual(redirects, [], "A failed cancellation keeps the admin on the editor");
    rejection = null;
    await act(async () => { completion = await cancellation().props.onConfirm("SAVE REGISTRATION"); });
    assert.equal(completion.status, "success");
    assert.deepEqual(redirects, ["/admin/tournaments/registration/registrants"], "Successful removal returns to the registrant list");
    console.log("Registration editors: saves, legacy gender/status, eligible events and singles partner controls passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.fetch = originalFetch;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
