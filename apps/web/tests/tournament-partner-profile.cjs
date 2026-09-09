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
  compiled.require = name => {
    const override = overrides[name];
    return override ? ("default" in override ? { __esModule: true, ...override } : override) : originalRequire(name);
  };
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}

const eligibility = load("lib/tournamentRegistrationEligibility.ts", {
  "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts")
});
const candidate = { id: "partner-profile", display_name: "Fixture Partner", doubles_skill: 3.4, dupr_id: "fixture-dupr" };
let lookupResponse = async () => ({ data: { profile_candidates: [candidate] } });
const lookups = [];
const submissions = [];
const api = {
  resolveClubTournamentPartnerProfile: async (club, payload) => { lookups.push({ club, ...payload }); return lookupResponse(); },
  resolveClubTournamentRegistrationProfile: async () => ({ data: { can_start_new: true, profile_candidates: [] } }),
  submitClubTournamentRegistration: async (_club, payload) => { submissions.push(payload); return { error: "Fixture stops before saving" }; },
  submitClubTournamentRegistrationEdit: async (_club, payload) => { submissions.push(payload); return { error: "Fixture stops before saving" }; }
};
const PartnerDetails = load("components/tournaments/TournamentPartnerDetails.tsx", {
  "@/lib/tournamentRegistrationApi": api,
  "@/lib/tournamentRegistrationEligibility": eligibility
}).default;
const shared = {
  "@/lib/tournamentRegistrationApi": api,
  "@/lib/tournamentRegistrationEligibility": eligibility,
  "@/components/tournaments/TournamentPartnerDetails": { default: PartnerDetails }
};
const base = "app/clubs/[clubSlug]/tournament-registration/";
const NewForm = load(base + "TournamentRegistrationForm.tsx", {
  ...shared, "@/lib/tournamentTeamCompetitionApi": {},
  "@/components/tournaments/FourPlayerTeamRegistrationCard": { default: () => null, TEAM_SLOTS: [] },
  "./EditLinkRequestForm": { default: () => null }, "./TournamentCommerceChooser": { default: () => null }
}).default;
const EditForm = load(base + "edit/EditTournamentRegistrationForm.tsx", {
  ...shared, "next/link": { default: ({ children }) => children },
  "@/components/interaction": { InteractionDialog: ({ children, actions }) => React.createElement("div", { role: "dialog" }, children, actions) },
  "../TournamentCommerceChooser": { default: () => null }
}).default;
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");
const event = { id: "mixed", registration_day_id: "day", event_family_label: "Mixed", division_name: "Below 9", event_type: "MIXED_DOUBLES", gender_restriction: "MIXED", partner_required: true, eligibility_mode: "COMBINED_RATING_CAP", combined_rating_cap: 9, selectable: true, price_usd: 20 };
const props = { clubSlug: "fixture-club", tournamentId: "t1", registrationSlug: "fixture-tournament", registrationOpen: true, days: [{ id: "day", label: "Day 1" }], events: [event] };
let renderer;
const button = name => renderer.root.findAllByType("button").find(node => content(node) === name);
const field = name => renderer.root.findByProps({ "aria-label": name });
const change = (name, value) => act(async () => field(name).props.onChange({ target: { value } }));
const choosePartner = () => act(async () => renderer.root.findAllByType("input").find(node => node.props.type === "radio" && content(node.parent).includes("Fixture Partner")).props.onChange());

async function testNewRegistration() {
  await act(async () => { renderer = create(React.createElement(NewForm, props)); });
  await act(async () => button("Start a registration").props.onClick());
  for (const [name, value] of [["First name", "Fixture"], ["Last name", "Player"], ["Email", "player@example.invalid"], ["Age", "40"], ["Gender", "Men"]]) await change(name, value);
  await change("Notes for tournament staff", "Private staff message");
  assert.match(content(renderer.root.findByProps({ id: "staff-notes-help" })), /do not appear on the public Partner Board/);
  await act(async () => button("Continue").props.onClick());
  await change("Doubles skill", "4.5");
  await act(async () => button("Continue to events").props.onClick());
  await act(async () => field("Mixed Below 9").props.onChange({ target: { checked: true } }));
  await change("Below 9 public partner note", "Public partner message");
  assert.match(content(renderer.root.findByProps({ id: "partner-note-help-mixed" })), /Visible to everyone/);
  await change("Below 9 partner plan", "HAS_PARTNER");
  await change("Below 9 partner name", "Fixture Partner");
  await act(async () => button("Find partner profile").props.onClick());
  assert.equal(lookups.at(-1).name, "Fixture Partner");
  assert.equal(lookups.at(-1).email, null, "A profile can be found before entering email, age or gender");
  await choosePartner();
  assert.equal(field("Below 9 partner skill").props.value, "3.4");
  assert.equal(renderer.root.findAllByProps({ "aria-label": "Below 9 partner DUPR ID" }).length, 0);
  // Adding contact details must preserve the explicitly selected profile.
  await change("Below 9 partner email", "Baumann");
  assert.equal(field("Below 9 partner skill").props.value, "3.4");
  await change("Below 9 partner age", "70");
  await change("Below 9 partner gender", "Women");
  await change("Below 9 partner skill", "3.4");
  await act(async () => button("Review registration").props.onClick());
  assert.match(content(renderer.root.findByProps({ role: "alert" })), /valid partner email/);
  await change("Below 9 partner email", "partner@example.invalid");
  assert.equal(field("Below 9 partner skill").props.value, "3.4");
  await act(async () => button("Review registration").props.onClick());
  await act(async () => button("Back").props.onClick());
  assert.equal(field("Below 9 partner skill").props.value, "3.4", "Selected profile survives review and back");
  assert.equal(renderer.root.findAllByProps({ "aria-label": "Below 9 partner DUPR ID" }).length, 0);
  await change("Below 9 partner skill", "4.6");
  await act(async () => button("Review registration").props.onClick());
  assert.match(content(renderer.root.findByProps({ role: "alert" })), /combined rating must be below 9/);
  await change("Below 9 partner skill", "3.4");
  await act(async () => button("Review registration").props.onClick());
  await act(async () => renderer.root.findAllByType("input").find(node => node.props.type === "checkbox").props.onChange({ target: { checked: true } }));
  await act(async () => button("Submit registration").props.onClick());
  const selection = submissions.at(-1).selections[0];
  assert.equal(selection.partner_name, candidate.display_name);
  assert.equal(selection.partner_skill, 3.4);
  assert.equal(selection.partner_dupr_id, candidate.dupr_id);
  assert.equal(selection.partner_email, "partner@example.invalid");
  assert.equal(selection.partner_note, "Public partner message");
  assert.equal(submissions.at(-1).notes, "Private staff message", "Public partner notes and private staff notes must remain separate when saving");
  assert.equal(selection.partner_player_id, undefined, "A public suggestion must not become a verified identity link");
  assert.equal(selection.profileId, undefined);
  await act(async () => renderer.unmount());
}

async function testEditing() {
  const registration = { id: "reg", email: "player@example.invalid", first_name: "Fixture", last_name: "Player", age: 40, gender: "Men", doubles_skill: 4.5, notes: "Existing private staff message" };
  await act(async () => { renderer = create(React.createElement(EditForm, { ...props, registration, editToken: "fixture", players: [], selections: [{ id: "selection", event_option_id: "mixed", partner_mode: "HAS_PARTNER", partner_name: "Fixture Partner", partner_email: "partner@example.invalid", partner_age: 70, partner_gender: "Female", partner_skill: 2.5 }] })); });
  await act(async () => button("Edit event").props.onClick());
  const staffNotes = renderer.root.findByProps({ name: "notes" }).props.defaultValue;
  assert.match(content(renderer.root.findByProps({ id: "edit-staff-notes-help" })), /do not appear on the public Partner Board/);
  assert.match(content(renderer.root.findByProps({ id: "edit-partner-note-help" })), /Visible to everyone/);
  await act(async () => renderer.root.findByProps({ "aria-describedby": "edit-partner-note-help" }).props.onChange({ target: { value: "Updated public partner message" } }));
  await act(async () => button("Find partner profile").props.onClick());
  await choosePartner();
  assert.equal(field("Below 9 partner skill").props.value, "3.4", "Profile selection updates controlled edit fields");
  assert.equal(field("Below 9 partner gender").props.value, "Women");
  await act(async () => button("Apply event changes").props.onClick());
  await act(async () => button("Edit event").props.onClick());
  assert.equal(renderer.root.findAllByProps({ "aria-label": "Below 9 partner DUPR ID" }).length, 0);
  await act(async () => button("Apply event changes").props.onClick());
  const original = global.FormData;
  global.FormData = class { get(name) { return { first_name: "Fixture", last_name: "Player", age: "40", gender: "Men", doubles_skill: "4.5", terms_accepted: "on", notes: staffNotes }[name] ?? null; } };
  try { await act(async () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} })); }
  finally { global.FormData = original; }
  const selection = submissions.at(-1).selections[0];
  assert.equal(selection.partner_skill, 3.4);
  assert.equal(selection.partner_gender, "Women");
  assert.equal(selection.partner_profile_id, undefined);
  assert.equal(selection.id, "selection");
  assert.equal(selection.partner_note, "Updated public partner message");
  assert.equal(submissions.at(-1).notes, "Existing private staff message", "Editing the public note must preserve the separate staff note");
  await act(async () => renderer.unmount());
}

async function testLookupRaceAndFallback() {
  function Host() {
    const [value, setValue] = React.useState({ name: "Fixture Partner", email: "", age: "", gender: "", skill: "", phone: "", duprId: "" });
    return React.createElement(PartnerDetails, { ...props, labelPrefix: "Partner", value, onChange: patch => setValue(current => ({ ...current, ...patch })) });
  }
  await act(async () => { renderer = create(React.createElement(Host)); });
  let finish;
  lookupResponse = () => new Promise(resolve => { finish = resolve; });
  let waiting;
  await act(async () => { waiting = button("Find partner profile").props.onClick(); });
  await change("Partner partner name", "Different Partner");
  await act(async () => { finish({ data: { profile_candidates: [candidate] } }); await waiting; });
  assert.equal(renderer.root.findAllByType("fieldset").length, 0, "A stale lookup must not suggest the previous partner");
  lookupResponse = async () => ({ data: { profile_candidates: [] } });
  await act(async () => button("Find partner profile").props.onClick());
  assert.match(content(renderer.root.findByProps({ role: "status" })), /No matching profile/);
  await change("Partner partner skill", "3.25");
  lookupResponse = async () => { throw Error("offline"); };
  await act(async () => button("Find partner profile").props.onClick());
  assert.match(content(renderer.root.findByProps({ role: "alert" })), /enter their details below/);
  assert.equal(field("Partner partner skill").props.value, "3.25", "Lookup failure preserves manual details");
  await act(async () => renderer.unmount());
}

async function main() {
  const previousWindow = global.window;
  global.window = { location: { hash: "" }, addEventListener() {}, removeEventListener() {} };
  try {
    await testNewRegistration();
    await testEditing();
    await testLookupRaceAndFallback();
    console.log("Partner profiles: new/edit flows, prefill, eligibility, identity boundary, stale responses and manual fallback passed.");
  } finally { if (renderer) await act(async () => renderer.unmount()); global.window = previousWindow; }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
