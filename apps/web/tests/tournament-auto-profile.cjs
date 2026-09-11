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
  const original = compiled.require.bind(compiled);
  compiled.require = name => {
    if (name === "@/components/tournaments/usePartnerInvitationRegistration") return { usePartnerInvitationRegistration: () => ({ token: "", invitation: null, error: "" }), PartnerInvitationRegistrationNotice: () => null };
    if (name === "@/lib/tournamentPartnerInvitations") return { invitationReturnPath: () => "" };
    if (name === "@/lib/tournamentRegistrationProfile") return profileHelpers;
    const value = overrides[name];
    return value ? { __esModule: true, ...value } : original(name);
  };
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}
const profileHelpers = load("lib/tournamentRegistrationProfile.ts");
const player = { id: "player-1", display_name: "Fixture Player", doubles_skill: 4.1, singles_skill: 3.8, dupr_id: "fixture-dupr" };
const duplicate = { ...player, id: "player-2", doubles_skill: 3.2 };
assert.equal(profileHelpers.automaticRegistrationProfile([player], "  FIXTURE   Player  "), player);
assert.equal(profileHelpers.automaticRegistrationProfile([player, duplicate], player.display_name, "name_exact"), null);
assert.equal(profileHelpers.automaticRegistrationProfile([player], "Someone Else", "email_exact"), null);
assert.equal(profileHelpers.automaticRegistrationProfile([player], "Fixture", "none"), null);
assert.equal(profileHelpers.automaticRegistrationProfile([], player.display_name), null);
assert.equal(profileHelpers.automaticRegistrationProfile([{ ...player, display_name: "Preferred Name" }], player.display_name, "name_exact").id, player.id);
const eligibility = load("lib/tournamentRegistrationEligibility.ts", { "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts") });
const result = (candidates = [player]) => ({ data: { can_start_new: true, profile_match_kind: "name_exact", profile_candidates: candidates } });
let resolveResponse = async () => result();
const requests = [];
const submissions = [];
const api = {
  resolveClubTournamentRegistrationProfile: async (_club, payload) => { requests.push(payload); return resolveResponse(payload); },
  resolveClubTournamentPartnerProfile: async (_club, payload) => { requests.push(payload); return resolveResponse(payload); },
  submitClubTournamentRegistration: async (_club, payload) => { submissions.push(payload); return { error: "Test stops before writing" }; },
  submitClubTournamentRegistrationEdit: async (_club, payload) => { submissions.push(payload); return { error: "Test stops before writing" }; }
};
const shared = {
  "@/lib/tournamentRegistrationApi": api,
  "@/lib/tournamentRegistrationEligibility": eligibility,
  "@/components/tournaments/TournamentPartnerDetails": { default: () => null },
  "@/lib/tournamentCommerceApi": {},
  "@/components/interaction": { InteractionDialog: ({ children, actions }) => React.createElement("section", { role: "dialog" }, children, actions) }
};
const base = "app/clubs/[clubSlug]/tournament-registration/";
const NewForm = load(base + "TournamentRegistrationForm.tsx", {
  ...shared, "@/lib/tournamentTeamCompetitionApi": {},
  "@/components/tournaments/FourPlayerTeamRegistrationCard": { default: () => null, TEAM_SLOTS: [] },
  "./EditLinkRequestForm": { default: () => null }, "./TournamentCommerceChooser": { default: () => null }
}).default;
const EditForm = load(base + "edit/EditTournamentRegistrationForm.tsx", {
  ...shared, "next/link": { default: ({ children }) => children }, "../TournamentCommerceChooser": { default: () => null }
}).default;
const event = { id: "singles", registration_day_id: "day", event_family_label: "Singles", division_name: "Open", event_type: "SINGLES", skill_mode: "OPEN", selectable: true };
const props = { clubSlug: "fixture", tournamentId: "t1", registrationSlug: "fixture-tournament", registrationOpen: true, days: [{ id: "day", label: "Day 1" }], events: [event] };
const registration = { id: "reg", first_name: "Fixture", last_name: "Player", display_name: "Fixture Player", email: "fixture@example.invalid", age: 70, gender: "Men", doubles_skill: 2.5, updated_at: "version-1" };
let renderer;
const text = node => typeof node === "string" ? node : (node.children || []).map(text).join("");
const button = label => renderer.root.findAllByType("button").find(node => text(node) === label);
const field = label => renderer.root.findByProps({ "aria-label": label });
const named = name => renderer.root.findAllByType("input").find(node => node.props.name === name);
const change = (label, value) => act(async () => field(label).props.onChange({ target: { value } }));
const chooseNone = () => act(async () => renderer.root.findAllByType("input").find(node => node.props.type === "radio" && text(node.parent).includes("None of these is me")).props.onChange());
const settle = () => act(async () => new Promise(resolve => setTimeout(resolve, 275)));
async function mount(Component, patch = {}) {
  if (renderer) await act(async () => renderer.unmount());
  submissions.length = 0;
  requests.length = 0;
  await act(async () => { renderer = create(React.createElement(Component, { ...props, ...patch }), {
    createNodeMock: node => node.type === "form" ? {} : node.type === "fieldset" ? { querySelectorAll: () => [] } : null
  }); });
}
async function startNew() {
  await mount(NewForm);
  await act(async () => button("Start a registration").props.onClick());
  for (const [label, value] of [["First name", "Fixture"], ["Last name", "Player"], ["Email", "fixture@example.invalid"], ["Age", "70"], ["Gender", "Men"]]) await change(label, value);
  await act(async () => button("Continue").props.onClick());
}
async function testNew() {
  await startNew();
  assert.equal(field("Doubles skill").props.value, "4.1");
  assert.equal(field("Singles skill").props.value, "3.8");
  assert.equal(renderer.root.findAllByProps({ "aria-label": "DUPR ID" }).length, 0);
  assert.equal(named("profile_candidate").props.checked, true, "Unique name match is selected before the player touches the choices");
  await act(async () => button("Continue to events").props.onClick());
  await act(async () => field("Singles Open").props.onChange({ target: { checked: true } }));
  await act(async () => button("Review registration").props.onClick());
  await act(async () => renderer.root.findAllByType("input").find(node => node.props.type === "checkbox").props.onChange({ target: { checked: true } }));
  await act(async () => button("Submit registration").props.onClick());
  assert.equal(submissions[0].doubles_skill, 4.1);
  assert.equal(submissions[0].dupr_id, player.dupr_id);
  assert.equal(submissions[0].player_id, null, "A public prefill must not bypass staff verification");

  await startNew();
  await chooseNone();
  await change("Doubles skill", "3.25");
  await act(async () => button("Back").props.onClick());
  await act(async () => button("Continue").props.onClick());
  assert.equal(field("Doubles skill").props.value, "3.25", "Going back preserves an explicit opt out and manual details");
  assert.equal(named("profile_candidate").props.checked, false);
  await act(async () => button("Back").props.onClick());
  await change("First name", "Other");
  resolveResponse = async () => result([]);
  await act(async () => button("Continue").props.onClick());
  assert.equal(field("Display name").props.value, "Other Player");
  assert.equal(field("Doubles skill").props.value, "", "Changing identity clears the old profile fields");

  resolveResponse = async () => result([player, duplicate]);
  await startNew();
  assert.equal(renderer.root.findAllByType("input").some(node => node.props.type === "radio" && node.props.checked), false);
  await act(async () => button("Continue to events").props.onClick());
  assert.match(text(renderer.root.findByProps({ role: "alert" })), /Choose your profile/);
  await chooseNone();
  await act(async () => button("Continue to events").props.onClick());
  assert.ok(button("Review registration"));
}
const editProps = patch => ({ registration, players: [], editToken: "fixture", selections: [{ id: "s1", event_option_id: "singles", partner_mode: "NONE", updated_at: "v1" }], ...patch });
const submitEdit = () => act(async () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} }));
async function testEdit() {
  resolveResponse = async () => result();
  await mount(EditForm, editProps());
  await submitEdit();
  assert.equal(submissions.length, 0, "Saving waits for the automatic match");
  await settle();
  assert.equal(named("doubles_skill").props.value, "4.1");
  assert.equal(named("registration_profile").props.checked, true);
  await act(async () => button("Edit event").props.onClick());
  await act(async () => button("Save changes").props.onClick());
  assert.equal(submissions.length, 1);
  assert.equal(submissions[0].doubles_skill, 4.1);
  assert.equal(submissions[0].singles_skill, 3.8);
  assert.equal(submissions[0].dupr_id, player.dupr_id);
  assert.equal(submissions[0].player_id, null);
  assert.equal(submissions[0].expected_updated_at, "version-1");

  await mount(EditForm, editProps({ registration: { ...registration, player_id: player.id }, players: [player] }));
  await settle();
  assert.equal(requests.length, 0, "An established link is never replaced by a name lookup");
  assert.equal(named("doubles_skill").props.value, 4.1);
  await submitEdit();
  assert.equal(submissions[0].player_id, player.id);

  resolveResponse = async () => result([player, duplicate]);
  await mount(EditForm, editProps());
  await settle();
  await submitEdit();
  assert.equal(submissions.length, 0, "Duplicate names require an explicit choice");
  await chooseNone();
  await submitEdit();
  assert.equal(submissions.length, 1);

  resolveResponse = async () => { throw Error("offline"); };
  await mount(EditForm, editProps());
  await settle();
  assert.equal(named("doubles_skill").props.value, "2.5");
  await submitEdit();
  assert.equal(submissions.length, 1, "A failed lookup still allows manual registration edits");
}
async function testStaleEditLookup() {
  let finish;
  resolveResponse = () => new Promise(resolve => { finish = resolve; });
  await mount(EditForm, editProps());
  await settle();
  await act(async () => named("first_name").props.onChange({ target: { value: "Different" } }));
  await act(async () => finish(result()));
  assert.equal(named("doubles_skill").props.value, "2.5", "The old name's lookup must not overwrite this person");
  resolveResponse = async () => result([]);
  await settle();
  assert.equal(renderer.root.findAllByProps({ name: "registration_profile" }).length, 0);

  resolveResponse = () => new Promise(resolve => { finish = resolve; });
  await mount(EditForm, editProps());
  await settle();
  await act(async () => named("doubles_skill").props.onChange({ target: { value: "3.6" } }));
  await act(async () => finish(result()));
  assert.equal(named("doubles_skill").props.value, "3.6", "Late matches cannot overwrite a newly typed value");
  assert.equal(named("registration_profile").props.checked, false);
}
async function main() {
  const originalWindow = global.window;
  const originalFormData = global.FormData;
  global.window = { location: { hash: "" }, addEventListener() {}, removeEventListener() {} };
  global.FormData = class { get(name) {
    const input = renderer.root.findAll(node => ["input", "select", "textarea"].includes(node.type) && node.props.name === name)[0];
    return input?.props.value ?? input?.props.defaultValue ?? null;
  } };
  try { await testNew(); await testEdit(); await testStaleEditLookup(); console.log("Automatic profiles: new/edit selection, save payloads, duplicates, opt out, existing links, stale responses and manual fallback passed."); }
  finally { if (renderer) await act(async () => renderer.unmount()); global.window = originalWindow; global.FormData = originalFormData; }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
