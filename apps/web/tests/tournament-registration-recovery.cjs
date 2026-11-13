const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");
global.IS_REACT_ACT_ENVIRONMENT = true;

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => {
    if (name === "@/lib/useRegistrationAttempt") return load("lib/useRegistrationAttempt.ts");
    if (name === "@/components/tournaments/RegistrationAttemptHistory") return { __esModule: true, default: () => null };
    if (name === "@/components/tournaments/usePartnerInvitationRegistration") return { usePartnerInvitationRegistration: () => ({ token: "", invitation: null, error: "" }), PartnerInvitationRegistrationNotice: () => null };
    if (name === "@/lib/tournamentPartnerInvitations") return { invitationReturnPath: () => "" };
    if (name === "@/lib/tournamentRegistrationProfile") return load("lib/tournamentRegistrationProfile.ts");
    if (name === "@/components/ClubDisplay") return load("components/ClubDisplay.tsx");
    if (name === "@/lib/clubSite") return load("lib/clubSite.ts");
    const override = overrides[name];
    return override ? ("default" in override ? { __esModule: true, ...override } : override) : originalRequire(name);
  };
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}

const registration = "app/clubs/[clubSlug]/tournament-registration/";
const api = { requestClubTournamentRegistrationEditLink: () => assert.fail("rendering must not send email") };
const EditForm = load(registration + "EditLinkRequestForm.tsx", {
  "@/lib/tournamentRegistrationApi": api,
  "@/components/interaction": { InteractionDialog: ({ title, children }) => React.createElement("section", { "data-testid": "edit-link-notice" }, title, children) }
}).default;
const Form = load(registration + "TournamentRegistrationForm.tsx", {
  "@/lib/tournamentRegistrationApi": api,
  "@/lib/tournamentRegistrationEligibility": load("lib/tournamentRegistrationEligibility.ts", {
    "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts")
  }),
  "@/lib/tournamentTeamCompetitionApi": {},
  "@/components/tournaments/TournamentPartnerDetails": { default: () => null },
  "@/components/tournaments/FourPlayerTeamRegistrationCard": { default: () => null, TEAM_SLOTS: [] },
  "./EditLinkRequestForm": { default: EditForm },
  "./TournamentCommerceChooser": { default: () => null }
}).default;
const pageData = { tournament: { id: "fixture", name: "Test tournament" }, registration_open: true, settings: {}, events: [], days: [] };
const Page = load(registration + "page.tsx", {
  "@/components/PublicTournamentSponsors": { default: () => null },
  "@/components/PublicTournamentNav": { default: () => null },
  "@/components/PublicClubLink": { default: ({ children }) => children },
  "@/lib/tournamentRegistrationApi": { getClubTournamentRegistration: async () => ({ data: pageData }) },
  "./TournamentRegistrationForm": { default: Form }
}).default;
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");

async function verifyEditLinkOutcomes() {
  const previousFormData = global.FormData;
  global.FormData = class { get() { return ""; } };
  const calls = [];
  let outcome = { error: { message: "Request timed out" } };
  api.requestClubTournamentRegistrationEditLink = async (_club, payload) => {
    calls.push(payload);
    if (outcome instanceof Error) throw outcome;
    return outcome;
  };
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(EditForm, { clubSlug: "fixture", tournamentId: "fixture", initialEmail: "player@example.com" })); });
    const submit = () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} });
    const button = () => renderer.root.findAllByType("button")[0];
    await act(submit);
    assert.match(content(renderer.toJSON()), /couldn’t confirm your request/);
    assert.match(content(renderer.toJSON()), /contact tournament staff/);
    assert.equal(renderer.root.findAllByProps({ role: "status" }).length, 0);
    assert.equal(button().props.disabled, false);

    outcome = new Error("Connection lost");
    await act(submit);
    assert.equal(button().props.disabled, false, "a rejected request cannot leave Sending stuck");
    assert.equal(calls[1].idempotency_key, calls[0].idempotency_key, "an uncertain HTTP retry reuses its request identity");

    outcome = { data: { ok: true, accepted: true } };
    await act(submit);
    assert.match(content(renderer.toJSON()), /Request received for/);
    assert.match(content(renderer.toJSON()), /If this address matches/);
    assert.match(content(renderer.toJSON()), /If no link arrives within a few minutes/);
    assert.doesNotMatch(content(renderer.toJSON()), /we.ve sent|has been sent|will arrive shortly/i);

    await act(submit);
    assert.notEqual(calls[3].idempotency_key, calls[2].idempotency_key, "a fresh acknowledged request gets a new identity for safe rejection recovery");

    let finish;
    api.requestClubTournamentRegistrationEditLink = (_club, payload) => {
      calls.push(payload);
      return new Promise(resolve => { finish = resolve; });
    };
    const count = calls.length;
    await act(async () => {
      const first = submit();
      const duplicate = submit();
      assert.equal(calls.length, count + 1, "rapid duplicate submissions share the in-flight request");
      finish(outcome);
      await Promise.all([first, duplicate]);
    });
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.FormData = previousFormData;
  }
}

async function main() {
  const previousWindow = global.window;
  const listeners = new Map();
  global.window = { location: { hash: "" }, addEventListener: (name, fn) => listeners.set(name, fn), removeEventListener: name => listeners.delete(name) };
  let renderer;
  try {
    await act(async () => { renderer = create(await Page({ params: { clubSlug: "fixture" } })); });
    const forms = () => renderer.root.findAllByProps({ "data-testid": "registration-edit-link-form" });
    const button = name => renderer.root.findAllByType("button").find(b => content(b) === name);
    assert.equal(forms().length, 0, "chooser does not show an extra email form");
    await act(async () => button("Edit my registration").props.onClick());
    assert.equal(forms().length, 1, "full page contains exactly one recovery form");
    await act(async () => button("Back to registration choices").props.onClick());
    assert.equal(forms().length, 0);
    await act(async () => {
      window.location.hash = "#manage-registration";
      listeners.get("hashchange")();
    });
    assert.equal(forms().length, 1, "existing confirmation anchor opens recovery");
    await act(async () => renderer.unmount());
    assert.equal(listeners.size, 0);

    pageData.registration_open = false;
    await act(async () => { renderer = create(await Page({ params: { clubSlug: "fixture" } })); });
    assert.equal(forms().length, 1, "direct anchor works when new registrations are closed");
    await act(async () => button("Back to registration choices").props.onClick());
    assert.equal(button("Start a registration"), undefined);
    await act(async () => button("Edit my registration").props.onClick());
    assert.equal(forms().length, 1);
    await verifyEditLinkOutcomes();
    console.log("Tournament registration recovery: navigation, honest acknowledgements, failed requests, retry identity and duplicate suppression passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.window = previousWindow;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
