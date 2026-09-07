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

const registration = "app/clubs/[clubSlug]/tournament-registration/";
const api = { requestClubTournamentRegistrationEditLink: () => assert.fail("rendering must not send email") };
const EditForm = load(registration + "EditLinkRequestForm.tsx", { "@/lib/tournamentRegistrationApi": api }).default;
const Form = load(registration + "TournamentRegistrationForm.tsx", {
  "@/lib/tournamentRegistrationApi": api,
  "@/lib/tournamentRegistrationEligibility": load("lib/tournamentRegistrationEligibility.ts", {
    "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts")
  }),
  "@/lib/tournamentTeamCompetitionApi": {},
  "@/components/tournaments/FourPlayerTeamRegistrationCard": { default: () => null, TEAM_SLOTS: [] },
  "./EditLinkRequestForm": { default: EditForm },
  "./TournamentCommerceChooser": { default: () => null }
}).default;
const pageData = { tournament: { id: "fixture", name: "Test tournament" }, registration_open: true, settings: {}, events: [], days: [] };
const Page = load(registration + "page.tsx", {
  "@/components/PublicTournamentSponsors": { default: () => null },
  "@/components/PublicTournamentNav": { default: () => null },
  "next/link": { default: ({ children }) => children },
  "@/lib/tournamentRegistrationApi": { getClubTournamentRegistration: async () => ({ data: pageData }) },
  "./TournamentRegistrationForm": { default: Form }
}).default;
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");

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
    console.log("Tournament registration recovery: single form, existing anchors and closed registration passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.window = previousWindow;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
