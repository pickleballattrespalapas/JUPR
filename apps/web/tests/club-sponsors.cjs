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
  const m = new Module(filename, module);
  m.filename = filename; m.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = m.require.bind(m); m.require = name => overrides[name] || original(name);
  m._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), { compilerOptions: {
    esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022
  } }).outputText, filename);
  return m.exports;
}
const interaction = load("components/interaction/types.ts");
const FormDialog = ({ children }) => children;
const Editor = load("app/admin/tournaments/setup/TournamentSponsorEditor.tsx", {
  "@/components/interaction": { ...interaction, FormDialog },
  "@/components/ConfirmAction": { ConfirmAction: () => null },
  "@/components/TournamentSponsorDisplay": { __esModule: true, default: () => null },
  "@/lib/tournamentSponsors": load("lib/tournamentSponsors.ts")
}).default;

async function main() {
  const profile = { id: "saved-homes", name: "Homes and Land of Baja", website: "https://example.com", public_description: "Local homes.",
    logo_path: "club/old/logo.webp", logo_url: "https://example.com/logo", contact_name: "Private contact", contact_email: "contact@example.com", contact_phone: "123", updated_at: "v1" };
  let rows = [], sharedSaves = [], draftSaves = [], uploads = 0, failDraft = false;
  let renderer;
  const props = () => ({ sponsors: rows, tournamentName: "New tournament", disabled: false,
    onListSaved: async () => ({ sponsors: [profile] }),
    onUseSaved: async id => ({ sponsor: { id, club_sponsor_id: id, name: profile.name, website: profile.website, public_description: profile.public_description,
      logo_path: "club/new/copied.webp", logo_url: profile.logo_url, tier: "supporting", level: "", notes: "", is_visible: true } }),
    onSaveShared: async (id, value) => { sharedSaves.push({ id, ...value }); return { sponsor: { ...profile, ...value, id, updated_at: "v2" } }; },
    onUpload: async () => { uploads++; throw Error("No upload should be needed"); },
    onSave: async value => { draftSaves.push(value); if (failDraft) throw Error("Draft failed"); rows = value; return true; }
  });
  await act(async () => { renderer = create(React.createElement(Editor, props())); });
  const root = renderer.root;
  const button = text => root.findAllByType("button").find(b => b.props.children === text);
  const dialog = () => root.findByType(FormDialog);
  const checkShared = () => root.findAllByType("input").filter(n => n.props.type === "checkbox")[1];
  try {
    await act(async () => button("Add sponsor").props.onClick());
    const picker = root.findAllByType("select")[0];
    assert.equal(picker.findAllByType("option")[1].props.children[0], profile.name);
    await act(async () => picker.props.onChange({ target: { value: profile.id } }));
    assert.equal(checkShared().props.checked, false, "Reusing a sponsor does not overwrite the shared profile");
    await act(async () => dialog().props.onSubmit());
    assert.equal(sharedSaves.length, 0);
    assert.equal(uploads, 0);
    assert.equal(rows[0].logo_path, "club/new/copied.webp");
    assert.equal(rows[0].club_sponsor_id, profile.id);
    assert.ok(!("contact_email" in rows[0]));
    await act(async () => { dialog().props.onAcknowledge(); renderer.update(React.createElement(Editor, props())); });
    await act(async () => button("Add sponsor").props.onClick());
    assert.equal(root.findAllByType("select")[0].findAllByType("option")[1].props.disabled, true, "Already-added sponsor cannot be added twice");
    assert.equal(checkShared().props.checked, true, "New sponsors are saved for future use by default");
    await act(async () => root.findAllByType("input").find(n => n.props.maxLength === 120 && n.props.required).props.onChange({ target: { value: "Los Barriles Life" } }));
    await act(async () => root.findAllByType("input").find(n => n.props.type === "email").props.onChange({ target: { value: "private@life.example" } }));
    failDraft = true;
    await act(async () => { await assert.rejects(dialog().props.onSubmit(), /Draft failed/); });
    const retryId = sharedSaves[0].id;
    assert.ok(retryId);
    failDraft = false;
    await act(async () => dialog().props.onSubmit());
    assert.equal(sharedSaves[1].id, retryId, "Retry preserves saved profile identity");
    assert.equal(sharedSaves[1].expected_updated_at, "v2");
    assert.equal(sharedSaves[0].contact_email, "private@life.example");
    assert.ok(!("contact_email" in draftSaves.at(-1).at(-1)), "Contacts only go to the club profile");
    assert.equal(rows.length, 2);
  } finally { act(() => renderer.unmount()); }
  console.log("Club sponsor checks passed: reuse, private contacts, duplicate prevention, and save retry.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
