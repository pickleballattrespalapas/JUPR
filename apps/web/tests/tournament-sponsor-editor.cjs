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
const { tournamentSetupActionError } = load("lib/tournamentSetupActionError.ts", {
  "@/components/interaction/types": interaction
});
const FormDialog = ({ children }) => children;
const Display = () => null;
const Editor = load("app/admin/tournaments/setup/TournamentSponsorEditor.tsx", {
  "@/components/interaction": { ...interaction, FormDialog },
  "@/components/ConfirmAction": { ConfirmAction: () => null },
  "@/components/TournamentSponsorDisplay": { default: Display, __esModule: true },
  "@/lib/tournamentSponsors": load("lib/tournamentSponsors.ts")
}).default;
const rejection = tournamentSetupActionError({ status: 409, headers: new Headers() }, {
  detail: "Tournament Admin data changed after it was loaded. Reload the authoritative detail, review the impact, and submit again."
});
assert.equal(rejection.kind, "conflict");
assert.equal(interaction.normalizeInteractionActionError(rejection), rejection);
assert.match(tournamentSetupActionError({ status: 401, headers: new Headers() }, {}).message, /sign-in expired/);
const serverError = tournamentSetupActionError({ status: 500, headers: new Headers({ "x-request-id": "request-123" }) }, { detail: "private database error" });
assert.match(serverError.message, /HTTP 500.*request-123/);
assert.doesNotMatch(serverError.message, /private database/);

async function main() {
  const originalFileReader = global.FileReader;
  const originalCreate = URL.createObjectURL;
  const originalRevoke = URL.revokeObjectURL;
  URL.createObjectURL = () => "blob:local-logo";
  URL.revokeObjectURL = () => {};
  global.FileReader = class {
    readAsDataURL() { this.result = "data:image/png;base64,aW1hZ2U="; this.onload(); }
  };
  let renderer;
  try {
    let uploads = 0;
    let rejectSave = true;
    const saves = [];
    const sponsor = { id: "baja", name: "Homes and Land of Baja", tier: "supporting", level: "", website: "homesandlandofbaja.com", notes: "", logo_path: "", is_visible: true };
    await act(async () => {
      renderer = create(React.createElement(Editor, {
        sponsors: [sponsor], tournamentName: "Baja Classic 2026", disabled: false,
        onUpload: async () => { uploads++; return { logo_path: "club/tournament/logo.webp", logo_url: "https://example.com/logo.webp" }; },
        onSave: async rows => { saves.push(rows); if (rejectSave) throw rejection; return true; }
      }));
    });
    const root = renderer.root;
    await act(async () => root.findAllByType("button").find(b => b.props.children === "Edit").props.onClick());
    await act(async () => root.findByType("select").props.onChange({ target: { value: "premier" } }));
    await act(async () => root.findAllByType("input").find(i => i.props.type === "file").props.onChange({ target: { files: [{ type: "image/png", size: 100 }] } }));
    assert.equal(root.findByType(Display).props.placement, "footer");
    assert.equal(root.findByType(Display).props.sponsors[0].tier, "premier");
    await act(async () => {
      await assert.rejects(root.findByType(FormDialog).props.onSubmit(), error => error === rejection);
    });
    assert.equal(uploads, 1);
    assert.equal(root.findByType(FormDialog).props.open, true, "Failure keeps the editor open");
    assert.equal(root.findByType(Display).props.sponsors[0].logo_url, "https://example.com/logo.webp");
    rejectSave = false;
    let result;
    await act(async () => { result = await root.findByType(FormDialog).props.onSubmit(); });
    assert.equal(result.status, "success");
    assert.equal(uploads, 1, "Retry must reuse the already uploaded logo");
    assert.equal(saves[1][0].name, sponsor.name);
    assert.equal(saves[1][0].tier, "premier");
    assert.equal(saves[1][0].logo_path, "club/tournament/logo.webp");
    assert.equal(saves[1][0].website, "https://homesandlandofbaja.com/");
    await act(async () => root.findByType("select").props.onChange({ target: { value: "presenting" } }));
    assert.equal(root.findByType(Display).props.placement, "header");
    await act(async () => root.findAllByType("input").find(i => i.props.inputMode === "url").props.onChange({ target: { value: "javascript:alert(1)" } }));
    await act(async () => {
      await assert.rejects(root.findByType(FormDialog).props.onSubmit(), error => error instanceof interaction.InteractionActionError && error.kind === "validation");
    });
    assert.equal(saves.length, 2, "Invalid website is rejected before saving");
  } finally {
    if (renderer) act(() => renderer.unmount());
    global.FileReader = originalFileReader;
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
  }
  console.log("Sponsor editor checks passed: actionable errors, tier preview, preserved edits, and upload reuse on retry.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
