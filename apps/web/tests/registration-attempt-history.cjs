const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");

function load(relative) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = compiled.require.bind(compiled);
  compiled.require = name => name === "@/lib/useAuthenticatedAutoLoad" ? load("lib/useAuthenticatedAutoLoad.ts") : original(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}

async function main() {
  const prior = { window: global.window, document: global.document, fetch: global.fetch };
  const listeners = new Map();
  global.window = { addEventListener: (k,v) => listeners.set(k,v), removeEventListener: k => listeners.delete(k) };
  global.document = { visibilityState: "visible", addEventListener: (k,v) => listeners.set(k,v), removeEventListener: k => listeners.delete(k) };
  const requests = [];
  global.fetch = async (url, init) => { requests.push({url, init, body: JSON.parse(init.body)}); throw new Error("offline"); };
  process.env.NEXT_PUBLIC_JUPR_API_BASE_URL = "http://fixture.invalid";
  const { useRegistrationAttempt } = load("lib/useRegistrationAttempt.ts");
  let id;
  const Recorder = props => { id = useRegistrationAttempt(props); return null; };
  const snapshot = { clubSlug: "fixture", tournamentId: "fixture-tournament", active: true, flow: "new", stage: "events",
    name: "Terri Fixture", email: "terri@example.invalid", error: "A player cannot register themselves as their own partner.",
    selections: [{event_option_id: "mixed35", partner_name: "Brent Fixture", same_email_as_player: true}],
    phone: "private-phone", edit_token: "private-edit", notes: "private-note" };
  let renderer;
  try {
    await act(async () => { renderer = create(React.createElement(Recorder,snapshot)); await new Promise(resolve => setTimeout(resolve,25)); });
    assert.equal(requests.length,1);
    const rejected = requests[0];
    assert.equal(rejected.body.attempt_id,id);
    assert.equal(rejected.body.event,"CLIENT_REJECTED");
    assert.equal(rejected.body.selections[0].same_email_as_player,true);
    assert.ok(!JSON.stringify(rejected.body).includes("private-"));
    assert.equal(rejected.init.keepalive,true);
    listeners.get("pagehide")();
    assert.equal(requests.at(-1).body.event,"LEFT_FORM");
    await act(async () => { renderer.update(React.createElement(Recorder,{...snapshot,completed:true})); });
    const count = requests.length;
    await act(async () => renderer.unmount());
    assert.equal(requests.length,count,"Completion must not be recorded as an abandoned attempt");
    const History = load("components/tournaments/RegistrationAttemptHistory.tsx").default;
    global.fetch = async () => ({ ok: true, json: async () => ({ attempts: [{attempt_id:"fixture",name:"Terri Fixture",email:"terri@example.invalid",outcome:"In progress / incomplete",stage:"events",last_seen_at:new Date().toISOString(),events:[]}] }) });
    await act(async () => { renderer = create(React.createElement(History,{ apiBase:"http://fixture.invalid",clubId:"club",tournamentId:"fixture",accessToken:"fixture-token" })); });
    await act(async () => renderer.root.findByType("form").props.onSubmit({preventDefault(){}}));
    assert.ok(JSON.stringify(renderer.toJSON()).includes("terri@example.invalid"));
    await act(async () => renderer.update(React.createElement(History,{apiBase:"http://fixture.invalid",clubId:"other-club",tournamentId:"fixture",accessToken:null})));
    assert.ok(!JSON.stringify(renderer.toJSON()).includes("terri@example.invalid"),"Private history must clear when the staff session/scope changes");
    await act(async () => renderer.unmount());
    console.log("Registration attempt history: rejection capture, safe snapshot, offline tolerance, page exit, completion and private staff scope passed.");
  } finally {
    global.window=prior.window; global.document=prior.document; global.fetch=prior.fetch;
    delete process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
