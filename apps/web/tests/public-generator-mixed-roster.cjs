const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const routes = [], router = { push(url) { routes.push(url); }, refresh() {} };
const stubs = {
  "@/components/PublicClubLink": ({ children }) => React.createElement("a", null, children),
  "next/navigation": { useRouter: () => router }
};
function load(name, parent = root) {
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts"].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const Workspace = load("@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace").default;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

(async () => {
  const storage = new Map();
  global.sessionStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  global.window = { sessionStorage };
  const names = ["Manuel", "Jose", "Victor", "Chay", "Beto"];
  const linked = { Jose: 102, Chay: 104 }, requests = [];
  let failure = null;
  const preview = {
    name: "3.5+ RR", generatorKind: "round_robin", playFormat: "doubles", totalRounds: 5,
    courtCount: 1, previewFingerprint: "preview-fingerprint",
    participants: names.map((name, i) => ({ id: `p-${i + 1}`, name, player_id: linked[name] })),
    rounds: names.map((_, i) => ({ number: i + 1, status: "preview", matches: [], byeParticipantIds: [`p-${i + 1}`] }))
  };
  global.fetch = async (url, options = {}) => {
    if (url.includes("/players?")) return reply({ players: [{ id: 102, name: "Jose" }, { id: 104, name: "Chay" }] });
    if (!options.body) return reply({ sessions: [] });
    const body = JSON.parse(options.body);
    requests.push({ url, body });
    assert.deepEqual(body.participant_names, names);
    assert.deepEqual(body.participant_player_ids, linked);
    assert.equal(body.total_rounds, 5);
    assert.equal(body.court_count, 1);
    if (url.endsWith("/preview")) return failure || reply({ ok: true, preview });
    assert.equal(body.preview_fingerprint, preview.previewFingerprint);
    return reply({ session: { session_key: "mixed-roster", current_round_number: 1 }, edit_token: "organizer-token" });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Workspace, { apiBase: "https://api.test", clubId: "club", generatorKind: "round_robin", status: { enabled: true, writes_enabled: true } })); });
  const count = tree.root.findAllByType("label").find(node => text(node).startsWith("Number of players")).findByType("select");
  await act(async () => count.props.onChange({ target: { value: "5" } }));
  for (const name of ["Jose", "Chay"]) {
    await act(async () => tree.root.findByProps({ placeholder: "Type at least 2 letters, then add a player" }).props.onChange({ target: { value: name } }));
    await act(async () => button(tree, "Add").props.onClick());
  }
  await act(async () => tree.root.findByType("textarea").props.onChange({ target: { value: names.join("\n") } }));
  assert.match(text(tree.toJSON()), /2 chosen from the club list · 3 entered manually/);
  assert.equal(button(tree, "Preview matchups").props.disabled, false);

  failure = reply({ detail: "Choose Jose from the player list again." }, 400);
  await act(async () => button(tree, "Preview matchups").props.onClick());
  assert.match(text(tree.toJSON()), /Choose Jose from the player list again\./);
  assert.equal(tree.root.findByType("textarea").props.value, names.join("\n"));
  for (const status of [422, 500]) {
    failure = reply({ detail: "internal database information" }, status);
    await act(async () => button(tree, "Preview matchups").props.onClick());
    assert.match(text(tree.toJSON()), /We couldn't complete that request/);
    assert.doesNotMatch(text(tree.toJSON()), /internal database information/);
  }
  failure = null;
  await act(async () => button(tree, "Preview matchups").props.onClick());
  assert.match(text(tree.toJSON()), /Preview ready: 5 rounds/);
  await act(async () => button(tree, "Start unrated session").props.onClick());
  assert.equal(requests.length, 5);
  assert.equal(routes.length, 1);
  assert.match(routes[0], /mixed-roster\/rounds\/1#edit=organizer-token$/);
  assert.equal(storage.get("public-generator-edit:club:mixed-roster"), "organizer-token");
  await act(async () => tree.unmount());
  console.log("PASS public mixed roster: club search + manual names, actionable validation, private server errors, preview and start.");
})().catch(error => { console.error(error); process.exit(1); });
