const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const routes = [], router = { push(url) { routes.push(url); }, refresh() {} };
let authenticated = true;
const stubs = {
  "@/components/PublicClubLink": ({ children, ...props }) => React.createElement("a", props, children),
  "next/link": ({ children }) => React.createElement("a", null, children),
  "@/lib/useAdminSession": { useAdminSession: () => authenticated ? ({ accessToken: "test-admin", session: {capabilities: {assignments: [{club_id: "club", permissions: ["manage_players"]}]}} }) : ({accessToken: "", session: null}) },
  "next/navigation": { useRouter: () => router },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel }) => React.createElement("button", null, triggerLabel) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) }
};
function load(name, parent = root) {
  if (name === "@/components/PublicControls") return require("./helpers/public-controls-module.cjs")(name);
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
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const reply = body => ({ok: true, status: 200, json: async () => structuredClone(body)});
const storage = new Map();
global.sessionStorage = {getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)};
global.window = {sessionStorage};
const change = async (node, value) => act(async () => node.props.onChange({target: {value}}));
(async () => {
  for (const file of ["@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace", "@/app/admin/play-generators/GeneratorWorkspace"]) {
    const Workspace = load(file).default;
    for (const scenario of ["success", "duplicate", "lost"]) {
      storage.clear(); authenticated = true;
      const players = Array.from({length: 5}, (_, i) => ({id: i+1, name: `Player ${i+1}`}));
      const requests = [], operations = new Map();
      global.fetch = async (url, options = {}) => {
        requests.push({url, options});
        if (url.endsWith("/players/editor/players") && options.method === "POST") {
          const body = JSON.parse(options.body);
          assert.equal(options.headers.Authorization, "Bearer test-admin");
          assert.equal(body.starting_jupr, 3.75);
          assert.equal(body.source, "generator_roster_setup");
          const player = {id: 6, name: body.name}; players.push(player);
          operations.set(body.idempotency_key, {player});
          if (scenario === "lost") throw new TypeError("Response lost");
          return reply({player});
        }
        if (url.includes("/players/editor/operations/")) return reply({status: "completed", result: operations.get(decodeURIComponent(url.split("/operations/")[1]))});
        if (url.endsWith("/players/editor/players/6")) return reply({player: players.find(player => player.id === 6)});
        if (url.endsWith("/players/editor/players")) {
          if (scenario === "duplicate" && players.length === 5) players.push({id: 6, name: "New Player"});
          return reply({players});
        }
        if (url.includes("/players?")) return reply({players});
        if (url.endsWith("/preview")) {
          const body = JSON.parse(options.body);
          assert.equal(body.participant_player_ids["New Player"], 6);
          assert.equal(body.participant_names.length, 6);
          assert.equal(Object.hasOwn(body, "ratings"), false);
          return reply({preview: {name: "Moneyball", eventFormat: "moneyball_expectations", participants: [], rounds: [], totalRounds: 6, courtCount: 1}});
        }
        return reply({sessions: []});
      };
      const props = {apiBase: "https://api.test", clubId: file.includes("Public") ? "club-slug" : "club", generatorKind: "round_robin", status: {enabled: true, writes_enabled: true, club: {id: "club"}}};
      let tree;
      const mount = async () => act(async () => {tree = create(React.createElement(Workspace, props));});
      await mount();
      await change(tree.root.findByProps({"aria-label": "Event format"}), "moneyball_expectations");
      await change(tree.root.findAllByType("label").find(node => text(node).startsWith("Number of players")).findByType("select"), "6");
      for (const player of players.slice()) {
        await change(tree.root.findByProps({placeholder: "Type at least 2 letters, then add a player"}), player.name);
        await act(async () => button(tree, "Add").props.onClick());
      }
      await act(async () => button(tree, "Create new player").props.onClick());
      const name = () => tree.root.findByProps({"aria-label": "New player name for this lineup"});
      const level = () => tree.root.findByProps({"aria-label": "New player starting level for this lineup"});
      assert.equal(name().props.value, "", "The form starts with a blank name");
      await change(name(), "New Player");
      assert.equal(button(tree, "Create and select player").props.disabled, true);
      await change(level(), "7.5");
      assert.equal(button(tree, "Create and select player").props.disabled, true);
      await change(level(), "3.75");
      const submit = async () => act(async () => tree.root.findByType("form").props.onSubmit({preventDefault(){}}));
      await submit();
      if (scenario === "duplicate") {
        assert.equal(requests.filter(r => r.options.method === "POST").length, 0);
        await act(async () => button(tree, "Use New Player").props.onClick());
      }
      if (scenario === "lost") {
        assert.equal(button(tree, "Preview matchups").props.disabled, true);
        await act(async () => tree.unmount()); await mount();
        assert.equal(button(tree, "Check player creation").props.disabled, false, "Recovery stays usable after reload");
        assert.equal(tree.root.findByType("textarea").props.disabled, true);
        await submit();
      }
      assert.equal(tree.root.findByType("textarea").props.value, [...players.slice(0, 5).map(p => p.name), "New Player"].join("\n"));
      assert.equal(button(tree, "Preview matchups").props.disabled, false);
      await act(async () => button(tree, "Preview matchups").props.onClick());
      assert.equal(requests.filter(r => r.url.endsWith("/players/editor/players") && r.options.method === "POST").length, scenario === "duplicate" ? 0 : 1);
      assert.ok(requests.filter(r => r.url.includes("/admin/clubs/")).every(r => r.url.includes("/admin/clubs/club/")), "Public slugs resolve to the canonical club ID");
      assert.ok(![...storage.keys()].some(k => k.startsWith("jupr-generator-player-create:")));
      await act(async () => tree.unmount());
    }
    storage.clear(); authenticated = false;
    let signedOut;
    await act(async () => {signedOut = create(React.createElement(Workspace, {apiBase: "https://api.test", clubId: "club", generatorKind: "round_robin", status: {enabled: true, writes_enabled: true}}));});
    await change(signedOut.root.findByProps({"aria-label": "Event format"}), "moneyball_expectations");
    assert.equal(button(signedOut, "Create new player"), undefined);
    assert.match(text(signedOut.toJSON()), /Sign in as club staff/);
    await act(async () => signedOut.unmount());
  }
  console.log("PASS Moneyball setup: public/staff creation, required starting level, canonical club, roster preservation, duplicate reuse, single-insert refresh recovery, and signed-out guidance.");
})().catch(error => {console.error(error); process.exitCode = 1;});
