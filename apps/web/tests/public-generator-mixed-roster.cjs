const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const routes = [], router = { push(url) { routes.push(url); }, refresh() {} };
const stubs = {
  "@/components/PublicClubLink": ({ children, ...props }) => React.createElement("a", props, children),
  "next/link": ({ children }) => React.createElement("a", null, children),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
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
    assert.equal(body.court_count, 0, "Automatic courts must stay automatic when the roster grows");
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
  assert.match(text(tree.toJSON()), /Starting schedule: 5 rounds/);
  await act(async () => button(tree, "Start unrated session").props.onClick());
  assert.equal(requests.length, 5);
  assert.equal(routes.length, 1);
  assert.match(routes[0], /mixed-roster\/rounds\/1#edit=organizer-token$/);
  assert.equal(storage.get("public-generator-edit:club:mixed-roster"), "organizer-token");
  await act(async () => tree.unmount());
  console.log("PASS public mixed roster: club search + manual names, actionable validation, private server errors, preview and start.");

  // Exercise actual preview/start requests in both workspaces. Mixed-format and
  // ladder allocations remain explicit; ordinary Round-Robins stay automatic.
  for (const file of ["@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace", "@/app/admin/play-generators/GeneratorWorkspace"]) {
    const Setup = load(file).default;
    for (const [kind, format, count, courts] of [["round_robin", "doubles", 6, 0], ["round_robin", "singles", 4, 0], ["round_robin", "doubles_singles", 6, 2], ["ladder", "doubles", 5, 1]]) {
      storage.clear();
      const bodies = [], historyRequests = [];
      global.fetch = async (_url, options = {}) => {
        if (!options.body) {
          if (_url.includes("/sessions?")) historyRequests.push(_url);
          return reply({ sessions: [{ session_key: "previous-session", title: "Previous club session", play_format: "doubles", status: "completed", current_round_number: 3 }], players: [] });
        }
        const body = JSON.parse(options.body);
        bodies.push(body);
        return _url.endsWith("/preview")
          ? reply({ preview: { ...preview, generatorKind: kind, playFormat: format, participants: [], rounds: [] } })
          : reply({ session: { session_key: "auto-courts", current_round_number: 1 }, edit_token: "test-organizer" });
      };
      await act(async () => { tree = create(React.createElement(Setup, { apiBase: "https://api.test", clubId: "club", generatorKind: kind, status: { enabled: true, writes_enabled: true } })); });
      const isPublic = file.includes("/clubs/");
      assert.equal(historyRequests.length, 1, "Both workspaces fetch saved sessions; public setup shows only this browser’s organizer sessions");
      assert.equal(text(tree.toJSON()).includes("Previous club session"), !isPublic, "Previous sessions remain in admin setup only");
      if (isPublic) assert.doesNotMatch(text(tree.toJSON()), /Recent sessions/);
      const labelSelect = label => tree.root.findAllByType("label").find(node => text(node).startsWith(label)).findByType("select");
      await act(async () => labelSelect("Play format").props.onChange({ target: { value: format } }));
      await act(async () => labelSelect("Number of players").props.onChange({ target: { value: String(count) } }));
      await act(async () => tree.root.findByType("textarea").props.onChange({ target: { value: Array.from({ length: count }, (_, i) => `Player ${i + 1}`).join("\n") } }));
      await act(async () => button(tree, "Preview matchups").props.onClick());
      await act(async () => button(tree, "Start unrated session").props.onClick());
      assert.equal(bodies.length, 2, `${file}: preview and start both reach the API`);
      for (const body of bodies) assert.equal(body.court_count, courts, `${file}: ${kind}/${format} court allocation`);
      await act(async () => tree.unmount());
    }
  }
  console.log("PASS public/admin preview and start: automatic Round-Robin courts, explicit ladder and mixed-format courts.");

  // A navigation failure after a confirmed start still has a direct link to that
  // session, without depending on a public history list or creating a duplicate.
  storage.clear();
  let starts = 0;
  global.fetch = async (_url, options = {}) => {
    if (!options.body) return reply({ players: [] });
    starts++;
    return reply({ session: { session_key: "just-started", current_round_number: 1 }, edit_token: "test-organizer" });
  };
  storage.set("public-play-generator-create:club:round_robin", JSON.stringify({ idempotency_key: "pending-create-123" }));
  router.push = () => { throw new Error("Navigation unavailable"); };
  await act(async () => { tree = create(React.createElement(Workspace, { apiBase: "https://api.test", clubId: "club", generatorKind: "round_robin", status: { enabled: true, writes_enabled: true } })); });
  await act(async () => button(tree, "Try starting again").props.onClick());
  assert.equal(starts, 1);
  assert.match(text(tree.toJSON()), /The session started, but it didn't open automatically/);
  const sessionLink = tree.root.findAllByType("a").find(node => text(node).includes("Open the session you just started"));
  assert.match(sessionLink.props.href, /just-started\/rounds\/1#edit=test-organizer$/);
  assert.doesNotMatch(text(tree.toJSON()), /Recent sessions/);
  assert.equal(button(tree, "Try starting again"), undefined);
  await act(async () => tree.unmount());

  // A remembered Team Match session must not appear on public setup, including
  // after a fresh visit with an existing browser-local session.
  const teamState = load("@/app/clubs/[clubSlug]/team-match-generator/teamMatchState");
  const TeamSetup = load("@/app/clubs/[clubSlug]/team-match-generator/TeamMatchGenerator").default;
  storage.clear();
  storage.set(teamState.lastTeamMatchSessionKey("club"), "old-team-session");
  global.window.localStorage = global.sessionStorage;
  await act(async () => { tree = create(React.createElement(TeamSetup, { clubId: "club" })); });
  assert.doesNotMatch(text(tree.toJSON()), /Resume most recent session|Recent sessions/);
  assert.equal(tree.root.findAllByType("a").length, 0);
  assert.equal(storage.get(teamState.lastTeamMatchSessionKey("club")), "old-team-session", "Hiding history never deletes previous sessions");
  await act(async () => tree.unmount());
  console.log("PASS public history hidden for all three generators, admin history preserved, and confirmed start navigation recovery.");
})().catch(error => { console.error(error); process.exit(1); });
