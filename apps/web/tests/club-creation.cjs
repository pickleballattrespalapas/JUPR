const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path");
const ts = require("typescript"), React = require("react");
const { act, create } = require("react-test-renderer");
const storage = new Map();
let restored, emailEnabled, network, signIns, redirects, authorized, tree, failRefresh, failAuthorize, createReply, signupReply;
const session = { access_token: "test-token", user: { id: "admin", email: "test@example.invalid" } };
const club = { club_id: "baja-pickleball", club_slug: "baja-pickleball", club_name: "Baja Pickleball", roles: ["administrator"] };
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
global.window = {
  location: { origin: "https://staging.example.test" },
  localStorage: {
    getItem: key => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key),
  },
};
const mocks = {
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "@/lib/adminAuthClient": {
    getAdminApiBaseUrl: () => "https://api.example.test",
    getAdminAuthConfig: () => ({ supabaseUrl: "https://auth.example.test", supabaseAnonKey: "public-test-key" }),
    consumeStaffInvitationSession: async () => restored,
    refreshAdminSession: async current => { if (failRefresh) throw new Error("expired"); return current; },
    signInWithPassword: async (email, password) => { signIns.push({ email, password }); return session; },
    authorizeAndSaveAdminSession: async (current, clubId) => {
      authorized.push({ current, clubId });
      if (failAuthorize) throw new Error("Admin access could not be verified. Try again.");
    },
  },
  "@/lib/adminWorkspace": { selectAdminWorkspace: workspace => redirects.push(workspace) },
};
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  new Function("require", "module", "exports", code)(name => {
    if (name.endsWith(".css")) return {};
    if (Object.hasOwn(mocks, name)) return mocks[name];
    if (name.startsWith("@/")) return load(name.slice(2) + ".ts");
    return require(name);
  }, module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const helpers = load("lib/clubCreationDraft.ts");
const Page = load("app/create-club/page.tsx").default;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = label => tree.root.findAllByType("button").find(node => text(node) === label);
const writes = () => network.filter(item => item.options.method === "POST");
const fill = async (id, value) => act(async () => tree.root.findByProps({ id }).props.onChange({ target: { value } }));
const click = async label => act(async () => button(label).props.onClick());
const submit = async () => act(async () => tree.root.findByType("form").props.onSubmit({ preventDefault() {} }));
const mount = async () => act(async () => { tree = create(React.createElement(Page)); });
const unmount = async () => act(async () => tree.unmount());
async function start(options = {}) {
  storage.clear(); restored = options.session || null; emailEnabled = options.emailEnabled || false;
  network = []; signIns = []; redirects = []; authorized = []; failRefresh = false; failAuthorize = false;
  createReply = response(club, 201); signupReply = response({});
  global.fetch = async (url, options = {}) => {
    network.push({ url, options });
    if (url.endsWith("signup-options")) return response({ email_enabled: emailEnabled });
    if (url.includes("/auth/v1/signup")) return signupReply;
    if (url.endsWith("/clubs/create")) return createReply;
    throw new Error(`Unexpected request: ${url}`);
  };
  await mount();
}
async function clubDetails() {
  await fill("club-name", "Baja Pickleball");
  assert.equal(tree.root.findByProps({ id: "club-address" }).props.value, "baja-pickleball");
  await submit();
}
async function signIn() {
  await act(async () => tree.root.findByProps({ type: "email" }).props.onChange({ target: { value: session.user.email } }));
  await act(async () => tree.root.findByProps({ type: "password" }).props.onChange({ target: { value: "test-password" } }));
  await submit();
}

(async () => {
  await start();
  assert.equal(tree.root.findAllByProps({ type: "email" }).length, 0, "First screen is club creation, not sign-in");
  assert.equal(tree.root.findAllByProps({ type: "password" }).length, 0);
  await fill("club-name", "   ");
  await submit();
  assert.match(text(tree.toJSON()), /Enter a club name/);
  await clubDetails();
  assert.match(text(tree.toJSON()), /existing test administrator/);
  assert.equal(button("New administrator").props.disabled, true, "Staging never enables account email");
  assert.equal(writes().length, 0, "No account or club is created by moving between steps");
  await click("← Club details");
  await fill("club-address", "baja-custom");
  await fill("club-name", "Baja Travelers");
  assert.equal(tree.root.findByProps({ id: "club-address" }).props.value, "baja-custom", "Custom address survives name edits");
  await submit();
  await unmount(); await mount();
  assert.match(text(tree.toJSON()), /Baja Travelers/);
  await click("← Club details");
  assert.equal(tree.root.findByProps({ id: "club-name" }).props.value, "Baja Travelers");
  assert.equal(tree.root.findByProps({ id: "club-address" }).props.value, "baja-custom");
  await submit(); await signIn();
  assert.match(text(tree.toJSON()), /Website visibilityUnpublished/);
  assert.equal(writes().length, 0, "Sign-in only opens review; it never silently creates a club");
  const saved = storage.get(helpers.CLUB_CREATION_DRAFT_KEY);
  assert.ok(!saved.includes("test-password") && !saved.includes("test-token") && !saved.includes(session.user.email));
  let finish;
  createReply = new Promise(resolve => { finish = resolve; });
  await act(async () => { button("Create club →").props.onClick(); button("Create club →").props.onClick(); });
  assert.equal(writes().length, 1, "Duplicate creation clicks are locked");
  assert.deepEqual(JSON.parse(writes()[0].options.body), { name: "Baja Travelers", slug: "baja-custom" });
  assert.equal(writes()[0].options.headers.Authorization, "Bearer test-token");
  await act(async () => finish(response(club, 201)));
  assert.deepEqual(redirects, [club]);
  assert.equal(storage.has(helpers.CLUB_CREATION_DRAFT_KEY), false, "Successful creation clears the local draft");
  await unmount();

  await start({ session }); await clubDetails();
  assert.equal(text(tree.root.findByProps({ id: "creation-step" })), "Administrator", "A signed-in administrator still sees Step 2");
  assert.match(text(tree.toJSON()), /test@example.invalid/);
  assert.equal(tree.root.findAllByProps({ type: "password" }).length, 0, "Existing accounts need confirmation, not another password");
  await unmount(); await mount();
  assert.equal(text(tree.root.findByProps({ id: "creation-step" })), "Administrator", "Refreshing Step 2 does not skip it");
  assert.equal(writes().length, 0);
  await click("Use this account →");
  await click("← Administrator");
  assert.equal(text(tree.root.findByProps({ id: "creation-step" })), "Administrator", "Review can return to administrator confirmation");
  await click("Use this account →");
  createReply = response({ detail: "That club address is already in use. Choose another address." }, 409);
  await click("Create club →");
  assert.equal(tree.root.findByProps({ id: "club-name" }).props.value, "Baja Pickleball");
  assert.match(text(tree.toJSON()), /already in use/);
  await fill("club-address", "another-club"); await submit();
  await click("Use this account →");
  failRefresh = true;
  await click("Create club →");
  assert.match(text(tree.toJSON()), /Sign in again/);
  assert.match(text(tree.toJSON()), /another-club/);
  assert.equal(writes().length, 1, "An expired session cannot send a creation request");
  await unmount();

  await start({ session }); await clubDetails();
  await click("Use this account →");
  failAuthorize = true;
  await click("Create club →");
  assert.match(text(tree.toJSON()), /Your club was created/);
  failAuthorize = false;
  await click("Open your new club →");
  assert.equal(writes().length, 1, "Opening a created club after a transient error does not create it twice");
  assert.deepEqual(redirects, [club]);
  await unmount();

  await start({ emailEnabled: true }); await clubDetails();
  assert.equal(button("New administrator").props["aria-pressed"], true, "New administrators are the default signup path");
  await act(async () => tree.root.findByProps({ type: "email" }).props.onChange({ target: { value: session.user.email } }));
  await act(async () => tree.root.findAllByProps({ type: "password" })[0].props.onChange({ target: { value: "new-password" } }));
  await act(async () => tree.root.findAllByProps({ type: "password" })[1].props.onChange({ target: { value: "not-matching" } }));
  await submit();
  assert.equal(writes().length, 0);
  await act(async () => tree.root.findAllByProps({ type: "password" })[1].props.onChange({ target: { value: "new-password" } }));
  await submit();
  assert.match(text(tree.toJSON()), /Check your email/);
  assert.match(text(tree.toJSON()), /Your club has not been created yet/);
  assert.equal(writes().length, 1);
  assert.equal(new URL(writes()[0].url).searchParams.get("redirect_to"), "https://staging.example.test/create-club");
  assert.equal(tree.root.findAllByProps({ type: "password" }).length, 0);
  await unmount(); restored = session; await mount();
  assert.match(text(tree.toJSON()), /Baja Pickleball/);
  assert.ok(button("Use this account →"), "Verification callback returns to administrator confirmation");
  await click("Use this account →");
  assert.ok(button("Create club →"), "Verification callback resumes review with club details");
  assert.equal(writes().length, 1, "Verification does not auto-create the club");
  await unmount();

  await start({ emailEnabled: true }); await clubDetails();
  signupReply = response({ ...session, expires_in: 3600 });
  await act(async () => tree.root.findByProps({ type: "email" }).props.onChange({ target: { value: session.user.email } }));
  await act(async () => { for (const input of tree.root.findAllByProps({ type: "password" })) input.props.onChange({ target: { value: "new-password" } }); });
  await submit(); await click("Create club →");
  assert.ok(authorized[0].current.expires_at > Date.now(), "Signup session expiry uses the app's milliseconds format");
  await unmount();

  assert.equal(helpers.suggestClubSlug("Club Los Barrilés!"), "club-los-barriles");
  storage.set(helpers.CLUB_CREATION_DRAFT_KEY, "bad-json");
  assert.equal(helpers.readClubCreationDraft(), null);
  helpers.saveClubCreationDraft({ name: "Old", slug: "old", slugEdited: false, step: 2, access_token: "must-not-persist" });
  assert.ok(!storage.get(helpers.CLUB_CREATION_DRAFT_KEY).includes("must-not-persist"));
  storage.set(helpers.CLUB_CREATION_DRAFT_KEY, JSON.stringify({ ...JSON.parse(storage.get(helpers.CLUB_CREATION_DRAFT_KEY)), savedAt: Date.now() - 25 * 60 * 60 * 1000 }));
  assert.equal(helpers.readClubCreationDraft(), null, "Old local drafts expire");
  const original = window.localStorage;
  window.localStorage = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() {} };
  await start(); await clubDetails();
  assert.match(text(tree.toJSON()), /Keep this tab open/);
  await signIn(); await click("Create club →");
  assert.deepEqual(redirects, [club], "Blocked browser storage does not block in-tab creation");
  await unmount(); window.localStorage = original;
  console.log("Club creation: club-first entry, validation, suggested/custom address, persisted details, account setup, verification return, explicit review, expired sessions, conflicts and duplicate protection passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
