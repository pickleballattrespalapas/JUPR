const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(relative, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", relative), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => {
    if (Object.hasOwn(mocks, name)) return mocks[name];
    if (name.endsWith(".css")) return {};
    return require(name);
  }, module, module.exports);
  return module.exports;
}

const events = new Map(), documentEvents = new Map(), timers = new Map();
let timerId = 0;
function eventTarget(listeners) {
  return {
    addEventListener(name, callback) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(callback);
    },
    removeEventListener(name, callback) { listeners.get(name)?.delete(callback); }
  };
}
global.document = { ...eventTarget(documentEvents), visibilityState: "visible" };
global.window = { ...eventTarget(events), setInterval(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; }, clearInterval(id) { timers.delete(id); } };
const auth = { getAdminApiBaseUrl: () => "http://127.0.0.1:9", clearAdminSession() { clearCalls++; } };
const api = load("lib/adminNotificationsApi.ts", { "@/lib/adminAuthClient": auth });
const Center = load("components/AdminNotificationCenter.tsx", {
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "@/lib/adminNotificationsApi": api,
  "@/lib/adminAuthClient": auth,
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts")
}).default;

let clubId = "alpha", accessToken = "token-a", clearCalls = 0, requests = [], override;
const categories = [
  { key: "generator_submissions", label: "Generator submissions", description: "Completed sessions to review.", kind: "action", href: "/admin/play-generators/submissions", status: "ready" },
  { key: "registrations", label: "New registrations", description: "Players signing up for play.", kind: "activity", href: "/admin/tournaments", status: "ready" }
];
const approval = { key: "generator:rr-1", category: "generator_submissions", kind: "action", title: "Monday round robin", description: "Joe submitted three rated games for approval.", href: "/admin/play-generators/submissions?session=rr-1", occurred_at: "2026-09-21T15:00:00Z", state: "new" };
const registration = { key: "registration:player-1", category: "registrations", kind: "activity", title: "Casey registered", description: "Casey joined the fall tournament.", href: "/admin/tournaments?tournament=fall", occurred_at: "2026-09-21T14:00:00Z", state: "new" };
let sourceItems, preferences, unavailable;
function reset() { sourceItems = [structuredClone(approval), structuredClone(registration)]; preferences = {}; unavailable = new Set(); requests = []; override = null; }
function feed(club = clubId) {
  return { club_id: club, checked_at: "2026-09-21T16:00:00Z", history_days: 30, truncated: false,
    categories: categories.map(category => ({ ...category, enabled: preferences[category.key] !== false, status: unavailable.has(category.key) ? "unavailable" : "ready", total_count: sourceItems.filter(item => item.category === category.key).length })),
    items: sourceItems.filter(item => preferences[item.category] !== false && !unavailable.has(item.category)).map(item => ({ ...item })) };
}
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
global.fetch = async (url, options = {}) => {
  const request = { url, options, body: options.body ? JSON.parse(options.body) : null };
  requests.push(request);
  if (override) { const handled = override(request); if (handled) return handled; }
  if (options.method === "PUT") {
    if (url.endsWith("/preferences")) preferences = { ...preferences, ...request.body.categories };
    else {
      const key = decodeURIComponent(url.split("/items/")[1]);
      const item = sourceItems.find(item => item.key === key);
      assert.ok(item, `Mutation targets a known notification: ${key}`);
      item.state = request.body.state;
    }
  }
  return reply(feed());
};
const text = node => typeof node === "string" || typeof node === "number" ? String(node) : node?.children?.map(text).join(" ") || "";
const content = tree => text(tree.toJSON());
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node).trim() === label || new RegExp(`^${label}\\s+\\d+$`).test(text(node).trim()) || node.props["aria-label"] === label);
const links = (tree, href) => tree.root.findAllByType("a").filter(node => node.props.href === href);
const element = compact => React.createElement(Center, { key: `${accessToken}\u0000${clubId}`, accessToken, clubId, compact });
async function mount(compact = false) { let tree; await act(async () => { tree = create(element(compact)); }); return tree; }
async function click(tree, label) { const node = button(tree, label); assert.ok(node, `Expected button ${label}`); await act(async () => node.props.onClick()); }
async function unmount(tree) { await act(async () => tree.unmount()); }
async function emit(listeners, event) { await act(async () => { for (const callback of [...listeners.get(event) || []]) callback(new Event(event)); }); }

// Behavior cases are below; the stateful transport intentionally retains item
// states and preferences when the component remounts, like the API does.
function row(tree, item) {
  let node = links(tree, item.href)[0];
  assert.ok(node, `Expected visible notification ${item.title}`);
  while (node && !node.findAllByType("button").length) node = node.parent;
  assert.ok(node, `Expected item controls for ${item.title}`);
  return node;
}
async function itemClick(tree, item, label) {
  const node = row(tree, item).findAllByType("button").find(node => text(node).trim() === label || node.props["aria-label"] === label);
  assert.ok(node, `Expected ${label} control for ${item.title}`);
  await act(async () => node.props.onClick());
}
function checkbox(tree, category) {
  return tree.root.findAllByType("label").find(node => text(node).includes(category)).findByType("input");
}

async function individualStatePersistsAndNewItemsRemainNew() {
  reset();
  let tree = await mount();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "http://127.0.0.1:9/admin/clubs/alpha/notifications");
  assert.equal(requests[0].options.headers.Authorization, "Bearer token-a");
  assert.ok(links(tree, approval.href).length);
  await itemClick(tree, registration, "Flag");
  assert.equal(sourceItems.find(item => item.key === registration.key).state, "flagged");
  assert.ok(content(tree).indexOf(registration.title) < content(tree).indexOf(approval.title), "Flagged work sorts ahead of newer unflagged work");
  await itemClick(tree, approval, "Clear");
  assert.equal(sourceItems.find(item => item.key === approval.key).state, "cleared");
  assert.equal(links(tree, approval.href).length, 0, "Clear removes just that notification from New");
  assert.ok(links(tree, categories[0].href).length, "Clearing does not remove access to the source approval queue");
  await click(tree, "Flagged");
  assert.ok(links(tree, registration.href).length);
  assert.equal(links(tree, approval.href).length, 0);
  await click(tree, "Cleared");
  assert.ok(links(tree, approval.href).length);
  await unmount(tree);
  tree = await mount();
  assert.equal(links(tree, approval.href).length, 0, "Cleared state survives reload");
  sourceItems.push({ ...approval, key: "generator:rr-2", title: "Tuesday round robin", href: "/admin/play-generators/submissions?session=rr-2" });
  await click(tree, "Refresh");
  assert.match(content(tree), /Tuesday round robin/, "A new source item is not suppressed by an earlier cleared notification");
  assert.equal(links(tree, approval.href).length, 0);
  await click(tree, "Cleared");
  await itemClick(tree, approval, "Restore");
  assert.equal(sourceItems.find(item => item.key === approval.key).state, "new");
  await click(tree, "Inbox");
  assert.ok(links(tree, approval.href).length);
  const writes = requests.filter(request => request.options.method === "PUT");
  assert.deepEqual(writes.map(request => request.body.state), ["flagged", "cleared", "new"]);
  assert.ok(writes.every(request => !request.url.includes("token-a")), "Bearer tokens never enter notification URLs");
  await unmount(tree);
}

async function failedStateChangeRetainsTheNotification() {
  reset();
  const tree = await mount();
  for (const status of [503, 403]) {
    override = request => request.options.method === "PUT" ? reply({ detail: "Unable to save notification." }, status) : null;
    await itemClick(tree, approval, "Clear");
    assert.ok(links(tree, approval.href).length, "Failed Clear keeps the original visible item");
    assert.equal(sourceItems[0].state, "new");
    assert.equal(clearCalls, 0, "Mutation denial keeps the signed-in session for reads and recovery");
    assert.ok(tree.root.findAllByProps({ role: "alert" }).length, "Failed persistence is visible");
  }
  override = null;
  await itemClick(tree, approval, "Clear");
  assert.equal(links(tree, approval.href).length, 0, "Clear can be retried after a save failure");
  await unmount(tree);
}

async function preferencesPersistOnlyAfterSuccessfulSave() {
  reset();
  let tree = await mount();
  await act(async () => tree.root.findByType("select").props.onChange({ target: { value: "registrations" } }));
  assert.equal(links(tree, approval.href).length, 0, "Category filter narrows the visible notices");
  await click(tree, "Notification settings");
  await act(async () => checkbox(tree, "New registrations").props.onChange({ target: { checked: false } }));
  assert.equal(requests.filter(request => request.options.method === "PUT").length, 0, "Editing settings waits for Save");
  override = request => request.url.endsWith("/preferences") ? reply({ detail: "Unable to save notification settings." }, 503) : null;
  await click(tree, "Save preferences");
  assert.ok(links(tree, registration.href).length, "A failed settings save does not hide existing notifications");
  assert.equal(preferences.registrations, undefined);
  assert.ok(tree.root.findAllByProps({ role: "alert" }).length);
  override = null;
  await click(tree, "Save preferences");
  assert.equal(preferences.registrations, false);
  assert.equal(tree.root.findByType("select").props.value, "all", "Disabling the selected category resets the filter to available categories");
  assert.equal(links(tree, registration.href).length, 0);
  assert.ok(links(tree, approval.href).length, "One disabled category does not hide unrelated approvals");
  await unmount(tree);
  tree = await mount();
  await click(tree, "Notification settings");
  assert.equal(checkbox(tree, "New registrations").props.checked, false, "Category choice survives reload");
  assert.equal(links(tree, registration.href).length, 0);
  await act(async () => checkbox(tree, "New registrations").props.onChange({ target: { checked: true } }));
  await click(tree, "Save preferences");
  assert.ok(links(tree, registration.href).length, "Reenabling returns recent category items");
  await unmount(tree);
}

async function unavailableDataNeverLooksClearAndRefreshes() {
  reset(); unavailable.add("generator_submissions");
  let tree = await mount();
  assert.ok(tree.root.findAllByProps({ role: "alert" }).length, "Unavailable categories have an explicit warning");
  assert.ok(links(tree, categories[0].href).length, "Unavailable categories still link to their action queue");
  assert.doesNotMatch(content(tree), /No new notifications|No pending work|All caught up/);
  await unmount(tree);
  reset(); override = () => reply({ detail: "Unavailable" }, 503);
  tree = await mount();
  assert.ok(tree.root.findAllByProps({ role: "alert" }).length);
  assert.doesNotMatch(content(tree), /No new notifications|All caught up/);
  override = null;
  await click(tree, "Refresh");
  assert.ok(links(tree, approval.href).length);
  const before = requests.length;
  await emit(events, "focus");
  assert.equal(requests.length, before + 1);
  document.visibilityState = "hidden";
  await emit(documentEvents, "visibilitychange");
  assert.equal(requests.length, before + 1);
  document.visibilityState = "visible";
  await emit(documentEvents, "visibilitychange");
  assert.equal(requests.length, before + 2);
  const timer = [...timers.values()].find(timer => timer.delay === 60_000);
  assert.ok(timer);
  await act(async () => timer.callback());
  assert.equal(requests.length, before + 3);
  await unmount(tree);
  assert.equal(timers.size, 0);
  assert.ok([...events.values(), ...documentEvents.values()].every(listeners => !listeners.size));
}

async function scopeChangesIgnoreLateReadAndMutation() {
  reset(); let resolveOld;
  const oldData = feed("alpha");
  override = () => new Promise(resolve => { resolveOld = resolve; });
  let tree = await mount();
  clubId = "beta"; override = null; sourceItems = [{ ...registration, title: "Beta registration" }];
  await act(async () => tree.update(element(false)));
  await act(async () => resolveOld(reply(oldData)));
  assert.match(content(tree), /Beta registration/);
  assert.doesNotMatch(content(tree), /Monday round robin/);
  let resolveMutation;
  override = request => request.options.method === "PUT" ? new Promise(resolve => { resolveMutation = resolve; }) : null;
  // Do not await this click: the mutation remains in flight while club changes.
  await act(async () => { void row(tree, registration).findAllByType("button").find(node => text(node).trim() === "Clear").props.onClick(); });
  clubId = "alpha"; override = null; sourceItems = [structuredClone(approval)];
  await act(async () => tree.update(element(false)));
  await act(async () => resolveMutation(reply({ ...oldData, club_id: "beta", items: [{ ...registration, title: "Old mutation result", state: "cleared" }] })));
  assert.doesNotMatch(content(tree), /Old mutation result|Beta registration/);
  assert.match(content(tree), /Monday round robin/);
  accessToken = "rotated-token";
  override = () => reply({ detail: "Access denied" }, 403);
  await act(async () => tree.update(element(false)));
  assert.equal(clearCalls, 1, "Denied notification read invalidates the session");
  assert.doesNotMatch(content(tree), /Monday round robin/);
  await unmount(tree);
  accessToken = "token-a"; clubId = "alpha";
}

(async () => {
  await individualStatePersistsAndNewItemsRemainNew();
  await failedStateChangeRetainsTheNotification();
  await preferencesPersistOnlyAfterSuccessfulSave();
  await unavailableDataNeverLooksClearAndRefreshes();
  await scopeChangesIgnoreLateReadAndMutation();
  console.log("Admin notifications: individual states, source action access, new items, preferences, persistence failures, refresh and scope isolation passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
