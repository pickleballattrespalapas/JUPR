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

const events = new Map();
const documentEvents = new Map();
const timers = new Map();
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
global.window = {
  ...eventTarget(events),
  setInterval(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
  clearInterval(id) { timers.delete(id); }
};

let clubId = "alpha";
let accessToken = "token-a";
let identity = "admin-a";
const administratorPermissions = ["manage_players", "manage_matches", "delete_matches", "run_replay", "manage_tournaments", "manage_subscriptions", "view_audit_log", "enter_scores", "manage_club_staff"];
let role = "administrator";
let permissions = administratorPermissions;
let scopes = [];
let authLoading = false;
let clearCalls = 0;
let requests = [];
let response;
const ok = data => ({ data, error: null, status: 200 });
const queue = (count = 3, status = "ready") => ({
  key: "generator_submissions", label: "Generator submissions", description: "Completed sessions waiting for approval.",
  href: "/admin/play-generators/submissions", count, status
});
const dashboard = (queues = [queue()], club = clubId) => ({ club_id: club, checked_at: "2026-09-21T16:00:00Z", queues });
const getDashboard = (token, club) => {
  requests.push({ token, club });
  return typeof response === "function" ? response(token, club) : Promise.resolve(response);
};
const Home = load("app/admin/AdminHome.tsx", {
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "@/lib/useAdminWorkspace": { useAdminWorkspace: () => ({ clubId, clubSlug: `${clubId}-club` }) },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken, loading: authLoading, session: accessToken ? { user: { id: identity }, capabilities: { assignments: [{ club_id: clubId, role, permissions, scopes }] } } : null }) },
  "@/lib/adminDashboardApi": { getAdminDashboard: getDashboard },
  "@/lib/adminAuthClient": { clearAdminSession() { clearCalls++; } },
  "@/lib/useAuthenticatedAutoLoad": load("lib/useAuthenticatedAutoLoad.ts")
}).default;

const text = node => typeof node === "string" || typeof node === "number" ? String(node) : node?.children?.map(text).join(" ") || "";
const content = tree => text(tree.toJSON());
const links = (tree, href) => tree.root.findAllByType("a").filter(node => node.props.href === href);
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node).includes(label));
const allClear = tree => /No pending|Nothing needs|You're all caught up|No actions need|No items need|All caught up/i.test(content(tree));
async function mount() { let tree; await act(async () => { tree = create(React.createElement(Home)); }); return tree; }
async function unmount(tree) { await act(async () => tree.unmount()); }
async function emit(listeners, event) { await act(async () => { for (const callback of [...listeners.get(event) || []]) callback(new Event(event)); }); }

async function pendingCountsAndApprovalRefresh() {
  response = ok(dashboard()); requests = [];
  const tree = await mount();
  assert.deepEqual(requests, [{ token: "token-a", club: "alpha" }], "Home loads one authenticated, club-scoped summary");
  assert.match(content(tree), /Admin Home/);
  assert.match(content(tree), /Needs attention/);
  assert.ok(links(tree, queue().href).some(node => /3/.test(text(node)) && /Generator submissions/.test(text(node))), "Pending count links directly to the approval queue");
  assert.equal(allClear(tree), false, "Pending approvals never display all clear");
  assert.doesNotMatch(content(tree), /Operations cockpit|Keep writes off|Streamlit|service.role|write.pilot/i, "Home does not expose obsolete migration controls");
  response = ok(dashboard([queue(0)]));
  assert.ok(button(tree, "Refresh"));
  await act(async () => button(tree, "Refresh").props.onClick());
  assert.equal(requests.length, 2);
  assert.ok(allClear(tree), "An approved queue becomes clear after refresh");
  assert.ok(!links(tree, queue().href).some(node => /\b3\b/.test(text(node))), "Old pending count is removed after approval");
  await unmount(tree);
}

async function failureIsNotAllClear() {
  response = ok(dashboard([queue(null, "unavailable"), { ...queue(0), key: "roster", label: "Roster review", href: "/admin/players" }]));
  let tree = await mount();
  assert.equal(allClear(tree), false, "An unavailable queue cannot produce an all-clear message");
  assert.match(content(tree), /unavailable|couldn.t|unable|could not|check failed/i);
  assert.ok(links(tree, queue().href).length, "Unavailable queues retain a route to inspect the work manually");
  await unmount(tree);
  response = { data: null, error: "Unable to check pending work.", status: 503 };
  tree = await mount();
  assert.equal(allClear(tree), false);
  assert.match(content(tree), /Unable to check pending work/);
  assert.ok(button(tree, "Refresh"), "A failed read can be retried");
  response = ok(dashboard());
  await act(async () => button(tree, "Refresh").props.onClick());
  assert.ok(links(tree, queue().href).some(node => /\b3\b/.test(text(node))));
  response = { data: null, error: "Unable to check pending work.", status: 503 };
  await act(async () => button(tree, "Refresh").props.onClick());
  assert.ok(!links(tree, queue().href).some(node => /\b3\b/.test(text(node))), "A failed refresh does not present stale counts as current");
  assert.equal(allClear(tree), false);
  await unmount(tree);
  response = ok(dashboard([]));
  tree = await mount();
  assert.equal(allClear(tree), false, "No authorized review queues is different from verified zero pending work");
  assert.match(content(tree), /No review queues are available for your role/);
  await unmount(tree);
}

async function returningToHomeRefreshesCounts() {
  response = ok(dashboard([queue(4)])); requests = [];
  const tree = await mount();
  response = ok(dashboard([queue(0)]));
  await emit(events, "focus");
  assert.equal(requests.length, 2, "Returning from an approval tab rechecks counts");
  assert.ok(allClear(tree));
  document.visibilityState = "hidden";
  await emit(documentEvents, "visibilitychange");
  assert.equal(requests.length, 2, "Hidden tabs do not refresh on visibility changes");
  document.visibilityState = "visible";
  response = ok(dashboard([queue(1)]));
  await emit(documentEvents, "visibilitychange");
  assert.equal(requests.length, 3);
  assert.equal(allClear(tree), false);
  const minuteTimer = [...timers.values()].find(timer => timer.delay === 60_000);
  assert.ok(minuteTimer, "Visible home refreshes once per minute");
  await act(async () => minuteTimer.callback());
  assert.equal(requests.length, 4);
  await unmount(tree);
  assert.equal(timers.size, 0, "Leaving home releases its polling timer");
  assert.ok([...events.values(), ...documentEvents.values()].every(listeners => listeners.size === 0), "Leaving home releases refresh listeners");
}

async function scopeChangesIgnoreOldResponses() {
  let resolveOld;
  response = () => new Promise(resolve => { resolveOld = resolve; });
  requests = [];
  const tree = await mount();
  clubId = "beta";
  response = ok(dashboard([{ ...queue(1), label: "Beta approval" }], "beta"));
  await act(async () => tree.update(React.createElement(Home)));
  assert.match(content(tree), /Beta approval/);
  await act(async () => resolveOld(ok(dashboard([{ ...queue(9), label: "Old club approval" }], "alpha"))));
  assert.doesNotMatch(content(tree), /Old club approval/);
  assert.match(content(tree), /Beta approval/);
  assert.deepEqual(requests.map(request => request.club), ["alpha", "beta"]);
  let resolveNew;
  accessToken = "rotated-token";
  response = () => new Promise(resolve => { resolveNew = resolve; });
  await act(async () => tree.update(React.createElement(Home)));
  assert.doesNotMatch(content(tree), /Beta approval/, "Token changes hide prior protected counts until reauthorized");
  await act(async () => resolveNew({ data: null, error: "Access denied", status: 403 }));
  assert.equal(clearCalls, 1, "Denied dashboard invalidates the admin session");
  await unmount(tree);
  clubId = "alpha"; accessToken = "token-a";
}

async function operatorShortcutsRespectAssignedPrograms() {
  role = "operator";
  // API capabilities include every operator permission; scopes narrow which
  // programs those permissions apply to.
  permissions = ["manage_players", "manage_matches", "manage_tournaments", "enter_scores"];
  response = ok(dashboard([]));
  scopes = [{ kind: "program_type", program_type: "tournaments", resource_id: "" }];
  let tree = await mount();
  assert.ok(links(tree, "/admin/tournaments").some(node => /Manage tournaments/.test(text(node))));
  for (const href of ["/admin/round-robin-generator", "/admin/ladder-generator", "/admin/league-manager", "/admin/interclub", "/admin/match-uploader", "/admin/match-log"]) {
    assert.equal(links(tree, href).length, 0, `Tournament operator does not see ${href}`);
  }
  assert.equal(links(tree, "/admin/players").length, 1, "Operators retain authorized player-directory access");
  await unmount(tree);
  scopes = [{ kind: "resource", program_type: "round_robin", resource_id: "assigned-session" }];
  tree = await mount();
  const rr = links(tree, "/admin/round-robin-generator");
  assert.equal(rr.length, 1);
  assert.match(text(rr[0]), /Open round robin/);
  assert.doesNotMatch(text(rr[0]), /Start a round robin/);
  assert.match(text(rr[0]), /assigned programs/);
  for (const href of ["/admin/tournaments", "/admin/ladder-generator", "/admin/league-manager", "/admin/interclub", "/admin/match-uploader", "/admin/match-log"]) {
    assert.equal(links(tree, href).length, 0, `Resource-only RR operator does not see ${href}`);
  }
  await unmount(tree);
  role = "administrator";
  permissions = administratorPermissions;
  scopes = [];
}

async function anonymousHomeDoesNotReadQueues() {
  accessToken = ""; requests = [];
  const tree = await mount();
  assert.equal(requests.length, 0);
  assert.equal(allClear(tree), false);
  assert.equal(links(tree, queue().href).some(node => /\b3\b/.test(text(node))), false);
  await emit(events, "focus");
  assert.equal(requests.length, 0, "Focus cannot fetch protected data for anonymous visitors");
  await unmount(tree);
  accessToken = "token-a";
}

(async () => {
  await pendingCountsAndApprovalRefresh();
  await failureIsNotAllClear();
  await returningToHomeRefreshesCounts();
  await scopeChangesIgnoreOldResponses();
  await operatorShortcutsRespectAssignedPrograms();
  await anonymousHomeDoesNotReadQueues();
  console.log("Admin Home: linked counts, approval refresh, honest failure states, auto refresh, authorization, club isolation and scoped operator shortcuts passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
