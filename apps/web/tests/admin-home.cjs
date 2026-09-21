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

let clubId = "alpha";
let accessToken = "token-a";
const administratorPermissions = ["manage_players", "manage_matches", "delete_matches", "run_replay", "manage_tournaments", "manage_subscriptions", "view_audit_log", "enter_scores", "manage_club_staff"];
let role = "administrator";
let permissions = administratorPermissions;
let scopes = [];
let authLoading = false;
const notificationMounts = [];
const NotificationCenter = props => {
  notificationMounts.push(props);
  return React.createElement("div", { "data-notification-club": props.clubId, "data-notification-token": props.accessToken, "data-compact": props.compact });
};
const Home = load("app/admin/AdminHome.tsx", {
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "@/lib/useAdminWorkspace": { useAdminWorkspace: () => ({ clubId, clubSlug: `${clubId}-club` }) },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken, loading: authLoading, session: accessToken ? { user: { id: "admin-a" }, capabilities: { assignments: [{ club_id: clubId, role, permissions, scopes }] } } : null }) },
  "@/components/AdminNotificationCenter": { __esModule: true, default: NotificationCenter }
}).default;
const text = node => typeof node === "string" || typeof node === "number" ? String(node) : node?.children?.map(text).join(" ") || "";
const content = tree => text(tree.toJSON());
const links = (tree, href) => tree.root.findAllByType("a").filter(node => node.props.href === href);
async function mount() { let tree; await act(async () => { tree = create(React.createElement(Home)); }); return tree; }
async function unmount(tree) { await act(async () => tree.unmount()); }

async function embedsScopedNotificationsBeforePlay() {
  const tree = await mount();
  assert.match(content(tree), /Admin Home/);
  assert.deepEqual(notificationMounts.at(-1), { accessToken: "token-a", clubId: "alpha", compact: true });
  const serialized = JSON.stringify(tree.toJSON());
  assert.ok(serialized.indexOf("data-notification-club") < serialized.indexOf("Run play"), "Notifications precede play shortcuts");
  assert.doesNotMatch(content(tree), /Operations cockpit|Keep writes off|Streamlit|service.role|write.pilot/i);
  assert.equal(links(tree, "/clubs/alpha-club").length, 1);
  clubId = "beta";
  accessToken = "token-b";
  await act(async () => tree.update(React.createElement(Home)));
  assert.deepEqual(notificationMounts.at(-1), { accessToken: "token-b", clubId: "beta", compact: true }, "Club/token switches supply the new notification scope");
  assert.equal(links(tree, "/clubs/beta-club").length, 1);
  await unmount(tree);
  clubId = "alpha"; accessToken = "token-a";
}

async function anonymousAndCheckingHomeDoNotMountNotifications() {
  for (const state of [{ token: "", loading: false }, { token: "token-a", loading: true }]) {
    accessToken = state.token; authLoading = state.loading;
    const previous = notificationMounts.length;
    const tree = await mount();
    assert.equal(notificationMounts.length, previous, "Notifications wait for authorized access");
    assert.equal(tree.root.findAllByType(NotificationCenter).length, 0);
    await unmount(tree);
  }
  accessToken = "token-a"; authLoading = false;
}

async function operatorShortcutsRespectAssignedPrograms() {
  role = "operator";
  // API capabilities include every operator permission; scopes narrow which
  // programs those permissions apply to.
  permissions = ["manage_players", "manage_matches", "manage_tournaments", "enter_scores"];
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

(async () => {
  await embedsScopedNotificationsBeforePlay();
  await operatorShortcutsRespectAssignedPrograms();
  await anonymousAndCheckingHomeDoNotMountNotifications();
  console.log("Admin Home: scoped notifications, authorization and operator shortcuts passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
