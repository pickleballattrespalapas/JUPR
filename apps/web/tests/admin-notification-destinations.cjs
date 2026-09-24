const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

function load(relative, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", relative), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const auth = load("lib/useAuthenticatedAutoLoad.ts");
let token = "staff-token";
const common = {
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: token, session: token ? { user: { id: "staff" } } : null }), adminSessionLabel: () => "Staff" },
  "@/lib/useAuthenticatedAutoLoad": auth,
  "@/components/ConfirmAction": { ConfirmAction: props => React.createElement("button", { disabled: props.disabled, onClick: () => props.onConfirm(props.confirmationText) }, props.triggerLabel) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ title, message }) },
  "next/link": ({ children, ...props }) => React.createElement("a", props, children),
  "next/navigation": { useSearchParams: () => new URLSearchParams(window.location.search) },
  "@/components/SearchablePlayerSelect": ({ children, onValueChange, ...props }) => React.createElement("select", { ...props, onChange: event => onValueChange(event.target.value) }, children)
};
const Generator = load("app/admin/play-generators/submissions/GeneratorSubmissions.tsx", common).default;
const Support = load("app/admin/support-requests/SupportRequestsPanel.tsx", common).default;
const Verified = load("app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx", common).default;
const Social = load("app/admin/tools/AdminToolsPanel.tsx", common).default;
const reply = payload => ({ ok: true, status: 200, json: async () => payload });
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join(" ");
const button = (tree, label) => tree.root.findAllByType("button").find(node => text(node) === label);
const props = { apiBase: "https://api.test", clubId: "alpha", status: { enabled: true, mutations_enabled: true } };
let scrolls, focuses, reads;
function setup(search, hash = "") {
  scrolls = []; focuses = []; reads = []; token = "staff-token";
  global.window = { location: { search, hash } };
  global.document = { getElementById: id => ({ scrollIntoView: () => scrolls.push(id) }) };
}
function nodeMock(element) {
  return { scrollIntoView: () => scrolls.push(element.props.id), focus: options => { assert.equal(options.preventScroll, true); focuses.push(element.props.id); } };
}
async function mount(Component) {
  let tree;
  await act(async () => { tree = create(React.createElement(Component, props), { createNodeMock: nodeMock }); });
  return tree;
}
function deferredFetch() {
  const pending = [];
  global.fetch = (url, options = {}) => { reads.push({ url, options }); return new Promise(resolve => pending.push({ url, resolve })); };
  return pending;
}
async function finish(request, payload) { await act(async () => request.resolve(reply(payload))); }
function assertReadOnly() {
  assert.ok(reads.every(({ url, options }) => url.includes("/clubs/alpha/") && options.headers.Authorization === "Bearer staff-token" || url.includes("/clubs/alpha/") && options.headers.get?.("Authorization") === "Bearer staff-token"));
  assert.ok(reads.every(({ options }) => !options.method || options.method === "GET"), "A notification opens a review without approving anything");
}

async function generatorDeepLink() {
  setup("?session=target-session");
  const pending = deferredFetch();
  const tree = await mount(Generator);
  assert.deepEqual(focuses, [], "Wait for the protected queue before focusing");
  const make = (id, session_key, title) => ({ id, session_key, title, version: 1, status: "pending", organizer_name: "Organizer", rating_mode: "unrated", match_date: "2026-09-23", match_count: 1, participants: [{ id: "p", name: "Player" }], matches: [] });
  await finish(pending[0], { submissions: [make("one", "other-session", "Another event"), make("two", "target-session", "Linked event")], players: [{ id: 12, name: "Player" }] });
  assert.equal(tree.root.findByProps({ id: "generator-submission-review" }).findByType("h2").children[0], "Linked event");
  assert.equal(tree.root.findAllByType("select")[0].props.value, "pending");
  assert.deepEqual(focuses, ["generator-submission-review"]);
  assert.ok(tree.root.findAllByType("select").some(node => node.props.value === 12), "Opening the requested item initializes player mapping");
  await act(async () => tree.root.findAllByType("button").find(node => text(node).includes("Another event")).props.onClick());
  assert.match(text(tree.root.findByProps({ id: "generator-submission-review" })), /Another event/);
  assert.deepEqual(focuses, ["generator-submission-review"], "Manual selection is not replaced by the notification target");
  assertReadOnly();
  token = "different-staff";
  await act(async () => tree.update(React.createElement(Generator, props)));
  assert.equal(tree.root.findAllByProps({ id: "generator-submission-review" }).length, 0, "Changing staff identity clears the old review while the new queue loads");
  await finish(pending[1], { submissions: [], players: [] });
  assert.equal(tree.root.findAllByProps({ id: "generator-submission-review" }).length, 0, "A deep link cannot reselect a prior identity's cached item");
  await act(async () => tree.unmount());
}

async function supportDeepLink() {
  setup("?status=in_review&request=request-two");
  const pending = deferredFetch();
  const tree = await mount(Support);
  assert.match(pending[0].url, /status=in_review/);
  const request = { id: "request-two", subject: "Correct my result", description: "Check score", status: "in_review", request_type: "data_correction", requester_name: "Traveler", admin_note: "Existing note" };
  await finish(pending[0], { requests: [{ ...request, id: "other", subject: "Other request" }, request], summary: { total: 2, by_status: {}, by_type: {} } });
  assert.match(text(tree.root.findByProps({ id: "support-request-review" })), /Correct my result/);
  assert.equal(tree.root.findByType("textarea").props.value, "Existing note");
  assert.deepEqual(focuses, ["support-request-review"]);
  await act(async () => tree.root.findByType("textarea").props.onChange({ target: { value: "My draft" } }));
  assert.equal(tree.root.findByType("textarea").props.value, "My draft", "A rerender preserves the admin's draft");
  assertReadOnly(); await act(async () => tree.unmount());
}

async function verifiedDeepLink() {
  setup("?request=request-two");
  const pending = deferredFetch();
  const tree = await mount(Verified);
  await finish(pending[0], { requests: [{ id: "request-one", player_name: "Other player", player_id: 1, request_status: "pending" }, { id: "request-two", player_name: "Linked player", player_id: 2, request_status: "pending" }], count: 2 });
  assert.deepEqual(focuses, ["verified-request-request-two"]);
  assert.equal(tree.root.findByProps({ id: "verified-request-request-two" }).props.style.borderColor, "#2563eb");
  await act(async () => { void button(tree, "Refresh requests").props.onClick(); });
  await finish(pending[1], { requests: [], count: 0 });
  assert.deepEqual(focuses, ["verified-request-request-two"], "Refreshing does not scroll the admin away from their place");
  assertReadOnly(); await act(async () => tree.unmount());
}

async function socialDeepLink() {
  setup("?submission=social-two", "#social-submissions");
  const pending = deferredFetch();
  const tree = await mount(Social);
  const queue = pending.find(request => request.url.includes("social-submissions"));
  const overview = pending.find(request => !request.url.includes("social-submissions"));
  const submission = { id: "social-two", name: "Linked social event", event_type: "round_robin", event_date: "2026-09-23", status: "pending", submitted_by_name: "Organizer", submission_mode: "unrated", summary_json: {}, raw_event_json: {} };
  await finish(queue, { submissions: [{ ...submission, id: "social-one", name: "Other social event" }, submission], status: "pending", warnings: [], confirmation_text: { approve: "APPROVE", reject: "REJECT" } });
  assert.deepEqual(focuses, [], "Review waits until the overview renders the queue");
  await finish(overview, { roles: [], activity: [], health: {}, role_options: [] });
  assert.match(text(tree.root.findByProps({ id: "social-submission-review" })), /Linked social event/);
  assert.deepEqual(focuses, ["social-submission-review"]);
  assert.equal(scrolls.at(-1), "social-submission-review", "The exact review is the final scroll destination");
  assertReadOnly(); await act(async () => tree.unmount());
}

async function absentTargetAndAnonymous() {
  setup("?session=not-in-this-club");
  let pending = deferredFetch();
  let tree = await mount(Generator);
  await finish(pending[0], { submissions: [], players: [] });
  assert.deepEqual(focuses, []); assert.equal(tree.root.findAllByProps({ id: "generator-submission-review" }).length, 0);
  await act(async () => tree.unmount());
  token = ""; pending = deferredFetch();
  tree = await mount(Generator);
  assert.equal(pending.length, 0, "Deep links do not load queues without authentication");
  await act(async () => tree.unmount());
}

(async () => {
  await generatorDeepLink(); await supportDeepLink(); await verifiedDeepLink(); await socialDeepLink(); await absentTargetAndAnonymous();
  console.log("PASS notification destinations: exact async review selection, focus and highlight, retained drafts, protected queues, read-only navigation");
})().catch(error => { console.error(error); process.exit(1); });
