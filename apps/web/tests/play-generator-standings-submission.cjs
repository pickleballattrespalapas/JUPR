const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const React = require("react");
const ts = require("typescript");
const { create, act } = require("react-test-renderer");

function load(relative, mocks = {}) {
  const source = fs.readFileSync(path.join(__dirname, "..", relative), "utf8");
  const code = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}

const submission = load("components/GeneratorSubmission.tsx");
const routerCalls = [];
const router = { push: url => routerCalls.push(url), refresh() {} };
const Link = ({ children, ...props }) => React.createElement("a", props, children);
const mocks = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "admin-token" }) },
  "@/components/GeneratorSubmission": submission,
  "@/components/PlayGeneratorStandingsTable": {
    __esModule: true,
    default: () => React.createElement("table", { "aria-label": "Standings" }),
    standingsSortLabel: () => "Wins"
  }
};
const PublicStandings = load("app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx", mocks).default;
const AdminStandings = load("app/admin/play-generators/GeneratorStandings.tsx", mocks).default;
const props = { apiBase: "https://api.test", clubId: "club", sessionKey: "session-1" };
const storageKey = "public-generator-edit:club:session-1";
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const content = tree => JSON.stringify(tree.toJSON());
const button = (tree, label) => tree.root.findAllByType("button").find(node => node.children.includes(label));
const baseline = {
  session_key: "session-1", title: "Tuesday RR", status: "active", version: "4",
  generator_kind: "round_robin", play_format: "doubles", rating_mode: "rated", scoring_mode: "scored",
  created_at: "2026-09-21T16:00:00Z", current_round_number: 1, total_rounds: 1, standings: [],
  event: { rounds: [{ number: 1, status: "saved" }] }
};

function publicEnvironment({ hashToken = "", storedToken = "" } = {}) {
  const values = new Map(storedToken ? [[storageKey, storedToken]] : []);
  global.sessionStorage = { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) };
  const replaced = [];
  global.window = {
    location: { hash: hashToken ? `#edit=${hashToken}` : "", pathname: "/clubs/club/round-robin-generator/sessions/session-1/standings", search: "" },
    history: { replaceState(state, title, url) { replaced.push(url); window.location.hash = ""; } }
  };
  return { values, replaced };
}

async function fillAndSubmit(tree) {
  await act(async () => tree.root.findByProps({ maxLength: 160 }).props.onChange({ target: { value: " Joe Organizer " } }));
  assert.equal(tree.root.findByProps({ type: "date" }).props.value, "2026-09-21");
  await act(async () => tree.root.findByType("form").props.onSubmit({ preventDefault() {} }));
}

async function organizerFinishesAndSubmitsWithoutAccount() {
  publicEnvironment({ storedToken: "organizer-secret" });
  let current = structuredClone(baseline);
  const calls = [];
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/complete")) current = { ...current, status: "completed", version: "5" };
    if (url.endsWith("/submit")) current = { ...current, version: "6", submission: { status: "pending", rating_mode: "rated" } };
    return reply({ session: current });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(PublicStandings, props)); });
  assert.ok(content(tree).includes("Rated"), "The selected rated mode remains visible on standings");
  assert.equal(tree.root.findAllByType("form").length, 0, "Results are submitted after finishing");
  await act(async () => button(tree, "Finish session").props.onClick());
  const finish = calls.find(call => call.url.endsWith("/complete"));
  assert.equal(finish.options.method, "POST");
  assert.equal(JSON.parse(finish.options.body).edit_token, "organizer-secret");
  assert.equal(JSON.parse(finish.options.body).expected_version, 4);
  assert.equal(calls.some(call => call.url.endsWith("/advance")), false, "Finishing is separate from advancing");
  assert.ok(button(tree, "Submit for approval"), "Finishing reveals submission on the same page");
  assert.equal(routerCalls.length, 0, "Organizer does not have to return to the round page");
  assert.ok(content(tree).indexOf("Submit for approval") < content(tree).indexOf("Standings"), "Submission is prominent above the table");
  await fillAndSubmit(tree);
  const sent = calls.find(call => call.url.endsWith("/submit"));
  assert.deepEqual(JSON.parse(sent.options.body), {
    edit_token: "organizer-secret", idempotency_key: "generator-submit:session-1", expected_version: 5,
    organizer_name: "Joe Organizer", match_date: "2026-09-21"
  });
  assert.ok(calls.every(call => !call.url.includes("organizer-secret")), "Organizer token stays out of URLs");
  assert.ok(content(tree).includes("Awaiting admin approval"));
  assert.equal(tree.root.findAllByType("form").length, 0, "Submitted results cannot be submitted twice");
  current = { ...current, submission: { status: "approved", approved_mode: "rated" } };
  await act(async () => button(tree, "Refresh approval status").props.onClick());
  assert.ok(content(tree).includes("Approved · Rated"));
  assert.ok(content(tree).includes("Ratings have been updated"));
  await act(async () => tree.unmount());
}

async function organizerFragmentAndFailedSubmission() {
  const environment = publicEnvironment({ hashToken: "fresh-organizer-secret" });
  const current = { ...baseline, status: "completed", rating_mode: "unrated" };
  let submitCalls = 0;
  global.fetch = async (url, options) => {
    if (url.endsWith("/submit")) { submitCalls++; return reply({ detail: "Changed" }, 409); }
    return reply({ session: current });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(PublicStandings, props)); });
  assert.equal(environment.values.get(storageKey), "fresh-organizer-secret");
  assert.deepEqual(environment.replaced, [window.location.pathname]);
  assert.equal(window.location.hash, "");
  await fillAndSubmit(tree);
  assert.equal(submitCalls, 1);
  assert.ok(content(tree).includes("This session changed. Refresh the page and try again."));
  assert.equal(tree.root.findByProps({ maxLength: 160 }).props.value, " Joe Organizer ", "Failed submission preserves organizer input");
  assert.ok(content(tree).includes("Unrated"));
  assert.ok(button(tree, "Submit for approval"), "Failed submission remains retryable");
  await act(async () => tree.unmount());
}

async function viewerCannotSubmitAndUnscoredHasNoApproval() {
  publicEnvironment();
  let current = { ...baseline, status: "completed" };
  global.fetch = async () => reply({ session: current });
  let tree;
  await act(async () => { tree = create(React.createElement(PublicStandings, props)); });
  assert.equal(tree.root.findAllByType("form").length, 0);
  assert.equal(button(tree, "Submit for approval"), undefined, "A public viewer cannot submit without the organizer token");
  assert.equal(button(tree, "Keep playing · Round 2"), undefined, "A public viewer cannot resume a session");
  await act(async () => tree.unmount());
  publicEnvironment({ storedToken: "organizer-secret" });
  current = { ...current, scoring_mode: "unscored" };
  await act(async () => { tree = create(React.createElement(PublicStandings, props)); });
  assert.ok(content(tree).includes("does not use standings"));
  assert.equal(button(tree, "Submit for approval"), undefined);
  await act(async () => tree.unmount());
}

async function adminFinishesAndSubmitsUnratedResults() {
  let current = { ...baseline, rating_mode: "unrated" };
  const calls = [];
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/complete")) current = { ...current, status: "completed", version: "5" };
    if (url.endsWith("/submit")) current = { ...current, version: "6", submission: { status: "pending", rating_mode: "unrated" } };
    return reply({ session: current });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(AdminStandings, props)); });
  await act(async () => button(tree, "Finish session").props.onClick());
  const finish = calls.find(call => call.url.endsWith("/complete"));
  assert.equal(finish.options.headers.Authorization, "Bearer admin-token");
  assert.equal(JSON.parse(finish.options.body).expected_version, "4");
  assert.equal(calls.some(call => call.url.endsWith("/advance")), false);
  assert.ok(button(tree, "Submit for approval"));
  assert.ok(tree.root.findByProps({ href: "/admin/play-generators/submissions" }));
  await fillAndSubmit(tree);
  const sent = calls.find(call => call.url.endsWith("/submit"));
  assert.equal(sent.url, "https://api.test/admin/clubs/club/play-generators/sessions/session-1/submit");
  assert.equal(sent.options.headers.Authorization, "Bearer admin-token");
  assert.deepEqual(JSON.parse(sent.options.body), { expected_version: "5", organizer_name: "Joe Organizer", match_date: "2026-09-21" });
  assert.ok(content(tree).includes("Awaiting admin approval"));
  current = { ...current, submission: { status: "approved", approved_mode: "unrated" } };
  await act(async () => button(tree, "Refresh approval status").props.onClick());
  assert.ok(content(tree).includes("Approved · Unrated"));
  assert.ok(content(tree).includes("Ratings are unchanged"));
  await act(async () => tree.unmount());
}

async function standingsCanContinueBeyondSchedule(Component, isPublic) {
  publicEnvironment({ storedToken: "organizer-secret" });
  routerCalls.length = 0;
  let current = {
    ...baseline, current_round_number: 8, total_rounds: 8,
    event: { rounds: [{ number: 7, status: "skipped" }, { number: 8, status: "saved" }] }
  };
  const calls = [];
  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/advance")) current = {
      ...current, status: "active", current_round_number: 9, version: "5",
      event: { rounds: [...current.event.rounds, { number: 9, status: "active" }] }
    };
    return reply({ session: current });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Component, props)); });
  assert.ok(button(tree, "Continue to Round 9"), "A saved round at the initial schedule limit still offers another round");
  assert.ok(button(tree, "Finish session"), "Finishing remains an explicit separate choice");
  const skippedLink = tree.root.findAllByType("a").find(node => node.props.href.endsWith("/rounds/7"));
  assert.ok(skippedLink.children.includes(" · Skipped"), "Skipped rounds remain visible and labeled in navigation");
  await act(async () => button(tree, "Continue to Round 9").props.onClick());
  assert.deepEqual(routerCalls, [isPublic
    ? "/clubs/club/round-robin-generator/sessions/session-1/rounds/9"
    : "/admin/round-robin-generator/sessions/session-1/rounds/9"]);
  const advance = calls.find(call => call.url.endsWith("/advance"));
  assert.equal(advance.options.method, "POST");
  if (isPublic) assert.equal(JSON.parse(advance.options.body).edit_token, "organizer-secret");
  else assert.equal(advance.options.headers.Authorization, "Bearer admin-token");
  await act(async () => tree.unmount());
}

async function completedSessionsCanResumeUntilSubmitted(Component) {
  publicEnvironment({ storedToken: "organizer-secret" });
  let current = { ...baseline, status: "completed" };
  global.fetch = async url => {
    if (url.endsWith("/advance")) current = {
      ...current, status: "active", current_round_number: 2,
      event: { rounds: [...current.event.rounds, { number: 2, status: "active" }] }
    };
    return reply({ session: current });
  };
  let tree;
  await act(async () => { tree = create(React.createElement(Component, props)); });
  assert.ok(button(tree, "Keep playing · Round 2"), "A finished session can still resume before submission");
  assert.equal(button(tree, "Finish session"), undefined);
  await act(async () => button(tree, "Keep playing · Round 2").props.onClick());
  assert.ok(content(tree).includes("In progress"));
  await act(async () => tree.unmount());

  for (const lock of [{ submission: { status: "pending" } }]) {
    current = { ...baseline, status: "completed", ...lock };
    await act(async () => { tree = create(React.createElement(Component, props)); });
    assert.equal(button(tree, "Keep playing · Round 2"), undefined, "Submitted sessions cannot be resumed");
    assert.equal(button(tree, "Finish session"), undefined);
    assert.equal(button(tree, "Submit for approval"), undefined, "Locked results cannot be submitted again");
    await act(async () => tree.unmount());
  }
}

async function cannotFinishWithReopenedRound(Component) {
  publicEnvironment({ storedToken: "organizer-secret" });
  const current = {
    ...baseline, current_round_number: 8, total_rounds: 8,
    event: { rounds: [{ number: 7, status: "active" }, { number: 8, status: "saved" }] }
  };
  global.fetch = async () => reply({ session: current });
  let tree;
  await act(async () => { tree = create(React.createElement(Component, props)); });
  assert.ok(button(tree, "Continue to Round 9"));
  assert.equal(button(tree, "Finish session"), undefined, "Reopened rounds must be resolved before finishing the session");
  await act(async () => tree.unmount());
}

(async () => {
  await organizerFinishesAndSubmitsWithoutAccount();
  await organizerFragmentAndFailedSubmission();
  await viewerCannotSubmitAndUnscoredHasNoApproval();
  await adminFinishesAndSubmitsUnratedResults();
  await standingsCanContinueBeyondSchedule(PublicStandings, true);
  await standingsCanContinueBeyondSchedule(AdminStandings, false);
  for (const Component of [PublicStandings, AdminStandings]) {
    await completedSessionsCanResumeUntilSubmitted(Component);
    await cannotFinishWithReopenedRound(Component);
  }
  console.log("Round-Robin standings submission behavior passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
