const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const { act, create } = require("react-test-renderer");

function load(file, mocks = {}) {
  const source = fs.readFileSync(path.join(__dirname, "..", file), "utf8");
  const code = ts.transpileModule(source, { compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true,
  }}).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => mocks[name] ?? require(name), module, module.exports);
  return module.exports;
}

const h = React.createElement;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const request = { request_type: "general_support", requester_name: "Alex", requester_email: "alex@example.test",
  subject: "Help", description: "Request details", consent_to_contact: true };

async function pageQueryParameters() {
  const { supportParam } = load("components/SupportIntakeGate.tsx", {
    "next/link": () => null,
    "@/lib/clubSiteServer": {},
    "@/lib/publicSupportContent": {},
    "@/lib/supportIntakeApi": {},
  });
  for (const [route, formModule] of [
    ["support", "./SupportRequestForm"],
    ["data-corrections", "./DataCorrectionForm"],
    ["profile-privacy", "./ProfilePrivacyRequestForm"],
  ]) {
    let gateProps, formProps;
    const Page = load(`app/${route}/page.tsx`, {
      "next/link": ({ children, ...props }) => h("a", props, children),
      "@/components/PublicControls": ({ children }) => children,
      "@/lib/publicSupportContent": { SUPPORT_EMAIL: "support@example.test" },
      "@/components/SupportIntakeGate": { __esModule: true, supportParam,
        default: props => { gateProps = props; return props.children; } },
      [formModule]: props => { formProps = props; return h("form"); },
    }).default;
    const params = { club: " beta ", player_id: "17", player_name: "Blair" };
    const markup = renderToStaticMarkup(await Page({ searchParams: Promise.resolve(params) }));
    assert.equal(gateProps.searchParams, params, `${route} passes resolved query parameters to the intake gate`);
    assert.equal(gateProps.path, `/${route}`);
    assert.equal(formProps.clubSlug, "beta", `${route} keeps the explicit club after resolving Next's Promise`);
    if (route === "profile-privacy") {
      assert.equal(formProps.playerId, "17");
      assert.equal(formProps.playerName, "Blair");
    }
    if (route !== "data-corrections") assert.match(markup, /href="\/data-corrections\?club=beta"/);
    renderToStaticMarkup(await Page({ searchParams: Promise.resolve({ club: ["alpha", "beta"] }) }));
    assert.equal(formProps.clubSlug, "", `${route} does not choose from ambiguous clubs`);
    renderToStaticMarkup(await Page({}));
    assert.equal(formProps.clubSlug, "", `${route} has no implicit default club`);
    assert.equal(gateProps.searchParams, undefined);
  }
}

async function gateBehavior() {
  let readinessCalls = [], directoryCalls = [], ready = true, missing = false, directoryBroken = false;
  const { default: Gate, supportParam } = load("components/SupportIntakeGate.tsx", {
    "next/link": ({ children, ...props }) => h("a", props, children),
    "@/lib/publicSupportContent": { SUPPORT_EMAIL: "support@example.test" },
    "@/lib/clubSiteServer": { getClubDirectory: async (q, offset) => {
      directoryCalls.push({ q, offset });
      if (directoryBroken) throw new Error("private internal detail");
      return { clubs: [{ slug: "alpha", name: "Alpha" }, { slug: "beta", name: "Beta" }], offset, limit: 100, total: 101 };
    } },
    "@/lib/supportIntakeApi": { getPublicSupportIntakeStatus: async slug => {
      readinessCalls.push(slug);
      return missing ? { data: null, status: 404 } : { data: { club: { id: slug, slug, name: "Beta" }, accepting_requests: ready } };
    } },
  });
  const render = async searchParams => renderToStaticMarkup(await Gate({ path: "/support", searchParams, children: h("form", { "data-intake": true }) }));
  let markup = await render({});
  assert.match(markup, /Choose the club/);
  assert.match(markup, /\/support\?club=beta/);
  assert.doesNotMatch(markup, /data-intake|tres-palapas/);
  assert.deepEqual(readinessCalls, []);
  assert.match(markup, /offset=100/);
  await render({ q: "Baja", offset: "100" });
  assert.deepEqual(directoryCalls.at(-1), { q: "Baja", offset: 100 });
  assert.equal(supportParam({ club: ["alpha", "beta"] }, "club"), "", "Ambiguous club parameters must never silently choose one");

  markup = await render({ club: "beta" });
  assert.deepEqual(readinessCalls, ["beta"]);
  assert.match(markup, /Send to Beta/);
  assert.match(markup, /data-intake/);
  ready = false;
  markup = await render({ club: "beta" });
  assert.doesNotMatch(markup, /data-intake/);
  assert.match(markup, /Your request has not been submitted/);
  assert.match(markup, /mailto:support@example.test/);
  missing = true;
  markup = await render({ club: "missing" });
  assert.match(markup, /club could not be found/);
  assert.doesNotMatch(markup, /data-intake/);
  directoryBroken = true;
  markup = await render({});
  assert.match(markup, /club list is temporarily unavailable/);
  assert.doesNotMatch(markup, /data-intake|private internal detail/);
}

async function apiBehavior() {
  process.env.JUPR_API_BASE_URL = "https://api.example.test";
  const api = load("lib/supportIntakeApi.ts");
  let calls = [], nextStatus = 503;
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(nextStatus === 503 ? { detail: "JUPR_PRIVATE_SETTING unavailable" }
      : { club: { id: "beta", slug: "beta", name: "Beta" }, accepting_requests: true, ok: true }), { status: nextStatus });
  };
  assert.match((await api.submitPublicSupportRequest("", request)).error, /Choose your club/);
  assert.equal(calls.length, 0);
  const unavailable = await api.submitPublicSupportRequest("beta", request);
  assert.match(unavailable.error, /contact your club organizer or email PCS support/i);
  assert.doesNotMatch(unavailable.error, /JUPR_/);
  assert.equal(unavailable.data, null);
  assert.equal(calls[0].url, "https://api.example.test/clubs/beta/support/intake");
  nextStatus = 200;
  const result = await api.getPublicSupportIntakeStatus("beta");
  assert.equal(result.data.accepting_requests, true);
  assert.equal(calls.at(-1).options.cache, "no-store");
  assert.match(calls.at(-1).url, /\/clubs\/beta\/support\/intake\/status$/);
}

async function formsKeepExplicitClubAndFailedInput() {
  const previousFormData = global.FormData;
  global.FormData = class { get() { return ""; } };
  try {
    for (const [file, type] of [
      ["app/support/SupportRequestForm.tsx", "general_support"],
      ["app/data-corrections/DataCorrectionForm.tsx", "data_correction"],
      ["app/profile-privacy/ProfilePrivacyRequestForm.tsx", "profile_privacy"],
    ]) {
      let tree, submission;
      const Form = load(file, { "@/lib/supportIntakeApi": { submitPublicSupportRequest: async (club, payload) => {
        submission = { club, payload };
        return { data: null, error: "Contact your club organizer or email PCS support." };
      } } }).default;
      await act(async () => { tree = create(h(Form, { clubSlug: "beta", playerId: "17", playerName: "Blair" })); });
      const fill = async (labelText, value) => {
        const label = tree.root.findAllByType("label").find(label => text(label).startsWith(labelText));
        const input = label.findAll(node => node.type === "input" || node.type === "textarea")[0];
        await act(async () => input.props.onChange({ target: { value } }));
      };
      await fill("Your name", "Blair");
      await fill("Your email", "blair@example.test");
      if (type === "profile_privacy") {
        await fill("Request details", "Review my public name.");
      } else {
        await fill("Short subject", "Please review");
        await fill(type === "general_support" ? "How can we help?" : "What looks wrong?", "Request details");
      }
      await act(async () => tree.root.findByProps({ type: "checkbox" }).props.onChange({ target: { checked: true } }));
      await act(async () => tree.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} }));
      assert.equal(submission.club, "beta");
      assert.equal(submission.payload.request_type, type);
      if (type === "profile_privacy") {
        assert.equal(submission.payload.player_id, "17");
        assert.equal(submission.payload.player_name, "Blair");
      }
      assert.equal(tree.root.findByProps({ type: "email" }).props.value, "blair@example.test", "Failed submission preserves entered details");
      assert.match(text(tree.root.findByProps({ role: "alert" })), /Contact your club organizer/);
      await act(async () => tree.unmount());
    }
  } finally { global.FormData = previousFormData; }
}

(async () => {
  await pageQueryParameters();
  await gateBehavior();
  await apiBehavior();
  await formsKeepExplicitClubAndFailedInput();
  console.log("Support intake: async page queries, explicit club selection, readiness gates, fallback contact, scoped forms and preserved failed input passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
