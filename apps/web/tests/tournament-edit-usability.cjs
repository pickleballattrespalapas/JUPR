const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] ? { __esModule: true, ...overrides[name] } : originalRequire(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}

const text = node => typeof node === "string" ? node : (node.children || []).map(text).join("");
const Dialog = ({ open, title, children, actions }) => open ? React.createElement("section", { role: "dialog", "aria-label": title }, children, actions) : null;
const Link = ({ children, ...props }) => React.createElement("a", props, children);
const eligibility = load("lib/tournamentRegistrationEligibility.ts", {
  "@/lib/tournamentSkillEligibility": load("lib/tournamentSkillEligibility.ts")
});
const base = "app/clubs/[clubSlug]/tournament-registration/";
const submissions = [];
const quotes = [];
let submitResponse;
let quoteResponse;
const Form = load(base + "edit/EditTournamentRegistrationForm.tsx", {
  "next/link": { default: Link },
  "@/components/interaction": { InteractionDialog: Dialog },
  "@/components/tournaments/TournamentPartnerDetails": { default: ({ value, onChange }) => React.createElement("input", { "aria-label": "Partner name", value: value.name, onChange: event => onChange({ name: event.target.value }) }) },
  "../TournamentCommerceChooser": { default: ({ onReviewChange, initialSelections }) => {
    React.useEffect(() => { onReviewChange(initialSelections || [], null); }, []);
    return null;
  } },
  "@/lib/tournamentRegistrationEligibility": eligibility,
  "@/lib/tournamentCommerceApi": {
    formatCommerceMoney: total => `$${(total / 100).toFixed(2)}`,
    quoteTournamentCommerce: async (_club, payload) => { quotes.push(payload); return quoteResponse(); }
  },
  "@/lib/tournamentRegistrationApi": {
    submitClubTournamentRegistrationEdit: async (_club, payload) => { submissions.push(payload); return submitResponse(); }
  }
}).default;
const registration = { id: "reg", first_name: "Fixture", last_name: "Player", email: "fixture@example.invalid", age: 70, gender: "Men", updated_at: "version-1" };
const events = [
  { id: "doubles", registration_day_id: "day", event_family_label: "Doubles", division_name: "Open", partner_required: true, partner_board_enabled: true, selectable: true, skill_mode: "OPEN", price_usd: 20 },
  { id: "singles", registration_day_id: "day", event_family_label: "Singles", division_name: "Singles", partner_required: false, selectable: true, skill_mode: "OPEN", price_usd: 20 }
];
const selections = [
  { id: "selection-doubles", event_option_id: "doubles", partner_mode: "NEEDS_PARTNER", show_on_partner_board: true, updated_at: "selection-1" },
  { id: "selection-singles", event_option_id: "singles", partner_mode: "NONE", updated_at: "selection-2" }
];
const props = { clubSlug: "fixture-club", tournamentId: "t1", registrationSlug: "fixture-tournament", editToken: "fixture-token", registration, selections, events, players: [], days: [{ id: "day", label: "Day 1" }] };
const formValues = { first_name: "Fixture", last_name: "Player", age: "70", gender: "Men", notes: "Private staff note", phone: "555-1234" };
const success = () => ({ data: { registration_id: "reg", confirmation_token: "confirmation", email_delivery: { status: "sent" } } });
const quote = (total = 4000) => ({ quote_fingerprint: `quote-${total}`, currency: "USD", total_minor: total, request: { event_option_ids: ["doubles", "singles"], item_selections: [] } });
let renderer;
const button = name => renderer.root.findAllByType("button").find(node => text(node) === name);
const dialog = () => renderer.root.findAllByProps({ role: "dialog" })[0];
const mount = async (patch = {}) => {
  if (renderer) await act(async () => renderer.unmount());
  submissions.length = 0;
  quotes.length = 0;
  submitResponse = success;
  quoteResponse = () => ({ data: { quote: quote() } });
  await act(async () => { renderer = create(React.createElement(Form, { ...props, ...patch }), {
    createNodeMock: element => element.type === "form" ? {} : element.type === "fieldset" ? { querySelectorAll: () => [] } : null
  }); });
  await act(async () => button("Edit event").props.onClick());
};

async function verifySaving() {
  await mount();
  assert.equal(dialog().findAllByType("select").length, 0, "The partner decision is visible, not hidden in a dropdown");
  await act(async () => renderer.root.findByProps({ value: "HAS_PARTNER", type: "radio" }).props.onChange());
  assert.equal(dialog().findAllByType("input").filter(node => node.props.type === "checkbox").length, 0, "Partner Board visibility is not offered after adding a partner");
  await act(async () => renderer.root.findByProps({ "aria-label": "Partner name" }).props.onChange({ target: { value: "Chosen Partner" } }));
  await act(async () => button("Save changes").props.onClick());
  assert.equal(submissions.length, 1, "The event Save changes button submits immediately");
  assert.equal(submissions[0].selections[0].partner_name, "Chosen Partner");
  assert.equal(submissions[0].selections[0].partner_mode, "HAS_PARTNER");
  assert.equal(submissions[0].selections[0].show_on_partner_board, false);
  assert.equal(submissions[0].selections[1].id, "selection-singles", "Other entries are retained");
  assert.equal(submissions[0].notes, formValues.notes);
  assert.equal(submissions[0].phone, formValues.phone);
  assert.equal(submissions[0].expected_updated_at, "version-1");
  assert.deepEqual(submissions[0].expected_selection_versions, selections.map(({ id, updated_at }) => ({ id, updated_at })));
  assert.equal(submissions[0].terms_accepted, true, "The save action accepts the agreement displayed beside it");
  assert.match(text(renderer.root), /Registration changes saved/);
  assert.equal(renderer.root.findAllByType("form").length, 0, "There is no second save after success");

  await mount();
  await act(async () => renderer.root.findByProps({ "aria-describedby": "edit-partner-board-help" }).props.onChange({ target: { checked: false } }));
  submitResponse = () => ({ error: "Registration changed. Refresh your edit link.", status: 409 });
  await act(async () => button("Save changes").props.onClick());
  assert.match(text(dialog().findByProps({ role: "alert" })), /Registration changed/);
  assert.equal(submissions[0].selections[0].partner_mode, "NEEDS_PARTNER", "Hiding a public listing must not change partner status");
  assert.equal(renderer.root.findByProps({ "aria-describedby": "edit-partner-board-help" }).props.checked, false, "Failure preserves the draft");
  submitResponse = () => { throw Error("offline"); };
  await act(async () => button("Save changes").props.onClick());
  assert.match(text(dialog().findByProps({ role: "alert" })), /connection/);
  assert.equal(button("Save changes").props.disabled, false, "Network errors release the save lock");

  await mount();
  let finish;
  submitResponse = () => new Promise(resolve => { finish = resolve; });
  const save = button("Save changes").props.onClick;
  let pending;
  await act(async () => { pending = save(); void save(); });
  assert.equal(submissions.length, 1, "Repeated clicks cannot submit a second write");
  assert.equal(button("Remove event").props.disabled, true);
  await act(async () => { finish(success()); await pending; });

  await mount();
  await act(async () => button("Remove event").props.onClick());
  assert.deepEqual(submissions[0].selections.map(row => row.event_option_id), ["singles"], "Removal persists without a bottom-of-page save");
  await mount({ selections: [selections[0]] });
  await act(async () => button("Remove event").props.onClick());
  assert.equal(submissions.length, 0);
  assert.match(text(dialog().findByProps({ role: "alert" })), /Keep at least one event/);

  await mount();
  submitResponse = () => ({ data: { registration_id: "reg" } });
  await act(async () => button("Save changes").props.onClick());
  assert.equal(renderer.root.findAllByType("form").length, 0, "A successful save without a receipt must not invite another write");
}

async function verifyPrices() {
  const commerce = { available: true, items: [], variants: [], bundles: [], bundle_components: [], promotions: [] };
  await mount({ commerce, commerceOrder: { total_minor: 4000, updated_at: "order-1" } });
  await act(async () => button("Save changes").props.onClick());
  assert.equal(quotes.length, 1);
  assert.equal(submissions.length, 1, "An unchanged total refreshes and saves in one click");
  assert.equal(submissions[0].commerce.expected_quote_fingerprint, "quote-4000");
  assert.equal(submissions[0].commerce.expected_order_updated_at, "order-1");

  await mount({ commerce, commerceOrder: { total_minor: 4000 } });
  quoteResponse = () => ({ data: { quote: quote(4500) } });
  await act(async () => button("Save changes").props.onClick());
  assert.equal(submissions.length, 0, "A changed price must be reviewed before writing");
  assert.match(text(dialog()), /\$45\.00/);
  await act(async () => button("Save changes").props.onClick());
  assert.equal(submissions.length, 1, "The changed price can be accepted in the same dialog");
  assert.equal(quotes.length, 1);
}

async function verifyRecovery() {
  if (renderer) await act(async () => renderer.unmount());
  const requests = [];
  let response = { data: { ok: true, accepted: true } };
  const EditLink = load(base + "EditLinkRequestForm.tsx", {
    "@/components/interaction": { InteractionDialog: Dialog },
    "@/lib/tournamentRegistrationApi": { requestClubTournamentRegistrationEditLink: async (_club, payload) => { requests.push(payload); return response; } }
  }).default;
  await act(async () => { renderer = create(React.createElement(EditLink, { ...props, initialEmail: "fixture@example.invalid" })); });
  await act(async () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} }));
  assert.equal(requests.length, 1);
  assert.match(text(dialog()), /fixture@example.invalid/);
  assert.match(text(dialog()), /Open your email and click/);
  assert.match(text(dialog()), /matches a registration/, "Do not expose whether an address is registered");
  await act(async () => button("OK, I’ll check my email").props.onClick());
  assert.equal(dialog(), undefined);
  assert.match(text(renderer.root.findByProps({ role: "status" })), /fixture@example.invalid/);
  response = { error: "Unavailable" };
  await act(async () => renderer.root.findByType("form").props.onSubmit({ preventDefault() {}, currentTarget: {} }));
  assert.equal(dialog(), undefined, "Failed sends do not show the success dialog");
  assert.match(text(renderer.root.findByProps({ role: "alert" })), /couldn’t send/);
}

async function verifyNavigation() {
  if (renderer) await act(async () => renderer.unmount());
  const nav = load("components/PublicTournamentNav.tsx", { "next/link": { default: Link }, "./PublicTournamentNav.module.css": { default: {} } });
  await act(async () => { renderer = create(React.createElement(nav.default, { ...props, active: "edit-registration" })); });
  const edit = renderer.root.findAllByType("a").find(node => text(node) === "Edit my registration");
  assert.equal(edit.props.href, "/clubs/fixture-club/tournament-registration/manage?tournament=fixture-tournament");
  assert.equal(edit.props["aria-current"], "page");
  assert.equal(nav.publicTournamentHref("fixture-club", "edit-registration", "t1"), "/clubs/fixture-club/tournament-registration/manage?tournament_id=t1");
  const requests = [];
  const ManagePage = load(base + "manage/page.tsx", {
    "@/components/PublicTournamentSponsors": { default: () => null },
    "@/components/PublicTournamentNav": { default: nav.default },
    "../EditLinkRequestForm": { default: () => React.createElement("form", { "data-testid": "edit-link" }) },
    "@/lib/tournamentRegistrationApi": { getClubTournamentRegistration: async (club, query) => { requests.push({ club, ...query }); return { data: { tournament: { id: "t1" }, registration_open: false, settings: {} } }; } }
  }).default;
  await act(async () => renderer.unmount());
  await act(async () => { renderer = create(await ManagePage({ params: { clubSlug: "fixture-club" }, searchParams: { tournament_id: "t1" } })); });
  assert.equal(requests[0].tournamentId, "t1");
  assert.equal(renderer.root.findAllByType("form").length, 1, "The edit tab opens directly to recovery even after new registration closes");
}

async function main() {
  const originalFormData = global.FormData;
  global.FormData = class { get(name) { return formValues[name] ?? null; } };
  try {
    await verifySaving();
    await verifyPrices();
    await verifyRecovery();
    await verifyNavigation();
    console.log("Registration editing: one-step saves, partner choices, errors, duplicate clicks, removal, quote review, email dialog and scoped navigation passed.");
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.FormData = originalFormData;
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
