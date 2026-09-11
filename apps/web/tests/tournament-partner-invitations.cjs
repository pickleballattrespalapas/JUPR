const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { act, create } = require("react-test-renderer");
const ts = require("typescript");
global.crypto ||= require("node:crypto").webcrypto;
function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, "..", relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename; compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] || original(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}
const content = node => typeof node === "string" ? node : (node.children || []).map(content).join("");
const Link = ({ children, ...props }) => React.createElement("a", props, children);
const invitation = { ok: true, role: "target", status: "PENDING", actions: ["accept", "decline"],
  requester_name: "Sam Sender", target_name: "Alex Player", message: "Can we play together?",
  tournament_name: "Fixture Tournament", division_name: "Mixed 3.5", expires_at: "2026-09-23T12:00:00Z",
  board_url: "/board", roster_url: "/roster" };

async function main() {
  const calls = [];
  let failDelivery = true;
  const FormDialog = ({ open, children }) => open ? React.createElement("section", null, children) : null;
  const Panel = load("app/clubs/[clubSlug]/tournament-partner-board/PartnerInvitationPanel.tsx", {
    "@/components/interaction/FormDialog": { FormDialog },
    "@/components/interaction/types": load("components/interaction/types.ts"),
    "@/lib/tournamentPartnerInvitations": { partnerInvitationRequest: async (_club, action, payload) => {
      calls.push({ action, payload });
      return { status: "PENDING", notification_status: { request_target: failDelivery ? "failed" : "dry_run" } };
    } }
  }).default;
  let renderer;
  await act(async () => { renderer = create(React.createElement(Panel, { apiBase: "http://fixture.local", clubSlug: "fixture", tournamentId: "fixture",
    entry: { player_name: "Alex Player", division: "Mixed 3.5", board_entry_key: "opaque-reference" } })); });
  assert.equal(calls.length, 0);
  await act(async () => renderer.root.findByType("button").props.onClick());
  const field = name => renderer.root.findAllByType("label").find(n => content(n).startsWith(name));
  await act(async () => field("Your full name").findByType("input").props.onChange({ target: { value: "Sam Sender" } }));
  await act(async () => field("Your email").findByType("input").props.onChange({ target: { value: "sender@example.com" } }));
  await act(async () => field("Your message").findByType("textarea").props.onChange({ target: { value: "Can we play together?" } }));
  await act(async () => { await assert.rejects(renderer.root.findByType(FormDialog).props.onSubmit, /email couldn’t be sent/); });
  failDelivery = false;
  let sent;
  await act(async () => { sent = await renderer.root.findByType(FormDialog).props.onSubmit(); });
  assert.equal(calls[0].payload.request_key, calls[1].payload.request_key, "A delivery retry retains the saved request identity");
  assert.equal(calls[1].payload.board_entry_key, "opaque-reference");
  assert.equal(calls[1].payload.email, "sender@example.com");
  assert.ok(!("target_email" in calls[1].payload));
  assert.match(JSON.stringify(sent), /Request sent to Alex Player/);
  assert.doesNotMatch(JSON.stringify(sent), /Check your email|confirmation link|Send my partner request/);
  assert.equal(content(renderer.root.findByType("button")), "Request sent");
  assert.doesNotMatch(content(renderer.root), /confirm your email/);
  await act(async () => renderer.unmount());

  global.window = { location: { hash: "#token=private-fixture-link" } };
  calls.length = 0;
  let state = structuredClone(invitation);
  const request = async (_club, action, payload) => {
    calls.push({ action, payload });
    if (action === "respond") state = { ...state, status: "COMPLETED", actions: [] };
    return structuredClone(state);
  };
  const ResponsePage = load("app/clubs/[clubSlug]/tournament-partner-request/PartnerInvitationResponse.tsx", {
    "next/link": { __esModule: true, default: Link },
    "@/lib/tournamentPartnerInvitations": { partnerInvitationRequest: request }
  }).default;
  await act(async () => { renderer = create(React.createElement(ResponsePage, { clubSlug: "fixture" })); });
  assert.deepEqual(calls.map(c => c.action), ["review"], "Opening an email link cannot accept a partnership or send mail");
  assert.match(content(renderer.root), /Confirm your partnership/);
  await act(async () => renderer.root.findAllByType("button").find(n => content(n) === "Confirm partnership").props.onClick());
  assert.equal(calls.at(-1).payload.action, "accept");
  assert.equal(calls.at(-1).payload.token, "private-fixture-link");
  assert.match(content(renderer.root), /You’re partnered up!/);
  assert.equal(renderer.root.findAllByType("a").find(n => content(n) === "View our team").props.href, "/roster");
  await act(async () => renderer.unmount());

  state = { ...invitation, role: "requester", status: "RESERVED", actions: ["cancel"],
    registration_url: "/registration#partner_invitation=private-fixture-link",
    registration_prefill: { name: "Sam Sender", email: "sender@example.com", event_option_id: "mixed" } };
  await act(async () => { renderer = create(React.createElement(ResponsePage, { clubSlug: "fixture" })); });
  assert.match(content(renderer.root), /Complete your registration for this division/);
  assert.equal(renderer.root.findAllByType("a").find(n => content(n) === "Complete registration").props.href, state.registration_url);
  assert.ok(!renderer.root.findAllByType("button").some(n => content(n) === "Confirm partnership"));
  await act(async () => renderer.unmount());
  const { usePartnerInvitationRegistration } = load("components/tournaments/usePartnerInvitationRegistration.tsx", {
    "@/lib/tournamentPartnerInvitations": { partnerInvitationRequest: request }
  });
  window.location.hash = "#partner_invitation=private-fixture-link";
  let registrationContext;
  function RegistrationProbe() { registrationContext = usePartnerInvitationRegistration("fixture"); return null; }
  await act(async () => { renderer = create(React.createElement(RegistrationProbe)); });
  assert.equal(registrationContext.token, "private-fixture-link");
  assert.deepEqual(registrationContext.invitation.registration_prefill, state.registration_prefill);
  await act(async () => renderer.unmount());
  const RosterPage = load("app/clubs/[clubSlug]/tournament-roster/page.tsx", {
    "next/link": { __esModule: true, default: Link },
    "next/navigation": { redirect: () => { throw Error("Unexpected redirect"); } },
    "@/components/PublicTournamentSponsors": { __esModule: true, default: () => null },
    "@/components/PublicTournamentModuleHeader": { __esModule: true, default: () => null },
    "@/lib/tournamentRegistrationApi": { getClubTournamentRoster: async () => ({ data: {
      tournament: { id: "fixture", name: "Fixture Tournament" }, settings: { registration_slug: "fixture-tournament" },
      summary: { total_registrations: 1, total_players: 1 },
      roster: { registrations_by_event: [{ public_entry_key: "opaque-entry", status: "Pending Registration", entry_type: "Team",
        event_day_label: "Day 1", event_family: "Mixed", division: "Below 9", members: [
          { display_name: "Alex Player" }, { display_name: "Sam Sender", registration_pending: true }
        ] }] }
    } }) }
  }).default;
  const rosterPage = await RosterPage({ params: { clubSlug: "fixture" }, searchParams: { tournament: "fixture-tournament", status: "Pending registration" } });
  await act(async () => { renderer = create(rosterPage); });
  assert.match(content(renderer.root), /Alex Player/);
  assert.match(content(renderer.root), /Sam Sender · Pending registration/);
  assert.match(content(renderer.root), /Showing 1 of 1 entries/);
  assert.ok(renderer.root.findAllByType("option").some(node => content(node) === "Pending registration"));
  await act(async () => renderer.unmount());
  console.log("Partner invitations: private form, retry identity, read-only email landing, explicit acceptance, roster confirmation and registration handoff passed.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
