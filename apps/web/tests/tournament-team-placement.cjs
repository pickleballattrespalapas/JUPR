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
  const original = compiled.require.bind(compiled);
  compiled.require = name => {
    if (name.endsWith(".css")) return {};
    if (overrides[name]) return { __esModule: true, ...overrides[name] };
    if (name.startsWith("@/")) return load(name.slice(2) + ".ts");
    return original(name);
  };
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText, filename);
  return compiled.exports;
}
const { playersNeedingTeam } = load("lib/tournamentTeamPlacement.ts");
const ids = snapshot => playersNeedingTeam(snapshot, "e1").map(row => row.id);
const registration = (id, extra = {}) => ({ id, email: `${id}@example.invalid`, status: "CONFIRMED", ...extra });
const playerIds = ["solo", "assigned", "invited", "captain", "declined", "removed", "other-division", "cancelled", "cancelled-selection", "inactive-selection"];
const snapshot = {
  registrations: playerIds.map(id => registration(id)),
  selections: playerIds.map(id => ({ registration_id: id, event_option_id: id === "other-division" ? "e2" : "e1" })),
  teams: [{ id: "t1", event_option_id: "e1", status: "CONFIRMED", captain_registration_id: "captain" }],
  members: [
    { team_id: "t1", registration_id: "assigned", status: "ACCEPTED" },
    { team_id: "t1", invited_email: " INVITED@example.invalid ", status: "INVITED" },
    { team_id: "t1", registration_id: "declined", status: "DECLINED" },
    { team_id: "t1", registration_id: "removed", status: "REMOVED" },
  ],
};
snapshot.registrations.find(row => row.id === "cancelled").status = "CANCELLED";
snapshot.selections.find(row => row.registration_id === "cancelled-selection").status = "CANCELLED";
snapshot.selections.find(row => row.registration_id === "inactive-selection").is_active = false;
assert.deepEqual(ids(snapshot), ["solo", "declined", "removed"]);
assert.deepEqual(playersNeedingTeam(snapshot, "e2").map(row => row.id), ["other-division"]);
snapshot.members.push({ team_id: "t1", registration_id: "solo", status: "ACCEPTED" });
assert.deepEqual(ids(snapshot), ["declined", "removed"], "Assignment removes the individual from the pool");
snapshot.members.pop();
snapshot.members.push({ team_id: "t1", registration_id: "different-id", invited_email: "solo@example.invalid", status: "INVITED" });
assert.ok(ids(snapshot).includes("solo"), "Email must not override a different registration link");
snapshot.teams[0].status = "WITHDRAWN";
assert.deepEqual(ids(snapshot), ["solo", "assigned", "invited", "captain", "declined", "removed"]);
assert.deepEqual(playersNeedingTeam(null, "e1"), []);
assert.deepEqual(playersNeedingTeam(snapshot, ""), []);

const eligibility = load("lib/tournamentRegistrationEligibility.ts");
const card = load("components/tournaments/FourPlayerTeamRegistrationCard.tsx", { "@/lib/tournamentRegistrationEligibility": eligibility });
const Recovery = load("app/clubs/[clubSlug]/tournament-registration/confirmation/FourPlayerTeamSetupRecovery.tsx", {
  "@/components/tournaments/FourPlayerTeamRegistrationCard": card,
  "@/lib/tournamentRegistrationEligibility": eligibility,
  "@/lib/tournamentTeamCompetitionApi": { createPublicFourPlayerTeam: () => assert.fail("rendering cannot write"), recoverPublicFourPlayerTeamSetup: () => assert.fail("rendering cannot fetch") },
}).default;
const recovery = { tournament: { id: "t1" }, captain: { registration_id: "solo", display_name: "Solo Player", email: "solo@example.invalid", gender: "Men", registration_status: "CONFIRMED" }, events: [{ id: "e1", event_family_label: "Team Tournament", division_name: "Mixed 5.0", setup_state: "SETUP_REQUIRED" }] };
const props = { clubSlug: "fixture", confirmationToken: "fixture-only", initialRecovery: recovery };
let renderer;
act(() => { renderer = create(React.createElement(Recovery, props)); });
assert.equal(renderer.root.findByType("details").props.open, undefined);
assert.equal(renderer.root.findAllByProps({ "data-testid": "team-placement-pending" }).length, 1);
act(() => renderer.unmount());
act(() => { renderer = create(React.createElement(Recovery, { ...props, needsAttention: true })); });
assert.equal(renderer.root.findByType("details").props.open, true, "Failed captain setup remains recoverable");
act(() => renderer.unmount());
recovery.events[0] = { ...recovery.events[0], setup_state: "COMPLETE", team: { id: "t1", name: "Assigned Team", status: "CONFIRMED", members: [{ member_id: "m1", slot: "MAN_2", display_name: "Solo Player", status: "ACCEPTED" }] } };
act(() => { renderer = create(React.createElement(Recovery, props)); });
assert.equal(renderer.root.findAllByType("details").length, 0);
assert.match(JSON.stringify(renderer.toJSON()), /Assigned Team/);
assert.equal(renderer.root.findAllByProps({ "data-testid": "team-placement-pending" }).length, 0);
act(() => renderer.unmount());
console.log("Tournament solo placement and confirmation passed");
