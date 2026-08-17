const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const read = (relative) => fs.readFileSync(path.join(root, relative), "utf8");

const panel = read(
  "app/admin/tournaments/live-operations/check-in/TournamentCheckInPanel.tsx"
);
const page = read("app/admin/tournaments/live-operations/check-in/page.tsx");
const api = read("lib/adminTournamentCheckInApi.ts");
const css = read(
  "app/admin/tournaments/live-operations/check-in/TournamentCheckInPanel.module.css"
);

assert.match(page, /initialDayId=/, "the route must hand the selected URL day to the client");
assert.match(panel, /Tournament day/, "check-in must expose a clearly labelled day selector");
assert.match(panel, /useRouter/, "day changes must be reflected in navigation state");
assert.match(panel, /params\.set\("day_id"/, "the selected day must be preserved in the URL");
assert.match(api, /dayId\?: string/, "the check-in read contract must accept a day id");
assert.match(api, /searchParams\.set\("day_id"/, "the API read must request one selected day");
assert.match(api, /attendance_status/, "each card needs an authoritative attendance bucket");
assert.match(panel, /attendanceStatus\(card\)/, "filters and badges must use attendance status");
assert.doesNotMatch(
  panel,
  /filter === "absent" && draft\.checkedIn/,
  "unchecked players must never be inferred to be absent"
);
assert.doesNotMatch(
  panel,
  /Absent \/ not checked in/,
  "the absent filter must not conflate absence with pending check-in"
);
assert.match(panel, /Not checked in/, "pending expected players need an honest status label");
assert.match(panel, /selectedDayLabel/, "readiness copy must identify the selected day");
assert.match(panel, /No players are scheduled for/, "an empty selected day needs an explicit state");
assert.match(css, /\.dayPicker/, "the day selector needs a responsive layout hook");
assert.match(css, /@media \(max-width: 760px\)/, "the workspace must retain its mobile layout");

console.log("tournament check-in day-scope frontend contract passed");
