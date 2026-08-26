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
assert.match(panel, /tournamentRouteHref/, "the selected day must be preserved in route context");
assert.match(api, /dayId\?: string/, "the check-in read contract must accept a day id");
assert.match(api, /searchParams\.set\("day_id"/, "the API read must request one selected day");
assert.match(api, /attendance_status/, "each player row needs an authoritative attendance bucket");
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
assert.match(panel, /Bulk check-in actions/, "operators need one bulk-action workspace");
assert.match(panel, /Select all shown/, "operators need a filtered select-all action");
assert.match(panel, /Attendance action/, "attendance must be editable for selected rows");
assert.match(panel, /Waiver action/, "waiver verification must be editable for selected rows");
assert.match(
  panel,
  /updateAdminTournamentCheckInBulk/,
  "selected players must submit through one transactional bulk command"
);
assert.match(panel, /expected_updated_at/, "every selected row must retain its stale-write fence");
assert.match(panel, /bulkOperation/, "the bulk command must retain idempotency evidence");
assert.match(
  panel,
  /Review and reselect players before another action/,
  "an uncertain response must require a fresh visible selection after authoritative reload"
);
assert.doesNotMatch(
  panel,
  /setSelectedIds\(selectedForReview\)/,
  "an error reload must not restore hidden or no-longer-selectable player ids"
);
assert.match(
  api,
  /check-in\/bulk\?day_id=/,
  "bulk check-in must use the day-scoped transactional endpoint"
);
assert.match(
  api,
  /tournament_registration_check_in_bulk_update/,
  "the transactional bulk response must be typed"
);
assert.match(panel, /<table/, "player check-in must render as a compact list table");
assert.doesNotMatch(panel, /Save check-in/, "per-player save-card actions must be removed");
assert.match(
  panel,
  /Tournament-day play remains available regardless/,
  "check-in must be presented as operational tracking, not a day-workspace gate"
);
assert.match(css, /\.dayPicker/, "the day selector needs a responsive layout hook");
assert.match(css, /\.bulkActionBar/, "bulk actions need a responsive layout hook");
assert.match(css, /\.playerTable/, "the selectable roster needs table styling");
assert.match(css, /@media \(max-width: 760px\)/, "the workspace must retain its mobile layout");

console.log("tournament check-in day-scope frontend contract passed");
