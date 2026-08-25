const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webRoot = path.resolve(__dirname, "..");
const read = (relativePath) => fs.readFileSync(path.join(webRoot, relativePath), "utf8");

const board = read("app/admin/league-manager/live/LeagueMovementCourtBoard.tsx");
const panel = read("app/admin/league-manager/live/LeagueLiveRoundPanel.tsx");
const css = read("app/admin/league-manager/live/LeagueMovementCourtBoard.module.css");
const domain = read("../../jupr_app/domain/league_live_orchestration.py");

assert.match(board, /DragDropContext/);
assert.match(board, /Droppable/);
assert.match(board, /Draggable/);
assert.match(board, /onAssignmentsChange\(assignmentsFor\(next\)\)/);
assert.match(board, /toSlot: column\.courtNumber == null \? null : index \+ 1/);
assert.match(board, /data-movement-direction=\{meta\.direction\}/);
assert.match(board, /Moved up from Court/);
assert.match(board, /Moved down from Court/);
assert.match(board, /Manual board change/);
assert.match(board, /Avg JUPR/);
assert.match(panel, /<LeagueMovementCourtBoard/);
assert.match(panel, /setMovementPlanStale\(true\)/);
assert.match(panel, /setBenchOverrideIds\(nextBench\)/);
assert.match(panel, /const automaticPlan = await requestMovementPlan/);
assert.match(panel, /The server-generated court board opens automatically/);
assert.match(panel, /Validate board changes/);
assert.doesNotMatch(panel, /Manual movement override reason/);
assert.doesNotMatch(panel, /Bench change reason/);
assert.doesNotMatch(panel, /override_reason: overrideReason/);
assert.match(panel, /override_reason: null/);
assert.match(panel, /bench_override_reason: null/);
assert.doesNotMatch(panel, /Final court for \$\{row\.player_name\}/);
assert.match(domain, /A reordered court board must assign every next-round player exactly once/);
assert.match(domain, /card order must use every slot/);
assert.match(domain, /def _assign_performance_slots/);
assert.match(domain, /Down-movers enter at the top/);
assert.match(domain, /Up-movers enter at the bottom/);
assert.doesNotMatch(domain, /Explain a manual movement override/);
assert.match(css, /\.playerCard\.up/);
assert.match(css, /\.playerCard\.down/);
assert.match(css, /cursor: grab/);
assert.match(css, /outline: 3px solid #60a5fa/);

console.log("league movement court-board contract: ok");
