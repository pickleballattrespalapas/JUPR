const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const read = (relative) => fs.readFileSync(path.join(root, relative), "utf8");

const roster = read("components/GeneratorRosterSetup.tsx");
const draft = read("lib/playGeneratorDraft.ts");
const adminWorkspace = read("app/admin/play-generators/GeneratorWorkspace.tsx");
const publicWorkspace = read("app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx");
const adminRunner = read("app/admin/play-generators/GeneratorRoundRunner.tsx");
const publicRunner = read("app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx");
const adminStandings = read("app/admin/play-generators/GeneratorStandings.tsx");
const publicStandings = read("app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx");

assert.match(roster, /type PlayFormat = "singles" \| "doubles" \| "doubles_singles"/);
assert.match(roster, /Doubles courts/);
assert.match(roster, /Singles courts/);
assert.match(roster, /recommendedMixedCourtSetup/);
assert.match(roster, /mixedRoundCount/);
assert.match(roster, /singles games, doubles games, partners, opponents, and byes balanced/);

assert.match(draft, /version: 2/);
assert.match(draft, /doublesCourtCount/);
assert.match(draft, /singlesCourtCount/);
assert.match(draft, /doubles_singles/);

for (const workspace of [adminWorkspace, publicWorkspace]) {
  assert.match(workspace, /Doubles \+ Singles Mix/);
  assert.match(workspace, /doubles_court_count/);
  assert.match(workspace, /singles_court_count/);
  assert.match(workspace, /matchFormatLabel/);
  assert.match(workspace, /\["Round", "Format", "Court"/);
  assert.match(workspace, /Download one-sheet PDF/);
}

for (const runner of [adminRunner, publicRunner]) {
  assert.match(runner, /Doubles \+ Singles Mix/);
  assert.match(runner, /matchFormatLabel\(match, event\.playFormat\)/);
  assert.match(runner, /Round Played/);
  assert.match(runner, /View standings and continue/);
}

assert.match(adminRunner, /Singles games publish\s+to singles ratings, and doubles games publish to doubles ratings/);
assert.match(adminStandings, /Doubles \+ Singles Mix/);
assert.match(publicStandings, /Doubles \+ Singles Mix/);

console.log("Mixed Round-Robin component contracts passed.");
