const fs = require("fs");
const assert = require("assert");

const uploader = fs.readFileSync("app/admin/match-uploader/MatchUploaderForm.tsx", "utf8");
const uploaderApi = fs.readFileSync("lib/adminMatchUploaderApi.ts", "utf8");
const leagueManager = fs.readFileSync("app/admin/league-manager/LeagueManagerPanel.tsx", "utf8");
const leagueApi = fs.readFileSync("lib/adminLeagueManagerApi.ts", "utf8");

assert.match(uploaderApi, /doubles_league_options\?: string\[\]/);
assert.match(uploaderApi, /singles_league_options\?: string\[\]/);
assert.match(uploader, /entryMethod === "singles" \? singlesLeagueOptions : doublesLeagueOptions/);
assert.match(uploader, /No active \{activeLeagueFormatLabel\} leagues/);
assert.match(uploader, /Create or activate a Singles league/);
assert.match(leagueManager, /League format/);
assert.match(leagueManager, /match_format: createMatchFormat/);
assert.match(leagueApi, /match_format\?: "doubles" \| "singles"/);
console.log("League match-format UI contract passed.");
