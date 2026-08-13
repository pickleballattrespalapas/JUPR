const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webRoot = path.resolve(__dirname, "..");
const playerEditor = fs.readFileSync(path.join(webRoot, "app", "admin", "players", "PlayerEditorPanel.tsx"), "utf8");
const socialEditor = fs.readFileSync(path.join(webRoot, "app", "admin", "match-log", "MatchLogSocialPanel.tsx"), "utf8");
const leagueLive = fs.readFileSync(path.join(webRoot, "app", "admin", "league-manager", "live", "LeagueLiveRoundPanel.tsx"), "utf8");

for (const [name, source] of [["Player Editor", playerEditor], ["Club Social", socialEditor], ["League Live", leagueLive]]) {
  assert.match(source, /sessionStorage\?\.setItem/, `${name} must retain an uncertain operation across a reload`);
  assert.match(source, /sessionStorage\?\.removeItem/, `${name} must clear only a conclusive operation`);
  assert.match(source, /Check and reconcile exact operation/, `${name} must expose an exact-operation recovery control`);
  assert.match(source, /\/reconcile/, `${name} must use a dedicated reconcile endpoint`);
}

for (const [name, source] of [["Player Editor", playerEditor], ["League Live", leagueLive]]) {
  assert.match(source, /const explicitlyFailed = detailRecord\?\.kind === "failed" && detailRecord\?\.recovery_required !== true/, `${name} must honor a server-proven failure`);
  assert.match(source, /!explicitlyFailed && uncertainStatus/, `${name} must reserve status-based uncertainty for responses without conclusive failure evidence`);
}

assert.match(socialEditor, /explicit\.kind === "failed" && explicit\.recovery_required !== true/, "Club Social must honor a server-proven failure before status-based uncertainty");

assert.match(playerEditor, /confirmation_text: "RECONCILE PLAYER OPERATION"/, "Player Editor recovery must carry explicit intent");
assert.match(playerEditor, /Boolean\(writeRecovery\)/, "Player Editor must block a new write while recovery is pending");
assert.match(playerEditor, /reconcilePlayerWrite\(\s*retainedRecovery[\s\S]*const recovery = retainedRecovery/, "Player Editor recovery must use the retained operation instead of a stale render closure");
assert.match(playerEditor, /\(\) => reconcilePlayerWrite\(pending\)/, "Player Editor follow-up must retain the exact pending operation object");
assert.doesNotMatch(playerEditor, /Retry exact league-rating update/, "Player Editor must not re-submit a mutation as recovery");

assert.match(socialEditor, /confirmation_text: "RECONCILE SOCIAL MATCH"/, "Club Social recovery must carry explicit intent");
assert.match(socialEditor, /Boolean\(writeRecovery\)/, "Club Social must block a new write while recovery is pending");
assert.doesNotMatch(socialEditor, /Retry exact save request/, "Club Social must not re-PATCH as recovery");

assert.match(leagueLive, /confirmation_text: "RECONCILE LIVE SESSION"/, "League Live recovery must carry explicit intent");
assert.match(leagueLive, /Boolean\(createRecovery\)/, "League Live must block a new session create while recovery is pending");
assert.doesNotMatch(leagueLive, /\(\) => createSession\(confirmationText\)/, "League Live must not re-POST session creation as recovery");

const matchUploader = fs.readFileSync(path.join(webRoot, "app", "admin", "match-uploader", "MatchUploaderForm.tsx"), "utf8");
assert.match(matchUploader, /const explicitlyFailed = detailRecord\.kind === "failed" && detailRecord\.recovery_required !== true/, "Match Uploader must honor a server-proven failure");
assert.match(matchUploader, /!explicitlyFailed && \(status >= 500/, "Match Uploader must retain only non-conclusive transport failures");

const teamLeagues = fs.readFileSync(path.join(webRoot, "app", "admin", "league-manager", "teams", "TeamLeaguesPanel.tsx"), "utf8");
assert.match(teamLeagues, /async function refreshAfterConfirmedWrite[\s\S]*?try \{[\s\S]*?await refresh\(\);[\s\S]*?\} catch \{[\s\S]*?return confirmedWriteRefreshWarning;/, "Team League follow-up refresh failures must not turn confirmed writes into failures");
assert.match(teamLeagues, /do not repeat the completed action/, "Team League refresh guidance must prevent a blind mutation retry");
assert.equal((teamLeagues.match(/await refreshAfterConfirmedWrite\(/g) || []).length, 6, "Every Team League mutation with a follow-up refresh must preserve its confirmed success");
assert.doesNotMatch(teamLeagues, /await (?:onSaved|refreshDetail)\(\);/, "Team League mutations must not let a follow-up refresh reject after a confirmed write");

console.log("Guarded operation recovery contracts passed.");
