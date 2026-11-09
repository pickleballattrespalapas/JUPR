const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const root = path.join(__dirname, '..');
const mod = {exports:{}};
const source = fs.readFileSync(path.join(root, 'lib/leagueLiveDateReview.ts'), 'utf8');
new Function('exports', ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(mod.exports);
const {leagueLiveToday, leagueLiveDateReview} = mod.exports;
const now = new Date('2026-11-09T02:00:00Z'); // November 8 at Tres Palapas, November 9 UTC.
assert.equal(leagueLiveToday(now), '2026-11-08');
assert.equal(leagueLiveToday(new Date('2026-11-09T07:00:00Z')), '2026-11-09');
assert.equal(leagueLiveDateReview('2026-11-08','session',1,now).needsReview, false);
const restored = leagueLiveDateReview('2026-11-01','session',1,now);
assert.equal(restored.needsReview, true);
assert.equal(leagueLiveDateReview('2026-11-15','session',1,now).needsReview, true);
// Acknowledging one historical round must not approve another session, round, date, or day.
for (const next of [
  leagueLiveDateReview('2026-11-01','other-session',1,now),
  leagueLiveDateReview('2026-11-01','session',2,now),
  leagueLiveDateReview('2026-11-02','session',1,now),
  leagueLiveDateReview('2026-11-01','session',1,new Date('2026-11-09T15:00:00Z')),
]) assert.notEqual(next.reviewKey, restored.reviewKey);
const panel = fs.readFileSync(path.join(root,'app/admin/league-manager/live/LeagueLiveRoundPanel.tsx'),'utf8');
const submit = panel.slice(panel.indexOf('async function submitRound('),panel.indexOf('async function submitRound(')+4200);
assert(submit.indexOf('currentDateReview.needsReview') < submit.indexOf('/submit'));
assert(panel.includes('!matchDateReviewed || !allSeriesComplete'));
assert(panel.includes('data-testid="league-live-date-review"'));
const persistedDraft = panel.slice(panel.indexOf('type StoredLeagueLiveRoundDraft'),panel.indexOf('function todayIso'));
assert(!persistedDraft.includes('acknowledgedMatchDateKey'));
console.log('League Live date review tests passed');
