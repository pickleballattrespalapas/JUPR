const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webRoot = path.resolve(__dirname, "..");
const routeRoot = path.join(webRoot, "app", "admin", "match-uploader");
const layout = fs.readFileSync(path.join(routeRoot, "layout.tsx"), "utf8");
const css = fs.readFileSync(path.join(routeRoot, "layout.module.css"), "utf8");
const form = fs.readFileSync(path.join(routeRoot, "MatchUploaderForm.tsx"), "utf8");

assert.match(layout, /className=\{styles\.root\}/, "route layout must scope Match Uploader styles");
assert.match(form, /league: string;/, "each manual match row must store its own league");
assert.match(form, /value=\{row\.league\}/, "official-league rows must expose an editable league selector");
assert.match(form, /context === "popup" \? "" : row\.weekTag/, "pop-up submissions must omit week tags");
assert.match(form, /context === "popup"[\s\S]*"overall_only"/, "pop-up submissions must be overall-only unless unrated");
assert.match(form, /context === "league" \? <label className=\{styles\.field\}><strong>Default week\/session/, "default week must only appear for official leagues");
assert.match(form, /styles\.metadataGrid/, "match metadata must use the non-overlapping grid");
assert.match(form, /styles\.teamsGrid/, "team and score entry must use the explicit layout grid");
assert.match(form, /Team 1 score[\s\S]*Team 2 score/, "team scores must appear together in the center section");
assert.match(form, /Starting JUPR must be between 1\.00 and 7\.00/, "invalid Starting JUPR must have a visible explanation");
assert.match(form, /Boolean\(startingJuprMessage\)/, "invalid Starting JUPR must disable player creation");
assert.match(form, /<strong>Context<\/strong>[\s\S]*Official League[\s\S]*Pop-Up \/ Social/, "singles and doubles must share context controls");
assert.match(form, /Singles match entry[\s\S]*styles\.metadataGrid[\s\S]*styles\.teamsGrid/, "singles entry must use the doubles metadata and player-score layout");
assert.match(form, /league: context === "popup" \? "POPUP" : \(singlesRow\.league \|\| defaultLeague\)/, "singles payload must preserve official or social context");
assert.match(form, /week_tag: context === "popup" \? "" : \(singlesRow\.weekTag \|\| defaultWeekTag\)/, "social singles must omit week tags");
assert.doesNotMatch(form, /Singles league\/tag/, "singles must not use its old one-off metadata labels");
assert.match(form, /rating changes always use the separate singles rating/, "singles context copy must clarify that ratings remain separate");
assert.match(form, /setSinglesValidationAttempted\(false\)/, "switching context or entry method must clear stale singles validation");
assert.match(css, /grid-template-columns: repeat\(4, minmax\(0, 1fr\)\)/, "desktop metadata fields must have explicit columns");
assert.match(css, /overflow-x: auto/, "narrow team layouts must scroll instead of overlapping");
assert.match(css, /grid-template-columns: minmax\(230px, 1fr\) minmax\(170px, 0\.65fr\) minmax\(230px, 1fr\)/, "teams and scores need three explicit columns");

console.log("Match Uploader context and responsive layout checks passed.");
