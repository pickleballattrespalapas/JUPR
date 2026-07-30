const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webRoot = path.resolve(__dirname, "..");
const routeRoot = path.join(webRoot, "app", "admin", "match-uploader");
const layout = fs.readFileSync(path.join(routeRoot, "layout.tsx"), "utf8");
const css = fs.readFileSync(path.join(routeRoot, "layout.module.css"), "utf8");
const form = fs.readFileSync(path.join(routeRoot, "MatchUploaderForm.tsx"), "utf8");

assert.match(layout, /className=\{styles\.root\}/, "route layout must scope the acceptance CSS");
assert.match(css, /input\[aria-readonly="true"\]/, "read-only league and match-type fields must be hidden");
assert.match(css, /select:disabled/, "the fixed POPUP league selector must be hidden");
assert.match(css, /section\[aria-label\$="Team 1"\]/, "Team 1 layout selector is required");
assert.match(css, /section\[aria-label\$="Team 2"\]/, "Team 2 layout selector is required");
assert.match(css, /Team 1 score/, "Team 1 score must be labelled in the center");
assert.match(css, /Team 2 score/, "Team 2 score must be labelled in the center");
assert.match(css, /overflow-x: auto/, "narrow screens must scroll rather than stack Team 2 below Team 1");
assert.match(form, /aria-readonly="true"/, "layout selectors must match the current internal fields");
assert.match(form, /aria-label=\{`Match \$\{index \+ 1\} Team 1`\}/, "layout selectors must match Team 1 markup");
assert.match(form, /aria-label=\{`Match \$\{index \+ 1\} Team 2`\}/, "layout selectors must match Team 2 markup");

console.log("Match Uploader layout acceptance checks passed.");
