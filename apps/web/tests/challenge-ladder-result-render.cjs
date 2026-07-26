const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const ts = require("typescript");

const componentPath = path.resolve(
  __dirname,
  "../components/ChallengeLadderResultDetails.tsx"
);
const source = fs.readFileSync(componentPath, "utf8");
const output = ts.transpileModule(source, {
  fileName: componentPath,
  compilerOptions: {
    esModuleInterop: true,
    jsx: ts.JsxEmit.ReactJSX,
    module: ts.ModuleKind.CommonJS,
    moduleResolution: ts.ModuleResolutionKind.NodeJs,
    target: ts.ScriptTarget.ES2022
  }
}).outputText;

const compiled = new Module(componentPath, module);
compiled.filename = componentPath;
compiled.paths = Module._nodeModulePaths(path.dirname(componentPath));
compiled._compile(output, componentPath);
const ChallengeLadderResultDetails = compiled.exports.default;

const challenger = {
  player_id: 3,
  player_name: "Casey Court",
  rank_at_create: 2,
  current_rank: 1,
  current_rating_jupr: 3.8
};
const defender = {
  player_id: 1,
  player_name: "Avery Ace",
  rank_at_create: 1,
  current_rank: 2,
  current_rating_jupr: 4.2
};
const challenge = {
  id: 81,
  tier_id: "PREM",
  status: "COMPLETED",
  bucket: "Recently Completed",
  challenger,
  defender,
  winner: challenger,
  completed_at: "2026-07-25T12:00:00Z"
};
const details = {
  version: 1,
  completeness: "full",
  rank_change: {
    swapped: true,
    challenger: {
      player_id: 3,
      player_name: "Casey Court",
      before: 2,
      after: 1,
      delta: -1
    },
    defender: {
      player_id: 1,
      player_name: "Avery Ace",
      before: 1,
      after: 2,
      delta: 1
    }
  },
  matches: [
    {
      slot: "a",
      match_id: 501,
      date: "2026-07-25T12:00:00Z",
      score_challenger_team: 22,
      score_defender_team: 15,
      challenger_partner: { player_id: 4, player_name: "Devon Dink" },
      defender_partner: { player_id: 5, player_name: "Emery Erne" },
      rating_changes: [
        {
          player_id: 3,
          player_name: "Casey Court",
          before_jupr: 3.75,
          after_jupr: 3.775,
          delta_jupr: 0.025
        }
      ]
    },
    {
      slot: "b",
      match_id: 502,
      date: "2026-07-25T12:30:00Z",
      score_challenger_team: 21,
      score_defender_team: 17,
      challenger_partner: { player_id: 5, player_name: "Emery Erne" },
      defender_partner: { player_id: 4, player_name: "Devon Dink" },
      rating_changes: [
        {
          player_id: 1,
          player_name: "Avery Ace",
          before_jupr: 4.225,
          after_jupr: 4.2,
          delta_jupr: -0.025
        }
      ]
    }
  ]
};

const html = renderToStaticMarkup(
  React.createElement(ChallengeLadderResultDetails, {
    challenge,
    details,
    clubSlug: "tres-palapas"
  })
);

assert.match(html, /data-result-completeness="full"/);
assert.match(
  html,
  /Position change:.*Casey Court.*#2.*→.*#1.*Avery Ace.*#1.*→.*#2/s
);
assert.match(
  html,
  /href="\/clubs\/tres-palapas\/matches\/501"[^>]*>Match A: 22–15/
);
assert.match(
  html,
  /href="\/clubs\/tres-palapas\/matches\/502"[^>]*>Match B: 21–17/
);
assert.match(
  html,
  /href="\/clubs\/tres-palapas\/players\/4"[^>]*>Devon Dink/
);
assert.match(
  html,
  /href="\/clubs\/tres-palapas\/players\/5"[^>]*>Emery Erne/
);
assert.doesNotMatch(html, /href="[^"]*challenge-ladder\?[^"]*player=/);
assert.match(
  html,
  /Match A: 22–15.*Challenger team:.*Casey Court.*Devon Dink.*Defender team:.*Avery Ace.*Emery Erne/s
);
assert.match(
  html,
  /Match B: 21–17.*Challenger team:.*Casey Court.*Emery Erne.*Defender team:.*Avery Ace.*Devon Dink/s
);
assert.match(html, /Casey Court.*3\.750.*→.*3\.775.*\(\+0\.025\)/s);
assert.match(html, /Avery Ace.*4\.225.*→.*4\.200.*\(-0\.025\)/s);
assert.doesNotMatch(html, /context_id|private-operation/);

console.log("Challenge Ladder full-result rendering contract passed.");
