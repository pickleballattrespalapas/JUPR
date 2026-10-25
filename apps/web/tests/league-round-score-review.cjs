const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");
global.IS_REACT_ACT_ENVIRONMENT = true;

function load(file, imports = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "../app/admin/league-manager/live", file), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  const mod = { exports: {} };
  new Function("require", "module", "exports", code)(name => imports[name] || require(name), mod, mod.exports);
  return mod.exports;
}
const helpers = load("leagueRoundScores.ts");
const review = load("LeagueRoundScoreReview.tsx", { "./leagueRoundScores": helpers });
const history = load("LeagueSavedRoundScores.tsx", { "./leagueRoundScores": helpers, "./LeagueRoundScoreReview": review });
const players = ["Shannon", "Dana", "Jane", "Suzi", "Shannon"].map((name, index) => ({ id: index + 1, name }));
const game = (key, ids, score, extra = {}) => ({ key, court: 7, gameLabel: key, team1: ids.slice(0, 2).map(id => players[id - 1]), team2: ids.slice(2).map(id => players[id - 1]), scoreT1: score[0], scoreT2: score[1], ...extra });
const games = [game("G1", [1, 2, 3, 4], [11, 4]), game("G2", [1, 3, 2, 4], [11, 8]), game("G3", [1, 4, 2, 3], [9, 11])];
assert.deepEqual(helpers.roundCourtStandings(games).find(row => row.id === 1), { id: 1, name: "Shannon", wins: 2, losses: 1, differential: 8 });
const corrected = games.map(row => row.key === "G3" ? { ...row, scoreT1: 11, scoreT2: 9, correctionUrl: "/admin/match-log/edit?match_id=103" } : row);
assert.equal(helpers.roundCourtStandings(corrected).find(row => row.id === 1).wins, 3);
assert.equal(helpers.roundCourtStandings([...corrected, game("unplayed", [1, 5, 2, 3], [null, null], { notPlayed: true })]).find(row => row.id === 5).wins, 0, "Distinct player IDs remain separate even with the same display name");

const savedRound = { round_number: 1, status: "submitted", score_readback_status: "current", matches_json: [{ court: 7, t1_p1: 1, t1_p2: 4, t2_p1: 2, t2_p2: 3, score_t1: 11, score_t2: 9, official_match_id: 103, context_id: "ctx", score_corrected: true }] };
const mapped = helpers.savedRoundScoreGames(savedRound, new Map(players.map(player => [player.id, player.name])));
assert.equal(mapped[0].correctionUrl, "/admin/match-log/edit?match_id=103");
assert.equal(mapped[0].corrected, true);
assert.equal(mapped[0].officialMatchId, 103);
assert.equal(helpers.savedRoundScoreGames({ ...savedRound, score_readback_status: "unavailable" }, new Map())[0].correctionUrl, undefined);
assert.equal(helpers.savedRoundScoreGames({ ...savedRound, score_readback_status: "unavailable" }, new Map())[0].officialMatchId, undefined);
const courtGames = games.map((row, index) => ({ ...row, officialMatchId: 101 + index }));
assert.equal(helpers.courtScoreCorrectionUrl([...courtGames, { ...games[0], notPlayed: true }]), "/admin/match-log/bulk?match_ids=101%2C102%2C103&selected_ids=101%2C102%2C103");
assert.equal(helpers.courtScoreCorrectionUrl([...courtGames, games[0]]), undefined, "Do not silently omit a game whose official match cannot be verified");
const text = node => node.children.map(child => typeof child === "string" ? child : text(child)).join("");

(async () => {
  let tree, edited;
  await act(async () => { tree = create(React.createElement(review.LeagueRoundScoreReview, { games: [...games, game("other", [5, 2, 3, 4], [11, 3], { court: 8 })], onEdit: row => { edited = row; } })); });
  await act(async () => tree.root.findByProps({ type: "search" }).props.onChange({ target: { value: "Jane" } }));
  await act(async () => tree.root.findByProps({ "aria-label": "Filter scores by court" }).props.onChange({ target: { value: "7" } }));
  const table = tree.root.findByType("table");
  assert.match(text(table), /Shannon2–1\+8/, "Filtering games must not truncate the full-round records");
  const edit = tree.root.findAllByType("button").find(node => text(node) === "Edit this game");
  await act(async () => edit.props.onClick());
  assert.equal(edited.key, "G1");
  await act(async () => tree.unmount());

  await act(async () => { tree = create(React.createElement(review.LeagueRoundScoreReview, { games: [...courtGames, game("other", [5, 2, 3, 4], [11, 3], { court: 8, officialMatchId: 104 })] })); });
  await act(async () => tree.root.findByProps({ type: "search" }).props.onChange({ target: { value: "Shannon" } }));
  const courtLink = tree.root.findByType("a");
  assert.equal(courtLink.props.href, "/admin/match-log/bulk?match_ids=101%2C102%2C103&selected_ids=101%2C102%2C103", "Court correction includes all three games and excludes the other court");
  assert.equal(courtLink.props.target, "_blank");
  await act(async () => tree.unmount());

  let refreshed = 0;
  await act(async () => { tree = create(React.createElement(history.LeagueSavedRoundScores, { rounds: [savedRound, { ...savedRound, round_number: 2, operation_key: "applied" }], playerNames: new Map(players.map(player => [player.id, player.name])), busy: false, onRefresh: () => { refreshed++; } })); });
  assert.match(text(tree.root), /already been applied/);
  await act(async () => tree.root.findAllByType("select")[0].props.onChange({ target: { value: "1" } }));
  assert.doesNotMatch(text(tree.root), /already been applied/);
  const link = tree.root.findByType("a");
  assert.equal(link.props.href, "/admin/match-log/edit?match_id=103");
  assert.equal(link.props.target, "_blank", "Leave the current round and draft scores open");
  assert.equal(refreshed, 0, "Browsing old rounds does not mutate or reload the active form");
  await act(async () => tree.root.findAllByType("button").find(node => text(node) === "Refresh saved scores").props.onClick());
  assert.equal(refreshed, 1);
  await act(async () => tree.unmount());
  console.log("League round scores: 2–1/3–0 records, identity separation, filtered games, direct edits, history navigation and safe correction links passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
