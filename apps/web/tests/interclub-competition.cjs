const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const { renderToStaticMarkup } = require('react-dom/server');
const base = 'app/admin/interclub/competition/';
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const registration = load('lib/interclubRegistration.ts');
const types = load('lib/interclubCompetition.ts', { './interclubRegistration': registration });
const css = new Proxy({}, { get: (_, key) => key === '__esModule' ? false : key });
const common = { '@/lib/interclubCompetition': types, './competition.module.css': css };
const ScoreEditor = load(base + 'ScoreEditor.tsx', common).default;
const PrintPacket = load(base + 'PrintPacket.tsx', common).PrintPacketContent;
const Standings = load(base + 'Standings.tsx', common).default;
let currentClub = 'alpha';
const workspace = load(base + 'CompetitionWorkspace.tsx', { ...common, 'next/link': ({ children, href }) => React.createElement('a', { href }, children), '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/adminWorkspace': { readBrowserWorkspace: () => ({ clubId: currentClub }) }, '@/lib/useAdminSession': { useAdminSession: () => ({}) }, '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: currentClub }) }, './ScoreEditor': ScoreEditor, './PrintPacket': PrintPacket, './Standings': Standings, './ScheduleMeet': () => null });
const nodeText = node => typeof node === 'string' ? node : node.children.map(nodeText).join('');
const button = (tree, label) => tree.root.findAllByType('button').find(node => nodeText(node) === label);
const text = tree => JSON.stringify(tree.toJSON());
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const copy = value => JSON.parse(JSON.stringify(value));
const clubName = id => ({ alpha: 'Alpha Club', beta: 'Beta Club' }[id] || id);
const players = Array.from({ length: 8 }, (_, index) => ({ entry_id: `entry-${index}`, name: index === 0 ? '<script>alert(1)</script>' : `Player ${index}`, gender: index % 4 < 2 ? 'female' : 'male', starting_rating: 3.7 }));
const teams = ['alpha', 'beta'].map((club_id, index) => ({ id: `team-${index}`, club_id, division: '3.5', revision: 1, name: clubName(club_id), roster: players.slice(index * 4, index * 4 + 4) }));
const game = (id, status = 'pending') => ({ id, status, a: status === 'completed' ? 11 : null, b: status === 'completed' ? 5 : null, winner: null, players_a: [], players_b: [], played_at: '2099-01-16T17:00:00Z', injury_reason: null });
const document = { schema_version: 1, meet_id: 'meet-1', phase: 'regular', format: 'gender', weather: 'normal', encounters: [{ id: 'encounter-1', division: '3.5', club_a: 'alpha', club_b: 'beta', rotation: 1, tiebreak: null, pairings: [
  { id: 'women', kind: 'women', court: 1, players_a: ['entry-0', 'entry-1'], players_b: ['entry-4', 'entry-5'], games: [1, 2, 3].map(n => game(`w${n}`)) },
  { id: 'men', kind: 'men', court: 2, players_a: ['entry-2', 'entry-3'], players_b: ['entry-6', 'entry-7'], games: [1, 2, 3].map(n => game(`m${n}`)) },
] }] };
const meet = { id: 'meet-1', host_club_id: 'alpha', club_ids: ['alpha', 'beta'], starts_at: '2099-01-16T17:00:00Z', roster_deadline: '2099-01-14T17:00:00Z', courts: 2 };
const batch = { meet_id: meet.id, phase: 'regular', revision: 4, state: 'draft', document, roster_sources: [{ team_id: 'team-0', revision: 1 }, { team_id: 'team-1', revision: 1 }], ratings_status: 'not_requested' };
const detail = { meet, batch, teams, is_organizer: true, can_manage: true, eligible_players: { alpha: players.slice(0, 4), beta: players.slice(4) } };
const context = { season: { id: 'season-1', details: { name: 'Southern BCS', divisions: ['3.5'], timezone: 'America/Mazatlan' } }, standings: { divisions: {}, qualification: {} }, club_cup: { standings: [], champions: [], status: 'provisional' } };
global.window = { addEventListener() {}, removeEventListener() {}, print() {} };

async function scoreEntry() {
  let next = copy(document), tree;
  const props = { detail, players: types.competitionPlayers(detail), clubName, disabled: false, onChange: value => { next = value; } };
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  const status = tree.root.findAllByType('select').find(node => node.props['aria-label'] === 'Women’s doubles game 1 status');
  await act(async () => status.props.onChange({ target: { value: 'forfeit' } }));
  assert.equal(next.encounters[0].pairings[0].games[0].a, null, 'Forfeits never invent score values');
  assert.equal(next.encounters[0].pairings[0].games[0].b, null);
  await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next })));
  assert.equal(tree.root.findByProps({ 'aria-label': 'women game 1 club A score' }).props.disabled, true);
  assert.ok(text(tree).includes('Game awarded to'));
  assert.ok(text(tree).includes('only because of injury'));
  await act(async () => tree.root.findAllByType('select').find(node => node.props['aria-label'] === 'Women’s doubles game 1 status').props.onChange({ target: { value: 'double_forfeit' } }));
  assert.equal(next.encounters[0].pairings[0].games[0].winner, null, 'Double forfeits award no invented game winner');
  assert.equal(next.encounters[0].pairings[0].games[0].a, null);
  assert.equal(types.playerNames([], new Map()), 'Pairing not fielded');
  const playerSelect = tree.root.findAllByType('select').find(node => node.props.value === 'entry-0');
  assert.ok(playerSelect.children.every(option => !option.props?.value?.startsWith('entry-4')), 'Starting lineup choices stay in represented club');
  await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next, disabled: true })));
  assert.ok(tree.root.findAllByType('fieldset').every(node => node.props.disabled), 'Approved/read-only document has no editable score fields');
  await act(async () => tree.unmount());
}

function printSafety() {
  const markup = renderToStaticMarkup(React.createElement(PrintPacket, { document, meet, seasonName: 'Southern BCS', timezone: 'America/Mazatlan', revision: 4, players: types.competitionPlayers(detail), clubName }));
  assert.ok(markup.includes('&lt;script&gt;alert(1)&lt;/script&gt;'), 'Player labels are escaped in print HTML');
  assert.ok(!markup.includes('<script>'));
  assert.ok(markup.includes('Court assignments') && markup.includes('Verified by club A') && markup.includes('Time played'));
  assert.ok(markup.includes('all three games') && markup.includes('injury / forfeit / unplayed'));
  assert.equal(types.gameCount(document).total, 6, 'Two club matchup has two three-game pairings, three games per player');
  const final = copy(document); final.phase = 'final'; final.encounters[0].division = '4.0';
  const finalsMarkup = renderToStaticMarkup(React.createElement(PrintPacket, { document: final, meet, seasonName: 'Southern BCS', timezone: 'America/Mazatlan', revision: 4, players: types.competitionPlayers(detail), clubName }));
  assert.ok(finalsMarkup.includes('full-court') && finalsMarkup.includes('after every four rallies'));
  assert.ok(finalsMarkup.includes('No individual rating effect'));
  assert.equal(types.singlesCourt('Open'), 'Full-court');
  const singlesEncounter = { pairings: [{ kind: 'mixed_a', players_a: ['w1', 'm1'], games: [{ players_a: ['substitute', 'm1'] }] }, { kind: 'mixed_b', players_a: ['w2', 'm2'], games: [{ players_a: [] }] }] };
  assert.deepEqual(types.activeSinglesPlayers(singlesEncounter, 'a'), ['substitute', 'm1', 'w2', 'm2'], 'An eligible injury replacement can join rotating singles');
}

async function revisionsAndStaleClub() {
  let saved = copy(detail), requests = [], finish, tree;
  const root = 'https://api.test/admin/clubs/alpha/interclub/competition/season-1/meets/meet-1/regular';
  global.fetch = async (url, options) => { requests.push({ url, options }); return options.method ? new Promise(resolve => { finish = resolve; }) : reply(saved); };
  const props = { root, clubId: 'alpha', accessToken: 'token-1', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
  assert.equal(button(tree, 'Print meet packet').props.disabled, false);
  assert.equal(button(tree, 'Review and submit meet').props.disabled, true, 'Incomplete meet cannot be submitted');
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(document), weather: 'delay' }));
  assert.equal(button(tree, 'Print meet packet').props.disabled, true, 'Packet must reflect a saved revision');
  await act(async () => tree.update(React.createElement(workspace.MeetOperations, { ...props, accessToken: 'token-2' })));
  await act(async () => { void button(tree, 'Save all draft scores').props.onClick(); void button(tree, 'Save all draft scores').props.onClick(); });
  assert.equal(requests.filter(request => request.options.method).length, 1, 'Double click cannot create two revisions');
  assert.equal(requests.at(-1).options.headers.Authorization, 'Bearer token-2');
  const body = JSON.parse(requests.at(-1).options.body);
  assert.equal(body.expected_revision, 4);
  assert.equal(body.document.weather, 'delay');
  saved = { ...saved, batch: { ...saved.batch, revision: 5, document: body.document } };
  await act(async () => finish(reply({ batch: saved.batch })));
  assert.equal(button(tree, 'Print meet packet').props.disabled, false);
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(saved.batch.document), weather: 'finalized_partial' }));
  currentClub = 'beta';
  await act(async () => button(tree, 'Save all draft scores').props.onClick());
  assert.equal(requests.filter(request => request.options.method).length, 1, 'A stale tab cannot save into a previous club');
  assert.ok(text(tree).includes('selected club changed'));
  await act(async () => tree.unmount()); currentClub = 'alpha';
}

async function approval() {
  let saved = copy(detail), requests = [], tree;
  saved.batch.state = 'submitted'; saved.batch.document.encounters.forEach(encounter => encounter.pairings.forEach(pairing => { pairing.games = pairing.games.map(value => game(value.id, 'completed')); }));
  global.fetch = async (url, options) => { requests.push({ url, options }); return options.method ? reply({ batch: { ...saved.batch, state: 'approved', revision: 6, ratings_status: 'pending' } }) : reply(saved); };
  const props = { root: 'https://api.test/meet', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
  assert.equal(button(tree, 'Save all draft scores'), undefined);
  await act(async () => button(tree, 'Review official approval').props.onClick());
  assert.equal(requests.filter(request => request.options.method).length, 0, 'Review is separate from official approval');
  assert.ok(text(tree).includes('Approve revision'));
  await act(async () => button(tree, 'Approve this revision').props.onClick());
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { expected_revision: 4 });
  assert.ok(text(tree).includes('Rating updates: ') && text(tree).includes('pending'));
  assert.ok(!text(tree).includes('Both league and represented-club updates are complete'));
  await act(async () => button(tree, 'Retry rating updates').props.onClick());
  assert.ok(requests.at(-1).url.endsWith('/retry-ratings'));
  await act(async () => tree.unmount());
}

function qualifyingDisplay() {
  const data = { ...context, standings: { divisions: { '3.5': [{ club_id: 'alpha', points: 3, pairings_won: 2, games_won: 4, point_differential: 8, meets_played: 1, position: 2, tied: true }] }, qualification: { '3.5': { qualifiers: [], playoff_required: ['alpha', 'beta'], status: 'playoff_required' } } }, club_cup: { standings: [], status: 'complete', champions: ['alpha', 'beta'] } };
  const markup = renderToStaticMarkup(React.createElement(Standings, { data, clubName }));
  assert.ok(markup.includes('Qualifying playoff needed') && markup.includes('no club advances on alphabetical order'));
  assert.ok(markup.includes('Joint Club Cup champions'));
}
function writePrintReview() {
  const output = process.env.PCS_PRINT_REVIEW_PATH;
  if (!output) return;
  const displayPlayers = new Map(players.map((player, index) => [player.entry_id, { ...player, name: ['Ana Rivera', 'Maria Costa', 'Luis Santos', 'Carlos Vega', 'Elena Moreno', 'Sofia Reyes', 'Mateo Cruz', 'Diego Luna'][index] }]));
  const render = value => renderToStaticMarkup(React.createElement(PrintPacket, { document: value, meet, seasonName: 'Southern BCS Interclub League', timezone: 'America/Mazatlan', revision: 4, players: displayPlayers, clubName }));
  const final = copy(document); final.phase = 'final'; final.format = 'mlp'; final.encounters[0].division = '4.0';
  const encounter = final.encounters[0];
  encounter.pairings.forEach(pairing => { pairing.games = [pairing.games[0]]; pairing.court = 1; });
  encounter.pairings.push({ id: 'mixed-a', kind: 'mixed_a', court: 1, players_a: ['entry-0', 'entry-2'], players_b: ['entry-4', 'entry-6'], games: [game('mixed-a')] }, { id: 'mixed-b', kind: 'mixed_b', court: 1, players_a: ['entry-1', 'entry-3'], players_b: ['entry-5', 'entry-7'], games: [game('mixed-b')] });
  const stylesheet = fs.readFileSync(path.join(__dirname, '..', base, 'competition.module.css'), 'utf8');
  const screenPreview = '@media screen {body {margin:0;background:#edf1f5;font:14px Arial;} .printPortal,.printRoot {display:block;} .printPage {box-sizing:border-box;max-width:210mm;margin:12px auto;padding:12mm;background:white;} .printRoot table{width:100%;border-collapse:collapse;} .printRoot td,.printRoot th{border:1px solid #555;padding:5px;}} .printRoot + .printRoot {break-before:page;}';
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Southern BCS paper packet review</title><style>' + stylesheet + screenPreview + '</style></head><body class="printBody"><div class="printPortal">' + render(document) + render(final) + '</div></body></html>');
  console.log('Print review fixture: ' + output);
}
(async () => { await scoreEntry(); printSafety(); await revisionsAndStaleClub(); await approval(); qualifyingDisplay(); writePrintReview(); console.log('PASS interclub competition: paper packet safety, scoped lineups, non-play scoring, exact revisions, stale club protection, approval and ratings status, qualification and joint Cup'); })().catch(error => { console.error(error); process.exit(1); });
