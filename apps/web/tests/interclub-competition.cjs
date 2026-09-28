const playerSearchModules = require("./helpers/player-search-modules.cjs");
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript'), { create, act } = require('react-test-renderer');
const { renderToStaticMarkup } = require('react-dom/server');
const base = 'app/admin/interclub/competition/';
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : (playerSearchModules(name) || require(name)), module, module.exports);
  return module.exports;
}
const registration = load('lib/interclubRegistration.ts');
const types = load('lib/interclubCompetition.ts', { './interclubRegistration': registration });
const css = new Proxy({}, { get: (_, key) => key === '__esModule' ? false : key });
const common = { '@/lib/interclubCompetition': types, './competition.module.css': css };
const substitutions = load('lib/interclubSubstitutions.ts', { './interclubCompetition': types });
common['@/lib/interclubSubstitutions'] = substitutions;
common['./SubstitutionRepair'] = load(base + 'SubstitutionRepair.tsx', common);
common['./InjurySubstitutionEditor'] = load(base + 'InjurySubstitutionEditor.tsx', common);
common['./CourtSchedule'] = load(base + 'CourtSchedule.tsx', common);
const ScoreEditor = load(base + 'ScoreEditor.tsx', common).default;
const PrintPacket = load(base + 'PrintPacket.tsx', common).PrintPacketContent;
const Standings = load(base + 'Standings.tsx', common).default;
let currentClub = 'alpha';
const workflow = load('app/admin/interclub/InterclubWorkflow.tsx', { 'next/link': ({ children, ...props }) => React.createElement('a', props, children), './workflow.module.css': css });
const registrationWindow = load('lib/interclubRegistrationWindow.ts');
const windowHook = load('lib/useRegistrationWindow.ts', { './interclubRegistrationWindow': registrationWindow });
let createPdf = async () => { throw new Error('PDF mock not set'); };
const workspace = load(base + 'CompetitionWorkspace.tsx', { ...common, '@/lib/interclubMeetPdf': { buildInterclubMeetPdf: (...args) => createPdf(...args) }, '@/lib/useRegistrationWindow': windowHook, '../InterclubWorkflow': workflow, 'next/link': ({ children, href }) => React.createElement('a', { href }, children), '@/lib/adminAuthClient': { getAdminApiBaseUrl: () => 'https://api.test' }, '@/lib/adminWorkspace': { readBrowserWorkspace: () => ({ clubId: currentClub }) }, '@/lib/useAdminSession': { useAdminSession: () => ({}) }, '@/lib/useAdminWorkspace': { useAdminWorkspace: () => ({ clubId: currentClub }) }, './ScoreEditor': ScoreEditor, './PrintPacket': PrintPacket, './Standings': Standings, './ScheduleMeet': () => null });
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
const closedRegistration = { opens_at: '2000-01-01T00:00:00Z', closes_at: '2000-02-01T00:00:00Z', revision: 1, status: 'closed', can_register: false, meet_planning_open: true };
const context = { season: { registration: closedRegistration, id: 'season-1', details: { name: 'Southern BCS', divisions: ['3.5'], timezone: 'America/Mazatlan' } }, standings: { divisions: {}, qualification: {} }, club_cup: { standings: [], champions: [], status: 'provisional' } };
global.window = { addEventListener() {}, removeEventListener() {}, print() {} };

async function qualifyingRoundRobin() {
  const qualified = copy(detail);
  qualified.batch.phase = 'qualifier'; qualified.batch.document.phase = 'qualifier';
  qualified.teams.push({ ...copy(teams[0]), id: 'team-gamma', club_id: 'gamma' });
  global.fetch = async () => reply(qualified);
  let tree;
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, {
    root: 'https://api.test/qualifier', clubId: 'alpha', accessToken: 'token', phase: 'qualifier',
    context: { ...context, standings: { divisions: {}, qualification: { '3.5': { playoff_required: ['alpha', 'beta', 'gamma'] } } } },
    clubName, onLock() {}, onSeasonChange() {},
  })); });
  const selects = tree.root.findAllByType('select').filter(node => node.props.required);
  await act(async () => { selects[0].props.onChange({ target: { value: 'alpha' } }); selects[1].props.onChange({ target: { value: 'gamma' } }); });
  assert.equal(button(tree, 'Generate pairings').props.disabled, false, 'A third club can play its qualifying matchup at the same skill level');
  await act(async () => { selects[0].props.onChange({ target: { value: 'beta' } }); selects[1].props.onChange({ target: { value: 'alpha' } }); });
  assert.equal(button(tree, 'Generate pairings').props.disabled, true, 'An existing pair is blocked in either order');
  await act(async () => tree.unmount());
}

async function missingLineups() {
  global.fetch = async () => reply({ ...copy(detail), batch: null, teams: [] });
  let tree;
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, {
    root: 'https://api.test/empty-meet', clubId: 'alpha', accessToken: 'token', phase: 'regular',
    context, clubName, onLock() {}, onSeasonChange() {},
  })); });
  assert.equal(button(tree, 'Generate pairings').props.disabled, true, 'Pairings cannot be generated before at least two clubs have approved lineups');
  const link = tree.root.findAllByType('a').find(node => nodeText(node) === 'Prepare lineups for this meet →');
  const url = new URL(link.props.href, 'https://example.test');
  assert.equal(url.searchParams.get('season'), 'season-1');
  assert.equal(url.searchParams.get('meet'), 'meet-1');
  assert.equal(url.searchParams.get('step'), 'lineups', 'Missing-roster message takes the organizer directly to that meet’s lineup workspace');
  assert.equal(tree.root.findAllByType('a').some(node => node.props['aria-label'] === 'Approve results'), false, 'Approval is unavailable until scores are submitted');
  await act(async () => tree.unmount());
}

async function staggeredSchedule() {
  let saved = { ...copy(detail), batch: null }, requests = [], tree;
  global.fetch = async (url, options) => {
    if (options.method) {
      requests.push({ url, body: JSON.parse(options.body) });
      saved = { ...saved, batch: { ...copy(batch), document: { ...copy(document), schedule_mode: requests.at(-1).body.schedule_mode } } };
      return reply({ batch: saved.batch });
    }
    return reply(saved);
  };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, {
    root: 'https://api.test/new-meet', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {},
  })); });
  const choice = tree.root.findByProps({ 'aria-label': 'Court schedule' });
  assert.equal(choice.props.value, 'staggered');
  assert.ok(nodeText(choice).includes('fit 2 courts'));
  await act(async () => choice.props.onChange({ target: { value: 'simultaneous' } }));
  assert.ok(text(tree).includes('Each opponent rotation runs all divisions together'));
  await act(async () => choice.props.onChange({ target: { value: 'staggered' } }));
  await act(async () => tree.root.findByType('form').props.onSubmit({ preventDefault() {} }));
  assert.deepEqual(requests[0].body, { expected_revision: 0, format: 'gender', schedule_mode: 'staggered' });
  assert.ok(text(tree).includes('Staggered starts') && text(tree).includes('Wave'));
  await act(async () => tree.unmount());

  const doc = { ...copy(document), schedule_mode: 'staggered' };
  const later = copy(doc.encounters[0]);
  later.id = 'later'; later.rotation = 2; later.division = '4.0';
  later.pairings.forEach(p => { p.id += '-later'; });
  doc.encounters.unshift(later); // Intentionally store the later wave first.
  const markup = renderToStaticMarkup(React.createElement(PrintPacket, { document: doc, meet, seasonName: 'Southern BCS', timezone: 'America/Mazatlan', revision: 4, players: types.competitionPlayers(detail), clubName }));
  assert.ok(markup.includes('Staggered starts') && markup.includes('>Wave</th>'));
  assert.ok(markup.includes('finish all three games, even at 2-0, before leaving'));
  assert.equal((markup.match(/<td>1-3<\/td>/g) || []).length, 4, 'Each court assignment reserves all three games in one wave');
  assert.equal((markup.match(/Games 1-3 on this court/g) || []).length, 4, 'Each score sheet keeps the pairing on its assigned court');
  assert.ok(markup.indexOf('Wave 1') < markup.indexOf('Wave 2'), 'Printed score sheets follow playing order');
  const schedule = types.scheduledEncounters(doc);
  assert.deepEqual(schedule.map(e => e.rotation), [1, 2]);
  assert.equal(doc.encounters[0].rotation, 2, 'Sorting the displayed schedule does not mutate saved data');
  const simultaneous = renderToStaticMarkup(React.createElement(common['./CourtSchedule'].default, { document, clubName }));
  assert.ok(simultaneous.includes('finish all three games, even at 2-0, before leaving'), 'Simultaneous starts also reserve a full three-game court block');
}

async function seasonRegistrationGate() {
  const originalWindow = global.window;
  global.window = new EventTarget(); window.print = () => {};
  const lockedWindows = [undefined,
    { opens_at: null, closes_at: null, revision: 0, status: 'unconfigured', can_register: false, meet_planning_open: false },
    { opens_at: '2099-01-01T00:00:00Z', closes_at: '2099-02-01T00:00:00Z', revision: 1, status: 'scheduled', can_register: false, meet_planning_open: false },
    { opens_at: '2000-01-01T00:00:00Z', closes_at: '2099-02-01T00:00:00Z', revision: 1, status: 'open', can_register: true, meet_planning_open: false },
  ];
  const props = { clubId: 'alpha', accessToken: 'token', initialSeasonId: 'season-1', initialMeetId: 'meet-1' };
  try {
    for (const registration of lockedWindows) {
      const requests = [], scopedContext = { ...context, season: { ...context.season, registration }, clubs: [], meets: [], batches: [], is_organizer: true };
      global.fetch = async url => { requests.push(url); return reply(url.endsWith('/competition') ? { seasons: [scopedContext.season] } : scopedContext); };
      let tree;
      await act(async () => { tree = create(React.createElement(workspace.CompetitionHome, props)); });
      assert.ok(text(tree).includes('Meet planning opens after registration closes'));
      assert.equal(tree.root.findAllByType(ScoreEditor).length, 0);
      assert.equal(tree.root.findAllByType(Standings).length, 0, 'The closed phase is required before operational standings controls mount');
      assert.equal(tree.root.findAllByType('select').length, 1, 'Only season choice remains; no meet settings are mounted');
      assert.ok(!requests.some(url => url.includes('/meets/')), 'Direct competition deep links do not fetch a meet during registration');
      const pool = tree.root.findAllByType('a').find(link => nodeText(link) === 'Go to season player pool');
      const url = new URL(pool.props.href, 'https://example.test');
      assert.equal(url.searchParams.get('season'), 'season-1'); assert.equal(url.searchParams.get('meet'), 'meet-1');
      await act(async () => tree.unmount());
    }
    const scopedContext = { ...context, clubs: [], meets: [meet], batches: [batch], is_organizer: true };
    const reads = [];
    global.fetch = async url => { reads.push(url); return reply(url.endsWith('/competition') ? { seasons: [context.season] } : url.includes('/meets/') ? detail : scopedContext); };
    let tree;
    await act(async () => { tree = create(React.createElement(workspace.CompetitionHome, props)); });
    assert.equal(tree.root.findAllByType(ScoreEditor).length, 1, 'Confirmed closed registration retains the complete score workflow');
    await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(document), weather: 'delay' }));
    const meetReads = reads.filter(url => url.includes('/meets/')).length;
    await act(async () => window.dispatchEvent(new Event('focus')));
    assert.equal(tree.root.findByType(ScoreEditor).props.document.weather, 'delay', 'A phase recheck preserves an unsaved score draft when registration is still closed');
    assert.equal(reads.filter(url => url.includes('/meets/')).length, meetReads, 'Background phase checks do not remount the current meet');
    await act(async () => tree.unmount());
  } finally { global.window = originalWindow; }
}

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

async function pairingControls() {
  let next = copy(document), tree;
  const props = { detail, players: types.competitionPlayers(detail), clubName, disabled: false, onChange: value => { next = value; } };
  const summaries = () => tree.root.findAllByType('summary').map(nodeText);
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  assert.ok(!summaries().some(label => /starting pairing|mixed partners/.test(label)), 'Gender doubles keep the already selected pairs without a redundant lineup editor');
  assert.equal(summaries().filter(label => label.startsWith('Substitute a player')).length, 6, 'Each game offers a clearly named injury substitution control');
  next.format = 'mixed';
  Object.assign(next.encounters[0].pairings[0], { kind: 'mixed_a', players_a: ['entry-0', 'entry-2'], players_b: ['entry-4', 'entry-6'] });
  Object.assign(next.encounters[0].pairings[1], { kind: 'mixed_b', players_a: ['entry-1', 'entry-3'], players_b: ['entry-5', 'entry-7'] });
  await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next })));
  assert.equal(summaries().filter(label => label === 'Change mixed partners before play').length, 2, 'Mixed teams can still arrange the two partnerships');
  for (const value of [0, 5]) {
    next.encounters[0].pairings[0].games[0].a = value;
    await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next })));
    assert.ok(!summaries().includes('Change mixed partners before play'), 'Even a partial or zero score locks the starting partnerships');
  }
  next.encounters[0].pairings[0].games[0].a = null;
  await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next, disabled: true })));
  assert.ok(!summaries().includes('Change mixed partners before play'), 'Read-only results cannot rearrange partners');
  await act(async () => tree.unmount());
}

async function preMeetRosterChange() {
  const writes = [];
  global.fetch = async (url, options) => {
    if (options.method === 'POST') { writes.push({ url, body: JSON.parse(options.body) }); return reply({ batch: { ...copy(batch), revision: 5 } }); }
    return reply(copy(detail));
  };
  let tree;
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, { root: 'https://api.test/meet-1', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} })); });
  const links = () => tree.root.findAllByType('a').filter(node => nodeText(node) === 'Change meet roster');
  assert.equal(links().length, 1);
  const target = new URL(links()[0].props.href, 'https://web.test');
  assert.equal(target.searchParams.get('season'), 'season-1');
  assert.equal(target.searchParams.get('meet'), 'meet-1');
  assert.equal(target.searchParams.get('step'), 'lineups');
  await act(async () => button(tree, 'Apply updated meet rosters').props.onClick());
  assert.equal(writes[0].url, 'https://api.test/meet-1/refresh-lineups');
  assert.equal(writes[0].body.expected_revision, 4, 'Applying roster changes keeps the existing revision guard');
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(document), weather: 'delay' }));
  assert.equal(links().length, 0, 'Unsaved work blocks the roster navigation link');
  assert.equal(button(tree, 'Apply updated meet rosters').props.disabled, true, 'Unsaved scores cannot be overwritten by a roster refresh');
  const started = copy(document);
  started.encounters[0].pairings[0].games[0] = game('w1', 'completed');
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange(started));
  assert.equal(button(tree, 'Apply updated meet rosters'), undefined, 'Once play is recorded, roster replacement gives way to per-game injury substitutions');
  await act(async () => tree.unmount());
}

async function automaticScoreEntry() {
  let next = copy(document), tree;
  next.encounters[0].pairings[0].games[0].played_at = null;
  const props = { detail, players: types.competitionPlayers(detail), clubName, disabled: false, onChange: value => { next = value; } };
  const current = () => next.encounters[0].pairings[0].games[0];
  const update = async (label, value) => act(async () => {
    tree.root.findByProps({ 'aria-label': label }).props.onChange({ target: { value } });
    tree.update(React.createElement(ScoreEditor, { ...props, document: next }));
  });
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  await update('women game 1 club A score', '11');
  assert.equal(current().status, 'pending', 'One score is not a complete result');
  const entryStartedAt = Date.now();
  await update('women game 1 club B score', '0');
  assert.equal(current().status, 'completed', '11-0 completes automatically without selecting a status');
  const entryTime = current().played_at;
  assert.ok(Date.parse(entryTime) >= entryStartedAt && Date.parse(entryTime) <= Date.now(), 'Completion records the current device time, independent of the future schedule');
  assert.equal(tree.root.findAllByProps({ type: 'datetime-local' }).length, 0, 'No per-game date or time entry');
  assert.equal(types.gameCount(next).entered, 1);
  const outcome = tree.root.findByProps({ 'aria-label': 'Women’s doubles game 1 status' });
  assert.equal(outcome.props.value, 'automatic');
  assert.equal(outcome.parent.parent.type, 'details', 'Status choices are outside the normal score row');
  assert.equal(outcome.findAllByType('option').some(option => option.props.value === 'completed'), false, 'Completion never requires choosing a dropdown option');
  await update('women game 1 club A score', '');
  assert.equal(current().status, 'pending');
  assert.equal(types.gameCount(next).entered, 0, 'Removing either score removes completion');
  await update('women game 1 club A score', '10');
  assert.equal(current().status, 'pending', '10-0 is not a final score');
  assert.ok(text(tree).includes('A final score must reach 11'));
  await update('women game 1 club B score', '12');
  assert.equal(current().status, 'completed', '10-12 is a valid win-by-two result');
  assert.equal(current().played_at, entryTime, 'Corrections preserve the first entry date');
  await update('women game 1 club B score', '13');
  assert.equal(current().status, 'pending', '13-10 is past the first winning score');
  await update('Women’s doubles game 1 status', 'retired');
  await update('women game 1 club A score', '7');
  await update('women game 1 club B score', '4');
  await act(async () => {
    tree.root.findAllByType('select').find(node => node.findAllByType('option').some(option => nodeText(option) === 'Choose winner')).props.onChange({ target: { value: 'b' } });
    tree.update(React.createElement(ScoreEditor, { ...props, document: next }));
  });
  await update('women game 1 club A score', '8');
  assert.equal(current().status, 'retired', 'An explicit injury outcome survives score changes');
  assert.equal(current().winner, 'b', 'The injury winner is not inferred from the leading score');
  await update('Women’s doubles game 1 status', 'forfeit');
  assert.equal(current().a, null); assert.equal(current().b, null);
  assert.equal(current().winner, null);
  assert.equal(types.gameCount(next).entered, 0, 'A one-sided forfeit needs a winner');
  await update('Women’s doubles game 1 status', 'double_forfeit');
  assert.equal(types.gameCount(next).entered, 1);
  await update('Women’s doubles game 1 status', 'automatic');
  assert.equal(current().status, 'pending');
  assert.equal(current().a, null, 'Switching back to scores never restores invented forfeit scores');
  await act(async () => tree.unmount());

  const championship = copy(document); championship.phase = 'final'; championship.format = 'mlp';
  championship.encounters[0].tiebreak = { status: 'pending', a: null, b: null, order_a: [], order_b: [] };
  next = championship;
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  await update('encounter-1 singles club A score', '21');
  await update('encounter-1 singles club B score', '19');
  assert.equal(next.encounters[0].tiebreak.status, 'completed', 'Singles completes automatically at its 21-point target');
  await update('encounter-1 singles club B score', '20');
  assert.equal(next.encounters[0].tiebreak.status, 'pending', 'Singles still requires a two-point margin');
  await update('encounter-1 singles club A score', '22');
  assert.equal(next.encounters[0].tiebreak.status, 'completed');
  await update('encounter-1 singles club B score', '');
  assert.equal(next.encounters[0].tiebreak.status, 'pending');
  await act(async () => tree.unmount());
}

async function savedScoreCompletion() {
  let saved = copy(detail), tree, requests = [];
  for (const pairing of saved.batch.document.encounters[0].pairings) for (const game of pairing.games) {
    game.a = 5; game.b = 11; game.played_at = null; // Reproduce saved pending scores without a recorded date.
  }
  const loadedAt = Date.now();
  const original = JSON.stringify(saved.batch.document);
  global.fetch = async (url, options) => {
    if (options.method) {
      const body = JSON.parse(options.body); requests.push({ url, body });
      saved = { ...saved, batch: { ...saved.batch, revision: saved.batch.revision + 1,
        ...(body.document ? { document: body.document } : { state: 'submitted' }) } };
      return reply({ batch: saved.batch });
    }
    return reply(saved);
  };
  const props = { root: 'https://api.test/saved-scores', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
  assert.equal(types.gameCount(tree.root.findByType(ScoreEditor).props.document).entered, 6);
  assert.equal(JSON.stringify(saved.batch.document), original, 'Loading does not rewrite the saved revision');
  assert.equal(button(tree, 'Save all draft scores').props.disabled, false);
  assert.equal(button(tree, 'Review and submit meet').props.disabled, true, 'Inferred statuses must be saved before submitting');
  await act(async () => { await button(tree, 'Save all draft scores').props.onClick(); });
  assert.equal(requests[0].body.expected_revision, 4);
  assert.ok(requests[0].body.document.encounters[0].pairings.every(pairing => pairing.games.every(game => game.status === 'completed' && game.a === 5 && game.b === 11 && Date.parse(game.played_at) >= loadedAt && Date.parse(game.played_at) <= Date.now())));
  assert.equal(button(tree, 'Review and submit meet').props.disabled, false);
  await act(async () => { button(tree, 'Review and submit meet').props.onClick(); });
  await act(async () => { await button(tree, 'Submit all official scores').props.onClick(); });
  assert.equal(requests.at(-1).body.expected_revision, 5, 'Submission still binds to the saved revision');
  assert.ok(requests.at(-1).url.endsWith('/submit'));
  await act(async () => tree.unmount());

  for (const [state, canManage] of [['approved', true], ['submitted', true], ['draft', false]]) {
    const locked = copy(detail); locked.batch.state = state; locked.can_manage = canManage;
    locked.batch.document.encounters[0].pairings[0].games[0].a = 5;
    locked.batch.document.encounters[0].pairings[0].games[0].b = 11;
    global.fetch = async () => reply(locked);
    await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
    assert.equal(tree.root.findByType(ScoreEditor).props.document.encounters[0].pairings[0].games[0].status, 'pending', 'Read-only and official documents retain their exact saved outcome');
    await act(async () => tree.unmount());
  }
  const recorded = { ...game('recorded'), status: 'completed', a: 11, b: 9, played_at: '2026-09-28T10:45:00-07:00' };
  assert.equal(types.automaticGameStatus(recorded, '2026-09-29T00:01:00Z').played_at, recorded.played_at, 'Later saves and day changes preserve already recorded dates');
  for (const status of ['pending','forfeit','double_forfeit','unplayed']) {
    assert.equal(types.automaticGameStatus({ ...game('unplayed'), status, a: null, b: null, played_at: null }).played_at, null, 'Unplayed outcomes have no entry timestamp');
  }
  assert.equal(types.automaticGameStatus({ ...game('injured'), status: 'retired', a: 4, b: 7, played_at: null }, recorded.played_at).played_at, recorded.played_at);
  for (const score of [[11,0],[11,9],[12,10],[102,100]]) assert.ok(types.isFinalScore(...score));
  for (const score of [[11,null],[null,0],[10,0],[11,10],[14,8],[11,-1],[11,1.5],[Infinity,0]]) assert.equal(types.isFinalScore(...score), false);
  for (const status of ['retired','forfeit','double_forfeit','unplayed']) assert.equal(types.automaticGameStatus({ ...game('special'), status, a: 11, b: 5 }).status, status, 'Explicit exceptional outcomes are never auto-completed');
}

async function playUpReplacementEligibility() {
  const low = { entry_id: 'play-up', name: 'Play Up Player', eligibility_rating: 2.9, rating: 4.8, division: '3.0', gender: 'female' };
  for (const division of ['3.0', '3.5', '4.0', '4.5', 'Open', '4.5/Open', 'oPeN']) {
    assert.equal(types.matchesSkillLevel(low, division), true, `A 2.9 player may play up into ${division}, regardless of a prior division label`);
  }
  assert.equal(types.matchesSkillLevel({ ...low, eligibility_rating: 3.49999 }, '3.0'), true, 'Use the exact ceiling rather than rounded display metadata');
  assert.equal(types.matchesSkillLevel({ ...low, eligibility_rating: 3.5, division: '3.0' }, '3.0'), false, 'A matching old division label cannot bypass its upper limit');
  assert.equal(types.matchesSkillLevel({ ...low, eligibility_rating: 4.0, division: '3.5' }, '3.5'), false);
  assert.equal(types.matchesSkillLevel({ entry_id: 'seed', name: 'Seed', starting_rating: 2.9 }, '3.5'), true, 'Roster fallback uses the starting rating when no later eligibility rating exists');
  assert.equal(types.matchesSkillLevel({ ...low, eligibility_rating: 9 }, '4.5/Open'), true, 'Open has no upper rating ceiling');
  for (const rating of [undefined, null, 0, -1, NaN, Infinity]) {
    for (const division of ['3.5', 'Open']) assert.equal(types.matchesSkillLevel({ entry_id: 'invalid', name: 'Invalid', eligibility_rating: rating }, division), false, 'A positive finite rating is required');
  }
  for (const division of ['', 'garbage', '3.1', '7.0', '4.5Open', ' Open ']) assert.equal(types.matchesSkillLevel(low, division), false, 'Unknown division labels do not silently become Open');
  const atLimit = { ...low, entry_id: 'at-limit', name: 'At Upper Limit', eligibility_rating: 4.0, division: '3.5' };
  const scoped = { ...detail, eligible_players: { ...detail.eligible_players, alpha: [...detail.eligible_players.alpha, low, atLimit] } };
  let next = copy(document), tree;
  const props = { detail: scoped, players: types.competitionPlayers(scoped), clubName, disabled: false, onChange: value => { next = value; } };
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  const selectors = tree.root.findAllByType('fieldset').filter(fieldset => fieldset.children.some(child => child.type === 'legend' && nodeText(child) === 'Alpha Club actual players')).flatMap(fieldset => fieldset.findAllByType('select'));
  assert.ok(selectors.length);
  for (const selector of selectors) {
    assert.ok(selector.findAllByType('option').some(option => option.props.value === low.entry_id), 'The injury replacement picker offers a lower-rated eligible player');
    assert.ok(!selector.findAllByType('option').some(option => option.props.value === atLimit.entry_id), 'The picker excludes a player at the division ceiling');
  }
  const before = copy(next);
  const secondGame = tree.root.findByProps({ id: 'interclub-game-w2' });
  const replacements = secondGame.findAllByType('fieldset').find(fieldset => fieldset.children.some(child => child.type === 'legend' && nodeText(child) === 'Alpha Club actual players'));
  await act(async () => replacements.findAllByType('select')[0].props.onChange({ target: { value: low.entry_id } }));
  assert.deepEqual(next.encounters[0].pairings[0].games[1].players_a, [low.entry_id, 'entry-1'], 'A substitute can come from outside the four selected players');
  assert.deepEqual(next.encounters[0].pairings[0].players_a, before.encounters[0].pairings[0].players_a, 'An injury substitution keeps the starting lineup intact');
  assert.deepEqual(next.encounters[0].pairings[0].games[0], before.encounters[0].pairings[0].games[0], 'An injury substitution never rewrites an earlier game');
  await act(async () => tree.update(React.createElement(ScoreEditor, { ...props, document: next })));
  const updatedGame = tree.root.findByProps({ id: 'interclub-game-w2' });
  assert.ok(updatedGame.findAllByType('p').some(node => nodeText(node).includes('Players for Game 2: Play Up Player')), 'The actual substituted lineup is visible beside its game');
  await act(async () => updatedGame.findAllByType('textarea').find(node => node.props.placeholder === 'Injury').props.onChange({ target: { value: 'Ankle injury; substitute entered before game 2.' } }));
  assert.equal(next.encounters[0].pairings[0].games[1].injury_reason, 'Ankle injury; substitute entered before game 2.');
  await act(async () => tree.unmount());
}


async function easySubstitutions() {
  const original = copy(document);
  for (const row of substitutions.competitionGameRows(original)) Object.assign(row.game, { status: 'completed', a: 11, b: 5 });
  const later = copy(original.encounters[0]); later.id = 'later'; later.rotation = 2; later.club_a = 'gamma'; later.club_b = 'alpha';
  for (const p of later.pairings) { p.id += '-later'; p.players_b = p.players_a; p.players_a = p.players_a.map(id => id + '-gamma'); p.games.forEach(g => { g.id += '-later'; }); }
  original.encounters.unshift(later);
  const legacy = copy(original), source = legacy.encounters[1].pairings[0].games[1];
  source.players_a = ['sub-one', 'entry-1'];
  const review = substitutions.reviewSubstitutions(legacy);
  assert.equal(review.length, 1, 'One incomplete substitution is grouped across opponents');
  assert.equal(review[0].missingReason, true);
  assert.deepEqual(review[0].returningGames, ['w3', 'w1-later', 'w2-later', 'w3-later']);
  const fixed = substitutions.substituteForRemainingGames(legacy, review[0]);
  assert.equal(fixed.changedGames, 5);
  assert.equal(substitutions.reviewSubstitutions(fixed.document).some(c => c.missingReason || c.returningGames.length), false);
  const rows = substitutions.competitionGameRows(fixed.document);
  assert.deepEqual(rows.find(r => r.game.id === 'w1').game, substitutions.competitionGameRows(original).find(r => r.game.id === 'w1').game, 'Earlier scored game is unchanged');
  for (const row of rows) {
    const old = substitutions.competitionGameRows(original).find(r => r.game.id === row.game.id);
    for (const key of ['a', 'b', 'status', 'winner', 'played_at']) assert.deepEqual(row.game[key], old.game[key], `Substitution preserves ${key}`);
    assert.deepEqual(row.pairing.players_a, old.pairing.players_a, 'Starting pair remains intact');
  }
  assert.deepEqual(fixed.document.encounters[0].pairings[0].games[0].players_b, ['sub-one', 'entry-1'], 'Replacement follows the club when it switches sides');
  assert.deepEqual(legacy.encounters[0].pairings[0].games[0].players_b, [], 'Repair does not mutate the source draft');
  const chain = copy(legacy); Object.assign(chain.encounters[0].pairings[0].games[1], { players_b: ['sub-two', 'entry-1'], injury_reason: 'Second injury' });
  const chained = substitutions.substituteForRemainingGames(chain, substitutions.reviewSubstitutions(chain)[0]).document;
  assert.deepEqual(chained.encounters[0].pairings[0].games[1].players_b, ['sub-two', 'entry-1'], 'Later legitimate replacement stays in place');
  assert.deepEqual(chained.encounters[0].pairings[0].games[2].players_b, ['sub-two', 'entry-1'], 'Repair respects later injury chains');
  const replay = copy(legacy); replay.encounters[0].pairings.forEach(p => { p.eligibility_deadline = '2099-02-01T00:00:00Z'; });
  assert.deepEqual(substitutions.substituteForRemainingGames(replay, review[0]).document.encounters[0], replay.encounters[0], 'A different replay cutoff has its own lineup');
  assert.equal(substitutions.gameFromError(legacy, "Skill level 3.5 · Rotation 1 · Court 1 · Women's doubles · Game 2: Record an injury reason."), 'w2');
  assert.throws(() => substitutions.substituteForRemainingGames(original, { gameId: 'w2', side: 'a', outgoing: 'entry-0', incoming: 'entry-1' }), /already playing/);
  const wrongPair = copy(legacy);
  wrongPair.encounters[0].pairings[1].players_b = ['sub-one', 'entry-3'];
  const correctedPair = substitutions.substituteForRemainingGames(wrongPair, { ...review[0], incoming: 'sub-two', previousIncoming: 'sub-one' }).document;
  assert.deepEqual(correctedPair.encounters[0].pairings[1], wrongPair.encounters[0].pairings[1], 'Correcting a mistaken substitute preserves their legitimate appearances in another pairing');

  const sub = { entry_id: 'sub-one', name: 'Available Substitute', eligibility_rating: 3.2, gender: 'female' };
  const scoped = { ...detail, eligible_players: { ...detail.eligible_players, alpha: [...detail.eligible_players.alpha, sub] } };
  let next = copy(original), tree;
  const props = { detail: scoped, players: types.competitionPlayers(scoped), clubName, disabled: false, onChange: value => { next = value; } };
  await act(async () => { tree = create(React.createElement(ScoreEditor, { ...props, document: next })); });
  const form = () => tree.root.findByProps({ id: 'interclub-game-w2' }).findByType(common['./InjurySubstitutionEditor'].default);
  await act(async () => form().findAllByType('select').find(node => !node.props.hidden).props.onChange({ target: { value: 'a:entry-0' } }));
  await act(async () => form().findAllByType('select').find(node => node.props.hidden).props.onChange({ target: { value: 'sub-one' } }));
  await act(async () => form().findAllByType('button').find(node => nodeText(node) === 'Apply substitution to remaining games').props.onClick());
  assert.deepEqual(next.encounters[1].pairings[0].games[2].players_a, ['sub-one', 'entry-1']);
  assert.equal(next.encounters[1].pairings[0].games[1].injury_reason, 'Injury', 'No free-text explanation is required for a declared injury');
  await act(async () => tree.unmount());

  let saved = { ...copy(scoped), batch: { ...copy(batch), document: legacy } }, writes = [];
  global.fetch = async (url, options) => {
    if (options.method) { writes.push(JSON.parse(options.body)); saved.batch = { ...saved.batch, document: writes.at(-1).document, revision: saved.batch.revision + 1 }; return reply({ batch: saved.batch }); }
    return reply(saved);
  };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, { root: 'https://api.test/meet-1', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} })); });
  assert.ok(text(tree).includes('One substitution needs attention'));
  await act(async () => button(tree, 'Show this substitution').props.onClick());
  assert.equal(tree.root.findByType(ScoreEditor).props.focusGame.id, 'w2');
  assert.equal(tree.root.findByType(ScoreEditor).props.divisionFilter, '', 'Jump clears any skill-level filter');
  await act(async () => button(tree, 'Carry substitute forward').props.onClick());
  assert.ok(!text(tree).includes('One substitution needs attention'));
  assert.equal(writes.length, 0, 'Repair stays a reviewable draft before saving');
  await act(async () => button(tree, 'Undo substitution update').props.onClick());
  assert.ok(text(tree).includes('One substitution needs attention'));
  await act(async () => button(tree, 'Carry substitute forward').props.onClick());
  await act(async () => button(tree, 'Save all draft scores').props.onClick());
  assert.equal(writes[0].expected_revision, 4);
  assert.equal(substitutions.reviewSubstitutions(writes[0].document).some(c => c.missingReason || c.returningGames.length), false);
  await act(async () => tree.unmount());
  const eligible = { ...sub, entry_id: 'sub-two', name: 'Eligible Substitute' };
  saved = { ...copy(scoped), eligible_players: { ...scoped.eligible_players, alpha: [...players.slice(0,4), { ...sub, gender: 'male' }, eligible] }, batch: { ...copy(batch), document: legacy } };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, { root: 'https://api.test/meet-1', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} })); });
  assert.ok(text(tree).includes('not eligible for women’s doubles'), 'An invalid old choice is explained before submission');
  assert.equal(button(tree, 'Apply replacement to remaining games').props.disabled, true);
  if (process.env.PCS_SUBSTITUTION_REVIEW_PATH) { const repairMarkup = renderToStaticMarkup(React.createElement(common['./SubstitutionRepair'].default, { ...tree.root.findByType(common['./SubstitutionRepair'].default).props })); fs.writeFileSync(process.env.PCS_SUBSTITUTION_REVIEW_PATH, '<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px Arial;background:#f5f7fa;margin:40px;max-width:1060px;}'+fs.readFileSync(path.join(__dirname, '..', base, 'competition.module.css'),'utf8')+'</style></head><body class="page"><h1>Meet score draft</h1><p>144 of 144 game outcomes entered · Saved</p><section class="warning"><h3>One substitution needs attention</h3><p>Your scores are kept. Finish recording each injury once here.</p>'+repairMarkup+'</section></body></html>'); }
  const repair = tree.root.findByType(common['./SubstitutionRepair'].default);
  await act(async () => repair.findByType('select').props.onChange({ target: { value: 'sub-two' } }));
  await act(async () => button(tree, 'Apply replacement to remaining games').props.onClick());
  assert.ok(!text(tree).includes('One substitution needs attention'));
  const repaired = tree.root.findByType(ScoreEditor).props.document;
  assert.deepEqual(repaired.encounters[1].pairings[0].games[1].players_a, ['sub-two', 'entry-1'], 'Correction replaces the invalid original substitute');
  assert.deepEqual(repaired.encounters[0].pairings[0].games[2].players_b, ['sub-two', 'entry-1'], 'Correction also carries the eligible replacement through later games');
  await act(async () => tree.unmount());
}

function printSafety() {
  const markup = renderToStaticMarkup(React.createElement(PrintPacket, { document, meet, seasonName: 'Southern BCS', timezone: 'America/Mazatlan', revision: 4, players: types.competitionPlayers(detail), clubName }));
  assert.ok(markup.includes('&lt;script&gt;alert(1)&lt;/script&gt;'), 'Player labels are escaped in print HTML');
  assert.ok(!markup.includes('<script>'));
  assert.ok(markup.includes('Court assignments') && markup.includes('Verified by club A') && !markup.includes('Time played'));
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
  const lineupLink = tree.root.findByProps({ 'aria-label': 'Lineups' });
  assert.ok(lineupLink.props.href.includes('season=season-1') && lineupLink.props.href.includes('meet=meet-1'), 'Back to lineups preserves both season and meet');
  assert.equal(button(tree, 'Review and submit meet').props.disabled, false, 'Review identifies missing results before submission');
  await act(async () => button(tree, 'Review and submit meet').props.onClick());
  assert.equal(button(tree, 'Submit all official scores').props.disabled, true, 'Incomplete meet cannot be submitted');
  assert.equal(tree.root.findAllByType('a').find(link => nodeText(link) === 'Go to the first incomplete game').props.href, '#interclub-game-w1');
  await act(async () => tree.root.findByType(ScoreEditor).props.onDivisionFilterChange('4.0'));
  await act(async () => tree.root.findAllByType('a').find(link => nodeText(link) === 'Go to the first incomplete game').props.onClick());
  assert.equal(tree.root.findByType(ScoreEditor).props.divisionFilter, '', 'Submission review reveals incomplete games hidden by the skill filter');
  await act(async () => button(tree, 'Keep reviewing').props.onClick());
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(document), weather: 'delay' }));
  assert.equal(button(tree, 'Print meet packet').props.disabled, true, 'Packet must reflect a saved revision');
  assert.equal(tree.root.findAllByType('a').filter(node => node.props.href.includes('/interclub/registrations')).length, 0, 'Dirty scores disable workflow links that would abandon the draft');
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

async function pdfDownloads() {
  let saved = copy(detail), tree, calls = [], downloads = [], release;
  global.fetch = async (url, options) => { assert.ok(!options.method, 'Export never mutates meet results'); return reply(saved); };
  const props = { root: 'https://api.test/meet', clubId: 'alpha', accessToken: 'token', phase: 'regular', context, clubName, onLock() {}, onSeasonChange() {} };
  createPdf = async (options, scope) => { calls.push({ options, scope }); return new Promise(resolve => { release = () => resolve({ filename: 'saved-r4.pdf', pdf: { save: async filename => downloads.push(filename), getNumberOfPages: () => 2 } }); }); };
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
  const download = button(tree, 'Download schedule PDF');
  await act(async () => { download.props.onClick(); download.props.onClick(); });
  assert.equal(calls.length, 1, 'Repeated clicks produce one PDF');
  assert.equal(calls[0].scope, 'schedule');
  assert.equal(calls[0].options.revision, 4);
  assert.deepEqual(calls[0].options.document, saved.batch.document, 'Download uses the saved document');
  assert.equal(button(tree, 'Download full packet PDF').props.disabled, true);
  await act(async () => release());
  assert.deepEqual(downloads, ['saved-r4.pdf']);
  assert.ok(text(tree).includes('Schedule PDF downloaded (2 pages).'));
  await act(async () => tree.root.findByType(ScoreEditor).props.onChange({ ...copy(document), weather: 'delay' }));
  assert.equal(button(tree, 'Download schedule PDF').props.disabled, true);
  assert.equal(button(tree, 'Download full packet PDF').props.disabled, true);
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(workspace.MeetOperations, props)); });
  createPdf = async () => { throw new Error('Unable to load PDF module'); };
  await act(async () => button(tree, 'Download full packet PDF').props.onClick());
  assert.ok(text(tree).includes('Could not create the PDF'));
  assert.equal(button(tree, 'Download full packet PDF').props.disabled, false, 'Failed download can be retried');
  createPdf = async (options, scope) => { calls.push({ options, scope }); return new Promise(resolve => { release = () => resolve({ filename: 'saved-r4.pdf', pdf: { save: async filename => downloads.push(filename), getNumberOfPages: () => 2 } }); }); };
  await act(async () => button(tree, 'Download full packet PDF').props.onClick());
  assert.equal(calls.at(-1).scope, 'packet');
  await act(async () => tree.unmount());
  await act(async () => release());
  assert.equal(downloads.length, 1, 'Leaving a meet cancels its pending download');
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
(async () => { await scoreEntry(); await pairingControls(); await preMeetRosterChange(); await automaticScoreEntry(); await savedScoreCompletion(); await playUpReplacementEligibility(); await easySubstitutions(); printSafety(); await staggeredSchedule(); await revisionsAndStaleClub(); await pdfDownloads(); await approval(); await qualifyingRoundRobin(); await missingLineups(); await seasonRegistrationGate(); qualifyingDisplay(); writePrintReview(); console.log('PASS interclub competition: fixed gender pairings, pre-meet roster changes, per-game injury substitutions, automatic score completion, saved pending-score recovery, zero scores, clearing, win-by-two, injury outcomes, singles, submission review, exact revisions, staging schedule and PDF controls'); })().catch(error => { console.error(error); process.exit(1); });
