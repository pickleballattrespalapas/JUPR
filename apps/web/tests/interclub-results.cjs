const assert = require('node:assert/strict'), React = require('react');
const { create, act } = require('react-test-renderer');
const load = require('./helpers/interclub-results-modules.cjs');
const League = load('components/PublicInterclubLeague.tsx').default;
const { CompetitionResults } = load('components/PublicInterclubCompetition.tsx');
const views = load('lib/interclubResultViews.ts');
const text = node => typeof node === 'string' ? node : node.children.map(text).join('');
const games = [0, 1, 2].map(index => ({ status: 'completed', a: 11, b: 7, winner: 'a', players_a: [index ? 'sub' : 'original', 'partner'], players_b: ['opponent', 'other'] }));
const encounter = (id, meet, division, clubA, clubB) => ({ id, meet_id: meet, division, club_a: clubA, club_b: clubB, phase: 'regular', weather: 'normal', pairings: [{ kind: 'women', games }], tiebreak: null });
const league = { document: { name: 'Southern BCS', start_date: '2026-12-01', end_date: '2027-03-31', timezone: 'America/Mazatlan', divisions: ['4.0','3.0','3.5'], scoring_version: 1, results: [], clubs: [{ id: 'a', name: 'Alpha' }, { id: 'b', name: 'Beta' }, { id: 'c', name: 'Cabo' }],
  meets: [{ id: 'm1', starts_at: '2026-12-17T17:00:00Z', host_club_id: 'a', club_ids: ['a','b','c'], courts: 8, duration_minutes: 180 }, { id: 'm2', starts_at: '2026-12-19T17:00:00Z', host_club_id: 'c', club_ids: ['a','b','c'], courts: 8, duration_minutes: 180 }],
  competition_results: [encounter('e1','m1','3.0','a','b'), encounter('e2','m1','3.5','b','c'), encounter('e3','m2','3.0','a','b')],
  players: [{ id:'original',name:'Ana Original',club_id:'a' },{ id:'sub',name:'Sara Replacement',club_id:'a' },{ id:'partner',name:'Pat Partner',club_id:'a' },{ id:'opponent',name:'Bea Opponent',club_id:'b' },{ id:'other',name:'Other Opponent',club_id:'b' }] }, standings: [{ division:'3.0',rows:[{ club_id:'a',points:3,games_won:6,pairings_won:2,meets_played:2,point_differential:-4,position:1 }] }, { division:'3.5',rows:[{ club_id:'a',points:3,games_won:4,pairings_won:1,meets_played:2,point_differential:3,position:1 }] }], club_cup:{ standings:[{ club_id:'a',position:1,points:12,regular_points:6,championship_points:6,meets_played:3,pairings_won:5,games_won:16,point_differential:14 }],champions:[],status:'provisional' } };
async function main() {
  let tree; await act(async () => { tree = create(React.createElement(League, { league })); });
  const tab = label => tree.root.findAllByProps({ role:'tab' }).find(node => text(node) === label);
  const visible = () => tree.root.findAllByProps({ role:'tabpanel' }).filter(node => !node.props.hidden);
  assert.equal(visible().length, 1); assert.ok(text(visible()[0]).includes('Overall Club Cup'));
  assert.equal(visible()[0].findAllByType('table').length, 1, 'First screen contains the overall table only');
  const cup = visible()[0].findByType('table');
  assert.deepEqual(cup.findByType('thead').findAllByType('th').map(text), ['Place','Club','Points','Meets','Pairings won','Games won','Point difference']);
  assert.deepEqual(cup.findByType('tbody').findAllByType('td').map(text), ['1','12','3','5','16','+14'], 'Display official Cup statistics including finals and distinct meets, not sums of division meet counts');
  const standings = tree.root.findByProps({ 'aria-label':'Standings' });
  assert.deepEqual(standings.findAllByType('option').map(node => node.props.value), ['', '3.0', '3.5', '4.0']);
  await act(async () => standings.props.onChange({ target:{value:'3.0'} }));
  assert.deepEqual(visible()[0].findByType('tbody').findAllByType('td').map(text), ['1','3','2','2','6','-4']);
  await act(async () => tree.root.findByProps({ 'aria-label':'View results for Alpha' }).props.onClick());
  assert.equal(tab('Results').props['aria-selected'], true);
  const select = name => tree.root.findAllByType('select').find(node => node.props['aria-label'] === name);
  assert.equal(select('Filter results by club').props.value, 'a');
  const gameRows = () => visible()[0].findAllByProps({ 'data-result-game':'' });
  assert.equal(gameRows().length, 6, 'A club selection immediately shows all its games across meets');
  assert.equal(visible()[0].findAllByType('details').length, 0, 'No matchup or pairing expansion is needed');
  await act(async () => select('Filter results by meet').props.onChange({ target:{value:'m1'} }));
  let panel = visible()[0]; assert.equal(gameRows().length, 3);
  const choosePlayer = value => panel.findAllByType('select').find(node => node.props.hidden).props.onChange({ target:{value} });
  await act(async () => choosePlayer('sub'));
  assert.ok(text(visible()[0]).includes('2 doubles games involving Sara Replacement'));
  assert.deepEqual(gameRows().map(row => text(row.findByType('th'))), ['Women’s doublesGame 2','Women’s doublesGame 3'], 'Substitute filter preserves original game numbers and excludes injured player’s earlier game');
  assert.ok(gameRows().every(row => text(row).includes('Sara Replacement') && !text(row).includes('Ana Original')));
  await act(async () => select('Filter results by club').props.onChange({ target:{value:'b'} }));
  const awayRow = gameRows()[0].findAllByType('td');
  assert.ok(text(awayRow[1]).startsWith('Beta'));
  assert.equal(text(awayRow[2]), '7–11Loss', 'The selected club’s score follows it to the left');
  assert.ok(text(awayRow[3]).startsWith('Alpha'));
  await act(async () => select('Filter results by club').props.onChange({ target:{value:'a'} }));
  await act(async () => select('Filter results by skill level').props.onChange({ target:{value:'3.5'} }));
  assert.ok(text(visible()[0]).includes('No results match these filters'));
  await act(async () => tree.root.findAllByType('button').find(node => text(node) === 'Clear filters').props.onClick());
  assert.equal(gameRows().length, 9);
  await act(async () => tab('Schedule').props.onClick());
  assert.equal(visible().length, 1); assert.ok(text(visible()[0]).includes('Meet schedule'));
  assert.equal(visible()[0].findAllByType('details').length, 0);
  await act(async () => tree.unmount());
  assert.equal(views.resultGameWinner({status:'retired',a:9,b:4,winner:'b'}), 'b');
  assert.equal(views.resultGameWinner({status:'double_forfeit',a:null,b:null,winner:null}), null);
  const special = { ...encounter('special','m1','3.5','a','b'), phase:'final', weather:'finalized_partial', pairings:[{ kind:'women',games:[
    { status:'retired',a:9,b:4,winner:'b',players_a:['original'],players_b:['opponent'] },
    { status:'forfeit',a:null,b:null,winner:'a' },
    { status:'double_forfeit',a:null,b:null,winner:null },
    { status:'unplayed',a:null,b:null,winner:null },
  ] }], tiebreak:{ status:'completed',a:21,b:19,players_a:['sub'],players_b:['other'] } };
  await act(async () => { tree = create(React.createElement(CompetitionResults, { results:[special],names:{a:'Alpha',b:'Beta'},players:league.document.players,initialClub:'b' })); });
  const specialRows = tree.root.findAllByProps({ 'data-result-game':'' });
  assert.equal(specialRows.length, 5, 'Doubles exceptions and completed singles share the visible table');
  assert.ok(text(specialRows[0]).includes('4–9Win'), 'Retirement preserves the awarded winner even when trailing');
  assert.ok(text(tree.root).includes('Beta awarded the game'));
  assert.ok(text(specialRows[1]).includes('—–—Loss'), 'No scores are invented for a forfeit');
  assert.ok(text(tree.root).includes('Both clubs forfeited') && text(tree.root).includes('Not played — weather'));
  assert.ok(text(specialRows[4]).includes('Rotating singles') && text(specialRows[4]).includes('19–21Loss'));
  assert.ok(text(tree.root).includes('4 doubles games · 1 tiebreak'));
  assert.equal(views.hasResultPlayer({ ...special,pairings:[],tiebreak:{...special.tiebreak,status:'pending'} }, 'sub'), false);
  await act(async () => tree.unmount());
  const clinched = { ...encounter('clinched','m1','3.5','a','b'), phase:'final', pairings: ['women','men','mixed_a','mixed_b'].map((kind,index) => ({ kind, games: [index < 3 ? games[0] : { status:'not_needed',a:null,b:null,winner:null,players_a:[],players_b:[] }] })) };
  await act(async () => { tree = create(React.createElement(CompetitionResults, { results:[clinched],names:{a:'Alpha',b:'Beta'},players:league.document.players,initialClub:'b' })); });
  assert.ok(text(tree.root).includes('3 doubles games'));
  const skipped = tree.root.findAllByProps({ 'data-result-game':'' })[3];
  assert.ok(text(tree.root).includes('Not needed — matchup decided 3–0'));
  assert.ok(!/Win|Loss|Draw|Result not entered/.test(text(skipped)), 'Skipped game has neither a result nor a missing-score warning');
  assert.equal(views.resultGameWinner(clinched.pairings[3].games[0]), null);
  assert.equal(views.hasResultPlayer({ ...clinched, pairings:[clinched.pairings[3]] }, 'original'), false);
  await act(async () => tree.unmount());
  console.log('Interclub Cup statistics, direct game tables, club orientation, filters, injury appearances and exceptional results passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
