const assert = require('node:assert/strict'), React = require('react');
const { create, act } = require('react-test-renderer');
const load = require('./helpers/interclub-results-modules.cjs');
const League = load('components/PublicInterclubLeague.tsx').default;
const views = load('lib/interclubResultViews.ts');
const text = node => typeof node === 'string' ? node : node.children.map(text).join('');
const games = [0, 1, 2].map(index => ({ status: 'completed', a: 11, b: 7, winner: 'a', players_a: [index ? 'sub' : 'original', 'partner'], players_b: ['opponent', 'other'] }));
const encounter = (id, meet, division, clubA, clubB) => ({ id, meet_id: meet, division, club_a: clubA, club_b: clubB, phase: 'regular', weather: 'normal', pairings: [{ kind: 'women', games }], tiebreak: null });
const league = { document: { name: 'Southern BCS', start_date: '2026-12-01', end_date: '2027-03-31', timezone: 'America/Mazatlan', divisions: ['4.0','3.0','3.5'], scoring_version: 1, results: [], clubs: [{ id: 'a', name: 'Alpha' }, { id: 'b', name: 'Beta' }, { id: 'c', name: 'Cabo' }],
  meets: [{ id: 'm1', starts_at: '2026-12-17T17:00:00Z', host_club_id: 'a', club_ids: ['a','b','c'], courts: 8, duration_minutes: 180 }, { id: 'm2', starts_at: '2026-12-19T17:00:00Z', host_club_id: 'c', club_ids: ['a','b','c'], courts: 8, duration_minutes: 180 }],
  competition_results: [encounter('e1','m1','3.0','a','b'), encounter('e2','m1','3.5','b','c'), encounter('e3','m2','3.0','a','b')],
  players: [{ id:'original',name:'Ana Original',club_id:'a' },{ id:'sub',name:'Sara Replacement',club_id:'a' },{ id:'partner',name:'Pat Partner',club_id:'a' },{ id:'opponent',name:'Bea Opponent',club_id:'b' },{ id:'other',name:'Other Opponent',club_id:'b' }] }, standings: [{ division:'3.0',rows:[{ club_id:'a',points:3,games_won:6,pairings_won:2,position:1 }] }], club_cup:{ standings:[{ club_id:'a',position:1,points:3,regular_points:3,championship_points:0 }],champions:[],status:'provisional' } };
async function main() {
  let tree; await act(async () => { tree = create(React.createElement(League, { league })); });
  const tab = label => tree.root.findAllByProps({ role:'tab' }).find(node => text(node) === label);
  const visible = () => tree.root.findAllByProps({ role:'tabpanel' }).filter(node => !node.props.hidden);
  assert.equal(visible().length, 1); assert.ok(text(visible()[0]).includes('Overall Club Cup'));
  assert.equal(visible()[0].findAllByType('table').length, 1, 'First screen contains the overall table only');
  const standings = tree.root.findByProps({ 'aria-label':'Standings' });
  assert.deepEqual(standings.findAllByType('option').map(node => node.props.value), ['', '3.0', '3.5', '4.0']);
  await act(async () => tree.root.findByProps({ 'aria-label':'View results for Alpha' }).props.onClick());
  assert.equal(tab('Results').props['aria-selected'], true);
  const select = name => tree.root.findAllByType('select').find(node => node.props['aria-label'] === name);
  assert.equal(select('Filter results by club').props.value, 'a');
  await act(async () => select('Filter results by meet').props.onChange({ target:{value:'m1'} }));
  let panel = visible()[0]; assert.equal(panel.findAllByType('details').length, 1);
  const choosePlayer = value => panel.findAllByType('select').find(node => node.props.hidden).props.onChange({ target:{value} });
  await act(async () => choosePlayer('sub'));
  assert.ok(text(visible()[0]).includes('2 doubles games involving Sara Replacement'));
  assert.deepEqual(visible()[0].findAllByType('tbody')[0].findAllByType('th').map(text), ['2','3'], 'Substitute filter preserves original game numbers and excludes injured player’s earlier game');
  assert.equal(visible()[0].findAllByType('details')[0].props.open, true);
  await act(async () => select('Filter results by skill level').props.onChange({ target:{value:'3.5'} }));
  assert.ok(text(visible()[0]).includes('No results match these filters'));
  await act(async () => tree.root.findAllByType('button').find(node => text(node) === 'Clear filters').props.onClick());
  assert.equal(visible()[0].findAllByType('details').length, 3);
  await act(async () => tab('Schedule').props.onClick());
  assert.equal(visible().length, 1); assert.ok(text(visible()[0]).includes('Meet schedule'));
  assert.equal(visible()[0].findAllByType('details').length, 0);
  await act(async () => tree.unmount());
  assert.equal(views.resultGameWinner({status:'retired',a:9,b:4,winner:'b'}), 'b');
  assert.equal(views.resultGameWinner({status:'double_forfeit',a:null,b:null,winner:null}), null);
  console.log('Interclub overall-first navigation, meet/club/skill/player filters and injury appearances passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
