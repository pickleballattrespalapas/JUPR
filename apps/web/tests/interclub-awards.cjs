const assert = require('node:assert/strict'), React = require('react');
const { create, act } = require('react-test-renderer');
const load = require('./helpers/interclub-results-modules.cjs');
const FinalResults = load('components/InterclubFinalResults.tsx').default;
const TrophyCase = load('components/ClubTrophyCase.tsx').default;
const { recipientGroups } = load('lib/interclubAwards.ts');
const text = node => typeof node === 'string' ? node : node.children.map(text).join('');
const award = (id, club, entry, kind) => ({ id, club_id: club, entry_id: entry, recipient_name: 'Same Name',
  recipient_type: entry ? 'player' : 'club', award_key: kind, division: kind === 'division_champion' ? '3.5' : '',
  title: kind === 'participation' ? 'Interclub Season Participant' : '3.5 Interclub Champion', season_name: 'Coastal season',
  earned_at: '2026-09-30T18:00:00Z', results_href: '/interclub/season/final-results' });
const awards = [award('1','a','p1','participation'), award('2','a','p1','division_champion'), award('3','b','p2','participation')];
const league = { id:'season',document:{ name:'Coastal season',clubs:[{id:'a',name:'Alpha',slug:'alpha'},{id:'b',name:'Beta',slug:'beta'}],
  divisions:['3.5'],players:[{id:'p1',club_id:'a',name:'Same Name'}] }, standings:[{division:'3.5',rows:[]}],
  club_cup:{champions:['a'],status:'complete',standings:[]},final_results:{complete:true,champions:['a'],clubs:2,players:2,meets:3,
    divisions:[{division:'3.5',winner:'a',runner_up:'b',games_won:3,games_lost:0,tiebreak:null,players:['p1']}]}};

async function main() {
  assert.equal(recipientGroups(awards).length, 2, 'Distinct players with the same name stay separate');
  assert.equal(recipientGroups(awards)[0].titles.length, 2, 'A player’s season honors share one card');
  let tree; await act(async () => { tree = create(React.createElement(FinalResults,{league,awards})); });
  assert.ok(text(tree.root).includes('Won 3–0') && text(tree.root).includes('Runner-up: Beta'));
  assert.ok(tree.root.findAllByType('a').some(node => node.props.href === '/interclub/season?view=results'));
  assert.ok(tree.root.findAllByType('a').some(node => node.props.href === '/clubs/alpha/trophies'));
  const players = tree.root.findByProps({'aria-label':'Player awards'});
  assert.equal(players.findAllByType('li').length, 2);
  await act(async () => players.findByType('select').props.onChange({target:{value:'b'}}));
  assert.equal(players.findAllByType('li').length, 1);
  assert.ok(!text(players.findByType('ul')).includes('Interclub Champion'));
  await act(async () => tree.unmount());
  const tied = structuredClone(league);
  tied.final_results.champions=['a','b']; tied.final_results.divisions[0].tiebreak={winner_score:21,runner_up_score:19};
  tied.final_results.divisions[0].games_won=2; tied.final_results.divisions[0].games_lost=2;
  await act(async () => { tree = create(React.createElement(FinalResults,{league:tied})); });
  assert.ok(text(tree.root).includes('Joint Club Cup champions') && text(tree.root).includes('21–19 after a 2–2 split'));
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(FinalResults,{league:{...league,final_results:{...league.final_results,complete:false}}})); });
  assert.ok(text(tree.root).includes('Final results are not ready yet'));
  assert.equal(tree.root.findAllByProps({'aria-label':'Skill-level champions'}).length, 0);
  await act(async () => tree.unmount());
  await act(async () => { tree = create(React.createElement(TrophyCase,{clubName:'Alpha',trophies:[award('4','a',null,'participation'),award('5','a',null,'division_champion')]})); });
  assert.equal(tree.root.findAllByProps({'data-club-trophy':''}).length, 2);
  await act(async () => tree.root.findByType('select').props.onChange({target:{value:'division_champion'}}));
  assert.equal(tree.root.findAllByProps({'data-club-trophy':''}).length, 1);
  await act(async () => tree.root.findByType('select').props.onChange({target:{value:'club_cup_champion'}}));
  assert.ok(text(tree.root).includes('No trophies in this category yet.'));
  await act(async () => tree.unmount());
  console.log('Interclub final results, champion summaries, player grouping, club filtering and trophy links passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
