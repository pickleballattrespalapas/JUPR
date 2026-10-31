const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const ts = require('typescript');
function load(file) {
  const module = {exports:{}};
  const code = ts.transpileModule(fs.readFileSync(file,'utf8'), {compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,esModuleInterop:true}}).outputText;
  new Function('require','module','exports',code)(require,module,module.exports);
  return module.exports;
}
const {buildTournamentPrintPdf,tournamentPrintDraws,tournamentPrintMatches,tournamentCardPages,tournamentTallyPages,printScoring,printSource} = load(path.join(__dirname,'../lib/tournamentPrintPacket.ts'));
function contents(pdf) {
  const bytes = Buffer.from(pdf.output('arraybuffer'));
  assert.equal(bytes.subarray(0,5).toString(),'%PDF-');
  return [...bytes.toString('latin1').matchAll(/stream\n([\s\S]*?)\nendstream/g)].map(match=>{
    try {return zlib.inflateSync(Buffer.from(match[1],'latin1')).toString('latin1');} catch {return match[1];}
  }).join('\n');
}
const names=['José Reyes / Elena Sánchez','Alex Montgomery / Beatriz de la Cruz','Carlos Mendoza / Diana López','Eva Torres / Francisco Rivera','Gabriela Ortega / Hugo Castillo','Isabel Moreno / Javier Ruiz','Karla Vargas / Luis Navarro','María González / Nicolás Álvarez'];
function fixture(n=8, id=`draw-${n}`) {
  const standings=Array.from({length:n},(_,i)=>({team_id:`${id}-team-${i+1}`,team_number:i+1,team_name:`Club team ${i+1}`,participant_names:(names[i] || `Player ${i*2+1} / Player ${i*2+2}`).split(' / '),seed:i+1,wins:0,losses:0,points_for:0,points_against:0,differential:0}));
  const side=i=>({team_id:standings[i-1].team_id,name:standings[i-1].team_name,participant_names:standings[i-1].participant_names});
  const games=[];
  const roster=Array.from({length:n+(n%2)},(_,i)=>i<n?i+1:null);
  for (let r=1;r<roster.length;r++) {
    let slot=1;
    for(let i=0;i<roster.length/2;i++) {
      const a=roster[i],b=roster[roster.length-1-i];
      if(a && b) games.push({id:`${id}-r${r}-s${slot}`,draw_id:id,draw_name:`Example ${n}-team division`,state:'UNQUEUED',stage:'ROUND_ROBIN',rr_round_number:r,rr_slot_number:slot,round_label:`Round ${r}`,slot_label:`Slot ${slot++}`,team_a:side(a),team_b:side(b),score_a:null,score_b:null,version:'1',scoring:{format:'GAME_TO_11',target:11,win_by_two:true},blockers:[],correction_readiness:{ready:false,blockers:[]}});
    }
    roster.splice(1,0,roster.pop());
  }
  const draw={id,name:`Example ${n}-team division`,state:'DRAFT',round_robin_summary:{standings,complete:false,ranking_policy:{description:'Teams are ranked by wins. Ties: head-to-head, point differential, points for, then team number.'}}};
  return {ok:true,scope:{club_id:'sample-club',tournament_id:'sample-tournament',registration_day_id:'sample-day'},tournament:{id:'sample-tournament',name:'SAMPLE - Tournament paper backups'},generated_at:'2026-10-31T16:30:00Z',day_scope:{selected_day_id:'sample-day',selected_day:{id:'sample-day',label:'Layout preview - example players',event_date:'2026-10-31'}},draws:[draw],games,courts:[]};
}

const canonicalSchedules=process.env.TOURNAMENT_PRINT_SCHEDULES ? JSON.parse(fs.readFileSync(process.env.TOURNAMENT_PRINT_SCHEDULES,'utf8')) : {};
function savedFixture(n) {
  const sample=fixture(n), saved=canonicalSchedules[n];
  if (saved) {
    const side=id=>{const s=sample.draws[0].round_robin_summary.standings[Number(id)-1];return {team_id:s.team_id,name:s.team_name,participant_names:s.participant_names};};
    const base=sample.games[0];
    sample.games=saved.map((game,index)=>({...base,id:`saved-${n}-${index}`,rr_round_number:game.rr_round_number,rr_slot_number:game.rr_slot_number,round_label:`Round ${game.rr_round_number}`,slot_label:`Slot ${game.rr_slot_number}`,team_a:side(game.team_a_id),team_b:side(game.team_b_id)}));
  }
  return sample;
}

(async()=>{
  for (const n of Object.keys(canonicalSchedules).map(Number)) {
    const snapshot=savedFixture(n), ids=[snapshot.draws[0].id], before=JSON.stringify(snapshot);
    const model=tournamentPrintDraws(snapshot,ids)[0];
    assert.deepEqual(model.rounds.flatMap(round=>round.games),snapshot.games,'the live generator schedule is preserved exactly');
    assert.deepEqual(tournamentPrintMatches(snapshot,[model],{drawIds:ids}).map(row=>row.game),snapshot.games,'single-division cards use the live generator order');
    await buildTournamentPrintPdf(snapshot,{drawIds:ids},'packet');
    assert.equal(JSON.stringify(snapshot),before);
  }
  for (const n of [...Array.from({length:15},(_,i)=>i+2),24,32]) {
    const data=fixture(n), before=JSON.stringify(data), options={drawIds:[data.draws[0].id]};
    const models=tournamentPrintDraws(data,options.drawIds);
    assert.equal(models[0].games.length,n*(n-1)/2);
    assert.equal(models[0].rounds.length,n%2?n:n-1);
    if(n%2) assert.ok(models[0].rounds.every(round=>round.byes.length===1));
    const cards=await buildTournamentPrintPdf(data,options,'cards');
    assert.equal(cards.pdf.getNumberOfPages(),Math.ceil(n*(n-1)/4),'exactly two cards per page, including odd final card');
    assert.equal(cards.pdf.internal.pageSize.getWidth(),612);
    assert.equal(cards.pdf.internal.pageSize.getHeight(),792);
    assert.ok(contents(cards.pdf).includes('José Reyes'));
    const tally=await buildTournamentPrintPdf(data,options,'admin');
    assert.equal(tally.pdf.internal.pageSize.getWidth(),792);
    assert.equal(tally.pdf.internal.pageSize.getHeight(),612);
    if (n<=8) assert.equal(tally.pdf.getNumberOfPages(),1,'small draws stay on one admin sheet');
    const pages=tournamentTallyPages(models[0]);
    assert.equal(tally.pdf.getNumberOfPages(),pages.length);
    assert.ok(pages.every(page=>page.teams.length<=8 && page.sections.reduce((sum,row)=>sum+row.bands,0)<=7),'no chart exceeds its horizontal or vertical page budget');
    for (const team of models[0].teams) {
      assert.deepEqual(pages.filter(page=>page.teams.some(t=>t.id === team.id)).flatMap(page=>page.sections.flatMap(row=>row.games.map(game=>game.id))),models[0].rounds.flatMap(round=>round.games.map(game=>game.id)),'every team receives the full saved schedule once, in order');
    }
    const text=contents(tally.pdf);
    for(const label of ['For','Against','Wins / Losses','Total points','Differential +/-','Seed / place']) assert.ok(text.includes(label));
    assert.equal(JSON.stringify(data),before,'printing never changes schedule, scores, roster or seeds');
  }

  // A single round can itself require multiple sheets; continuation must not
  // falsely shade a team as having a bye when it plays on the following sheet.
  const huge=fixture(130); huge.games=huge.games.filter(game=>game.rr_round_number===1);
  const hugeModel=tournamentPrintDraws(huge,[huge.draws[0].id])[0];
  const hugePages=tournamentTallyPages(hugeModel);
  assert.ok(hugePages.every(page=>page.sections.reduce((sum,row)=>sum+row.bands,0)<=7));
  const hugeText=contents((await buildTournamentPrintPdf(huge,{drawIds:[huge.draws[0].id]},'admin')).pdf);
  assert.ok(hugeText.includes('(cont.)') || hugeText.includes('cont.'));
  assert.ok(hugeText.includes('Other'),'teams on a continuation sheet are not labeled as byes');

  for (const n of [1,2,3,7,10,29]) {
    const entries=Array.from({length:n},(_,i)=>i+1);
    const cut=tournamentCardPages(entries,'cut-stack');
    assert.deepEqual([...cut.map(page=>page[0]),...cut.map(page=>page[1])].filter(Boolean),entries,'cutting the full stack preserves card order');
    assert.deepEqual(tournamentCardPages(entries,'consecutive').flat().filter(Boolean),entries,'page-by-page mode preserves card order');
  }

  const poolA=fixture(4,'pool-a'),poolB=fixture(3,'pool-b');
  const combined={...poolA,draws:[...poolA.draws,...poolB.draws],games:[...poolB.games,...poolA.games].reverse()};
  const allDrawIds=combined.draws.map(draw=>draw.id);
  const ordered=()=>tournamentPrintMatches(combined,tournamentPrintDraws(combined,allDrawIds),{drawIds:allDrawIds}).map(row=>row.game.id);
  assert.deepEqual(ordered().slice(0,3),[...poolA.games.filter(game=>game.rr_round_number===1),...poolB.games.filter(game=>game.rr_round_number===1)].map(game=>game.id),'all divisions round one prints before later rounds');
  const prePlayModels=tournamentPrintDraws(combined,allDrawIds);
  const prePlayCards=tournamentPrintMatches(combined,prePlayModels,{drawIds:allDrawIds});
  const roundNumbers=prePlayCards.map(row=>row.game.rr_round_number);
  assert.deepEqual(roundNumbers,[...roundNumbers].sort((a,b)=>a-b),'advance printing keeps every saved round in ascending play order');
  for (const model of prePlayModels) {
    const cardIds=prePlayCards.filter(row=>row.game.draw_id===model.draw.id).map(row=>row.game.id);
    const tallyIds=tournamentTallyPages(model).filter(page=>page.teams[0].id===model.teams[0].id).flatMap(page=>page.sections.flatMap(section=>section.games.map(game=>game.id)));
    assert.deepEqual(cardIds,tallyIds,'each division tally has the same saved match order as its advance-printed cards');
  }
  const onCourt=poolA.games[0], waiting=poolB.games[0], reserved=poolA.games[1];
  onCourt.court_id='c2'; reserved.reserved_court_id='c2';
  combined.courts=[{id:'c2',label:'Court 2',position:2,current_assignment:{game_id:onCourt.id},next_assignment:{game_id:reserved.id}}];
  combined.eligible_queue=[{game_id:waiting.id,position:1}];
  combined.reserved_queue=[{game_id:reserved.id,position:2,reserved_court_id:'c2'}];
  assert.deepEqual(ordered().slice(0,3),[onCourt.id,waiting.id,reserved.id],'current courts then the server unified ready/reserved queue');
  const queueCards=contents((await buildTournamentPrintPdf(combined,{drawIds:allDrawIds},'cards')).pdf);
  const courtFields=[...queueCards.matchAll(/\(Court: ([^)]*)\)/g)].map(match=>match[1]);
  assert.equal(courtFields.length,combined.games.length);
  assert.ok(courtFields.every(value=>value==='__________'),'all courts stay writable, including current, reserved and unassigned matches');
  assert.ok(!queueCards.includes('Team 1 -') && !queueCards.includes('Winner \\(circle\\):'),'cards identify teams by names only');

  const snapshot=fixture(4), id=snapshot.draws[0].id, opts={drawIds:[id]};
  // Use a changed saved schedule and non-contiguous slots. A count-based
  // template would silently replace these assignments.
  snapshot.games[0].rr_slot_number=8; snapshot.games[0].slot_label='Slot 8';
  snapshot.games[0].round_label='Round 10'; snapshot.games[0].rr_round_number=10;
  const m=tournamentPrintDraws(snapshot,[id])[0];
  assert.equal(m.rounds.at(-1).label,'Round 10');
  assert.equal(m.rounds.at(-1).games[0].id,snapshot.games[0].id);
  const customText=contents((await buildTournamentPrintPdf(snapshot,opts,'admin')).pdf);
  assert.ok(customText.includes('Slot 8') && customText.includes('1 v 4'),'saved altered pairings and slot labels remain on the left');

  snapshot.games[0].state='COMPLETED'; snapshot.games[0].finalized_at='2026-10-31T17:00:00Z'; snapshot.games[0].score_a=11; snapshot.games[0].score_b=0;
  snapshot.draws[0].round_robin_summary.standings[0].wins=1;
  const remaining=await buildTournamentPrintPdf(snapshot,{...opts,unplayedOnly:true,blankCards:2},'cards');
  assert.equal(remaining.pdf.getNumberOfPages(),4,'five remaining matches plus two spare cards');
  assert.ok(!contents(remaining.pdf).includes('Round 10 / Slot 8'));
  const fullAdmin=await buildTournamentPrintPdf(snapshot,{...opts,unplayedOnly:true,includeResults:true},'admin');
  assert.ok(contents(fullAdmin.pdf).includes('Slot 8'),'card filter never drops matches from the tally');
  assert.ok(contents(fullAdmin.pdf).includes('(0) Tj'),'a recorded zero remains a score');

  const second=fixture(3,'other-draw');
  const multi={...snapshot,draws:[...snapshot.draws,...second.draws],games:[...snapshot.games,...second.games]};
  const scoped=await buildTournamentPrintPdf(multi,opts,'packet');
  assert.ok(!contents(scoped.pdf).includes('Example 3-team'),'pool selection does not leak other pools');
  assert.throws(()=>tournamentPrintDraws(multi,['missing-draw']),/no longer/);
  assert.throws(()=>tournamentPrintDraws(multi,[]),/Select at least/);
  const empty={...snapshot,games:[]};
  await assert.rejects(buildTournamentPrintPdf(empty,opts,'cards'),/No matches/);
  assert.ok(contents((await buildTournamentPrintPdf(empty,opts,'admin')).pdf).includes('No round-robin games'));

  const base=snapshot.games[1];
  const playoffs=[
    {...base,id:'sf1',stage:'PLAYOFF',round_label:'SF',playoff_game_code:'P1',playoff_round:'SF',team_a_source:{seed:1},team_b_source:{seed:4}},
    {...base,id:'final',stage:'PLAYOFF',round_label:'FINAL',playoff_game_code:'P3',playoff_round:'FINAL',team_a:{team_id:null,name:'To be determined',participant_names:[]},team_b:{team_id:null,name:'To be determined',participant_names:[]},team_a_source:{winnerOf:'P1'},team_b_source:{winnerOf:'P2'},scoring:{format:'BEST_2_OF_3',individual_game_target:11,individual_game_win_by_two:true}},
  ];
  const bracket={...snapshot,games:[...snapshot.games,...playoffs]};
  const bracketText=contents((await buildTournamentPrintPdf(bracket,opts,'packet')).pdf);
  for(const label of ['Winner of P1','Winner of P2','Best 2 of 3','Game 3','playoff bracket worksheet']) assert.ok(bracketText.includes(label),label);
  assert.equal(printSource({loserOf:'P1'}),'Loser of P1');
  assert.match(printScoring({...base,scoring:{format:'GAME_TO_15',target:15,win_by_two:false}}),/15, win by 1/);
  assert.match(printScoring({...base,scoring:{}}),/target: ______/,'never invent a scoring target');
  assert.match(printScoring({...base,scoring:{format:'BEST_2_OF_3',individual_game_target:15,individual_game_win_by_two:false}}),/Best 2 of 3 to 15, win by 1/);
  const dependency={...bracket,games:[...bracket.games,{...playoffs[0],id:'sf2',playoff_game_code:'P2'}].map(game=>game.id==='final'?{...game,playoff_game_code:'A-FIRST',playoff_round:'UNKNOWN',round_label:'UNKNOWN'}:game).reverse()};
  const dependencyOrder=tournamentPrintMatches(dependency,tournamentPrintDraws(dependency,[id]),opts).map(row=>row.game.id);
  assert.ok(dependencyOrder.indexOf('sf1')<dependencyOrder.indexOf('final') && dependencyOrder.indexOf('sf2')<dependencyOrder.indexOf('final'),'dependencies precede a final even when its code sorts first');
  snapshot.games[0].result_type='FORFEIT';
  assert.ok(contents((await buildTournamentPrintPdf(snapshot,{...opts,includeResults:true},'admin')).pdf).includes('(F) Tj'));

  // A deliberately long but realistic four-player name must fit without losing names.
  const long=fixture(8); long.games[0].team_a.participant_names=['Alexandria María Fernández de la Cruz','Christopher Santiago Montgomery Rivera'];
  await buildTournamentPrintPdf(long,{drawIds:[long.draws[0].id]},'packet');

  if (process.env.TOURNAMENT_PRINT_PREVIEW) {
    const previewFixture = n => {
      const sample = savedFixture(n);
      sample.tournament.name='Tres Palapas Halloween Tournament';
      sample.day_scope.selected_day.label='Saturday';
      const division=({3:'Mixed Doubles 3.0 - Pool A',4:'Mixed Doubles 3.5 - Pool A',6:'Mixed Doubles 3.0',7:'Mixed Doubles 4.0',8:'Mixed Doubles 3.5'})[n] || 'Mixed Doubles 4.5';
      sample.draws[0].name=division;
      sample.games.forEach(game=>{game.draw_name=division;});
      return sample;
    };
    const sample=previewFixture(8);
    sample.courts=[{id:'court-1',label:'Court 1',position:1},{id:'court-2',label:'Court 2',position:2}];
    sample.games[0].court_id='court-1';sample.games[1].court_id='court-2';
    const cards=await buildTournamentPrintPdf({...sample,games:sample.games.slice(0,2)},{drawIds:[sample.draws[0].id]},'cards');
    // Separate generated files are intermediate QA only. The delivered preview
    // is assembled from the app's exact PDF output, not redrawn for the demo.
    fs.mkdirSync(process.env.TOURNAMENT_PRINT_PREVIEW,{recursive:true});
    fs.writeFileSync(path.join(process.env.TOURNAMENT_PRINT_PREVIEW,'cards.pdf'),Buffer.from(cards.pdf.output('arraybuffer')));
    for (const n of [3,4,6,7,8,16]) {
      const sample=previewFixture(n);
      const admin=await buildTournamentPrintPdf(sample,{drawIds:[sample.draws[0].id]},'admin');
      fs.writeFileSync(path.join(process.env.TOURNAMENT_PRINT_PREVIEW,`admin-${n}.pdf`),Buffer.from(admin.pdf.output('arraybuffer')));
    }
    const series={...sample,games:sample.games.slice(0,2).map((game,i)=>({...game,stage:'PLAYOFF',round_label:'Semifinal',playoff_round:'SF',playoff_game_code:`P${i+1}`,scoring:{format:'BEST_2_OF_3',individual_game_target:11,individual_game_win_by_two:true}}))};
    const seriesCards=await buildTournamentPrintPdf(series,{drawIds:[sample.draws[0].id]},'cards');
    fs.writeFileSync(path.join(process.env.TOURNAMENT_PRINT_PREVIEW,'series-cards.pdf'),Buffer.from(seriesCards.pdf.output('arraybuffer')));
  }
  console.log('PASS tournament paper packet: 2-16/24/32 teams, oversized-round continuation, saved schedules, multi-division play order, live queue/courts, playoff dependencies, cut-stack and consecutive layouts, byes, filters, scoring formats and no mutations.');
})().catch(error=>{console.error(error);process.exit(1);});
