const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), zlib = require('node:zlib');
const ts = require('typescript');
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file).exports;
  const module = { exports: {} }; cache.set(file, module);
  const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText;
  new Function('require','module','exports',code)(name => name.startsWith('.') ? load(path.resolve(path.dirname(file), name + '.ts')) : require(name), module, module.exports);
  return module.exports;
}
const { buildInterclubMeetPdf } = load(path.join(__dirname, '../lib/interclubMeetPdf.ts'));
function content(pdf) {
  const bytes = Buffer.from(pdf.output('arraybuffer'));
  assert.equal(bytes.subarray(0,5).toString(), '%PDF-');
  const raw = bytes.toString('latin1');
  return Array.from(raw.matchAll(/stream\n([\s\S]*?)\nendstream/g), match => {
    try { return zlib.inflateSync(Buffer.from(match[1], 'latin1')).toString('latin1'); } catch { return match[1]; }
  }).join('\n');
}
const players = new Map(Array.from({ length: 8 }, (_, i) => [`p${i}`, { entry_id: `p${i}`, name: i ? `Player ${i}` : 'José Reyes' }]));
const game = (id, changes = {}) => ({ id, status: 'pending', a: null, b: null, winner: null, players_a: [], players_b: [], played_at: null, ...changes });
const document = { schema_version: 1, meet_id: 'meet-1234', phase: 'regular', format: 'gender', schedule_mode: 'staggered', weather: 'normal', encounters: Array.from({ length: 24 }, (_, i) => ({
  id: `matchup-reference-${i}`, division: ['3.0','3.5','4.0','4.5'][i % 4], club_a: 'alpha', club_b: 'beta', rotation: Math.floor(i/4)+1, tiebreak: null,
  pairings: ['women','men'].map((kind,j) => ({ id: `pair-${i}-${j}`, kind, court: (i%4)*2+j+1, players_a: [`p${j*2}`,`p${j*2+1}`], players_b: [`p${j*2+4}`,`p${j*2+5}`], games: [1,2,3].map(g => game(`${i}-${j}-${g}`)) })),
})) };
const options = { document, meet: { id: document.meet_id, starts_at: '2026-12-17T07:00:00Z', roster_deadline: '2026-12-16T07:00:00Z', host_club_id: 'alpha', courts: 8 }, seasonName: 'Southern BCS', timezone: 'America/Mazatlan', revision: 7, players, clubName: id => id === 'alpha' ? 'Alpha Club' : 'Beta Club' };
(async () => {
  const before = JSON.stringify(document);
  const schedule = await buildInterclubMeetPdf(options, 'schedule');
  const packet = await buildInterclubMeetPdf(options, 'packet');
  assert.equal(schedule.pdf.getNumberOfPages(), 2, 'The four-club draw is a compact two-page schedule');
  assert.equal(packet.pdf.getNumberOfPages(), 27, 'Full export includes schedule, instructions and all 24 score sheets');
  assert.match(schedule.filename, /^Southern-BCS-2026-12-17-meet-123-schedule-r7\.pdf$/);
  assert.match(content(schedule.pdf), /Court schedule/);
  const scheduleText = content(schedule.pdf);
  assert.ok(scheduleText.includes('Games 1-3 against the same opponents'));
  assert.equal((scheduleText.match(/\(1-3\) Tj/g) || []).length, 48, 'Every scheduled pairing reserves its court for all three games');
  const full = content(packet.pdf);
  assert.equal((full.match(/Games 1-3 on this court/g) || []).length, 48, 'Every regular score sheet keeps all games on its assigned court');
  for (const encounter of document.encounters) assert.ok(full.includes(encounter.id), `Missing score sheet ${encounter.id}`);
  assert.ok(full.includes('José Reyes'), 'Accented player names remain readable');
  assert.ok(full.includes('Page 27 of 27'));
  assert.ok(!full.includes('Time played'), 'Paper sheets require scores, not individual game times');
  assert.ok(full.includes('Game dates are recorded automatically'));
  assert.equal(JSON.stringify(document), before, 'Exports do not change the saved draw');

  const incidents = structuredClone(document); incidents.encounters = [incidents.encounters[0]];
  const pairing = incidents.encounters[0].pairings[0]; pairing.court = null; pairing.players_b = [];
  pairing.games[0] = game('injury', { status: 'retired', a: 7, b: 4, winner: 'b', players_a: ['p7','p1'], played_at: '2026-09-28T18:00:00Z', injury_reason: 'Replacement after injury - actual score retained.' });
  pairing.games[1].status = 'forfeit'; pairing.games[1].winner = 'a';
  const incidentPdf = await buildInterclubMeetPdf({ ...options, document: incidents }, 'packet');
  const incidentText = content(incidentPdf.pdf);
  for (const label of ['Forfeit','Injury','Replacement after injury','Player 7','11:00 AM']) assert.ok(incidentText.includes(label), `Missing incident detail ${label}`);

  const final = structuredClone(incidents); final.phase = 'final'; final.format = 'mlp'; final.schedule_mode = 'simultaneous';
  final.encounters[0].pairings.forEach(pairing => { pairing.games = [pairing.games[0]]; });
  final.encounters[0].tiebreak = { status: 'completed', a: 23, b: 21, order_a: ['p0','p1','p2','p3'], order_b: ['p4','p5','p6','p7'] };
  const finalText = content((await buildInterclubMeetPdf({ ...options, document: final }, 'packet')).pdf);
  assert.ok(finalText.includes('Skinny rotating singles') && finalText.includes('A: 23 B: 21'), 'Completed championship tiebreak is preserved');
  assert.ok(!finalText.includes('Staggered starts'));
  assert.ok(!finalText.includes('Games 1-3'), 'Championship pairings remain one game each');
  console.log('PASS interclub PDF: compact and full files, every score sheet, saved revision, accented names, injury substitutions, non-play results, timezone and championship tiebreak');
})().catch(error => { console.error(error); process.exit(1); });
