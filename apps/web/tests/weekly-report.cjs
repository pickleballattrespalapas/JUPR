const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
function load(file) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname,'..',file),'utf8'), {compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.CommonJS}}).outputText;
  const module = {exports:{}}; new Function('module','exports',code)(module,module.exports); return module.exports;
}
const {buildWeeklyReportEmail,emailUrl,reportLink,reportPath} = load('lib/weeklyReportEmail.ts');
const {weeklyReportDatePreset} = load('lib/clubDate.ts');
const fixture = require('./fixtures/weekly-report.json');
const options = {origin:'https://staging.example.com',clubSlug:'tres-palapas',published:true};
const compact = buildWeeklyReportEmail(fixture,options);
assert.match(compact,/max-width:620px/);
for (const color of ['#123e37','#f8f3e9','#17665c','#cb794f']) assert.ok(compact.includes(color));
assert.match(compact,/Tuesday Moneyball/);
assert.match(compact,/Summer Social/);
assert.ok(!compact.includes('Baja League · 4.0'), 'Compact email features three events');
assert.ok(!compact.includes('coverage') && !compact.includes('Example pending session'), 'Private review data is not exported');
assert.match(compact,/week=2026-10-05/);
assert.ok(!compact.includes('<html'), 'Section can be embedded in the existing newsletter');
const full = buildWeeklyReportEmail(fixture,{...options,compact:false,document:true});
assert.match(full,/<!doctype html>/); assert.match(full,/Baja League · 4.0/); assert.match(full,/Mixed 3.5/);
assert.match(full,/https:\/\/app.courtreserve.com/);
const draft = buildWeeklyReportEmail(fixture,{...options,published:false});
assert.match(draft,/DRAFT PREVIEW/); assert.ok(!draft.includes('View the full weekly report'));
const attack = structuredClone(fixture);
attack.report.opening = '<img src=x onerror=alert(1)> & "text"';
attack.report.upcoming = [{text:'Sign up',url:'javascript:alert(1)'}];
const escaped = buildWeeklyReportEmail(attack,options);
assert.match(escaped,/&lt;img/); assert.ok(!escaped.includes('<img') && !escaped.includes('href="javascript:'));
assert.equal(emailUrl('data:text/html,evil'),''); assert.equal(emailUrl('https://u:p@example.com'),'');
assert.equal(reportLink(options.origin,'tres','//evil.com'),'');
assert.equal(reportPath('tres','match-explorer'),'');
assert.equal(reportPath('tres','league-results'),'');
assert.equal(reportPath('tres','leagues/Fall%203.5/weekly-history?week=4'),'/clubs/tres/leagues/Fall%203.5/weekly-history?week=4');
assert.equal(reportLink(options.origin,'tres','/interclub/season'),'https://staging.example.com/interclub/season');
assert.equal(reportPath('tres','round-robin-generator/shared/session/standings'),'/clubs/tres/round-robin-generator/shared/session/standings');
const league = structuredClone(fixture);
league.report.event_results = [{title:'Fall League',kind:'League',summary:'',details:[],href:'league-results',weekly_reports:[{label:'Weekly report · Week 4',href:'leagues/Fall/weekly-history?week=4'}]}];
const leagueEmail = buildWeeklyReportEmail(league,{...options,compact:true});
assert.match(leagueEmail,/Weekly report · Week 4/);
assert.match(leagueEmail,/https:\/\/staging.example.com\/clubs\/tres-palapas\/leagues\/Fall\/weekly-history\?week=4/);
assert.ok(!leagueEmail.includes('league-results'));
assert.deepEqual(weeklyReportDatePreset('last_week',new Date('2026-10-16T17:00:00Z')),{start:'2026-10-05',end:'2026-10-11'});
assert.deepEqual(weeklyReportDatePreset('this_week',new Date('2026-10-16T17:00:00Z')),{start:'2026-10-12',end:'2026-10-16'});
assert.deepEqual(weeklyReportDatePreset('last_seven',new Date('2026-10-16T17:00:00Z')),{start:'2026-10-10',end:'2026-10-16'});
assert.deepEqual(weeklyReportDatePreset('last_week',new Date('2026-10-12T05:00:00Z')),{start:'2026-09-28',end:'2026-10-04'}, 'Sunday night in Mazatlan stays in the correct week');
console.log('Weekly report email, escaping, compact/full variants, and club date presets: passed');
for (const parameter of ['tournament_id', 'tournament']) {
  const href = `tournament-results?${parameter}=event-123`;
  assert.equal(reportPath('tres', href), `/clubs/tres/${href}`);
  for (const suffix of ['&redirect=https://evil.example', '#fragment', '/../../admin', '&tournament_id=other']) {
    assert.equal(reportPath('tres', href + suffix), '');
  }
}

const briefTools = load('lib/weeklyReportEmail.ts');
const teaserSource = {start_date:"2026-10-26",report:{headline:"This week on the courts",edition:[
  {key:"events",items:[{title:"Mixed 3.5",kind:"Halloween Tournament",lines:["1. Team A"],text:"Podium"}]}],
  engagement:{mode:"automatic",featured:"poll",poll:{question:"Which drill?",options:["Drops","Resets"]},
    subject:"",body:""}}};
const short = buildWeeklyReportEmail(teaserSource,{...options,teaser:true});
assert.match(short,/Team A/);assert.match(short,/Which drill/);
assert.equal((short.match(/<a href=/g)||[]).length,1);
assert.match(short,/weekly-recap\?week=2026-10-26&amp;from=email/);
assert.doesNotMatch(short,/Cast my vote|<form|View results/);
assert.doesNotMatch(buildWeeklyReportEmail(teaserSource,{...options,teaser:true,published:false}),/<a href=/);
teaserSource.report.engagement.subject="Our own story";teaserSource.report.engagement.body="A <story>.\n\nJoin us.";
assert.match(buildWeeklyReportEmail(teaserSource,{...options,teaser:true}),/A &lt;story&gt;/);
assert.deepEqual(briefTools.shortRecapEmail(teaserSource).paragraphs,["A <story>.","Join us."]);
teaserSource.report.engagement.mode="review";teaserSource.report.engagement.body="";
assert.doesNotMatch(buildWeeklyReportEmail(teaserSource,{...options,teaser:true}),/Which drill/);
console.log("Short email preserves edits, respects approval and links once to the dated recap.");

const historical = structuredClone(fixture);
historical.report.achievements = ["Retired badge should stay hidden"];
assert.doesNotMatch(buildWeeklyReportEmail(historical, { ...options, compact: false }), /Retired badge|NEW ACHIEVEMENTS/);
historical.report.edition = [
  { key: "achievements", title: "Hidden badge collection", items: [{ id: "badge", title: "Retired badge", lines: [] }] },
  { key: "events", title: "Event results", items: [{ id: "result", title: "Tournament podium", lines: [] }] },
];
historical.report.email_edition = historical.report.edition;
for (const compact of [true, false]) {
  const content = buildWeeklyReportEmail(historical, { ...options, compact });
  assert.doesNotMatch(content, /Hidden badge|Retired badge/);
  assert.match(content, /Tournament podium/);
}
