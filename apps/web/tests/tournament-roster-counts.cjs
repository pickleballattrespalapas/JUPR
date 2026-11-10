const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const member = (name, key) => ({ display_name: name, event_entry_key: key });
const entries = [
  { public_entry_key: 'team', event_day_label: 'Thursday', event_family: 'Doubles', division: '4.0',
    event_label: 'Doubles 4.0', status: 'Registered', entry_type: 'Team',
    members: [member('Alex Lee', 'alex-doubles'), member('Alex Lee', 'other-alex-doubles')] },
  { public_entry_key: 'single', event_day_label: 'Wednesday', event_family: 'Singles', division: '4.0',
    event_label: 'Singles 4.0', status: 'Registered', entry_type: 'Registration',
    members: [member('Alex Lee', 'alex-singles')] },
  { public_entry_key: 'reservation', event_day_label: 'Thursday', event_family: 'Doubles', division: '4.0',
    event_label: 'Doubles 4.0', status: 'Pending Registration', entry_type: 'Team',
    members: [member('Sam Player', 'sam-doubles'), { display_name: 'Guest Player', registration_pending: true }] },
];
const summary = { total_registrations: 3, total_players: 3, total_event_entries: 4, players_needing_partners: 0 };
const data = {
  tournament: { id: 't1', name: 'Count Fixture', start_date: '2026-11-18', end_date: '2026-11-22' },
  settings: { registration_slug: 'count-fixture', registration_status: 'open' },
  tournaments: [], days: [], events: [], players: [], available: true, registration_open: true,
  summary, roster_summary: summary, roster: { registrations_by_event: entries },
};
const mocks = {
  '@/lib/tournamentRegistrationApi': {
    getClubTournamentRoster: async () => ({ data }),
    getClubTournamentRegistration: async () => ({ data }),
  },
  '@/components/PublicClubLink': { __esModule: true, default: ({ children, ...props }) => React.createElement('a', props, children) },
  '@/components/PublicTournamentModuleHeader': { __esModule: true, default: () => null },
  '@/components/PublicTournamentSponsors': { __esModule: true, default: () => null },
  '@/components/PublicTournamentNav': { __esModule: true, default: () => null, publicTournamentHref: () => '/fixture' },
  'next/navigation': { redirect: () => { throw new Error('Unexpected redirect'); } },
};
function load(file) {
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    if (mocks[name]) return mocks[name];
    if (name.startsWith('@/')) return load(name.slice(2) + '.ts');
    return require(name);
  }, module, module.exports);
  return module.exports;
}
const text = html => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');

(async () => {
  const { countIndividualEventEntries: count } = load('lib/tournamentRosterCounts.ts');
  assert.equal(count(entries), 4, 'two names on a team are two entries; an unregistered guest is excluded');
  assert.equal(count([...entries, entries[0]]), 4, 'duplicate presentation does not duplicate individual entries');
  assert.equal(count([{ ...entries[0], members: [{ display_name: 'Same Name' }, { display_name: 'Same Name' }] }]), 2);
  assert.equal(count([]), 0);

  const props = filters => ({ params: Promise.resolve({ clubSlug: 'fixture' }), searchParams: Promise.resolve({ tournament: 'count-fixture', ...filters }) });
  const RosterPage = load('app/clubs/[clubSlug]/tournament-roster/page.tsx').default;
  const roster = text(renderToStaticMarkup(await RosterPage(props({}))));
  assert.match(roster, /Players 3 Individual event entries 4/);
  assert.match(roster, /Showing 4 of 4 individual event entries/);
  assert.match(roster, /Thursday · Doubles · 4.0 3 player entries/);
  assert.match(roster, /Wednesday · Singles · 4.0 1 player entry/);
  assert.doesNotMatch(roster, /Total entries|Sign-ups/);

  const filtered = text(renderToStaticMarkup(await RosterPage(props({ day: 'Wednesday' }))));
  assert.match(filtered, /Showing 1 of 4 individual event entries/);
  assert.doesNotMatch(filtered, /Guest Player/);
  const pending = text(renderToStaticMarkup(await RosterPage(props({ status: 'Pending registration' }))));
  assert.match(pending, /Showing 1 of 4 individual event entries/);
  assert.match(pending, /Guest Player · Pending registration/);

  const HomePage = load('app/clubs/[clubSlug]/tournaments/page.tsx').default;
  const home = text(renderToStaticMarkup(await HomePage(props({}))));
  assert.match(home, /Players 3 Individual event entries 4 Players needing partners 0/);
  console.log('Tournament counts: individual players, per-event totals, filters, repeated rows, and pending guests passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
