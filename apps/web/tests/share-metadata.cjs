const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const root = path.resolve(__dirname, '..');
function load(name, mocks = {}) {
  const filename = path.join(root, name);
  const module = new Module(filename, moduleParent);
  module.filename = filename;
  module.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = module.require.bind(module);
  module.require = (id) => Object.hasOwn(mocks, id) ? mocks[id] : original(id);
  const result = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX },
    reportDiagnostics: true
  });
  assert.equal((result.diagnostics || []).filter((d) => d.category === ts.DiagnosticCategory.Error).length, 0);
  module._compile(result.outputText, filename);
  return module.exports;
}
const moduleParent = module;
const core = load('lib/shareMetadataCore.ts');
const club = { name: 'Tres Palapas', slug: 'tres-palapas', description: 'Pickleball in Baja.', location: 'Los Barriles', accent: '#c45299' };
const origin = 'https://pickleballclubsandwich.com';
const prefix = '/clubs/tres-palapas';
let count = 0;
const check = async (name, fn) => { await fn(); count++; console.log(`PASS ${name}`); };
async function resolve(suffix, query = '', responses = {}, c = club) {
  const calls = [];
  const share = await core.resolveClubShare({ path: `/clubs/${c.slug}${suffix}`, query: new URLSearchParams(query), origin }, c, async (p) => { calls.push(p); return responses[p] || null; });
  return { share, calls };
}
const event = { id: 't1', name: 'Baja Classic 2026', start_date: '2026-11-18', end_date: '2026-11-22' };
const tournament = { tournament: event, settings: { registration_slug: 'bajaclassic26', location_name: 'Los Barriles' }, registration_open: true };
const sponsors = { tournament_id: 't1', sponsors: [{ tier: 'presenting', name: 'Homes and Land of Baja' }] };
const responses = {
  [`${prefix}/tournament-registration?registration_slug=bajaclassic26`]: tournament,
  [`${prefix}/tournament-registration?tournament_id=t1`]: tournament,
  [`${prefix}/tournaments/t1/sponsors`]: sponsors,
};
(async () => {
  await check('tournament index promotes the club, never the first tournament', async () => {
    const { share, calls } = await resolve('/tournaments', '', responses);
    assert.equal(share.title, 'Tournaments at Tres Palapas'); assert.deepEqual(calls, []); assert.match(share.description, /tournaments at Tres Palapas/);
  });
  await check('selected tournament carries real dates, location and sponsor', async () => {
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26', responses);
    assert.equal(share.title, 'Baja Classic 2026 | Tres Palapas'); assert.match(share.description, /Nov 18–22, 2026/);
    assert.match(share.description, /Los Barriles/); assert.match(share.description, /Homes and Land of Baja/);
    assert.equal(share.siteName, 'Tres Palapas');
  });
  await check('existing ID and deep links remain supported', async () => {
    assert.equal((await resolve('/tournaments', 'tournament_id=t1', responses)).share.title, 'Baja Classic 2026 | Tres Palapas');
    assert.match((await resolve('/tournaments/t1/results', '', responses)).share.title, /Baja Classic 2026 — Results/);
  });
  await check('tournament tabs describe the selected page', async () => {
    for (const [tab, title] of [['register','Registration'], ['roster','Player Roster'], ['partners','Partner Board'], ['results','Results']]) {
      const { share } = await resolve(`/tournaments/${tab}`, 'tournament=bajaclassic26', responses);
      assert.match(share.title, new RegExp(title)); assert.match(share.title, /Baja Classic 2026/);
    }
  });
  await check('wrong or missing selectors cannot advertise an unrelated event', async () => {
    const bad = { [`${prefix}/tournament-registration?registration_slug=wrong`]: tournament };
    const { share } = await resolve('/tournaments', 'tournament=wrong', bad);
    assert.equal(share.title, 'Tournament unavailable | Tres Palapas'); assert.equal(share.noindex, true);
  });
  await check('conflicting slug and id are rejected', async () => {
    const bad = { [`${prefix}/tournament-registration?registration_slug=bajaclassic26&tournament_id=other`]: tournament };
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26&tournament_id=other', bad);
    assert.equal(share.noindex, true); assert.doesNotMatch(share.title, /Baja Classic/);
  });
  await check('league and RR overviews promote the section and club', async () => {
    assert.equal((await resolve('/leagues')).share.title, 'Leagues at Tres Palapas');
    assert.equal((await resolve('/round-robin-generator')).share.title, 'Round Robins at Tres Palapas');
  });
  await check('league names with spaces and special characters resolve exactly', async () => {
    const name = 'Monday 4.0 & Friends';
    const fixture = { [`${prefix}/league-results?${new URLSearchParams({ league_name: name })}`]: { selected_league: name, league: { name } } };
    assert.equal((await resolve(`/leagues/${encodeURIComponent(name)}/standings`, '', fixture)).share.title, `${name} — Standings | Tres Palapas`);
    assert.match((await resolve('/league-results', new URLSearchParams({ league_name: name, tab: 'weekly' }).toString(), fixture)).share.title, /Weekly Results/);
  });
  await check('invalid league cannot silently fall back to a different league', async () => {
    const fixture = { [`${prefix}/league-results?league_name=missing`]: { selected_league: 'Other' } };
    assert.equal((await resolve('/leagues/missing', '', fixture)).share.noindex, true);
  });
  await check('RR rounds and standings promote the individual session', async () => {
    const fixture = { [`${prefix}/play-generators/sessions/rr1`]: { session: { session_key: 'rr1', title: 'Friday Social Round Robin' } } };
    assert.equal((await resolve('/round-robin-generator/sessions/rr1/rounds/2', '', fixture)).share.title, 'Friday Social Round Robin — Round 2 | Tres Palapas');
    assert.match((await resolve('/round-robin-generator/sessions/rr1/standings', '', fixture)).share.title, /Standings/);
  });
  await check('live sessions promote the individual event', async () => {
    const fixture = { [`${prefix}/live-sessions/live1`]: { session: { session_key: 'live1', title: 'Wednesday Ladder' } } };
    assert.equal((await resolve('/live/live1', '', fixture)).share.title, 'Wednesday Ladder | Tres Palapas');
  });
  await check('profiles use only the public display identity, not contact details', async () => {
    const fixture = { [`${prefix}/players/5?recent_limit=1&history_limit=1`]: { player: { id: 5, name: 'Internal name', email: 'private@example.com' }, identity: { display_name: 'Public Player' } } };
    const { share } = await resolve('/players/5', '', fixture);
    assert.match(share.title, /Public Player/); assert.doesNotMatch(JSON.stringify(share), /private@example|Internal name/);
  });
  await check('each club uses its own identity and logo', async () => {
    const other = { name: 'Other Club', slug: 'other-club', image: 'https://images.example.com/other.png' };
    const { share } = await resolve('/tournaments', '', {}, other);
    assert.equal(share.title, 'Tournaments at Other Club'); assert.equal(share.image, other.image); assert.doesNotMatch(JSON.stringify(share), /Tres|PCS|Sandwich/);
  });
  await check('event artwork takes precedence over the club logo', async () => {
    const fixture = { ...responses, [`${prefix}/tournament-registration?registration_slug=bajaclassic26`]: { ...tournament, tournament: { ...event, event_tags: { poster_url: 'https://images.example.com/event.jpg' } } } };
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26', fixture, { ...club, image: 'https://images.example.com/club.jpg' });
    assert.equal(share.image, 'https://images.example.com/event.jpg');
  });
  await check('signed images, unsafe protocols and private hosts are not advertised', async () => {
    for (const image of ['javascript:alert(1)', 'http://127.0.0.1/private', 'https://x.supabase.co/storage/v1/object/sign/private?token=secret', 'https://images.example.com/logo?token=secret']) assert.equal(core.publicImage(image, origin), undefined);
  });
  await check('secret query values never appear in metadata or canonical links', async () => {
    const { share, calls } = await resolve('/tournaments', 'tournament=bajaclassic26&edit_token=SECRET&email=private@example.com', responses);
    assert.doesNotMatch(JSON.stringify(share), /SECRET|private@example/); assert.equal(share.noindex, true); assert.deepEqual(calls, []); assert.equal(share.query, '');
  });
  await check('tracking and unsupported query parameters are discarded', async () => {
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26&utm_source=whatsapp&evil=SECRET', responses);
    assert.equal(share.query, 'tournament=bajaclassic26');
  });
  await check('staff and confirmation pages cannot expose private details', async () => {
    for (const p of ['/admin/score-entry', '/tournaments/confirmation', '/tournaments/edit']) {
      const { share, calls } = await resolve(p, '', responses); assert.equal(share.noindex, true); assert.deepEqual(calls, []);
    }
  });
  await check('unpublished sites and unlisted discovery are respected', async () => {
    assert.equal((await resolve('/tournaments', '', {}, { ...club, available: false })).share.title, 'Club unavailable');
    assert.equal((await resolve('/tournaments', '', {}, { ...club, discoverable: false })).share.noindex, true);
  });
  await check('published custom pages use their own title, text and image', async () => {
    const { share } = await resolve('/pages/visit', '', {}, { ...club, page: { title: 'Plan Your Visit', description: '<p>Welcome to <b>Baja</b>.</p>', image: 'https://images.example.com/courts.jpg' } });
    assert.equal(share.title, 'Plan Your Visit | Tres Palapas'); assert.doesNotMatch(share.description, /<p>|<b>/); assert.match(share.image, /courts/);
  });
  await check('every standard club section avoids generic PCS promotion', async () => {
    for (const section of ['leaderboards','players','matches','play','live','ladder-generator','team-match-generator','challenge-ladder','weekly-recap','badge-codex','match-explorer','interclub']) {
      const { share } = await resolve(`/${section}`); assert.match(share.title, /Tres Palapas/); assert.doesNotMatch(share.title, /PCS|Sandwich/);
    }
  });
  await check('dates do not move to the previous day and invalid dates are omitted', async () => {
    assert.equal(core.dateRange('2026-11-18', '2026-11-22'), 'Nov 18–22, 2026'); assert.equal(core.dateRange('2026-02-30', null), '');
  });
  await check('API failure falls back safely without a fabricated event', async () => {
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26'); assert.equal(share.noindex, true); assert.doesNotMatch(share.title, /Baja/);
  });
  const server = load('lib/shareMetadata.ts', {
    'next/headers': { headers: () => new Headers() }, react: { cache: (fn) => fn },
    './shareClub': { loadSharingClub: async () => club }, './shareMetadataCore': core
  });
  await check('complete Open Graph, Twitter and canonical fields preserve selected page', async () => {
    const { share } = await resolve('/tournaments', 'tournament=bajaclassic26', responses);
    const meta = server.sharingMetadata(share, origin);
    assert.equal(meta.openGraph.title, share.title); assert.equal(meta.twitter.title, share.title); assert.equal(meta.openGraph.siteName, club.name);
    assert.equal(meta.openGraph.url, `${origin}${prefix}/tournaments?tournament=bajaclassic26`);
    assert.match(meta.openGraph.images[0].url, /share-image/); assert.equal(meta.openGraph.images[0].width, 1200);
    assert.equal(meta.title.absolute, share.title); assert.equal(meta.alternates.canonical, meta.openGraph.url);
  });
  await check('preview hosts cannot be poisoned by an arbitrary domain', async () => {
    assert.equal(server.sharingOrigin('https://jupr-test.vercel.app'), 'https://jupr-test.vercel.app');
    assert.notEqual(server.sharingOrigin('https://pickleballclubsandwich.com.evil.example'), 'https://pickleballclubsandwich.com.evil.example');
  });
  console.log(`Sharing metadata: ${count} checks passed.`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
