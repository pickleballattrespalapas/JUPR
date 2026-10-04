const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const React = require('react'), ts = require('typescript');
const { act, create } = require('react-test-renderer');
const Link = ({ children, ...props }) => React.createElement('a', props, children);
const modules = new Map();
function load(file) {
  if (modules.has(file)) return modules.get(file);
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    if (name === 'next/link') return Link;
    if (name.endsWith('.css')) return new Proxy({}, { get: (_, key) => key === '__esModule' ? false : String(key) });
    if (name === '@/lib/adminAuthClient') return { getAdminApiBaseUrl: () => 'https://api.test' };
    if (name === '@/lib/useAdminSession' || name === '@/lib/useAdminWorkspace') return {};
    if (name.startsWith('@/')) return load(`${name.slice(2)}.ts`);
    if (name.startsWith('.')) return load(`${path.join(path.dirname(file), name)}.ts`);
    return require(name);
  }, module, module.exports);
  modules.set(file, module.exports); return module.exports;
}
const helpers = load('lib/interclubSeasonGuide.ts');
const { SeasonGuideView, GuideLoader } = load('app/admin/interclub/season/SeasonGuide.tsx');
const season = { id: 'season-1', organizer_club_id: 'alpha', details: { name: 'Southern BCS — Winter Interclub League', timezone: 'America/Mazatlan', start_date: '2099-01-01', end_date: '2099-03-01' }, registration: { status: 'closed', meet_planning_open: true } };
const meets = ['regular', 'final'].map((phase, i) => ({ id: `meet-${i}`, competition_phase: phase, host_club_id: 'alpha', starts_at: `2099-0${i + 1}-01T17:00:00Z` }));
const batches = meets.map(meet => ({ meet_id: meet.id, phase: meet.competition_phase, state: 'approved', revision: 3, approved_revision: 3, ratings_status: 'completed', document: { weather: 'normal' } }));
const finished = { registration: { season, is_organizer: true, season_complete: true, clubs: [{ id: 'alpha', name: 'Tres Palapas' }] }, competition: { season, is_organizer: true, meets, batches, club_cup: { status: 'complete' } }, awards: { ready: true, current: true, revision: 1, problems: [] }, history: { can_start: true, current: { source_id: season.id }, seasons: [{ source_id: season.id, position: 1, label: '2026' }] } };
const clone = () => structuredClone(finished);
function stateChecks() {
  let data = clone();
  assert.equal(helpers.buildSeasonGuide(data).stage, 5, 'A season played ahead of its dates is finished');
  assert.equal(helpers.buildSeasonGuide(data).complete, true);
  data.history.seasons.push({ source_id: 'next', label: '2027', position: 2, admin_href: '/admin/interclub?season=next' });
  data.history.can_start = false;
  assert.equal(helpers.buildSeasonGuide(data).action.href, '/admin/interclub?season=next', 'Resume the linked draft rather than propose a duplicate');
  data = clone(); data.awards.current = false;
  assert.equal(helpers.buildSeasonGuide(data).stage, 4, 'Published scores alone do not skip trophy review');
  data.awards.ready = false; data.awards.problems = ['Resolve championship tie'];
  assert.deepEqual(helpers.buildSeasonGuide(data).instructions, ['Resolve championship tie']);
  for (const patch of [{ ratings_status: 'failed' }, { ratings_status: 'pending' }, { state: 'draft' }, { state: 'submitted' }, { approved_revision: 2 }, { document: { weather: 'rescheduled' } }]) {
    data = clone(); Object.assign(data.competition.batches[0], patch);
    const guide = helpers.buildSeasonGuide(data);
    assert.equal(guide.stage, 2, 'An unresolved regular meet takes priority over published results');
    assert.equal(guide.complete, false);
    assert.ok(guide.action.href.includes('meet=meet-0'));
  }
  data = clone(); data.competition.meets.pop(); data.competition.batches.pop(); data.competition.club_cup.status = 'provisional';
  assert.equal(helpers.buildSeasonGuide(data).stage, 3, 'Approved regular meets lead to qualification and championship setup');
  assert.equal(helpers.buildSeasonGuide(data).action.label, 'Open championship setup');
  data = clone(); data.competition.batches.pop();
  assert.equal(helpers.buildSeasonGuide(data).stage, 3);
  assert.ok(helpers.buildSeasonGuide(data).action.href.includes('meet=meet-1'));
  data = clone(); data.competition.meets = []; data.competition.batches = [];
  assert.equal(helpers.buildSeasonGuide(data).stage, 2, 'No meets is not a completed regular season');
  for (const status of ['unconfigured', 'scheduled', 'open']) {
    data = clone(); data.registration.season.registration = { status, meet_planning_open: false }; data.competition = null; data.awards = null;
    assert.equal(helpers.buildSeasonGuide(data).stage, 1);
    assert.ok(helpers.buildSeasonGuide(data).action.href.includes('/registrations?'));
  }
  data = clone(); data.registration.is_organizer = false; data.registration.season_complete = false; data.registration.own_participation = { status: 'invited' }; data.competition = null;
  assert.match(helpers.buildSeasonGuide(data).action.href, /#invitation-title$/);
  data.registration.own_participation.status = 'accepted'; data.competition = clone().competition; data.awards = null; data.history = null;
  assert.equal(helpers.buildSeasonGuide(data).complete, false, 'A participant’s own finished meets do not mark the whole season complete');
  data.competition.batches.pop();
  assert.equal(helpers.buildSeasonGuide(data).stage, 3, 'A participating club can find its upcoming final');
  assert.ok(helpers.buildSeasonGuide(data).action.href.includes('meet=meet-1'));
  assert.ok(helpers.buildSeasonGuide(clone()).hrefs[2].includes('meet=meet-0'), 'Regular meet review opens a regular meet rather than the championship landing');
}

async function renderingAndLoading() {
  let tree;
  await act(async () => { tree = create(React.createElement(SeasonGuideView, { data: clone() })); });
  assert.equal(tree.root.findAllByType('h3').length, 6);
  assert.equal(tree.root.findAllByProps({ 'aria-current': 'step' }).length, 1);
  assert.ok(tree.root.findAllByType('a').some(a => a.props.href.includes('/event-history?')));
  await act(async () => tree.unmount());
  let data = clone(), requests = [], failAwards = false;
  global.fetch = async (url, options) => {
    requests.push({ url, options });
    const body = url.includes('/registrations/') ? data.registration : url.includes('/competition/') ? data.competition : url.includes('/event-seasons?') ? data.history : failAwards ? { detail: 'Awards unavailable' } : data.awards;
    return { ok: !(failAwards && url.endsWith('/awards')), status: failAwards && url.endsWith('/awards') ? 503 : 200, json: async () => body };
  };
  await act(async () => { tree = create(React.createElement(GuideLoader, { clubId: 'alpha', seasonId: season.id, accessToken: 'test-token' })); });
  assert.equal(requests.length, 4);
  assert.ok(requests.every(r => !r.options.method && r.url.includes('/clubs/alpha/')), 'Guide only reads the selected club’s authorized endpoints');
  failAwards = true;
  await act(async () => tree.root.findByType('button').props.onClick());
  assert.equal(tree.root.findAllByProps({ role: 'alert' }).length, 1);
  assert.equal(tree.root.findAllByType(SeasonGuideView).length, 0, 'A failed refresh removes stale completion and next-season advice');
  await act(async () => tree.unmount());
  data = clone(); data.registration.is_organizer = false; data.registration.own_participation = { status: 'accepted' }; requests = []; failAwards = false;
  await act(async () => { tree = create(React.createElement(GuideLoader, { clubId: 'beta', seasonId: season.id, accessToken: 'test-token' })); });
  assert.equal(requests.length, 2, 'A participant never reads organizer-only awards or draft history');
  await act(async () => tree.unmount());
  data.registration.own_participation.status = 'invited'; requests = [];
  await act(async () => { tree = create(React.createElement(GuideLoader, { clubId: 'beta', seasonId: season.id, accessToken: 'test-token' })); });
  assert.equal(requests.length, 1, 'An unaccepted invitation never opens competition');
  await act(async () => tree.unmount());
}

stateChecks();
renderingAndLoading().then(() => {
  if (process.env.SEASON_GUIDE_RENDER) {
    const css = fs.readFileSync(path.join(__dirname, '../app/admin/interclub/season/season.module.css'), 'utf8');
    const html = require('react-dom/server').renderToStaticMarkup(React.createElement('main', { className: 'page' }, React.createElement(SeasonGuideView, { data: clone() })));
    fs.writeFileSync(process.env.SEASON_GUIDE_RENDER, `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;background:#f8fafc;font-family:Arial,sans-serif}${css}</style></head><body>${html}</body></html>`);
  }
  console.log('Season guide lifecycle, corrections, ratings, awards, rollover, access boundaries and refresh checks passed');
}).catch(error => { console.error(error); process.exit(1); });
