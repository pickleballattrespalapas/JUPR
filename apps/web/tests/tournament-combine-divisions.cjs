const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const React = require('react');
const { act, create } = require('react-test-renderer');
const ts = require('typescript');
global.IS_REACT_ACT_ENVIRONMENT = true;

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, '..', relative);
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => overrides[name] || originalRequire(name);
  compiled._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { esModuleInterop: true, jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, filename);
  return compiled.exports;
}

const confirmationMock = { ConfirmAction: props => React.createElement('button', {
  disabled: props.disabled || props.busy,
  'data-review': props.description,
  onClick: () => props.onConfirm(props.confirmationText),
}, props.triggerLabel) };
const combineModule = load('app/admin/tournaments/ops/CombineTournamentDivisionsPanel.tsx', { '@/components/ConfirmAction': confirmationMock });
const Panel = combineModule.default;
const source = { id: 'four', event_family_label: 'Mixed doubles', division_name: '4.0', event_type: 'DOUBLES', gender_restriction: 'MIXED', partner_required: true, enabled: true, registration_day_id: 'tonight', competition_format: 'STANDARD' };
const higher = { ...source, id: 'four-five', division_name: '4.5' };
const next = { ...source, id: 'five', division_name: '5.0' };
const otherDay = { ...source, id: 'other-day', registration_day_id: 'tomorrow', division_name: 'Other day' };
const otherFamily = { ...source, id: 'other-family', event_family_label: "Men's doubles", gender_restriction: 'MALE', division_name: 'Men' };
const unavailable = { ...source, id: 'disabled', enabled: false, division_name: 'Disabled' };
const draw = { id: 'draw-four', name: 'Mixed 4.0', event_option_id: source.id, registration_day_id: 'tonight', status: 'DRAFT', updated_at: 'v1' };
const eventOptions = [source, higher, next, otherDay, otherFamily, unavailable];
function text(node) { return typeof node === 'string' || typeof node === 'number' ? String(node) : (node.children || []).map(text).join(''); }
const button = root => root.findAllByType('button').find(node => text(node) === 'Combine selected divisions');
const checkbox = (root, label) => root.findAllByType('label').find(node => text(node).includes(label)).findByType('input');

async function testPanel() {
  let renderer;
  const writes = [];
  const props = { draw, draws: [draw], eventOptions, hasGames: false, disabled: false, busy: false, onCombine: async (...args) => { writes.push(args); return { status: 'success', title: 'Combined', description: 'Combined' }; } };
  await act(async () => { renderer = create(React.createElement(Panel, props)); });
  assert.equal(button(renderer.root).props.disabled, true);
  assert.equal(checkbox(renderer.root, '4.0').props.disabled, true, 'the original source cannot be removed');
  assert.doesNotMatch(text(renderer.root), /Other day|Men's|Disabled/);
  await act(async () => checkbox(renderer.root, '4.5').props.onChange({ target: { checked: true } }));
  assert.equal(button(renderer.root).props.disabled, false);
  assert.match(button(renderer.root).props['data-review'], /one shared podium/);
  assert.match(button(renderer.root).props['data-review'], /Original registrations remain unchanged/);
  assert.equal(renderer.root.findAllByType('input').find(node => node.props.maxLength).props.value, 'Mixed doubles / 4.0 / 4.5');
  await act(async () => button(renderer.root).props.onClick());
  assert.deepEqual(writes[0], [['four', 'four-five'], 'Mixed doubles / 4.0 / 4.5', 'COMBINE DIVISIONS']);

  await act(async () => renderer.root.findAllByType('input').find(node => node.props.maxLength).props.onChange({ target: { value: '   ' } }));
  assert.equal(button(renderer.root).props.disabled, true, 'a meaningful draw name is required');

  for (const [override, message] of [
    [{ hasGames: true }, /after games have been generated/],
    [{ draw: { ...draw, pool_config: { version: 1, pools: [] } } }, /Remove and save.*pool split/],
    [{ draw: { ...draw, status: 'COMPLETED' } }, /draft or ready draw/],
  ]) {
    await act(async () => renderer.update(React.createElement(Panel, { ...props, ...override, key: String(message) })));
    assert.equal(button(renderer.root).props.disabled, true);
    assert.match(text(renderer.root), message);
  }

  const combined = { ...draw, source_event_option_ids: ['four', 'four-five'] };
  await act(async () => renderer.update(React.createElement(Panel, { ...props, draw: combined, key: 'combined' })));
  assert.equal(checkbox(renderer.root, '4.5').props.disabled, true);
  await act(async () => checkbox(renderer.root, '5.0').props.onChange({ target: { checked: true } }));
  await act(async () => button(renderer.root).props.onClick());
  assert.deepEqual(writes[1][0], ['four', 'four-five', 'five']);
  await act(async () => renderer.update(React.createElement(Panel, { ...props, draws: [draw, { ...draw, id: 'other-combined', event_option_id: 'four-five', source_event_option_ids: ['four-five', 'five'] }], key: 'occupied' })));
  assert.equal(checkbox(renderer.root, '4.5').props.disabled, true);
  assert.match(text(renderer.root), /already belongs to another combined draw/);
  await act(async () => renderer.unmount());
  assert.equal(combineModule.compatibleDivision({ ...higher, scheduled_day_ids: ['tomorrow'] }, source, 'tonight'), false);
  assert.equal(combineModule.compatibleDivision({ ...higher, competition_format: 'TEAM' }, source, 'tonight'), false);
  assert.equal(combineModule.compatibleDivision({ ...higher, event_family_key: 'different' }, source, 'tonight'), false);
}

async function testOpsIntegration() {
  const Ops = load('app/admin/tournaments/ops/TournamentOpsPanel.tsx', {
    '@/components/SearchablePlayerSelect': { __esModule: true, default: () => null },
    './TournamentDrawPoolsPanel': { __esModule: true, default: () => null },
    './TournamentPlayerLinkReview': { __esModule: true, default: () => null },
    './CombineTournamentDivisionsPanel': combineModule,
    '@/components/ConfirmAction': confirmationMock,
    'next/link': { __esModule: true, default: ({ children }) => children },
    '@/components/interaction': { InteractionActionError: class extends Error {}, actionSuccess: (title, description) => ({ status: 'success', title, description }) },
    '@/lib/useAdminSession': { useAdminSession: () => ({ accessToken: 'fixture-token' }) },
    '@/lib/useAuthenticatedAutoLoad': load('lib/useAuthenticatedAutoLoad.ts'),
    '@/lib/tournamentRouteContext': { tournamentRouteHref: value => value },
    '@/lib/tournamentRoundRobinIdempotency.mjs': require('../lib/tournamentRoundRobinIdempotency.mjs'),
  }).default;
  let snapshot = { ok: true, tournament: { id: 'tourney', name: 'Tonight' }, draws: [draw], teams: [], games: [], source_game_versions: [], podium: [], event_options: eventOptions, players: [], summary: { draws: 1, teams: 0, games: 0, podium: 0 }, state_fingerprint: 'reviewed-before-combine' };
  const requests = [];
  const priorFetch = global.fetch;
  global.fetch = async (url, options = {}) => {
    requests.push({ url, options });
    if (options.method === 'POST') {
      snapshot = { ...snapshot, draws: [{ ...draw, name: 'Mixed doubles / 4.0 / 4.5', source_event_option_ids: ['four', 'four-five'], updated_at: 'v2' }], state_fingerprint: 'after-combine' };
      return new Response(JSON.stringify({ ok: true, draw: snapshot.draws[0], moved_team_count: 2, combined_source_draw_ids: ['draw-higher'] }), { headers: { 'Content-Type': 'application/json' } });
    }
    return new Response(JSON.stringify(snapshot), { headers: { 'Content-Type': 'application/json' } });
  };
  let renderer;
  try {
    const props = { apiBase: 'https://fixture.invalid', clubId: 'club', initialTournamentId: 'tourney', initialDrawId: draw.id, workflow: 'draws', status: { enabled: true, mutation_runtime: { service_role_ready: true, surface_flags: { operations: { enabled: true } } }, operations_runtime: { operations_mutations_enabled: true } } };
    await act(async () => { renderer = create(React.createElement(Ops, props)); });
    await act(async () => checkbox(renderer.root.findByProps({ 'data-testid': 'combine-tournament-divisions' }), '4.5').props.onChange({ target: { checked: true } }));
    await act(async () => button(renderer.root).props.onClick());
    const write = requests.find(row => row.options.method === 'POST');
    assert.equal(write.url, 'https://fixture.invalid/admin/clubs/club/tournaments/admin/tournaments/tourney/draws/draw-four/combine-divisions');
    assert.deepEqual(JSON.parse(write.options.body), { event_option_ids: ['four', 'four-five'], name: 'Mixed doubles / 4.0 / 4.5', expected_state_fingerprint: 'reviewed-before-combine', confirmation_text: 'COMBINE DIVISIONS', source: 'next_tournament_ops_combine_divisions' });
    assert.equal(requests.filter(row => row.options.method === 'POST').length, 1);
    assert.match(text(renderer.root), /Combined registration divisions: Mixed doubles \/ 4.0 \+ Mixed doubles \/ 4.5/);
    assert.match(text(renderer.root), /Imports confirmed registration teams from all included divisions/);
    const importCard = renderer.root.findAllByType('article').find(node => node.findAllByType('h2').some(heading => text(heading) === 'Import confirmed registrations'));
    assert.equal(importCard.findByType('select').props.value, 'APPEND', 'merging preserves manual roster entries by default');
  } finally {
    if (renderer) await act(async () => renderer.unmount());
    global.fetch = priorFetch;
  }
}

(async () => {
  await testPanel();
  await testOpsIntegration();
  console.log('Combine divisions: selection, guards, review, request scope, roster-preserving import, and refreshed sources passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
