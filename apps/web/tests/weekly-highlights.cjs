const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
const { renderToStaticMarkup } = require('react-dom/server');
global.IS_REACT_ACT_ENVIRONMENT = true;
function load(file, mocks = {}) {
  const source = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => mocks[name] ? (mocks[name].default ? { __esModule: true, ...mocks[name] } : mocks[name]) : require(name), module, module.exports);
  return module.exports;
}
const Cards = load('components/WeeklyHighlightCards.tsx').default;
const Editor = load('app/admin/league-manager/results/WeeklyHighlightsEditor.tsx', { '@/components/WeeklyHighlightCards': { default: Cards } }).default;
const cards = ['Alex', 'Blair', 'Casey', 'Devon', 'Eden', 'Frank', 'Gale'].map((name, i) => ({ id: `wins:${i + 1}`, category: 'wins', title: 'Weekly wins', name, metric: `${8 - i} wins`, detail: 'Week-specific record', player_ids: [i + 1], value: 8 - i, sample: 9 }));
const data = week => ({ week_num: week, mode: 'automatic', cards: cards.slice(0, 3), automatic_cards: cards.slice(0, 3), selected_ids: [], revision: 0, max_highlights: 6, available: true, warnings: [], unavailable_ids: [], categories: [{ id: 'wins', title: 'Weekly wins', qualification: 'At least 3 wins.', candidates: cards }] });
const props = { apiBase: 'https://example.test', clubId: 'club', leagueName: 'Open', week: 2, accessToken: 'test', writesEnabled: true };
const text = node => typeof node === 'string' ? node : node?.children?.map(text).join(' ').replace(/\s+/g, ' ') || '';
let requests = [], conflict = false;
global.fetch = async (url, options = {}) => {
  requests.push({ url, ...options });
  const week = Number(url.match(/weeks\/(\d+)/)[1]);
  if (options.method === 'PUT') {
    const body = JSON.parse(options.body);
    return { ok: !conflict, json: async () => conflict ? { detail: 'Another admin changed this week. Reload before saving.' } : { ...body, week_num: week, revision: 1 } };
  }
  return { ok: true, json: async () => data(week) };
};
const findButton = (tree, label) => tree.root.findAllByType('button').find(node => text(node).includes(label));
const checkbox = (tree, index) => tree.root.findAllByType('input').filter(node => node.props.type === 'checkbox')[index];
async function click(button) { await act(async () => button.props.onClick()); }
(async () => {
  let tree;
  await act(async () => { tree = create(React.createElement(Editor, props)); });
  assert.match(text(tree.toJSON()), /Choose Week 2 highlights/);
  assert.equal(findButton(tree, 'Save Week 2').props.disabled, true);
  await act(async () => tree.root.findAllByType('input').find(node => node.props.type === 'radio').props.onChange());
  await act(async () => checkbox(tree, 3).props.onChange());
  await click(tree.root.findAllByType('button').find(node => node.props['aria-label'] === 'Move highlight 4 up'));
  await act(async () => tree.update(React.createElement(Editor, { ...props, week: 1 })));
  assert.match(text(tree.toJSON()), /Choose Week 1 highlights/);
  await act(async () => tree.update(React.createElement(Editor, props)));
  assert.equal(checkbox(tree, 3).props.checked, true, 'unsaved choices survive week switching');
  await click(findButton(tree, 'Save Week 2'));
  let sent = requests.find(request => request.method === 'PUT');
  assert.deepEqual(JSON.parse(sent.body), { mode: 'curated', selected_ids: ['wins:1','wins:2','wins:4','wins:3'], expected_revision: 0 });
  assert.match(sent.url, /clubs\/club\/league-manager\/leagues\/Open\/weeks\/2\/highlights$/);
  assert.match(text(tree.toJSON()), /Week 2 highlights saved/);
  assert.equal(findButton(tree, 'Save Week 2').props.disabled, true);
  for (let i = 4; i < 6; i++) await act(async () => checkbox(tree, i).props.onChange());
  assert.equal(checkbox(tree, 6).props.disabled, true, 'limit prevents a seventh selection');
  conflict = true;
  await click(findButton(tree, 'Save Week 2'));
  assert.match(text(tree.toJSON()), /Another admin changed this week/);
  assert.equal(checkbox(tree, 5).props.checked, true, 'conflict preserves edits');
  conflict = false;
  while (tree.root.findAllByType('button').some(node => /^Remove highlight/.test(node.props['aria-label'] || ''))) {
    await click(tree.root.findAllByType('button').find(node => /^Remove highlight/.test(node.props['aria-label'] || '')));
  }
  assert.match(text(tree.toJSON()), /hides the highlights section for Week 2/);
  await click(findButton(tree, 'Save Week 2'));
  assert.deepEqual(JSON.parse(requests.filter(request => request.method === 'PUT').at(-1).body).selected_ids, []);
  await act(async () => tree.unmount());

  const result = {
    selected_league: 'Open', selected_week: 2, past_leagues: [], weeks: [{ week_num: 2, week_label: 'Week 2', has_results: true }],
    weekly_results: [{ week_num: 2, player_id: 1, player_name: 'Ranking player', rank: 1, wins: 4, losses: 0, games: 4, rating_jupr: 3 }],
    weekly_highlight_selection: { week_num: 2, mode: 'curated', cards: [cards[1], cards[0]] }
  };
  const Page = load('app/clubs/[clubSlug]/leagues/[leagueName]/weekly-history/page.tsx', {
    '@/components/WeeklyHighlightCards': { default: Cards },
    '@/components/PublicControls': { default: ({children}) => React.createElement('details', null, children) },
    '@/components/PublicClubLink': { default: ({children, ...p}) => React.createElement('a', p, children) },
    '@/components/PublicLeagueNav': { default: () => null },
    '@/lib/api': { getClubLeagueResults: async () => ({ data: result, error: null }) }
  }).default;
  const pageProps = { params: Promise.resolve({ clubSlug: 'club', leagueName: 'Open' }), searchParams: Promise.resolve({ week: '2' }) };
  let html = renderToStaticMarkup(await Page(pageProps));
  assert.ok(html.indexOf('Weekly highlights') < html.indexOf('Ranking player'), 'curated highlights precede rankings');
  assert.ok(html.indexOf('Blair') < html.indexOf('Alex'), 'public order matches saved order');
  assert.match(html, /7 wins/);
  result.weekly_highlight_selection.cards = [];
  html = renderToStaticMarkup(await Page(pageProps));
  assert.doesNotMatch(html, /Weekly highlights/);
  assert.match(html, /Ranking player/, 'empty selection only hides highlights');
  console.log('Weekly highlights UI: selection, order, drafts per week, exact-week save, limit, conflict, empty state and public placement passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
