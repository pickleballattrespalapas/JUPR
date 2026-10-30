const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const root = path.join(__dirname, '..');
const modules = new Map();
function compile(file) {
  const full = path.resolve(file);
  if (modules.has(full)) return modules.get(full).exports;
  const compiled = {exports: {}}; modules.set(full, compiled);
  const code = ts.transpileModule(fs.readFileSync(full, 'utf8'), {compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true}}).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    if (name.endsWith('.module.css')) return new Proxy({}, {get: (_, key) => String(key)});
    if (name === '@/components/SearchablePlayerSelect') return function Select({onValueChange, ...props}) { return React.createElement('select', {...props, onChange: event => onValueChange(event.target.value)}); };
    if (name.startsWith('@/')) {
      const target = path.join(root, name.slice(2));
      return compile(fs.existsSync(target + '.tsx') ? target + '.tsx' : target + '.ts');
    }
    return require(name);
  }, compiled, compiled.exports);
  return compiled.exports;
}
const parser = compile(path.join(root, 'lib/courtReserveRosterImport.ts'));
const Component = compile(path.join(root, 'components/CourtReserveRosterImport.tsx')).default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const button = (tree, value) => tree.root.findAllByType('button').find(node => text(node) === value);
const players = [{id: 601, name: 'Shannon Lovell'}, {id: 120, name: 'Suzanne Johnston'}, {id: 4, name: 'José Díaz'}, {id: 5, name: 'Jamie Same'}, {id: 6, name: 'Jamie Same (P)'}];
const raw = '\ufeffEnter Your Skill Level,Full Name\r\n3.5,Shannon Lovell\r\n3.0,Suzanne Johnston (P)\r\n3.75–4.0,Jose Diaz\r\n';
const rows = parser.parseCourtReserveReport(raw);
assert.deepEqual(rows.map(row => row.name), ['Shannon Lovell', 'Suzanne Johnston', 'Jose Diaz']);
assert.equal(rows[2].requestedRating, '3.75–4.0');
assert.deepEqual(parser.matchCourtReserveRows(rows, players).map(row => row.playerId), [601, 120, 4]);
assert.equal(parser.parseCourtReserveReport('Full Name;Skill Level\nPlayer One;3.5')[0].requestedRating, '3.5');
assert.equal(parser.parseCourtReserveReport('Name\tRating\n"Player,\nTwo"\t3.25')[0].name, 'Player, Two');
assert.equal(parser.parseCourtReserveReport('Full Name,Skill Level\n"Sam ""Ace"" Jones",3.5')[0].name, 'Sam "Ace" Jones');
assert.throws(() => parser.parseCourtReserveReport('Full Name,Skill Level\n"No end,3.5'), /quoted/);
assert.throws(() => parser.parseCourtReserveReport('Shannon Lovell,3.5'), /headings/);
assert.throws(() => parser.parseCourtReserveReport('Full Name,Skill Level\n,3.5'), /no player name/);
assert.throws(() => parser.parseCourtReserveReport('Full Name,Skill Level\n' + Array.from({length:501}, (_, i) => `Player ${i},3.0`).join('\n')), /500/);
assert.equal(parser.matchCourtReserveRows([{rowNumber:2,name:'Jamie Same',requestedRating:'3'}], players)[0].matchStatus, 'ambiguous');
assert.equal(parser.matchCourtReserveRows([{rowNumber:2,name:'Shannon Love',requestedRating:'3'}], players)[0].matchStatus, 'unmatched', 'No fuzzy profile selection');
assert.equal(parser.cleanCourtReserveName('Sarah Jones (Guest)'), 'Sarah Jones (Guest)', 'Only known P suffix is removed');
assert.throws(() => parser.buildCourtReserveSelection(rows, {2:601,3:601,4:4}), /more than once/);
assert.deepEqual(parser.buildCourtReserveSelection(rows, {2:601,3:601,4:4},new Set([3])), {playerIds:[601,4],requestedRatings:{'601':'3.5','4':'3.75–4.0'}});
assert.throws(() => parser.buildCourtReserveSelection(rows, {2:601,3:null,4:4}), /Choose a PCS profile/);

(async () => {
  let imported = [], requests = [], tree;
  global.fetch = async (url, options) => {
    const body = JSON.parse(options.body); requests.push({url,body});
    return {ok:true,json:async () => ({players:body.player_ids.map(id => ({player_id:id,name:players.find(p=>p.id===id).name,rating_jupr:3.4,starting_jupr:3,rated_matches:12,has_rated_history:true,review_required:false,requested_rating:body.requested_ratings[id],requested_jupr:null,requested_rating_differs:false,state_fingerprint:'a',confirmed_at:null}))})};
  };
  await act(async () => {tree=create(React.createElement(Component,{apiBase:'https://api.test',clubId:'club',accessToken:'token',players,onImport:value=>imported.push(value)}));});
  await act(async () => tree.root.findByProps({'aria-label':'CourtReserve report cells'}).props.onChange({target:{value:raw}}));
  await act(async () => button(tree,'Preview report').props.onClick());
  assert.equal(requests[0].url,'https://api.test/admin/clubs/club/starting-rating-review/preview');
  assert.deepEqual(requests[0].body.player_ids,[4,120,601]);
  assert.equal(button(tree,'Use these players (3)').props.disabled,false);
  assert.equal(tree.root.findAllByType('input').filter(node=>node.props.type==='number').length,0,'Established imports have no rating edit controls');
  await act(async () => button(tree,'Use these players (3)').props.onClick());
  assert.deepEqual(imported[0],{playerIds:[601,120,4],requestedRatings:{'601':'3.5','120':'3.0','4':'3.75–4.0'}});
  // Editing a matched identity immediately invalidates previous authoritative readiness.
  await act(async () => tree.root.findByProps({'aria-label':'PCS profile for report row 3: Suzanne Johnston'}).props.onValueChange('601'));
  assert.equal(button(tree,'Use these players').props.disabled,true);
  assert.match(text(tree.toJSON()),/more than once/);
  await act(async () => tree.unmount());

  requests=[]; imported=[];
  await act(async () => {tree=create(React.createElement(Component,{apiBase:'https://api.test',clubId:'club',accessToken:'token',players,requireRatingReview:false,onImport:value=>imported.push(value)}));});
  await act(async () => tree.root.findByProps({'aria-label':'CourtReserve report cells'}).props.onChange({target:{value:'Full Name\tSkill Level\nShannon Lovell\t3.5\nUnknown Name\t3.0'}}));
  await act(async () => button(tree,'Preview report').props.onClick());
  assert.equal(button(tree,'Use these players').props.disabled,true,'Unmatched import cannot silently create a profile');
  await act(async () => tree.root.findByProps({'aria-label':'Include report row 3: Unknown Name'}).props.onChange({target:{checked:false}}));
  assert.equal(button(tree,'Use these players (1)').props.disabled,false);
  await act(async () => button(tree,'Use these players (1)').props.onClick());
  assert.equal(requests.length,0,'Unrated roster import does not require rating confirmation');
  assert.deepEqual(imported[0].playerIds,[601]);
  await act(async () => tree.unmount());
  console.log('PASS CourtReserve import: CSV/TSV quotes, ranges, accents, P suffix, identity collisions, duplicates, explicit omissions, rated-history preservation, unrated roster.');
})().catch(error=>{console.error(error);process.exitCode=1;});
