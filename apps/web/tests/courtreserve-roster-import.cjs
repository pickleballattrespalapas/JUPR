const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { create, act } = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const storage = new Map();
global.sessionStorage = {getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)};
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
assert.equal(parser.courtReserveStartingLevel(' 3.5 '),'3.5');
for (const value of ['', '3.5/4.0', '3.75–4.0', '3.5+', '0', '7.5', 'NaN', 'Infinity']) assert.equal(parser.courtReserveStartingLevel(value),'');

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

  // Exercise the real creation component, guarded endpoint contract and import together.
  const report = 'Full Name\tSkill Level\nShannon Lovell\t3.5\nJavier Gonzalez-Bringas\t3.25\nSkip Me\t3.0';
  for (const scenario of ['success','rated','duplicate','lost','detail-lost','denied']) {
    storage.clear(); imported=[]; requests=[];
    let created=null, savedOperation=null, detailFailed=false, confirmed=false;
    const reply = payload=>({ok:true,status:200,json:async()=>structuredClone(payload)});
    global.fetch = async (url, options={}) => {
      const body=options.body ? JSON.parse(options.body) : null;
      requests.push({url,body,method:options.method});
      assert.equal(options.headers.Authorization,'Bearer token');
      if (url.endsWith('/players/editor/players') && options.method==='POST') {
        assert.equal(body.name,'Javier Gonzalez-Bringas'); assert.equal(body.starting_jupr,3.25);
        assert.equal(body.source,'courtreserve_roster_import');
        if (scenario==='denied') return {ok:false,status:403,json:async()=>({detail:'Player creation is not permitted.'})};
        created={id:900,name:body.name,club_id:'club',rating:1300,rating_jupr:3.25};
        savedOperation={key:body.idempotency_key,player:created};
        if (scenario==='lost') throw new TypeError('Response lost after save');
        return reply({player:created});
      }
      if (url.endsWith('/players/editor/players')) {
        if (scenario==='duplicate') created={id:900,name:'Javier Gonzalez-Bringas',rating:1600,rating_jupr:4};
        return reply({players:[...players,...(created?[created]:[])]});
      }
      if (url.includes('/players/editor/operations/')) {
        assert.equal(decodeURIComponent(url.split('/operations/')[1]),savedOperation.key);
        return reply({status:'completed',result:{player:savedOperation.player}});
      }
      if (url.endsWith('/players/editor/players/900')) {
        if (scenario==='detail-lost' && !detailFailed) { detailFailed=true; throw new TypeError('Detail response lost'); }
        return reply({player:created});
      }
      const reviews=()=>body.player_ids.map(id=>({player_id:id,name:id===900?created.name:players.find(p=>p.id===id).name,
        rating_jupr:id===900 ? (confirmed?3.75:3.25) : 4.2,starting_jupr:id===900 ? (confirmed?3.75:3.25) : 4.2,
        has_rated_history:id!==900,rated_matches:id===900?0:20,review_required:id===900&&!confirmed,
        requested_rating:body.requested_ratings[id],requested_jupr:id===900?3.25:3.5,requested_rating_differs:false,state_fingerprint:'current'}));
      if (url.endsWith('/starting-rating-review/preview')) return reply({players:reviews()});
      if (url.endsWith('/starting-rating-review/confirm')) {
        assert.deepEqual(body.reviews.map(row=>[row.player_id,row.rating_jupr]),[[900,3.75]],'Established ratings are never changed');
        confirmed=true; return reply({players:[]});
      }
      throw new Error(`Unexpected request ${url}`);
    };
    const props={apiBase:'https://api.test',clubId:'club',accessToken:'token',players,requireRatingReview:scenario==='rated',onImport:value=>imported.push(value),onCancel:()=>{}};
    async function mountReport() {
      await act(async()=>{tree=create(React.createElement(Component,props));});
      await act(async()=>tree.root.findByProps({'aria-label':'CourtReserve report cells'}).props.onChange({target:{value:report}}));
      await act(async()=>button(tree,'Preview report').props.onClick());
      if (!tree.root.findByProps({'aria-label':'Include report row 4: Skip Me'}).props.disabled) await act(async()=>tree.root.findByProps({'aria-label':'Include report row 4: Skip Me'}).props.onChange({target:{checked:false}}));
    }
    const submitCreation=async()=>act(async()=>tree.root.findByProps({'aria-label':'New club player for Javier Gonzalez-Bringas'}).props.onSubmit({preventDefault(){}}));
    await mountReport();
    await act(async()=>tree.root.findByProps({'aria-label':'Create new player for Javier Gonzalez-Bringas'}).props.onClick());
    assert.equal(tree.root.findByProps({'aria-label':'New player name for Javier Gonzalez-Bringas'}).props.value,'Javier Gonzalez-Bringas');
    assert.equal(tree.root.findByProps({'aria-label':'New player starting level for Javier Gonzalez-Bringas'}).props.value,'3.25');
    await submitCreation();
    if (scenario==='duplicate') {
      assert.equal(requests.filter(r=>r.method==='POST').length,0,'Fresh matching profile is offered instead of duplicated');
      await act(async()=>button(tree,'Use Javier Gonzalez-Bringas').props.onClick());
    }
    if (scenario==='lost' || scenario==='detail-lost') {
      assert.equal(button(tree,'Cancel import').props.disabled,true);
      assert.equal(tree.root.findByProps({'aria-label':'CourtReserve report cells'}).props.disabled,true);
      assert.equal(tree.root.findByProps({'aria-label':'Include report row 3: Javier Gonzalez-Bringas'}).props.disabled,true);
      await act(async()=>tree.unmount()); await mountReport();
      assert(button(tree,scenario==='lost'?'Check player creation':'Use created player'),'Creation recovery survives reimport');
      await submitCreation();
      await act(async()=>tree.root.findByProps({'aria-label':'Include report row 4: Skip Me'}).props.onChange({target:{checked:false}}));
    }
    if (scenario==='denied') {
      assert.match(text(tree.toJSON()),/not permitted/);
      assert.equal(button(tree,'Use these players').props.disabled,true);
      assert.equal(storage.size,0,'Definitive rejection clears pending operation');
    } else {
      assert.equal(tree.root.findByProps({'aria-label':'PCS profile for report row 2: Shannon Lovell'}).props.value,'601','Other profile choices survive creation');
      assert.equal(tree.root.findByProps({'aria-label':'Include report row 4: Skip Me'}).props.checked,false,'Skipped rows stay excluded');
      assert.equal(tree.root.findByProps({'aria-label':'PCS profile for report row 3: Javier Gonzalez-Bringas'}).props.value,'900','New profile is selected immediately');
      if (scenario==='rated') {
        assert.equal(button(tree,'Use these players (2)').props.disabled,true,'New profile still needs normal rating review');
        await act(async()=>tree.root.findByProps({'aria-label':'Javier Gonzalez-Bringas doubles starting rating'}).props.onChange({target:{value:'3.75'}}));
        const confirm=tree.root.findAllByType('button').find(node=>text(node).startsWith('Confirm starting'));
        assert(confirm); await act(async()=>confirm.props.onClick());
      }
      assert.equal(button(tree,'Use these players (2)').props.disabled,false);
      await act(async()=>button(tree,'Use these players (2)').props.onClick());
      assert.deepEqual(imported[0].playerIds,[601,900]);
      assert.equal(imported[0].additionalPlayers[0].id,900,'Parent roster receives the new profile without reloading');
      assert.equal(imported[0].additionalPlayers[0].rating,scenario==='rated'?1500:scenario==='duplicate'?1600:1300);
      assert.equal(requests.filter(r=>r.url.endsWith('/players/editor/players')&&r.method==='POST').length,scenario==='duplicate'?0:1,'Recovery never repeats player insertion');
      assert.equal(storage.size,0);
    }
    await act(async()=>tree.unmount());
  }
  console.log('PASS inline CourtReserve creation: prefilled name/estimate, profile selection, existing-player reuse, ratings review, preserved choices, denied writes and reload recovery without duplicates.');
  console.log('PASS CourtReserve import: CSV/TSV quotes, ranges, accents, P suffix, identity collisions, duplicates, explicit omissions, rated-history preservation, unrated roster.');
})().catch(error=>{console.error(error);process.exitCode=1;});
