const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), {act, create} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const h = React.createElement;
const source = ts.transpileModule(fs.readFileSync(path.join(__dirname,'../app/admin/round-robin-hub/HubEditPlayersDialog.tsx'),'utf8'),{
  compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}
}).outputText;
function Dialog({title,children,description,actions,onRequestClose}) {return h('section',{'data-dialog':title,onRequestClose},description,children,actions);}
function Search(props) {return h('input',props);}
const compiled={exports:{}};
new Function('require','module','exports',source)(name=>{
  if(name==='@/components/interaction') return {InteractionDialog:Dialog};
  if(name==='@/components/PlayerSearchInput') return Search;
  if(name==='@/lib/playerSearch') return {fetchAllPublicPlayers:async()=>[{id:42,name:'Club Profile'}]};
  if(name==='@/lib/adminRoundRobinHubApi') return {hubPlayerLabel:p=>p.display_name||p.name};
  if(name.endsWith('.module.css')) return {};
  return require(name);
},compiled,compiled.exports);
const Component=compiled.exports.default;
const text=node=>typeof node==='string'?node:(node?.children||[]).map(text).join('');
const player=(id,extra={})=>({participant_id:id,name:id,identity:id,active:true,active_from_round:1,...extra});
const match={session_key:'rr',match_id:'game',round_number:2,side_a:[{id:'a',name:'a'},{id:'b',name:'b'}],side_b:[{id:'c',name:'c'},{id:'d',name:'d'}]};
(async()=>{
  let tree,closed=0,retried=0,reviewed=0;const calls=[];
  let props={apiBase:'http://localhost',clubId:'club',match,
    group:{title:'Level 3.5',current_round_number:2,can_add_player:true,players:[...['a','b','c','d'].map(id=>player(id,{on_court:true})),player('bye',{on_bye:true}),
      player('busy',{on_court:true}),player('reserved',{reserved:true}),player('paused',{paused:true}),player('left',{active:false,inactive_from_round:2})]},
    stale:false,available:true,busy:false,disabled:false,pending:false,error:'',onSubmit:p=>calls.push(p),onClose:()=>closed++,onRetry:()=>retried++,onReview:()=>reviewed++};
  const update=async patch=>{props={...props,...patch};await act(async()=>tree.update(h(Component,props)));};
  const select=label=>tree.root.findByProps({'aria-label':label});
  const choose=async(label,value)=>act(async()=>select(label).props.onChange({target:{value}}));
  const button=label=>tree.root.findAllByType('button').find(n=>text(n)===label);
  const submit=()=>tree.root.findByType('form').props.onSubmit({preventDefault(){}});
  await act(async()=>{tree=create(h(Component,props));});
  assert(button('Save players').props.disabled);
  await choose('Player to replace or swap','a');
  const replacement=select('Replacement');
  for(const id of ['a','busy','reserved','paused','left']) assert(replacement.findByProps({value:id}).props.disabled);
  assert(!replacement.findByProps({value:'b'}).props.disabled,'Other slots in this match can swap');
  await choose('Replacement','bye');
  await act(async()=>submit());
  assert.deepEqual(calls,[{participant_id:'a',replacement_participant_id:'bye'}]);
  await update({stale:true});
  assert(button('Save players').props.disabled);
  await act(async()=>{submit();button('Reload current players').props.onClick();});
  assert.equal(calls.length,1);assert.equal(reviewed,1);
  await update({stale:false});
  await choose('Replacement','new');
  await act(async()=>select('New player name or club profile').props.onTextChange('Guest'));
  await update({pending:true,error:'Response not confirmed'});
  assert(button('Cancel').props.disabled && button('Save players').props.disabled);
  await act(async()=>{submit();tree.root.findByType(Dialog).props.onRequestClose();button('Retry saved action').props.onClick();});
  assert.equal(calls.length,1);assert.equal(closed,0);assert.equal(retried,1);
  await update({pending:false,error:''});
  await act(async()=>select('New player name or club profile').props.onPick({value:'42',label:'Club Profile'}));
  await act(async()=>submit());
  assert.deepEqual(calls[1],{participant_id:'a',name:'Club Profile',player_id:42});
  await act(async()=>select('New player name or club profile').props.onTextChange('bye'));
  assert(button('Save players').props.disabled);
  assert.match(text(tree.toJSON()),/already on the roster/);
  await update({available:false});
  assert(button('Save players').props.disabled);
  await act(async()=>submit());
  assert.equal(calls.length,2);
  await act(async()=>tree.unmount());
  console.log('PASS court player editor: explicit slot, safe choices, guest/profile input, stale review, and frozen exact retry.');
})().catch(error=>{console.error(error);process.exitCode=1;});
