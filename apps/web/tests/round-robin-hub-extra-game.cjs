const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const React = require('react'), {act,create} = require('react-test-renderer');
global.IS_REACT_ACT_ENVIRONMENT = true;
const h = React.createElement;
const source = ts.transpileModule(fs.readFileSync(path.join(__dirname,'../app/admin/round-robin-hub/HubExtraGameDialog.tsx'),'utf8'),{
  compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}
}).outputText;
function Dialog({title,children,description,actions,onRequestClose}) {return h('section',{'data-dialog':title,onRequestClose},description,children,actions);}
const compiled={exports:{}};
new Function('require','module','exports',source)(name=>{
  if(name==='@/components/interaction')return {InteractionDialog:Dialog};
  if(name.endsWith('.module.css'))return new Proxy({},{get:(_target,key)=>key});
  return require(name);
},compiled,compiled.exports);
const Extra=compiled.exports.default;
const player=(id,identity=id)=>({participant_id:id,name:id,identity,active:true,paused:false,on_court:false,reserved:false});
const groups=[{session_key:'low',title:'Lower',status:'active',event_format:'standard',players:[player('Yaki')]},
  {session_key:'high',title:'Higher',status:'active',event_format:'standard',players:[player('Gaby'),player('Stephen'),player('Marianne'),player('Alias Yaki','Yaki')]}];
(async()=>{
  let tree, closed=0,retried=0;const calls=[];
  let props={groups,busy:false,disabled:false,pending:false,error:'',onClose:()=>closed++,onRetry:()=>retried++,onSubmit:value=>calls.push(value)};
  const render=async patch=>{props={...props,...patch};await act(async()=>{if(tree)tree.update(h(Extra,props));else tree=create(h(Extra,props));});};
  const text=node=>typeof node==='string'?node:Array.isArray(node)?node.map(text).join(' '):node?text(node.children):'';
  const button=label=>tree.root.findAllByType('button').find(node=>text(node.props.children)===label);
  const select=index=>tree.root.findAllByType('select')[index];
  const choose=async(index,group,player)=>act(async()=>select(index).props.onChange({target:{value:JSON.stringify([group,player])}}));
  const submit=()=>tree.root.findByType('form').props.onSubmit({preventDefault(){}});
  await render({});
  assert.match(text(tree.toJSON()),/Exhibition only/);
  assert(button('Create extra game').props.disabled);
  await choose(0,'low','Yaki');
  const alias=select(1).findAllByType('option').find(node=>node.props.value===JSON.stringify(['high','Alias Yaki']));
  assert(alias.props.disabled,'Same person in another division cannot occupy a second slot');
  await choose(1,'high','Gaby');await choose(2,'high','Stephen');await choose(3,'high','Marianne');
  assert.equal(button('Create extra game').props.disabled,false);
  await act(async()=>submit());
  assert.deepEqual(calls,[{side_a:[{session_key:'low',participant_id:'Yaki'},{session_key:'high',participant_id:'Gaby'}],side_b:[{session_key:'high',participant_id:'Stephen'},{session_key:'high',participant_id:'Marianne'}],standings_policy:'exhibition'}]);
  await render({pending:true,error:'Result not confirmed'});
  assert(button('Create extra game').props.disabled && button('Cancel').props.disabled);
  await act(async()=>{submit();tree.root.findByType(Dialog).props.onRequestClose();button('Retry saved action').props.onClick();});
  assert.equal(calls.length,1);assert.equal(closed,0);assert.equal(retried,1);
  const changed=structuredClone(groups);changed[1].players[0].on_court=true;
  await render({pending:false,error:'',groups:changed});
  assert(button('Create extra game').props.disabled);
  assert.match(text(tree.toJSON()),/Finish or release/);
  await act(async()=>submit());assert.equal(calls.length,1);
  await act(async()=>tree.unmount());
  console.log('PASS extra game: explicit exhibition policy, exact teams, identity protection, stale availability, and frozen recovery.');
})().catch(error=>{console.error(error);process.exitCode=1;});
