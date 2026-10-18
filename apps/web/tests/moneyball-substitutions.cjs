const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map(), h = React.createElement;
const navigation = [];
const router = { push: url => navigation.push(url), replace() {}, refresh() {} };
const Link = ({ children, ...props }) => h("a", props, children);
const stubs = {
  "next/link": Link,
  "@/components/PublicClubLink": Link,
  "next/navigation": { useRouter: () => router },
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel, description, onConfirm, disabled }) => h("button", { onClick: onConfirm, disabled, "data-description": description }, triggerLabel) },
  "@/components/GeneratorSharing": () => null,
  "@/components/GeneratorSubmission": { __esModule: true, default: () => null, generatorResultLabel: () => "Unrated" },
  "@/components/SearchablePlayerSelect": { __esModule: true, default: ({value,onValueChange,children,...props}) => h("select",{...props,value,onChange:event=>onValueChange(event.target.value)},children) },
};
function load(name, parent = root) {
  if (name === "@/components/PublicControls") return require("./helpers/public-controls-module.cjs")(name);
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts", path.join(base, "index.ts")].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree,label) => tree.root.findAllByType("button").find(node=>text(node)===label);
(async()=>{
  for (const file of ["@/app/admin/play-generators/GeneratorRoundRunner","@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner"]) {
    global.sessionStorage={getItem:()=>"organizer",setItem(){},removeItem(){}};
    global.window={location:{hash:"",pathname:"/test",search:""},history:{replaceState(){}}};
    const session={session_key:"s",title:"Moneyball",status:"active",version:1,generator_kind:"round_robin",play_format:"doubles",scoring_mode:"scored",rating_mode:"unrated",current_round_number:1,total_rounds:3,
      event:{eventFormat:"moneyball_expectations",generatorKind:"round_robin",playFormat:"doubles",currentRoundNumber:1,totalRounds:3,courtCount:1,status:"active",participants:Array.from({length:4},(_,i)=>({id:`p-${i+1}`,name:`Player ${i+1}`,roster_order:i+1})),rounds:[1,2,3].map(number=>({number,status:number===1?"active":"preview",byeParticipantIds:[],matches:[{id:`r${number}-c1`,court:1,sideA:["p-1","p-2"],sideB:["p-3","p-4"],scoreA:null,scoreB:null}]}))}};
    session.event.moneyball = {targetScore:15,modelNote:"Experimental 15-point extension",conversionNote:"Matchups and saved scores were preserved."};
    session.event.moneyballExpectations = session.event.participants.map(p=>({participantId:p.id,name:p.name,scheduledGames:3,expectedWins:1.5,expectedLosses:1.5,expectedDifferential:0}));
    const requests=[];
    global.fetch=async(url,options={})=>{
      if(options.body){const body=JSON.parse(options.body);requests.push(body);assert.equal(body.action,"substitute");assert.equal(body.participant_id,"p-1");assert.equal(body.name,"Replacement");assert.equal(body.substitute_scope,"rest");session.version++;}
      return {ok:true,status:200,json:async()=>({session:JSON.parse(JSON.stringify(session))})};
    };
    let tree;
    await act(async()=>{tree=create(h(load(file).default,{apiBase:"https://api.test",clubId:"club",generatorKind:"round_robin",sessionKey:"s",roundNumber:1}));});
    assert.ok(JSON.stringify(tree.toJSON()).includes("Both the original player and substitute become ineligible"));
    assert.equal(button(tree,"Skip round"),undefined);
    const expectations=tree.root.findByProps({"aria-label":"Moneyball player expectations"});
    assert.equal(expectations.findByType("tbody").findAllByType("tr").length,4);
    assert.ok(text(expectations).includes("Games to 15, win by two") && text(expectations).includes("including games already played"));
    assert.ok(text(expectations).includes("Experimental 15-point extension"));
    assert.ok(!text(expectations).includes("Expectations lock when you start"));
    assert.ok(text(tree.toJSON()).includes("Fixed Moneyball rotation: complete every scheduled game"));
    assert.ok(!text(tree.toJSON()).includes("you add extra rounds"));
    const change=tree.root.findAllByType("select").find(node=>node.findAllByType("option").some(opt=>opt.props.value==="substitute"));
    assert.deepEqual(change.findAllByType("option").map(opt=>opt.props.value),["substitute"]);
    await act(async()=>tree.root.findAllByType("select").find(node=>node.props["aria-label"]==="Current player").props.onChange({target:{value:"p-1"}}));
    const label=tree.root.findAllByType("label").find(node=>text(node).includes("New player name"));
    await act(async()=>label.findByType("input").props.onChange({target:{value:"Replacement"}}));
    const submit=()=>button(tree,"Apply substitution · both players ineligible");
    assert.equal(submit().props.disabled,false);
    const score=tree.root.findByProps({"aria-label":"r1-c1 side A score"});
    await act(async()=>score.props.onChange({target:{value:"11"}}));
    assert.equal(submit().props.disabled,true,"Protect local score drafts before changing a current Moneyball slot");
    await act(async()=>score.props.onChange({target:{value:""}}));
    await act(async()=>submit().props.onClick());
    assert.equal(requests.length,1);
    assert.ok(JSON.stringify(tree.toJSON()).includes("Both players are ineligible to win; expectations remain fixed"));
    await act(async()=>tree.unmount());
  }
  console.log("Moneyball public/staff substitutions: restricted actions, both-player DQ explanation, current-round scope, draft protection and request payload passed.");
})().catch(e=>{console.error(e);process.exitCode=1;});
