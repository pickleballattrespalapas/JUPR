const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const routes = [], router = { push(url) { routes.push(url); }, refresh() {} };
const stubs = {
  "@/components/PublicClubLink": ({ children, ...props }) => React.createElement("a", props, children),
  "next/link": ({ children }) => React.createElement("a", null, children),
  "@/lib/useAdminSession": { useAdminSession: () => ({ accessToken: "test-admin" }) },
  "next/navigation": { useRouter: () => router },
  "@/components/ConfirmAction": { ConfirmAction: ({ triggerLabel }) => React.createElement("button", null, triggerLabel) },
  "@/components/interaction": { actionSuccess: (title, message) => ({ status: "success", title, message }) }
};
function load(name, parent = root) {
  if (name === "@/components/PublicControls") return require("./helpers/public-controls-module.cjs")(name);
  if (Object.hasOwn(stubs, name)) return stubs[name];
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts"].find(p => fs.existsSync(p) && fs.statSync(p).isFile());
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
const PublicWorkspace = load("@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace").default;
const AdminWorkspace = load("@/app/admin/play-generators/GeneratorWorkspace").default;
const Table = load("@/components/PlayGeneratorStandingsTable").default;
const Pool = load("@/components/GeneratorMoneyball").default;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const button = (tree,label) => tree.root.findAllByType("button").find(node => text(node) === label);
const reply = body => ({ok:true,status:200,json:async()=>body});
(async () => {
  for (const Workspace of [PublicWorkspace,AdminWorkspace]) {
    for (const playerCount of [8,6]) {
    const storage = new Map();
    global.sessionStorage = {getItem:key=>storage.get(key)||null,setItem:(key,val)=>storage.set(key,val),removeItem:key=>storage.delete(key)};
    global.window = {sessionStorage};
    const players = Array.from({length:playerCount},(_,i)=>({id:i+1,name:`Player ${i}`}));
    const calls = [];
    global.fetch = async (url,options={}) => {
      if (url.includes("/players?")) return reply({players});
      if (!options.body) return reply({sessions:[]});
      calls.push(JSON.parse(options.body));
      return reply({preview:{generatorKind:"round_robin",playFormat:"doubles",eventFormat:"moneyball_expectations",name:"Moneyball",totalRounds:7,courtCount:2,participants:[],rounds:[],previewFingerprint:"frozen"}});
    };
    let tree;
    await act(async()=>{tree=create(React.createElement(Workspace,{apiBase:"https://api.test",clubId:"club",generatorKind:"round_robin",status:{enabled:true,writes_enabled:true}}));});
    await act(async()=>tree.root.findByProps({"aria-label":"Event format"}).props.onChange({target:{value:"moneyball_expectations"}}));
    const count=tree.root.findAllByType("label").find(node=>text(node).startsWith("Number of players")).findByType("select");
    assert.equal(count.props.value,8); assert.ok(!count.props.disabled);
    const counts=count.findAllByType("option").map(node=>Number(node.props.value));
    assert.deepEqual(counts,Array.from({length:37},(_,i)=>i+4).filter(n=>n===6||n%4===0||n%4===1));
    assert.ok(!JSON.stringify(tree.toJSON()).includes("Standings ranked by"));
    assert.ok(!JSON.stringify(tree.toJSON()).includes("You can keep adding rounds"));
    assert.ok(JSON.stringify(tree.toJSON()).includes("The starting lineup and scheduled slots stay fixed"));
    if (playerCount === 6) {
      await act(async()=>count.props.onChange({target:{value:"6"}}));
      const body=text(tree.toJSON());
      assert.ok(body.includes("4 games with 4 different partners and 2 byes each"));
      assert.ok(body.includes("no consecutive byes") && body.includes("not a perfect round robin"));
      assert.ok(!body.includes("Everyone partners once and faces each opponent twice"));
      assert.ok(body.includes("600-peso pool"));
    }
    for (const player of players) {
      await act(async()=>tree.root.findByProps({placeholder:"Type at least 2 letters, then add a player"}).props.onChange({target:{value:player.name}}));
      await act(async()=>button(tree,"Add").props.onClick());
    }
    await act(async()=>button(tree,"Preview matchups").props.onClick());
    assert.equal(calls.at(-1).event_format,"moneyball_expectations");
    assert.equal(calls.at(-1).total_rounds,playerCount === 6 ? 6 : 7);
    assert.equal(calls.at(-1).court_count,0); // Server resolves the fixed court count. assert.equal(calls.at(-1).scoring_mode,"scored");
    assert.ok(!Object.hasOwn(calls.at(-1),"ratings"));
    assert.ok([...storage.values()].some(value=>value.includes('"eventFormat":"moneyball_expectations"')));
    await act(async()=>tree.unmount());
    }
  }
  const rows=[{participantId:"p-1",name:"Example",rank:1,matches:7,wins:2,losses:5,differential:-11,expectedWins:2.63,expectedDifferential:-11.88,moneyballScore:54.246,moneyballFinal:true}];
  let tree;
  await act(async()=>{tree=create(React.createElement(Table,{rows,sortMode:"moneyball_expectations"}));});
  const body=JSON.stringify(tree.toJSON());
  assert.ok(body.includes("54.25") && body.includes("Expected wins") && body.includes("Expected point diff."));
  assert.ok(!body.includes("Prize") && !body.includes("starting order"));
  await act(async()=>{tree=create(React.createElement(Pool,{value:{entryFeeMxn:100,prizePoolMxn:800,completedGames:14,totalGames:14,final:true,winners:[{participantId:"p-1",name:"Example",payoutMxn:800}]}}));});
  assert.ok(JSON.stringify(tree.toJSON()).includes("Example: 800.00 pesos"));
  console.log("Moneyball: public/staff format selection, perfect player counts, saved draft, request payload, hidden ratings, expectations table and winner settlement passed.");
})().catch(error=>{console.error(error);process.exitCode=1;});
