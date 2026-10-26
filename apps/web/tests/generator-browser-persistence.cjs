const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");
const root = path.resolve(__dirname, ".."), cache = new Map();
const routes = [], router = { push(url) { routes.push(url); }, refresh() {} };
let authenticated = true;
global.IS_REACT_ACT_ENVIRONMENT = true;
const stubs = {
  "@/components/PublicClubLink": ({ children, ...props }) => React.createElement("a", props, children),
  "next/link": ({ children }) => React.createElement("a", null, children),
  "@/lib/useAdminSession": { useAdminSession: () => authenticated ? ({ accessToken: "test-admin", session: {capabilities: {assignments: [{club_id: "club", permissions: ["manage_players"]}]}} }) : ({accessToken: "", session: null}) },
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

const storage = () => {
  const values = new Map();
  return { getItem: k => values.get(k) ?? null, setItem: (k,v) => values.set(k,v), removeItem: k => values.delete(k), clear: () => values.clear() };
};
global.localStorage = storage();
global.sessionStorage = storage();
global.window = { localStorage, sessionStorage };
const store = load('@/lib/generatorBrowserStorage');
const drafts = load('@/lib/playGeneratorDraft');
const starts = load('@/lib/adminGeneratorStart');
const scores = load('@/lib/generatorScoreDraft');
const newTab = () => { global.sessionStorage = storage(); window.sessionStorage = sessionStorage; };
const tokenKey = 'public-generator-edit:club:session';
sessionStorage.setItem(tokenKey, 'old-organizer');
assert.equal(store.readGeneratorEditToken('club', 'session'), 'old-organizer');
assert.equal(localStorage.getItem(tokenKey), 'old-organizer', 'Existing tab-only owner token is migrated');
newTab();
assert.equal(store.readGeneratorEditToken('club', 'session'), 'old-organizer', 'Closing the tab must not lose editing access');
assert.equal(store.readGeneratorEditToken('other-club', 'session'), '');
assert.equal(store.readGeneratorEditToken('club', 'other-session'), '');
assert.equal(store.readGeneratorEditToken('club', 'session', true), '', 'Owner access never grants shared scorekeeper access');
store.rememberGeneratorEditToken('club', 'shared', 'scorekeeper', true);
newTab();
assert.equal(store.readGeneratorEditToken('club', 'shared', true), '', 'Shared scorekeeper access remains temporary');
const draftKey = 'public-play-generator-draft:club:round_robin';
const draft = {title:'Morning RR', playFormat:'doubles',targetCount:4,participantText:'A\nB\nC\nD',linkedPlayerIds:{},preview:{rounds:[{number:1}]}};
drafts.writePlayGeneratorDraft(draftKey, draft);
newTab();
assert.equal(drafts.readPlayGeneratorDraft(draftKey).title, draft.title);
assert.deepEqual(drafts.readPlayGeneratorDraft(draftKey).preview, draft.preview);
drafts.clearPlayGeneratorDraft(draftKey); newTab();
assert.equal(drafts.readPlayGeneratorDraft(draftKey), null);
const start = {status:'pending',hubId:null,body:{idempotency_key:'same-start',title:'Morning RR'}};
assert.equal(starts.writeGeneratorStart('start', start), true);
newTab();
assert.deepEqual(starts.readGeneratorStart('start').body, start.body, 'Interrupted start retains exact idempotency key across tabs');
const pending = {context:'public:club:session:1',path:'/scores',body:JSON.stringify({idempotency_key:'same-save',scores:[{score_a:11,score_b:7}]})};
scores.rememberScoreSave(pending.context, pending); newTab();
assert.deepEqual(scores.storedScoreSave(pending.context), pending);
const newer = {...pending,body:JSON.stringify({idempotency_key:'newer-save',scores:[]})};
scores.rememberScoreSave(pending.context, newer);
scores.rememberScoreSave(pending.context, null, pending);
assert.deepEqual(scores.storedScoreSave(pending.context), newer, 'Late response never deletes newer pending save');
scores.rememberScoreSave(pending.context, null, newer); newTab();
assert.equal(scores.storedScoreSave(pending.context), null);
scores.rememberScoreSave("shared-context", {...pending,context:"shared-context"}, undefined, false);
newTab();
assert.equal(scores.storedScoreSave("shared-context", false), null, "Shared pending saves must not persist scorekeeper credentials");
store.forgetGeneratorEditToken('club', 'session'); newTab();
assert.equal(store.readGeneratorEditToken('club', 'session'), '');
const durable = localStorage;
global.localStorage = {getItem(){throw Error('blocked')},setItem(){throw Error('quota')},removeItem(){throw Error('blocked')}};
assert.equal(store.rememberGeneratorEditToken('club','fallback','token'), false, 'Do not claim persistence when only tab storage works');
assert.equal(store.readGeneratorEditToken('club','fallback'), 'token');
assert.equal(starts.writeGeneratorStart('fallback-start',start), true);
assert.equal(starts.writeGeneratorStart('fallback-start',null), true);
global.sessionStorage = {getItem(){throw Error('blocked')},setItem(){throw Error('blocked')},removeItem(){throw Error('blocked')}};
assert.equal(store.readGeneratorEditToken('club','missing'), '');
assert.equal(starts.writeGeneratorStart('blocked',start), false);
global.localStorage = durable; newTab();
localStorage.clear();
localStorage.setItem(tokenKey, 'organizer');
localStorage.setItem('public-generator-edit:club:completed', 'organizer-completed');
const Resume = load('@/components/GeneratorResumeSessions').default;
const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join('');
const props = {apiBase:'http://local.test',clubId:'club',generatorKind:'round_robin'};
(async () => {
  global.fetch = async () => ({ok:true,json:async()=>({sessions:[
    {session_key:'completed',title:'Finished RR',status:'completed',generator_kind:'round_robin',current_round_number:4},
    {session_key:'session',title:'Morning RR',status:'active',generator_kind:'round_robin',current_round_number:3},
    {session_key:'another',title:'Someone else',status:'active',generator_kind:'round_robin',current_round_number:1}
  ]})});
  let tree;
  await act(async()=> {tree=create(React.createElement(Resume,props));});
  const links = tree.root.findAllByType('a');
  assert.equal(links.length,2, 'Show only sessions this browser can manage');
  assert.equal(text(links[0]), 'Resume session');
  assert.equal(links[0].props.href, '/clubs/club/round-robin-generator/sessions/session/rounds/3');
  assert(!JSON.stringify(tree.toJSON()).includes('organizer'), 'Never render bearer tokens in the list');
  await act(async()=> tree.update(React.createElement(Resume,{...props,clubId:'other-club'})));
  assert.equal(tree.toJSON(),null, 'Changing clubs must not show the old club sessions');
  await act(async()=> tree.unmount());
  global.fetch = async () => { throw new Error('offline'); };
  await act(async()=> {tree=create(React.createElement(Resume,props));});
  assert.match(text(tree.toJSON()), /couldn’t load your saved sessions/, 'List failure is not silently reported as no sessions');
  await act(async()=> tree.unmount());
  console.log('PASS generator persistence: new tab, legacy migration, draft/start/save recovery, storage failures, token isolation, and resume navigation');
})().catch(error=>{console.error(error);process.exitCode=1;});
