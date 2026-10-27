const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");

global.IS_REACT_ACT_ENVIRONMENT = true;
const frames = new Map();
let frameId = 0;
global.window = {
  requestAnimationFrame(callback) { frames.set(++frameId, callback); return frameId; },
  cancelAnimationFrame(id) { frames.delete(id); }
};
function paint() {
  const pending = [...frames.values()];
  frames.clear();
  for (const callback of pending) callback();
}
class Element {
  constructor(name) { this.name = name; this.isConnected = true; this.tabIndex = 0; this.attributes = new Map(); }
  matches(selector) { return selector === ":disabled" && Boolean(this.disabled); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); }
  addEventListener() {}
  focus() { document.activeElement = this; }
}
global.HTMLElement = Element;
const trigger = new Element("open score");
const first = new Element("Side A score");
const second = new Element("Side B score");
const outcome = new Element("saved result");
const main = new Element("main");
let phaseTarget = null;
const dialog = {
  open: false,
  showModal() { this.open = true; first.focus(); },
  close() { this.open = false; },
  querySelector(selector) { return selector === "[data-dialog-focus]" ? phaseTarget : null; },
  querySelectorAll() { return [first, second]; }
};
global.document = { body: {}, activeElement: trigger, querySelector: () => main };
const moduleObject = { exports: {} };
const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "../components/interaction/InteractionDialog.tsx"), "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
}).outputText;
new Function("require", "module", "exports", code)(name => {
  if (name === "react-dom") return { createPortal: children => children };
  if (name.endsWith(".module.css")) return {};
  return require(name);
}, moduleObject, moduleObject.exports);
const { InteractionDialog } = moduleObject.exports;
const props = { open: true, phase: "ready", title: "Enter score", children: null, actions: null,
  initialFocusRef: { current: first }, onRequestClose() {} };

async function run() {
  let tree;
  await act(async () => { tree = create(React.createElement(InteractionDialog, props), {
    createNodeMock: element => element.type === "dialog" ? dialog : null
  }); });
  // Users can move to another field before the browser delivers its next frame.
  second.focus();
  paint();
  assert.equal(document.activeElement, second, "Opening a dialog must not steal focus back after the user moves to the second score");
  await act(async () => tree.update(React.createElement(InteractionDialog, { ...props, dirty: true })));
  paint();
  assert.equal(document.activeElement, second, "Editing a score must retain the selected field");
  phaseTarget = outcome;
  await act(async () => tree.update(React.createElement(InteractionDialog, { ...props, phase: "success" })));
  paint();
  assert.equal(document.activeElement, outcome, "An outcome must still receive accessible phase focus");
  await act(async () => tree.update(React.createElement(InteractionDialog, { ...props, open: false })));
  paint();
  assert.equal(document.activeElement, trigger, "Closing restores the original trigger");
  await act(async () => tree.unmount());
  console.log("PASS dialog focus: quick field changes, retained drafts, outcome focus, and trigger restoration");
}
run().catch(error => { console.error(error); process.exitCode = 1; });
