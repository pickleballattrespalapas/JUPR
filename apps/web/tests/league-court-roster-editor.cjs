const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), ts = require("typescript");
const React = require("react"), { act, create } = require("react-test-renderer");

// Keep the actual controlled editor and its drag-result handler. The mock only
// supplies the DnD render props that normally require a browser layout engine.
const dnd = {
  DragDropContext: ({ children, ...props }) => React.createElement("mock-dnd-context", props, children),
  Droppable: ({ children, ...props }) => React.createElement("mock-droppable", props, children({
    innerRef() {}, droppableProps: { "data-droppable-id": props.droppableId }, placeholder: null
  }, { isDraggingOver: false })),
  Draggable: ({ children, ...props }) => React.createElement("mock-draggable", props, children({
    innerRef() {}, draggableProps: { "data-draggable-id": props.draggableId, style: {} }, dragHandleProps: {}
  }, { isDragging: false }))
};

const webRoot = path.resolve(__dirname, ".."), modules = new Map();
function load(name, parent = webRoot) {
  if (name === "@hello-pangea/dnd") return dnd;
  if (name.endsWith(".module.css")) return new Proxy({}, { get: (_, key) => String(key) });
  if (!name.startsWith("@/") && !name.startsWith(".")) return require(name);
  const base = name.startsWith("@/") ? path.join(webRoot, name.slice(2)) : path.resolve(parent, name);
  const file = [base, base + ".tsx", base + ".ts"].find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
  if (!file) throw new Error(`Missing test module: ${name}`);
  if (modules.has(file)) return modules.get(file);
  const mod = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
  }).outputText;
  new Function("require", "module", "exports", code)(dependency => load(dependency, path.dirname(file)), mod, mod.exports);
  modules.set(file, mod.exports);
  return mod.exports;
}

const Editor = load("@/app/admin/league-manager/live/LeagueCourtRosterEditor").LeagueCourtRosterEditor;
const text = node => typeof node === "string" ? node : (node?.children || []).map(text).join("");
const names = courts => courts.map(court => court.playerNames.split("\n").map(name => name.trim()).filter(Boolean));
const baseCourts = () => [
  { court: "1", formatType: "4-Player", playerNames: "Alex\nBlair\nCasey\nDrew" },
  { court: "2", formatType: "5-Player", playerNames: "Emery\nFinley\nGray\nHarper\nIndigo" }
];
async function mount(overrides = {}) {
  let tree, currentCourts, resetCourts, changeCount = 0, editRosterCount = 0;
  const initial = overrides.courts || baseCourts();
  function Harness() {
    const [courts, setCourts] = React.useState(() => structuredClone(initial));
    currentCourts = courts;
    resetCourts = setCourts;
    return React.createElement(Editor, {
      formats: ["4-Player", "5-Player"],
      playerNames: names(initial).flat(),
      disabled: false,
      ...overrides,
      courts,
      onChange(nextCourts) { changeCount++; setCourts(nextCourts); },
      onEditRoster() { editRosterCount++; }
    });
  }
  await act(async () => { tree = create(React.createElement(Harness)); });
  return {
    tree,
    get courts() { return currentCourts; },
    get changes() { return changeCount; },
    get rosterEdits() { return editRosterCount; },
    async reset(courts) { await act(async () => resetCourts(structuredClone(courts))); },
    async close() { await act(async () => tree.unmount()); }
  };
}
function control(harness, label) {
  const matches = harness.tree.root.findAll(node => typeof node.type === "string" && node.props["aria-label"] === label);
  assert.equal(matches.length, 1, `Expected one control labeled ${label}`);
  return matches[0];
}
function button(harness, label, within = harness.tree.root) {
  const matches = within.findAllByType("button").filter(node => text(node) === label);
  assert.equal(matches.length, 1, `Expected one button ${label}`);
  return matches[0];
}
async function click(node) {
  assert.notEqual(node.props.disabled, true, `Control must be enabled: ${node.props["aria-label"] || text(node)}`);
  await act(async () => node.props.onClick());
}
async function change(node, value) {
  assert.notEqual(node.props.disabled, true, `Control must be enabled: ${node.props["aria-label"] || node.type}`);
  await act(async () => node.props.onChange({ target: { value } }));
}
async function drag(harness, name, from, to, sourceIndex = 0) {
  const result = {
    draggableId: `player:${name.replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US")}`,
    source: { droppableId: from, index: sourceIndex },
    destination: to == null ? null : { droppableId: to, index: 0 },
    reason: to == null ? "CANCEL" : "DROP", mode: "FLUID", type: "DEFAULT", combine: null
  };
  await act(async () => harness.tree.root.findByType("mock-dnd-context").props.onDragEnd(result));
}

(async () => {
  // Interaction checks below intentionally assert the committed court assignments,
  // not helper implementation details or source-code patterns.
  {
    const harness = await mount();
    const initial = structuredClone(harness.courts);
    assert.equal(button(harness, "Undo last edit").props.disabled, true);
    await click(control(harness, "Select Alex on Court 1, position 1"));
    assert.equal(harness.changes, 0, "Selecting a player must not alter the roster");
    await click(control(harness, "Swap Alex with Emery on Court 2"));
    assert.deepEqual(names(harness.courts), [
      ["Emery", "Blair", "Casey", "Drew"],
      ["Alex", "Finley", "Gray", "Harper", "Indigo"]
    ]);
    assert.deepEqual(names(harness.courts).map(list => list.length), [4, 5]);
    assert.deepEqual(names(harness.courts).flat().sort(), names(initial).flat().sort(), "A swap preserves every player exactly once");
    assert.deepEqual(harness.courts.map(court => court.formatType), ["4-Player", "5-Player"]);
    await click(button(harness, "Undo last edit"));
    assert.deepEqual(harness.courts, initial, "Undo restores the full assignment including court formats");
    assert.equal(button(harness, "Undo last edit").props.disabled, true);
    await harness.close();
  }

  {
    const harness = await mount();
    assert.equal(control(harness, "Move Alex up on Court 1").props.disabled, true);
    assert.equal(control(harness, "Move Drew down on Court 1").props.disabled, true);
    await click(control(harness, "Move Casey up on Court 1"));
    assert.deepEqual(names(harness.courts)[0], ["Alex", "Casey", "Blair", "Drew"]);
    assert.deepEqual(names(harness.courts)[1], names(baseCourts())[1], "Reordering does not change another court");
    await click(control(harness, "Select Alex on Court 1, position 1"));
    await click(control(harness, "Swap Alex with Drew on Court 1"));
    assert.deepEqual(names(harness.courts)[0], ["Drew", "Casey", "Blair", "Alex"], "Two taps also swap positions within one court");
    await harness.close();
  }

  {
    const harness = await mount();
    await click(control(harness, "Select Indigo on Court 2, position 5"));
    await change(control(harness, "Move Indigo to court"), "0");
    assert.deepEqual(names(harness.courts), [
      ["Alex", "Blair", "Casey", "Drew", "Indigo"],
      ["Emery", "Finley", "Gray", "Harper"]
    ]);
    assert.deepEqual(harness.courts.map(court => court.formatType), ["5-Player", "4-Player"], "Moving from five to four players updates both formats");
    assert.deepEqual(names(harness.courts).flat().sort(), names(baseCourts()).flat().sort());
    await harness.close();
  }

  {
    const harness = await mount();
    await change(control(harness, "Court position for Court 1"), "1");
    assert.deepEqual(names(harness.courts), [names(baseCourts())[1], names(baseCourts())[0]], "Changing court position moves the whole group in its existing player order");
    assert.deepEqual(harness.courts.map(court => court.court), ["1", "2"], "Court positions remain unique and contiguous");
    assert.deepEqual(harness.courts.map(court => court.formatType), ["5-Player", "4-Player"], "Court formats travel with their groups");
    await click(button(harness, "Undo last edit"));
    assert.deepEqual(harness.courts, baseCourts());
    await harness.close();
  }

  {
    const harness = await mount();
    await click(control(harness, "Unassign Alex from Court 1"));
    assert.deepEqual(names(harness.courts)[0], ["Blair", "Casey", "Drew"]);
    assert.match(text(harness.tree.toJSON()), /Unassigned players \(1\)/);
    await change(control(harness, "Assign Alex to court"), "0");
    assert.deepEqual(names(harness.courts)[0], ["Blair", "Casey", "Drew", "Alex"]);
    assert.equal(harness.tree.root.findAll(node => node.type === "select" && String(node.props["aria-label"]).startsWith("Assign ")).length, 0, "No players remain unassigned; the empty drop zone may remain available");
    assert.deepEqual(names(harness.courts).flat().sort(), names(baseCourts()).flat().sort(), "Unassigning and reassigning cannot drop an attendee");
    await click(button(harness, "Edit session roster"));
    assert.equal(harness.rosterEdits, 1, "The roster action reaches its parent callback");
    assert.equal(harness.changes, 2, "Opening the roster editor does not mutate court assignments");
    await harness.close();
  }

  {
    const harness = await mount();
    await click(button(harness, "Remove court", control(harness, "Court 1 roster")));
    assert.equal(harness.courts.length, 1);
    assert.deepEqual(names(harness.courts), [names(baseCourts())[1]]);
    assert.match(text(harness.tree.toJSON()), /Unassigned players \(4\)/);
    for (const name of names(baseCourts())[0]) control(harness, `Assign ${name} to court`);
    assert.equal(button(harness, "Remove court").props.disabled, true, "At least one court remains available");
    await click(button(harness, "Add court"));
    assert.deepEqual(harness.courts.map(court => court.court), ["1", "2"]);
    for (const name of names(baseCourts())[0]) await change(control(harness, `Assign ${name} to court`), "1");
    assert.deepEqual(names(harness.courts), [names(baseCourts())[1], names(baseCourts())[0]]);
    assert.equal(harness.tree.root.findAll(node => node.type === "select" && String(node.props["aria-label"]).startsWith("Assign ")).length, 0, "All players from the removed court have been reassigned");
    assert.deepEqual(names(harness.courts).flat().sort(), names(baseCourts()).flat().sort(), "Removing a court retains all players for reassignment");
    await harness.close();
  }

  {
    const harness = await mount();
    const initial = structuredClone(harness.courts);
    const bulk = () => control(harness, "Players, one per line for Court 1");
    const apply = () => button(harness, "Apply player list", control(harness, "Court 1 roster"));
    for (const duplicate of ["Alex\nBlair\nCasey\nEmery", "Alex\n  alex  \nCasey\nDrew"]) {
      await change(bulk(), duplicate);
      await click(apply());
      assert.match(text(harness.tree.root.findByProps({ role: "alert" })), /listed more than once/);
      assert.deepEqual(harness.courts, initial, "Duplicate bulk assignments cannot overwrite a valid court arrangement");
      assert.equal(harness.changes, 0);
    }
    await change(bulk(), "Alex\nBlair\nCasey\nUnknown Attendee");
    await click(apply());
    assert.match(text(harness.tree.root.findByProps({ role: "alert" })), /not in the active session roster/);
    assert.deepEqual(harness.courts, initial, "Bulk edit cannot introduce an attendee absent from the session roster");
    await change(bulk(), "Drew\nCasey\nBlair\nAlex");
    await click(apply());
    assert.deepEqual(names(harness.courts)[0], ["Drew", "Casey", "Blair", "Alex"]);
    assert.equal(harness.changes, 1, "A valid list can be applied after correcting an error");
    await harness.close();
  }

  {
    const harness = await mount();
    await click(control(harness, "Move Blair up on Court 1"));
    assert.equal(button(harness, "Undo last edit").props.disabled, false);
    await click(control(harness, "Select Alex on Court 1, position 2"));
    await change(control(harness, "Players, one per line for Court 1"), "Stale unsaved draft");
    const externalCourts = baseCourts();
    externalCourts[1].playerNames = "Indigo\nHarper\nGray\nFinley\nEmery";
    await harness.reset(externalCourts);
    assert.equal(button(harness, "Undo last edit").props.disabled, true, "A newly loaded court arrangement cannot undo into an earlier session");
    assert.deepEqual(harness.courts, externalCourts);
    assert.equal(harness.tree.root.findAll(node => node.props["aria-label"] === "Move Alex to court").length, 0, "External reload clears selected positions");
    assert.equal(control(harness, "Players, one per line for Court 1").props.value, externalCourts[0].playerNames, "External reload clears stale bulk drafts");
    await harness.close();
  }

  {
    const harness = await mount({ disabled: true, playerNames: [...names(baseCourts()).flat(), "Jordan"] });
    const controls = harness.tree.root.findAll(node => ["button", "select", "input", "textarea"].includes(node.type));
    assert.ok(controls.length > 20, "The disabled check includes player, court, bulk, roster and unassigned controls");
    for (const node of controls) assert.equal(node.props.disabled, true, `Disabled editor exposed an enabled control: ${node.props["aria-label"] || text(node)}`);
    assert.equal(harness.changes, 0);
    assert.deepEqual(harness.courts, baseCourts());
    for (const node of harness.tree.root.findAllByType("mock-draggable")) assert.equal(node.props.isDragDisabled, true, "Locked score entry also disables every draggable card");
    for (const node of harness.tree.root.findAllByType("mock-droppable")) assert.equal(node.props.isDropDisabled, true, "Locked score entry also disables every drop target");
    await drag(harness, "Alex", "court:0:slot:0", "court:1:slot:0");
    assert.equal(harness.changes, 0, "A delayed drag-end event cannot mutate a locked editor");
    assert.deepEqual(harness.courts, baseCourts());
    await harness.close();
  }

  {
    const courts = baseCourts();
    courts[1].playerNames = "Jose\nFinley\nGray\nHarper\nIndigo";
    courts.push({ court: "3", formatType: "4-Player", playerNames: "Jitka\nKim\nLeah\nMorgan" });
    const harness = await mount({ courts, playerRatings: [{ name: "  JITKA  ", rating: 1600 }, { name: "Jose", rating: 1824 }] });
    assert.match(text(control(harness, "Select Jitka on Court 3, position 1")), /Rating\s*4\.00/);
    assert.match(text(control(harness, "Select Jose on Court 2, position 1")), /Rating\s*4\.56/);
    assert.match(text(control(harness, "Select Alex on Court 1, position 1")), /Rating\s*—/, "An unknown rating is explicitly missing, not zero");
    const jitkaDraggable = harness.tree.root.findAllByType("mock-draggable").find(node => node.props.draggableId === "player:jitka");
    assert.ok(jitkaDraggable, "Each player has a stable draggable identity independent of court and position");
    assert.equal(jitkaDraggable.props.disableInteractiveElementBlocking, true, "Dragging directly from the player-name button must be enabled");
    await drag(harness, "Jitka", "court:2:slot:0", "court:1:slot:0");
    assert.deepEqual(names(harness.courts), [
      ["Alex", "Blair", "Casey", "Drew"],
      ["Jitka", "Finley", "Gray", "Harper", "Indigo"],
      ["Jose", "Kim", "Leah", "Morgan"]
    ], "Dragging Jitka onto Jose exchanges only their occupied positions");
    assert.deepEqual(names(harness.courts).map(list => list.length), [4, 5, 4], "An occupied-slot drop never makes a court overfull");
    assert.deepEqual(names(harness.courts).flat().sort(), names(courts).flat().sort(), "Dragging cannot lose or duplicate an attendee");
    assert.match(text(control(harness, "Select Jitka on Court 2, position 1")), /Rating\s*4\.00/, "Ratings follow the player after the swap");
    assert.match(text(control(harness, "Select Jose on Court 3, position 1")), /Rating\s*4\.56/);
    await click(button(harness, "Undo last edit"));
    assert.deepEqual(harness.courts, courts, "Undo restores a drag swap exactly");
    assert.match(text(control(harness, "Select Jitka on Court 3, position 1")), /Rating\s*4\.00/);
    await harness.close();
  }

  {
    const harness = await mount();
    await drag(harness, "Alex", "court:0:slot:0", "court:0:slot:2");
    assert.deepEqual(names(harness.courts)[0], ["Blair", "Casey", "Alex", "Drew"], "Within-court dragging reorders intervening players instead of exchanging two positions");
    await drag(harness, "Alex", "court:0:slot:2", "court:0:slot:4");
    assert.deepEqual(names(harness.courts)[0], ["Blair", "Casey", "Drew", "Alex"], "The empty end slot moves a player to the end of their court");
    assert.deepEqual(names(harness.courts)[1], names(baseCourts())[1]);
    const unchanged = structuredClone(harness.courts), previousChanges = harness.changes;
    await drag(harness, "Alex", "court:0:slot:3", "court:0:slot:3");
    await drag(harness, "Alex", "court:0:slot:3", null);
    await drag(harness, "Missing player", "court:0:slot:3", "court:1:slot:0");
    await drag(harness, "Alex", "court:99:slot:3", "court:1:slot:0");
    assert.deepEqual(harness.courts, unchanged, "Same-position, cancelled, stale-source and invalid-source drops do not alter assignments");
    assert.equal(harness.changes, previousChanges);
    await harness.close();
  }

  {
    const harness = await mount();
    await drag(harness, "Indigo", "court:1:slot:4", "court:0:slot:4");
    assert.deepEqual(names(harness.courts), [
      ["Alex", "Blair", "Casey", "Drew", "Indigo"],
      ["Emery", "Finley", "Gray", "Harper"]
    ], "Dropping on an empty end slot moves instead of replacing a player");
    assert.deepEqual(harness.courts.map(court => court.formatType), ["5-Player", "4-Player"]);
    await click(button(harness, "Undo last edit"));
    await click(button(harness, "Add court"));
    await drag(harness, "Alex", "court:0:slot:0", "court:2:slot:0");
    assert.deepEqual(names(harness.courts), [["Blair", "Casey", "Drew"], names(baseCourts())[1], ["Alex"]], "A newly added empty court accepts a direct drop");
    await harness.close();
  }

  {
    const harness = await mount({ playerNames: [...names(baseCourts()).flat(), "Jordan"], playerRatings: [{ name: "Jordan", rating: 1400 }, { name: "Alex", rating: 1600 }] });
    await drag(harness, "Jordan", "unassigned", "court:0:slot:0");
    assert.deepEqual(names(harness.courts)[0], ["Jordan", "Blair", "Casey", "Drew"], "An unassigned player can replace an occupied court position");
    control(harness, "Assign Alex to court");
    assert.match(text(control(harness, "Select Jordan on Court 1, position 1")), /Rating\s*3\.50/);
    await drag(harness, "Alex", "unassigned", "court:0:slot:4");
    assert.deepEqual(names(harness.courts)[0], ["Jordan", "Blair", "Casey", "Drew", "Alex"]);
    assert.equal(harness.courts[0].formatType, "5-Player");
    await drag(harness, "Jordan", "court:0:slot:0", "unassigned");
    assert.deepEqual(names(harness.courts)[0], ["Blair", "Casey", "Drew", "Alex"]);
    assert.equal(harness.courts[0].formatType, "4-Player");
    control(harness, "Assign Jordan to court");
    assert.deepEqual(names(harness.courts).flat().sort(), names(baseCourts()).flat().sort(), "Moving to unassigned preserves all other court players");
    await click(button(harness, "Undo last edit"));
    assert.deepEqual(names(harness.courts)[0], ["Jordan", "Blair", "Casey", "Drew", "Alex"], "Undo restores an unassignment drag including its court format");
    assert.equal(harness.courts[0].formatType, "5-Player");
    await harness.close();
  }
  console.log("PASS League court roster editor: click and drag swaps, within-court drag order, occupied/empty/unassigned drops, ratings and undo, 4/5-player formats, roster preservation, stale/cancelled/disabled drag guards, bulk validation and external reset.");
})().catch(error => { console.error(error); process.exitCode = 1; });
