const assert = require("node:assert/strict");
const fs = require("node:fs"),
  path = require("node:path"),
  ts = require("typescript");
const React = require("react");
const { create, act } = require("react-test-renderer");
function load(file, mocks = {}) {
  const code = ts.transpileModule(
    fs.readFileSync(path.join(__dirname, "..", file), "utf8"),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.CommonJS,
        jsx: ts.JsxEmit.ReactJSX,
        esModuleInterop: true,
      },
    },
  ).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(
    (name) => (Object.hasOwn(mocks, name) ? mocks[name] : require(name)),
    module,
    module.exports,
  );
  return module.exports;
}
const builderLib = load("lib/recapBuilder.ts");
const { RecapContentEditor, RecapStoryEditor } = load(
  "app/admin/weekly-recap/RecapContentEditor.tsx",
  {
    "@/lib/recapBuilder": builderLib,
    "@/components/RecapEdition": () => null,
    "./recap-builder.module.css": {},
  },
);
const ConfirmAction = (props) => React.createElement("confirm-action", props);
const LayoutManager = load("app/admin/weekly-recap/RecapLayoutManager.tsx", {
  "@/components/ConfirmAction": { ConfirmAction },
  "@/components/interaction": {
    actionSuccess: (title, description) => ({ title, description }),
  },
  "./recap-builder.module.css": {},
}).default;
const fixture = require("./fixtures/weekly-recap-builder.json");
const rich = require("./fixtures/weekly-recap-stories.json");
const text = (node) =>
  typeof node === "string" ? node : (node?.children || []).map(text).join(" ");
(async () => {
  let builder = builderLib.normalizeBuilder(rich.config),
    tree;
  const render = (Component) =>
    React.createElement(Component, {
      builder,
      catalog: { ...fixture.catalog, custom: builder.custom_blocks },
      disabled: false,
      onChange: (value) => {
        builder = value;
        tree.update(render(Component));
      },
    });
  await act(async () => {
    tree = create(render(RecapContentEditor));
  });
  const website = [...builder.order],
    websiteItems = [...builder.sections.custom.item_ids];
  await act(async () =>
    tree.root
      .findByProps({ "aria-label": "Move Club stories up in email" })
      .props.onClick(),
  );
  assert.equal(builder.email_order[0], "custom");
  assert.deepEqual(
    builder.order,
    website,
    "Email order does not move website sections",
  );
  await act(async () =>
    tree.root
      .findByProps({ "aria-label": "Move Saturday smiles earlier in email" })
      .props.onClick(),
  );
  assert.deepEqual(builder.sections.custom.email_item_ids, [
    "photo",
    "register",
  ]);
  assert.deepEqual(
    builder.sections.custom.item_ids,
    websiteItems,
    "Email item order leaves website choices intact",
  );
  await act(async () => tree.unmount());
  await act(async () => {
    tree = create(render(RecapStoryEditor));
  });
  const photo = tree.root.findAllByType("fieldset")[0];
  await act(async () =>
    photo
      .findAllByType("input")
      .find((i) => i.props.type === "checkbox")
      .props.onChange({ target: { checked: false } }),
  );
  let projection = builderLib.projectEditions(
    { ...fixture.catalog, custom: builder.custom_blocks },
    builder,
  );
  assert(
    !JSON.stringify(projection).includes("Saturday smiles"),
    "Hiding a story removes it from both editions",
  );
  const hidden = tree.root.findAllByType("fieldset")[2];
  await act(async () =>
    hidden
      .findAllByType("input")
      .find((i) => i.props.type === "checkbox")
      .props.onChange({ target: { checked: true } }),
  );
  assert(builder.sections.custom.item_ids.includes("hidden"));
  await act(async () =>
    tree.root
      .findAllByType("button")
      .find((b) => text(b).replace(/\s+/g, " ") === "+ Featured event")
      .props.onClick(),
  );
  const created = builder.custom_blocks.at(-1);
  assert.equal(created.layout, "promotion");
  assert.equal(created.link_label, "Register now");
  assert(
    builder.sections.custom.item_ids.includes(created.id),
    "New stories join explicit selections",
  );
  assert(builder.sections.custom.email_item_ids.includes(created.id));
  await act(async () => tree.unmount());

  const prefs = {
    revision: 7,
    settings: {
      templates: [
        { id: "a", name: "Weekly", builder: builderLib.normalizeBuilder() },
        { id: "b", name: "Social", builder: builderLib.normalizeBuilder() },
      ],
      default_template_id: "a",
    },
    leagues: [],
  };
  const calls = [];
  let failure = "";
  await act(async () => {
    tree = create(
      React.createElement(LayoutManager, {
        preferences: prefs,
        disabled: false,
        onSave: async (...args) => {
          if (failure) throw Error(failure);
          calls.push(args);
        },
        onReload: async () => {
          calls.push(["reload"]);
        },
      }),
    );
  });
  const rename = () =>
    tree.root
      .findAllByType("button")
      .find((b) => text(b) === "Rename saved layout");
  await act(async () =>
    tree.root
      .findByType("input")
      .props.onChange({ target: { value: "Member update" } }),
  );
  await act(async () => {
    rename().props.onClick();
  });
  assert.equal(calls[0][0].templates[0].id, "a");
  assert.equal(calls[0][0].templates[0].name, "Member update");
  assert.equal(calls[0][1], 7, "Rename uses the exact loaded revision");
  await act(async () =>
    tree.root
      .findByType("input")
      .props.onChange({ target: { value: " social " } }),
  );
  await act(async () => {
    rename().props.onClick();
  });
  assert.match(text(tree.toJSON()), /already exists/);
  assert.equal(calls.length, 1);
  failure = "Club templates changed. Reload before saving.";
  await act(async () =>
    tree.root
      .findByType("input")
      .props.onChange({ target: { value: "Updated" } }),
  );
  await act(async () => {
    rename().props.onClick();
  });
  assert.match(text(tree.toJSON()), /Reload before saving/);
  assert.equal(calls.length, 1);
  failure = "";
  await act(async () =>
    tree.root
      .findAllByType("select")[0]
      .props.onChange({ target: { value: "b" } }),
  );
  assert.equal(calls.at(-1)[0].default_template_id, "b");
  await act(async () =>
    tree.root.findByType(ConfirmAction).props.onConfirm("DELETE LAYOUT"),
  );
  assert.deepEqual(
    calls.at(-1)[0].templates.map((t) => t.id),
    ["b"],
  );
  assert.equal(
    calls.at(-1)[0].default_template_id,
    "",
    "Deleting the selected default restores Standard",
  );
  await act(async () => tree.unmount());
  // The Content picker must preserve exact evidence and explicit email choices.
  const EvidencePicker = load(
    "app/admin/weekly-recap/RecapEvidencePicker.tsx",
    {
      "@/lib/clubSite": load("lib/clubSite.ts"),
      "@/lib/recapBuilder": builderLib,
      "./recap-builder.module.css": {},
    },
  ).default;
  const key = "METRIC_most_wins",
    categoryId = `spotlight-${key}`;
  const candidates = {
    [key]: Array.from({ length: 10 }, (_, i) => ({
      candidate_id: `player:${i}`,
      label: "Most wins",
      display: `Player ${i} — ${12 - i} wins`,
    })),
  };
  const courts = require("./fixtures/weekly-recap-courts.json");
  const evidenceCatalog = { ...fixture.catalog, events: [...fixture.catalog.events, ...courts.events.filter(i => i.kind === "Ladder court")] };
  let highlights = {
    [key]: { include: false, players: [], description: "", order: 1 },
  };
  builder = builderLib.normalizeBuilder({
    sections: { spotlight: { item_ids: [], detail: "brief" } },
  });
  let preview;
  const renderEvidence = () => {
    preview = builderLib.previewRecap(
      { builder_catalog: evidenceCatalog },
      {
        builder,
        spotlight_overrides: highlights,
        looking_ahead: [],
        scope: { league_names: [] },
        editorial: {
          headline: "Test",
          opening: "",
          event_notes: "",
          upcoming: [],
        },
      },
      candidates,
    );
    return React.createElement(EvidencePicker, {
      builder,
      catalog: preview.builder_catalog,
      candidates,
      spotlights: highlights,
      disabled: false,
      onBuilderChange: (value) => {
        builder = value;
        tree.update(renderEvidence());
      },
      onSpotlightChange: (key, patch) => {
        highlights = { ...highlights, [key]: { ...highlights[key], ...patch } };
        tree.update(renderEvidence());
      },
    });
  };
  await act(async () => {
    tree = create(renderEvidence());
  });
  const checkbox = (label) => tree.root.findByProps({ "aria-label": label });
  await act(async () =>
    checkbox("Include Round-robin winners").props.onChange({
      target: { checked: false },
    }),
  );
  const selectedEvents = preview.report.edition.find(
    (b) => b.key === "events",
  ).items;
  assert(
    !selectedEvents.some((item) => /round robin|moneyball/i.test(item.kind)),
  );
  assert(
    selectedEvents.some((item) => /tournament/i.test(item.kind)),
    "Other categories are preserved",
  );
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-rr" }).length, 0);
  await act(async () => checkbox("Include Round-robin winners").props.onChange({ target: { checked: true } }));
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-rr" }).length, 1);
  await act(async () => checkbox("Include New achievements").props.onChange({ target: { checked: false } }));
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-achievements" }).length, 0);
  await act(async () => checkbox("Include New achievements").props.onChange({ target: { checked: true } }));
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-achievements" }).length, 1);
  const courtPicker = () => tree.root.findByProps({ "data-testid": "recap-evidence-courts" });
  assert(text(courtPicker()).includes("Alex ×3"));
  assert(!text(courtPicker()).includes("point difference"));
  await act(async () => checkbox("Include Alex ×3").props.onChange({ target: { checked: false } }));
  assert(!builder.sections.events.item_ids.includes("a1") && !builder.sections.events.item_ids.includes("a3"));
  await act(async () => checkbox("Include Alex ×3").props.onChange({ target: { checked: true } }));
  await act(async () => checkbox("Feature Alex ×3 in email").props.onChange({ target: { checked: true } }));
  assert(["a1", "a2", "a3"].every(id => builder.sections.events.email_item_ids.includes(id)));
  await act(async () => checkbox("Include Ladder court winners").props.onChange({ target: { checked: false } }));
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-courts" }).length, 0);
  await act(async () => tree.root.findByProps({ type: "search" }).props.onChange({ target: { value: "Taylor" } }));
  assert.match(text(tree.toJSON()), /No results match/);
  await act(async () => tree.root.findByProps({ type: "search" }).props.onChange({ target: { value: "" } }));
  await act(async () => checkbox("Include Ladder court winners").props.onChange({ target: { checked: true } }));
  assert.equal(tree.root.findAllByProps({ "data-testid": "recap-evidence-courts" }).length, 1);
  await act(async () =>
    checkbox("Include Most wins").props.onChange({ target: { checked: true } }),
  );
  assert(
    builder.sections.spotlight.item_ids.includes(categoryId),
    "Enabling restores a category previously excluded by layout",
  );
  const players = () =>
    tree.root
      .findByProps({ "data-testid": `recap-evidence-${key}` })
      .findAllByType("input")
      .filter((n) => !n.props["aria-label"] && n.props.type === "checkbox");
  await act(async () =>
    players()[0].props.onChange({ target: { checked: false } }),
  );
  await act(async () =>
    players()[5].props.onChange({ target: { checked: true } }),
  );
  await act(async () =>
    checkbox("Include Most wins in email").props.onChange({
      target: { checked: true },
    }),
  );
  for (const channel of ["edition", "email_edition"]) {
    const block = preview.report[channel].find((b) => b.key === "spotlight");
    assert.deepEqual(
      block.items[0].lines,
      [1, 2, 5].map((i) => candidates[key][i].display),
      "Every explicitly selected player survives compact layout and email",
    );
  }
  await act(async () =>
    tree.root
      .findByProps({ type: "search" })
      .props.onChange({ target: { value: "Player 5" } }),
  );
  assert.equal(players().length, 1);
  assert.equal(
    players()[0].props.checked,
    true,
    "Search does not alter selections",
  );
  await act(async () =>
    tree.root
      .findByProps({ type: "search" })
      .props.onChange({ target: { value: "Missing person" } }),
  );
  assert.match(text(tree.toJSON()), /No results match/);
  await act(async () =>
    tree.root
      .findByProps({ type: "search" })
      .props.onChange({ target: { value: "" } }),
  );
  for (let index = 0; index < 10; index++) {
    if (!players()[index].props.checked)
      await act(async () =>
        players()[index].props.onChange({ target: { checked: true } }),
      );
  }
  assert.equal(highlights[key].players.length, 10);
  for (let index = 0; index < 10; index++) {
    await act(async () =>
      players()[index].props.onChange({ target: { checked: false } }),
    );
  }
  assert.equal(highlights[key].players.length, 0);
  assert.equal(
    players().length,
    10,
    "Clearing selections keeps all choices available",
  );
  assert(checkbox("Include Most wins").props.checked);

  await act(async () =>
    checkbox("Include Most wins").props.onChange({
      target: { checked: false },
    }),
  );
  assert(!preview.report.edition.some((b) => b.key === "spotlight"));
  await act(async () => tree.unmount());
  console.log(
    "Recap customization controls: independent ordering, story inclusion, promotions, layout management and conflicts passed",
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
