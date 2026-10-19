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
  console.log(
    "Recap customization controls: independent ordering, story inclusion, promotions, layout management and conflicts passed",
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
