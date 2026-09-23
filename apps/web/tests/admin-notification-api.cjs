const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

function load(relative, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, "..", relative), "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS }
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(name => mocks[name] ?? require(name), module, module.exports);
  return module.exports;
}

const events = load("lib/adminNotificationsEvents.ts");
const api = load("lib/adminNotificationsApi.ts", {
  "@/lib/adminAuthClient": { getAdminApiBaseUrl: () => "http://127.0.0.1:9" },
  "@/lib/adminNotificationsEvents": events
});
const key = "a".repeat(64);
const feed = state => ({ club_id: "alpha", categories: [], items: [{ key, href: "/admin/tools", state }] });
const reply = data => ({ ok: true, status: 200, json: async () => data });

async function mutationOutranksOverlappingReads({ readBeforeMutation, finishReadFirst }) {
  const calls = [], published = [];
  global.fetch = (url, options) => new Promise(resolve => calls.push({ url, options, resolve }));
  const unsubscribe = events.subscribeAdminNotifications("token", "alpha", data => published.push(data.items[0].state));
  let read;
  if (readBeforeMutation) read = api.getAdminNotifications("token", "alpha");
  const mutation = api.clearAdminNotifications("token", "alpha", [key]);
  if (!readBeforeMutation) read = api.getAdminNotifications("token", "alpha");
  assert.equal(calls.length, 2);
  const get = calls.find(call => call.options.method === "GET");
  const put = calls.find(call => call.options.method === "PUT");
  if (finishReadFirst) {
    get.resolve(reply(feed("new")));
    await read;
    put.resolve(reply(feed("cleared")));
    await mutation;
  } else {
    put.resolve(reply(feed("cleared")));
    await mutation;
    get.resolve(reply(feed("new")));
    await read;
  }
  assert.equal(published.at(-1), "cleared", "The confirmed mutation must be the last sidebar update even when a poll overlaps it");
  if (!finishReadFirst) assert.deepEqual(published, ["cleared"], "A late pre-mutation snapshot cannot restore cleared badges");

  const freshRead = api.getAdminNotifications("token", "alpha");
  calls.at(-1).resolve(reply(feed("flagged")));
  await freshRead;
  assert.equal(published.at(-1), "flagged", "A read begun after mutation completion can publish fresh source changes");
  unsubscribe();
}

async function overlappingSidebarAndCenterReadsAreShared() {
  const calls = [];
  global.fetch = (url, options) => new Promise(resolve => calls.push({ url, options, resolve }));
  const sidebar = api.getAdminNotifications("token", "alpha");
  const center = api.getAdminNotifications("token", "alpha");
  assert.equal(calls.length, 1, "Sidebar and center share one overlapping feed request");
  calls[0].resolve(reply(feed("new")));
  const results = await Promise.all([sidebar, center]);
  assert.equal(results[0].data.items[0].state, "new");
  assert.equal(results[1].data.items[0].state, "new");
}

(async () => {
  await overlappingSidebarAndCenterReadsAreShared();
  for (const readBeforeMutation of [true, false]) {
    for (const finishReadFirst of [true, false]) {
      await mutationOutranksOverlappingReads({ readBeforeMutation, finishReadFirst });
    }
  }
  console.log("Admin notification API: shared reads and all read/mutation completion orders preserve confirmed sidebar state.");
})().catch(error => { console.error(error); process.exitCode = 1; });
