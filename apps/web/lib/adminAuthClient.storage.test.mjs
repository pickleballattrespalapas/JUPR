import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

const compiledModulePath = process.env.JUPR_ADMIN_AUTH_CLIENT_MODULE;
const authModuleUrl = compiledModulePath
  ? pathToFileURL(resolve(compiledModulePath)).href
  : new URL("./adminAuthClient.ts", import.meta.url).href;

const {
  ADMIN_SESSION_STORAGE_KEY,
  adminSessionStorageEventIsRelevant
} = await import(authModuleUrl);

test("only admin-session storage changes or localStorage.clear invalidate the admin session", () => {
  assert.equal(adminSessionStorageEventIsRelevant(ADMIN_SESSION_STORAGE_KEY), true);
  assert.equal(adminSessionStorageEventIsRelevant(null), true);
  assert.equal(adminSessionStorageEventIsRelevant("jupr_tournament_live_pending_tres_palapas"), false);
  assert.equal(adminSessionStorageEventIsRelevant("unrelated"), false);
});
