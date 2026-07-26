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
  adminSessionStorageEventIsRelevant,
  restoreAuthorizedAdminSession
} = await import(authModuleUrl);

test("only admin-session storage changes or localStorage.clear invalidate the admin session", () => {
  assert.equal(adminSessionStorageEventIsRelevant(ADMIN_SESSION_STORAGE_KEY), true);
  assert.equal(adminSessionStorageEventIsRelevant(null), true);
  assert.equal(adminSessionStorageEventIsRelevant("jupr_tournament_live_pending_tres_palapas"), false);
  assert.equal(adminSessionStorageEventIsRelevant("unrelated"), false);
});

test("two consumers coalesce an expired-session restore before a losing refresh can clear it", async () => {
  const originalWindow = globalThis.window;
  const originalCustomEvent = globalThis.CustomEvent;
  const originalFetch = globalThis.fetch;
  const originalEnv = {
    apiBase: process.env.NEXT_PUBLIC_JUPR_API_BASE_URL,
    supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
    supabaseAnonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  };
  const storage = new Map();
  const dispatchedEvents = [];
  const expiredSession = {
    access_token: "expired-access",
    refresh_token: "shared-refresh",
    expires_at: 1,
    token_type: "bearer",
    user: { id: "user-1", email: "admin@example.com" }
  };
  storage.set(ADMIN_SESSION_STORAGE_KEY, JSON.stringify(expiredSession));

  let releaseSuccessfulRefresh;
  const successfulRefreshGate = new Promise((resolve) => {
    releaseSuccessfulRefresh = resolve;
  });
  let refreshCalls = 0;
  let capabilityCalls = 0;

  process.env.NEXT_PUBLIC_JUPR_API_BASE_URL = "https://api.staging.test";
  process.env.NEXT_PUBLIC_SUPABASE_URL = "https://auth.staging.test";
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = "public-anon";
  globalThis.CustomEvent = class {
    constructor(type, init = {}) {
      this.type = type;
      this.detail = init.detail;
    }
  };
  globalThis.window = {
    localStorage: {
      getItem(key) {
        return storage.has(key) ? storage.get(key) : null;
      },
      setItem(key, value) {
        storage.set(key, String(value));
      },
      removeItem(key) {
        storage.delete(key);
      }
    },
    dispatchEvent(event) {
      dispatchedEvents.push(event);
      return true;
    }
  };
  globalThis.fetch = async (url) => {
    const target = String(url);
    if (target.includes("grant_type=refresh_token")) {
      refreshCalls += 1;
      if (refreshCalls === 1) {
        await successfulRefreshGate;
        return new Response(
          JSON.stringify({
            access_token: "refreshed-access",
            refresh_token: "refreshed-token",
            expires_in: 3600,
            token_type: "bearer",
            user: { id: "user-1", email: "admin@example.com" }
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        );
      }
      return new Response(
        JSON.stringify({ message: "invalid refresh token" }),
        { status: 401, headers: { "content-type": "application/json" } }
      );
    }
    if (target.includes("/admin/auth/capabilities")) {
      capabilityCalls += 1;
      return new Response(
        JSON.stringify({
          authorized: true,
          user: { email: "admin@example.com" },
          requested_club_id: "tres_palapas",
          assignments: [
            {
              club_id: "tres_palapas",
              role: "club_owner",
              permissions: ["manage_matches"]
            }
          ]
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    throw new Error(`Unexpected request: ${target}`);
  };

  try {
    const firstRestore = restoreAuthorizedAdminSession(undefined, {
      changeSource: "consumer-a"
    });
    const secondRestore = restoreAuthorizedAdminSession(undefined, {
      changeSource: "consumer-b"
    });

    await Promise.resolve();
    assert.equal(
      refreshCalls,
      1,
      "the second consumer must share the first restore instead of attempting a losing refresh"
    );
    releaseSuccessfulRefresh();
    const [firstSession, secondSession] = await Promise.all([
      firstRestore,
      secondRestore
    ]);

    assert.equal(firstSession?.access_token, "refreshed-access");
    assert.equal(secondSession?.access_token, "refreshed-access");
    assert.equal(refreshCalls, 1);
    assert.equal(capabilityCalls, 1);
    assert.equal(
      JSON.parse(storage.get(ADMIN_SESSION_STORAGE_KEY)).access_token,
      "refreshed-access"
    );
    assert.equal(dispatchedEvents.length, 1);
    assert.equal(dispatchedEvents[0].detail.source, "consumer-a");
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
    if (originalCustomEvent === undefined) delete globalThis.CustomEvent;
    else globalThis.CustomEvent = originalCustomEvent;
    globalThis.fetch = originalFetch;
    for (const [name, value] of Object.entries({
      NEXT_PUBLIC_JUPR_API_BASE_URL: originalEnv.apiBase,
      NEXT_PUBLIC_SUPABASE_URL: originalEnv.supabaseUrl,
      NEXT_PUBLIC_SUPABASE_ANON_KEY: originalEnv.supabaseAnonKey
    })) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
});
