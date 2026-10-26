// Generator recovery must survive closing a tab. Migrate the old tab-only
// records on first use without ever requiring storage to render a session.
export function readGeneratorBrowserValue(key: string): string | null {
  try {
    const value = localStorage.getItem(key);
    if (value !== null) return value;
  } catch { /* Storage may be blocked. Try the existing tab record. */ }
  try {
    const value = sessionStorage.getItem(key);
    if (value !== null) {
      try {
        localStorage.setItem(key, value);
        sessionStorage.removeItem(key);
      } catch { /* Keep the old record until durable storage is available. */ }
    }
    return value;
  } catch { return null; }
}

export function writeGeneratorBrowserValue(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    try { sessionStorage.removeItem(key); } catch { /* Durable copy is saved. */ }
    return true;
  } catch {
    // A full local store may still allow reads and removals. Do not let its old
    // value shadow the newer tab fallback on the next recovery attempt.
    try { localStorage.removeItem(key); } catch { /* Confirm the fallback below. */ }
    try {
      sessionStorage.setItem(key, value);
      return readGeneratorBrowserValue(key) === value;
    } catch { return false; }
  }
}

export function removeGeneratorBrowserValue(key: string): boolean {
  // Attempt both even if one storage area is blocked.
  let removed = false;
  try { localStorage.removeItem(key); removed = true; } catch { /* Try the tab copy too. */ }
  try { sessionStorage.removeItem(key); removed = true; } catch { /* May be unavailable. */ }
  return removed && readGeneratorBrowserValue(key) === null;
}

function editKey(clubId: string, sessionKey: string, shared: boolean): string {
  return `${shared ? "shared-generator-edit" : "public-generator-edit"}:${clubId}:${sessionKey}`;
}

export function readGeneratorEditToken(clubId: string, sessionKey: string, shared = false): string {
  const key = editKey(clubId, sessionKey, shared);
  if (!shared) return readGeneratorBrowserValue(key) || "";
  try { return sessionStorage.getItem(key) || ""; } catch { return ""; }
}

// Shared scorekeepers remain tab-scoped. Only the original organizer's access
// persists on this browser, and the API still validates every write.
export function rememberGeneratorEditToken(clubId: string, sessionKey: string, token: string, shared = false): boolean {
  const key = editKey(clubId, sessionKey, shared);
  if (shared) {
    try { sessionStorage.setItem(key, token); return true; } catch { return false; }
  }
  writeGeneratorBrowserValue(key, token);
  try { return localStorage.getItem(key) === token; } catch { return false; }
}

export function forgetGeneratorEditToken(clubId: string, sessionKey: string, shared = false): void {
  const key = editKey(clubId, sessionKey, shared);
  if (!shared) removeGeneratorBrowserValue(key);
  else { try { sessionStorage.removeItem(key); } catch { /* Already unavailable. */ } }
}
