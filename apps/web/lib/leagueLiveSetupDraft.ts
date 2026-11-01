// Setup is saved before a League Live session or court preview exists. Keep
// drafts separate by operator, club, and league; never persist auth tokens.
export function leagueLiveSetupDraftKey(clubId: string, operatorId: string, leagueName: string): string {
  return operatorId && leagueName
    ? `jupr-league-live-setup:${JSON.stringify([clubId, operatorId, leagueName])}`
    : "";
}

export type SetupDraftSaveState = "saved" | "tab_only" | "unavailable";
type Envelope<T> = { version: 1; savedAt: string; data: T };

export function readLeagueLiveSetupDraft<T>(key: string, validate: (value: unknown) => value is T): Envelope<T> | null {
  if (!key) return null;
  // If persistent storage was full, a newer tab copy may exist. Read both.
  const candidates: Envelope<T>[] = [];
  for (const area of ["sessionStorage", "localStorage"] as const) {
    try {
      const value = JSON.parse(globalThis[area]?.getItem(key) || "null") as Envelope<unknown> | null;
      if (value?.version === 1 && Number.isFinite(Date.parse(value.savedAt)) && validate(value.data)) {
        candidates.push(value as Envelope<T>);
      }
    } catch { /* A corrupt or unavailable store must not prevent setup. */ }
  }
  return candidates.sort((a, b) => b.savedAt.localeCompare(a.savedAt))[0] || null;
}

export function writeLeagueLiveSetupDraft<T>(key: string, data: T): SetupDraftSaveState {
  if (!key) return "unavailable";
  const raw = JSON.stringify({ version: 1, savedAt: new Date().toISOString(), data } satisfies Envelope<T>);
  try {
    if (!globalThis.localStorage) throw new Error("Storage unavailable");
    globalThis.localStorage.setItem(key, raw);
    try { globalThis.sessionStorage?.removeItem(key); } catch { /* Persistent copy is safe. */ }
    return "saved";
  } catch {
    try {
      if (!globalThis.sessionStorage) throw new Error("Storage unavailable");
      globalThis.sessionStorage.setItem(key, raw);
      return "tab_only";
    } catch { return "unavailable"; }
  }
}

export function clearLeagueLiveSetupDraft(key: string): boolean {
  if (!key) return true;
  let cleared = true;
  for (const area of ["localStorage", "sessionStorage"] as const) {
    try { globalThis[area]?.removeItem(key); } catch { cleared = false; }
  }
  return cleared;
}
