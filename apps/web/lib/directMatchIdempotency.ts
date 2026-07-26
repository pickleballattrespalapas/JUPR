"use client";

type PendingDirectMatchWrite = {
  fingerprint: string;
  idempotencyKey: string;
};

const pendingDirectMatchWrites = new Map<string, PendingDirectMatchWrite>();

function newIdempotencyKey(): string {
  if (
    typeof globalThis.crypto !== "undefined"
    && typeof globalThis.crypto.randomUUID === "function"
  ) {
    return `direct-match:${globalThis.crypto.randomUUID()}`;
  }
  return `direct-match:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

function storageKey(scope: string): string {
  return `jupr-direct-match-write:${scope}`;
}

export function directMatchIdempotencyKey(
  scope: string,
  request: Record<string, unknown>
): string {
  const fingerprint = JSON.stringify(request);
  const key = storageKey(scope);
  const memoryPending = pendingDirectMatchWrites.get(key);
  if (memoryPending?.fingerprint === fingerprint) {
    return memoryPending.idempotencyKey;
  }
  try {
    const raw = globalThis.sessionStorage?.getItem(key);
    if (raw) {
      const pending = JSON.parse(raw) as Partial<PendingDirectMatchWrite>;
      if (
        pending.fingerprint === fingerprint
        && typeof pending.idempotencyKey === "string"
      ) {
        pendingDirectMatchWrites.set(key, {
          fingerprint,
          idempotencyKey: pending.idempotencyKey
        });
        return pending.idempotencyKey;
      }
    }
  } catch {
    // A browser can block session storage; the in-flight request is still safe.
  }

  const idempotencyKey = newIdempotencyKey();
  pendingDirectMatchWrites.set(key, { fingerprint, idempotencyKey });
  try {
    globalThis.sessionStorage?.setItem(
      key,
      JSON.stringify({ fingerprint, idempotencyKey })
    );
  } catch {
    // Continue with the generated key for this request.
  }
  return idempotencyKey;
}

export function clearDirectMatchIdempotencyKey(
  scope: string,
  idempotencyKey: string
): void {
  const key = storageKey(scope);
  if (pendingDirectMatchWrites.get(key)?.idempotencyKey === idempotencyKey) {
    pendingDirectMatchWrites.delete(key);
  }
  try {
    const raw = globalThis.sessionStorage?.getItem(key);
    if (!raw) return;
    const pending = JSON.parse(raw) as Partial<PendingDirectMatchWrite>;
    if (pending.idempotencyKey === idempotencyKey) {
      globalThis.sessionStorage?.removeItem(key);
    }
  } catch {
    // The confirmed server receipt is authoritative even if cleanup is blocked.
  }
}
