const pendingRoundRobinGenerations = new Map();

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function storageKey(scope) {
  return `jupr_tournament_ops_round_robin_generate_pending_v1:${scope}`;
}

function canonicalReviewedRequest(request) {
  const teamVersions = (Array.isArray(request?.expected_team_versions)
    ? request.expected_team_versions
    : [])
    .map((row) => ({
      id: String(row?.id || ""),
      updated_at: String(row?.updated_at || "")
    }))
    .sort((left, right) => (
      left.id.localeCompare(right.id)
      || left.updated_at.localeCompare(right.updated_at)
    ));

  return {
    expected_state_fingerprint: String(request?.expected_state_fingerprint || ""),
    expected_draw_updated_at: String(request?.expected_draw_updated_at || ""),
    expected_team_versions: teamVersions
  };
}

export function roundRobinGenerationReviewedFingerprint(request) {
  return JSON.stringify(canonicalReviewedRequest(request));
}

function sessionStorageFrom(options) {
  if (Object.prototype.hasOwnProperty.call(options, "storage")) return options.storage;
  return globalThis.sessionStorage;
}

function createCanonicalUuid(options) {
  const createUuid = options.createUuid || globalThis.crypto?.randomUUID?.bind(globalThis.crypto);
  if (!createUuid) {
    throw new Error("Secure UUID generation is unavailable in this browser.");
  }
  const idempotencyKey = createUuid();
  if (!UUID_PATTERN.test(idempotencyKey)) {
    throw new Error("Secure UUID generation returned an invalid operation key.");
  }
  return idempotencyKey;
}

export function stableRoundRobinGenerationIdempotencyKey(scope, request, options = {}) {
  const fingerprint = roundRobinGenerationReviewedFingerprint(request);
  const key = storageKey(scope);
  let storage;

  try {
    storage = sessionStorageFrom(options);
    const raw = storage?.getItem(key);
    if (raw) {
      const pending = JSON.parse(raw);
      if (
        pending?.version === 1
        && pending.fingerprint === fingerprint
        && typeof pending.idempotencyKey === "string"
        && UUID_PATTERN.test(pending.idempotencyKey)
      ) {
        pendingRoundRobinGenerations.set(key, pending);
        return pending.idempotencyKey;
      }
    }
  } catch {
    // Browser storage can be blocked; in-memory reuse still protects this page.
  }

  const memoryPending = pendingRoundRobinGenerations.get(key);
  if (memoryPending?.fingerprint === fingerprint && UUID_PATTERN.test(memoryPending.idempotencyKey)) {
    return memoryPending.idempotencyKey;
  }

  const pending = {
    version: 1,
    fingerprint,
    idempotencyKey: createCanonicalUuid(options)
  };
  pendingRoundRobinGenerations.set(key, pending);
  try {
    storage = storage ?? sessionStorageFrom(options);
    storage?.setItem(key, JSON.stringify(pending));
  } catch {
    // Keep the generated key in memory for the lifetime of this page.
  }
  return pending.idempotencyKey;
}

export function clearRoundRobinGenerationIdempotencyKey(scope, idempotencyKey, options = {}) {
  const key = storageKey(scope);
  if (pendingRoundRobinGenerations.get(key)?.idempotencyKey === idempotencyKey) {
    pendingRoundRobinGenerations.delete(key);
  }
  try {
    const storage = sessionStorageFrom(options);
    const raw = storage?.getItem(key);
    if (!raw) return;
    const pending = JSON.parse(raw);
    if (pending?.idempotencyKey === idempotencyKey) storage?.removeItem(key);
  } catch {
    // A conclusive server result remains authoritative if cleanup is blocked.
  }
}
