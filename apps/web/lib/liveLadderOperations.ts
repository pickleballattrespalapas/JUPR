export type OperationKeyRegistry = Record<string, string>;

function randomKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `fallback-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function idempotencyKeyFor(registry: OperationKeyRegistry, scope: string): string {
  if (!registry[scope]) registry[scope] = randomKey();
  return registry[scope];
}

export function rotateIdempotencyKey(registry: OperationKeyRegistry, scope: string): void {
  delete registry[scope];
}

export async function deriveLiveLadderOperationKey(input: {
  clubId: string;
  surface: "challenge_ladder" | "moneyball" | "jupr_live_admin";
  operationType: string;
  entityId: string;
  idempotencyKey: string;
}): Promise<string> {
  const scope = [input.clubId, input.surface, input.operationType, input.entityId, input.idempotencyKey]
    .map((value) => String(value).trim())
    .join("\n");
  const bytes = new TextEncoder().encode(scope);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}
