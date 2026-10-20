export type PendingGeneratorStart = {
  status: "pending";
  body: Record<string, unknown> & { idempotency_key: string };
  hubId: string | null;
};

export type CreatedGeneratorStart = {
  status: "created";
  sessionKey: string;
  roundNumber: number;
  hubId: string | null;
};

export type GeneratorStartRecovery = PendingGeneratorStart | CreatedGeneratorStart;

export function readGeneratorStart(key: string): GeneratorStartRecovery | null {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(key) || "null");
    if (!value || value.version !== 1 || !(value.hubId === null || typeof value.hubId === "string")) return null;
    if (value.status === "pending" && value.body && typeof value.body.idempotency_key === "string") return value;
    if (value.status === "created" && typeof value.sessionKey === "string" && value.sessionKey && Number.isInteger(value.roundNumber) && value.roundNumber > 0) return value;
  } catch { /* Missing or unavailable browser storage has no recoverable request. */ }
  return null;
}

export function writeGeneratorStart(key: string, value: GeneratorStartRecovery | null): boolean {
  try {
    if (value) window.sessionStorage.setItem(key, JSON.stringify({ version: 1, ...value }));
    else window.sessionStorage.removeItem(key);
    return true;
  } catch { return false; }
}

export class GeneratorRequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); this.name = "GeneratorRequestError"; }
}

export function isRejectedGeneratorStart(error: unknown): boolean {
  // A 409 may mean a domain write succeeded but its response needs recovery.
  return error instanceof GeneratorRequestError && [400, 401, 403, 404, 422].includes(error.status);
}

export function generatorStartFailure(error: unknown, recovering = false): string {
  if (recovering) return "This saved start still needs confirmation. Sign in again if needed, then choose Retry this start to check the same request. Do not start another copy.";
  if (error instanceof Error && /No domain mutation was attempted/i.test(error.message)) {
    return "The session could not be started. Your draft is saved. Choose Retry this start to try again.";
  }
  if (isRejectedGeneratorStart(error)) return `${error instanceof Error ? error.message : "The start was rejected."} Your draft is saved. Review the setup and try again.`;
  return "We could not confirm whether this session started. Your draft and exact request are saved. Choose Retry this start to check the same request without creating another session.";
}
