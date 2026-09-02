export type RoundRobinGenerationReviewedRequest = {
  expected_state_fingerprint?: unknown;
  expected_draw_updated_at?: unknown;
  expected_team_versions?: readonly {
    id?: unknown;
    updated_at?: unknown;
  }[];
};

export type RoundRobinGenerationIdempotencyOptions = {
  storage?: Pick<Storage, "getItem" | "setItem" | "removeItem"> | null;
  createUuid?: () => string;
};

export function roundRobinGenerationReviewedFingerprint(
  request: RoundRobinGenerationReviewedRequest
): string;

export function stableRoundRobinGenerationIdempotencyKey(
  scope: string,
  request: RoundRobinGenerationReviewedRequest,
  options?: RoundRobinGenerationIdempotencyOptions
): string;

export function clearRoundRobinGenerationIdempotencyKey(
  scope: string,
  idempotencyKey: string,
  options?: RoundRobinGenerationIdempotencyOptions
): void;
