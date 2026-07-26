import { expect, test, type APIRequestContext, type APIResponse } from "@playwright/test";
import { bootstrapStagingContext, expectedApiOrigin } from "./support/staging";

const adminToken = String(process.env.STAGING_ADMIN_BEARER_TOKEN || "").trim();
const clubId = String(process.env.JUPR_MATCH_EXCLUSION_FIXTURE_CLUB_ID || "").trim();
const duplicateKeepIdRaw = String(process.env.JUPR_MATCH_EXCLUSION_DUPLICATE_KEEP_ID || "").trim();
const duplicateTargetIdRaw = String(process.env.JUPR_MATCH_EXCLUSION_DUPLICATE_TARGET_ID || "").trim();
const duplicateTargetVersionRaw = String(
  process.env.JUPR_MATCH_EXCLUSION_DUPLICATE_TARGET_ROW_VERSION || ""
).trim();
const distinctMatchIdRaw = String(process.env.JUPR_MATCH_EXCLUSION_DISTINCT_MATCH_ID || "").trim();
const distinctVersionRaw = String(process.env.JUPR_MATCH_EXCLUSION_DISTINCT_ROW_VERSION || "").trim();
const staleKey = String(process.env.JUPR_MATCH_EXCLUSION_STALE_IDEMPOTENCY_KEY || "").trim();
const duplicateKey = String(process.env.JUPR_MATCH_EXCLUSION_DUPLICATE_IDEMPOTENCY_KEY || "").trim();
const directKey = String(process.env.JUPR_MATCH_EXCLUSION_DIRECT_IDEMPOTENCY_KEY || "").trim();
const allowMutationEvidence = /^(1|true|yes|on)$/i.test(
  String(process.env.JUPR_MATCH_EXCLUSION_ALLOW_MUTATION_E2E || "")
);

const fixtureSource = "staging_parity_match_exclusion_recovery";
const fullReset = "ALL (Full System Reset)";
const maxResponseBytes = 512 * 1024;
const maxDiagnosticCharacters = 2_000;
const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const sensitiveKeyPattern =
  /(^|_)(authorization|token|secret|cookie|password|api_key|service_role_key)($|_)/i;

type JsonObject = Record<string, unknown>;

function requiredPositiveInteger(raw: string, name: string): number {
  if (!/^[1-9][0-9]*$/.test(raw)) {
    throw new Error(`${name} must be a positive integer fixture identifier.`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} is outside the safe integer range.`);
  }
  return value;
}

function requireFixtureContract(): {
  duplicateKeepId: number;
  duplicateTargetId: number;
  duplicateTargetVersion: number;
  distinctMatchId: number;
  distinctVersion: number;
} {
  if (!allowMutationEvidence) {
    throw new Error("The isolated match-exclusion mutation gate is not enabled.");
  }
  if (!adminToken || !clubId || clubId === "tres_palapas") {
    throw new Error("An isolated fixture club and prepared admin bearer are required.");
  }
  for (const [name, value] of [
    ["stale idempotency key", staleKey],
    ["duplicate idempotency key", duplicateKey],
    ["direct idempotency key", directKey]
  ] as const) {
    if (!uuidPattern.test(value)) throw new Error(`${name} is invalid.`);
  }
  if (new Set([staleKey, duplicateKey, directKey]).size !== 3) {
    throw new Error("Fixture request identities must be distinct.");
  }
  return {
    duplicateKeepId: requiredPositiveInteger(duplicateKeepIdRaw, "duplicate keep ID"),
    duplicateTargetId: requiredPositiveInteger(duplicateTargetIdRaw, "duplicate target ID"),
    duplicateTargetVersion: requiredPositiveInteger(
      duplicateTargetVersionRaw,
      "duplicate target row version"
    ),
    distinctMatchId: requiredPositiveInteger(distinctMatchIdRaw, "distinct match ID"),
    distinctVersion: requiredPositiveInteger(distinctVersionRaw, "distinct match row version")
  };
}

function authHeaders(): Record<string, string> {
  return {
    Authorization: `Bearer ${adminToken}`,
    Accept: "application/json"
  };
}

function matchLogUrl(matchId?: number): string {
  const query = new URLSearchParams({ limit: "20" });
  if (matchId != null) query.set("match_id", String(matchId));
  return `${expectedApiOrigin}/admin/clubs/${encodeURIComponent(clubId)}/match-log?${query}`;
}

function exclusionUrl(): string {
  return `${expectedApiOrigin}/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclude`;
}

function duplicateCleanupUrl(): string {
  return `${expectedApiOrigin}/admin/clubs/${encodeURIComponent(clubId)}/match-log/duplicates/cleanup`;
}

function operationUrl(operationId: string): string {
  return `${expectedApiOrigin}/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclusions/${encodeURIComponent(operationId)}`;
}

async function boundedJson(response: APIResponse, label: string): Promise<JsonObject> {
  const contentType = response.headers()["content-type"] || "";
  expect(contentType, `${label} did not return JSON`).toContain("application/json");
  const body = await response.body();
  expect(body.byteLength, `${label} returned an oversized body`).toBeLessThanOrEqual(maxResponseBytes);
  let payload: unknown;
  try {
    payload = JSON.parse(body.toString("utf-8"));
  } catch {
    throw new Error(`${label} returned invalid JSON.`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(`${label} did not return a JSON object.`);
  }
  return payload as JsonObject;
}

function diagnosticJson(payload: JsonObject): string {
  const serialized = JSON.stringify(payload, (key, value: unknown) => {
    const normalizedKey = key
      .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
      .replace(/[-.\s]+/g, "_")
      .toLowerCase();
    if (sensitiveKeyPattern.test(normalizedKey)) return "[redacted]";
    if (typeof value !== "string") return value;
    return value
      .replace(/Bearer\s+\S+/gi, "Bearer [redacted]")
      .replace(/\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b/g, "[redacted-token]");
  });
  return serialized.slice(0, maxDiagnosticCharacters);
}

async function loadActiveMatches(request: APIRequestContext): Promise<JsonObject[]> {
  const response = await request.get(matchLogUrl(), {
    headers: authHeaders(),
    maxRedirects: 0,
    failOnStatusCode: false
  });
  expect(response.status(), "Match Log fixture readback failed").toBe(200);
  const payload = await boundedJson(response, "Match Log fixture readback");
  expect(payload.enabled).toBe(true);
  expect(payload.apply_enabled).toBe(true);
  expect(Array.isArray(payload.matches)).toBe(true);
  return payload.matches as JsonObject[];
}

async function loadActiveMatch(
  request: APIRequestContext,
  matchId: number
): Promise<JsonObject> {
  const response = await request.get(matchLogUrl(matchId), {
    headers: authHeaders(),
    maxRedirects: 0,
    failOnStatusCode: false
  });
  expect(response.status(), "Exact Match Log readback failed").toBe(200);
  const payload = await boundedJson(response, "Exact Match Log readback");
  expect(payload.matches).toEqual(expect.any(Array));
  const matches = payload.matches as JsonObject[];
  expect(matches).toHaveLength(1);
  expect(matches[0].id).toBe(matchId);
  expect(Number(matches[0].row_version)).toBeGreaterThanOrEqual(1);
  return matches[0];
}

function assertTerminalOperation(
  payload: JsonObject,
  {
    operationMode,
    excludedId,
    idempotent
  }: {
    operationMode: "duplicates_cleaned" | "matches_excluded";
    excludedId: number;
    idempotent: boolean;
  }
): string {
  expect(payload.ok).toBe(true);
  expect(payload.atomic).toBe(true);
  expect(payload.mode).toBe(operationMode);
  expect(payload.operation_status).toBe("succeeded");
  expect(payload.recovery_stage).toBeNull();
  expect(payload.idempotent).toBe(idempotent);
  expect(payload.excluded_count).toBe(1);
  expect(payload.excluded_ids).toEqual([excludedId]);
  expect(payload.affected_player_ids).toEqual(expect.any(Array));
  expect(new Set(payload.affected_player_ids as number[]).size).toBe(4);
  expect(payload.replay_job_id).toMatch(uuidPattern);
  expect(payload.replay_status).toBe("succeeded");
  expect(payload.replay_result).toEqual(
    expect.objectContaining({
      target_reset: fullReset,
      players_updated: true,
      activity_players_updated: 4,
      activity_players_with_matches: 4,
      activity_players_without_matches: 0,
      singles_replay_supported: true
    })
  );
  expect(payload.badge_reconcile).toEqual(
    expect.objectContaining({
      ok: true,
      contract_version: expect.any(String),
      player_ids: expect.arrayContaining(payload.affected_player_ids as number[]),
      processed_player_ids: expect.arrayContaining(payload.affected_player_ids as number[])
    })
  );
  expect(payload.operation_id).toMatch(uuidPattern);
  return String(payload.operation_id);
}

test.describe("atomic Match Log exclusion/recovery staging evidence", () => {
  test.describe.configure({ mode: "serial" });

  test.beforeEach(async ({ context }) => {
    await bootstrapStagingContext(context);
  });

  test("isolated CAS, duplicate cleanup, replay, badges, retry, exclusion, and recovery are durable", async ({
    request
  }) => {
    const {
      duplicateKeepId,
      duplicateTargetId,
      duplicateTargetVersion,
      distinctMatchId,
      distinctVersion
    } = requireFixtureContract();
    expect(new Set([duplicateKeepId, duplicateTargetId, distinctMatchId]).size).toBe(3);

    const initialMatches = await loadActiveMatches(request);
    expect(initialMatches.map((row) => Number(row.id)).sort((a, b) => a - b)).toEqual(
      [duplicateKeepId, duplicateTargetId, distinctMatchId].sort((a, b) => a - b)
    );
    expect(Number((await loadActiveMatch(request, duplicateTargetId)).row_version)).toBe(
      duplicateTargetVersion
    );
    expect(Number((await loadActiveMatch(request, distinctMatchId)).row_version)).toBe(
      distinctVersion
    );

    const staleResponse = await request.post(exclusionUrl(), {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false,
      data: {
        targets: [
          {
            match_id: distinctMatchId,
            expected_row_version: distinctVersion + 1
          }
        ],
        idempotency_key: staleKey,
        confirmation_text: "DELETE",
        note: `Disposable staging fixture ${clubId}`,
        source: fixtureSource
      }
    });
    const stalePayload = await boundedJson(staleResponse, "Stale match exclusion");
    expect(
      staleResponse.status(),
      `Stale match exclusion was not rejected. Response: ${diagnosticJson(stalePayload)}`
    ).toBe(409);
    expect(stalePayload.detail).toEqual(
      expect.objectContaining({ code: "MATCH_EXCLUSION_STALE" })
    );
    const afterStale = await loadActiveMatches(request);
    expect(afterStale).toHaveLength(3);
    expect(Number((await loadActiveMatch(request, distinctMatchId)).row_version)).toBe(
      distinctVersion
    );

    const duplicateRequest = {
      targets: [
        {
          match_id: duplicateTargetId,
          expected_row_version: duplicateTargetVersion
        }
      ],
      idempotency_key: duplicateKey,
      confirmation_text: "DELETE",
      note: `Disposable staging fixture ${clubId}`,
      source: fixtureSource
    };
    const duplicateResponse = await request.post(duplicateCleanupUrl(), {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false,
      data: duplicateRequest
    });
    expect(duplicateResponse.status(), "Atomic duplicate cleanup failed").toBe(200);
    const duplicatePayload = await boundedJson(
      duplicateResponse,
      "Atomic duplicate cleanup"
    );
    const duplicateOperationId = assertTerminalOperation(duplicatePayload, {
      operationMode: "duplicates_cleaned",
      excludedId: duplicateTargetId,
      idempotent: false
    });
    expect(duplicatePayload.deleted_count).toBe(1);
    expect(duplicatePayload.deleted_ids).toEqual([duplicateTargetId]);

    const duplicateRetryResponse = await request.post(duplicateCleanupUrl(), {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false,
      data: duplicateRequest
    });
    expect(duplicateRetryResponse.status(), "Exact duplicate retry failed").toBe(200);
    const duplicateRetry = await boundedJson(
      duplicateRetryResponse,
      "Exact duplicate retry"
    );
    const retryOperationId = assertTerminalOperation(duplicateRetry, {
      operationMode: "duplicates_cleaned",
      excludedId: duplicateTargetId,
      idempotent: true
    });
    expect(retryOperationId).toBe(duplicateOperationId);
    expect(duplicateRetry.replay_job_id).toBe(duplicatePayload.replay_job_id);

    const operationResponse = await request.get(operationUrl(duplicateOperationId), {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false
    });
    expect(operationResponse.status(), "Exclusion operation lookup failed").toBe(200);
    const operationPayload = await boundedJson(
      operationResponse,
      "Exclusion operation lookup"
    );
    expect(operationPayload).toEqual(
      expect.objectContaining({
        ok: true,
        atomic: true,
        operation_id: duplicateOperationId,
        operation_status: "succeeded",
        recovery_stage: null,
        excluded_ids: [duplicateTargetId],
        replay_status: "succeeded",
        error_text: null,
        targets: [
          {
            match_id: duplicateTargetId,
            expected_row_version: duplicateTargetVersion
          }
        ]
      })
    );
    expect(operationPayload.badge_ids).toEqual(expect.any(Array));
    expect((operationPayload.badge_ids as unknown[]).length).toBeGreaterThan(0);
    expect(operationPayload.badge_contract_version).toEqual(expect.any(String));
    expect(operationPayload.finished_at).toEqual(expect.any(String));

    const freshDistinct = await loadActiveMatch(request, distinctMatchId);
    const freshDistinctVersion = Number(freshDistinct.row_version);
    expect(Number.isSafeInteger(freshDistinctVersion)).toBe(true);
    expect(freshDistinctVersion).toBeGreaterThanOrEqual(distinctVersion);
    const directRequest = {
      targets: [
        {
          match_id: distinctMatchId,
          expected_row_version: freshDistinctVersion
        }
      ],
      idempotency_key: directKey,
      confirmation_text: "DELETE",
      note: `Disposable staging fixture ${clubId}`,
      source: fixtureSource
    };
    const directResponse = await request.post(exclusionUrl(), {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false,
      data: directRequest
    });
    expect(directResponse.status(), "Atomic direct match exclusion failed").toBe(200);
    const directPayload = await boundedJson(
      directResponse,
      "Atomic direct match exclusion"
    );
    const directOperationId = assertTerminalOperation(directPayload, {
      operationMode: "matches_excluded",
      excludedId: distinctMatchId,
      idempotent: false
    });
    expect(directOperationId).not.toBe(duplicateOperationId);

    const recoveryResponse = await request.post(`${operationUrl(directOperationId)}/recover`, {
      headers: authHeaders(),
      maxRedirects: 0,
      failOnStatusCode: false,
      data: {
        confirmation_text: "RECOVER",
        source: fixtureSource
      }
    });
    expect(recoveryResponse.status(), "Completed-operation recovery retry failed").toBe(200);
    const recoveryPayload = await boundedJson(
      recoveryResponse,
      "Completed-operation recovery retry"
    );
    expect(recoveryPayload).toEqual(
      expect.objectContaining({
        ok: true,
        atomic: true,
        mode: "already_recovered",
        operation_id: directOperationId,
        operation_status: "succeeded",
        recovery_stage: null,
        idempotent: true,
        excluded_ids: [distinctMatchId],
        replay_status: "succeeded"
      })
    );
    expect(recoveryPayload.replay_job_id).toBe(directPayload.replay_job_id);

    const finalMatches = await loadActiveMatches(request);
    expect(finalMatches).toHaveLength(1);
    expect(Number(finalMatches[0].id)).toBe(duplicateKeepId);
    expect(finalMatches[0].deleted_at ?? null).toBeNull();
    expect(Number(finalMatches[0].row_version)).toBeGreaterThanOrEqual(1);
  });
});
