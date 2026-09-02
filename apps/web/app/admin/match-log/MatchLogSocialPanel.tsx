"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import type { AdminSocialMatchLogResponse, AdminSocialMatchLogRow, AdminMatchLogWriteResult, AdminSocialMatchOperationResponse } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogSocialPanelProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
};

type SocialEditState = {
  eventName: string;
  playedOn: string;
  roundNumber: string;
  courtNumber: string;
  miniRoundNumber: string;
  scoreT1: string;
  scoreT2: string;
};

type Feedback = {
  tone: "success" | "error";
  text: string;
};

type LoadRowsOptions = {
  announce?: "load" | "refresh" | false;
  preferredId?: string;
};

type SocialSaveRequest = {
  socialMatchId: string;
  contextKey: string;
  changedFields: string[];
  body: Record<string, unknown> & {
    expected_current: Record<string, unknown>;
    idempotency_key: string;
    confirmation_text: string;
  };
};

type SocialWriteRecovery = {
  operationKey: string;
  socialMatchId: string;
  changedFields: string[];
  status: string;
  message: string;
};

type StoredSocialWriteRecovery = SocialWriteRecovery & { version: 1 };

class SocialApiRequestError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "SocialApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function rowId(row: AdminSocialMatchLogRow | null): string {
  return String(row?.social_match_id ?? row?.id ?? "");
}

function operationKey(action: string, entityId: string): string {
  const suffix =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${action}:${entityId}:${suffix}`;
}

function socialApiErrorMessage(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    const message = String((detail as { message?: unknown }).message || "").trim();
    if (message) return message;
  }
  return `API error (${status})`;
}

function isUncertainSocialError(error: unknown): boolean {
  if (!(error instanceof SocialApiRequestError)) return true;
  if (error.detail && typeof error.detail === "object") {
    const explicit = error.detail as { kind?: unknown; recovery_required?: unknown };
    if (explicit.kind === "failed" && explicit.recovery_required !== true) return false;
  }
  if (error.status >= 500 || [408, 425, 429].includes(error.status)) return true;
  if (!error.detail || typeof error.detail !== "object") return false;
  const detail = error.detail as {
    code?: unknown;
    kind?: unknown;
    recovery_required?: unknown;
  };
  return (
    detail.code === "RECOVERY_REQUIRED" ||
    detail.kind === "uncertain" ||
    detail.recovery_required === true
  );
}

function socialErrorOperationKey(error: unknown, fallback: string): string {
  if (!(error instanceof SocialApiRequestError) || !error.detail || typeof error.detail !== "object") return fallback;
  const operationKeyValue = (error.detail as { operation_key?: unknown }).operation_key;
  return typeof operationKeyValue === "string" && operationKeyValue ? operationKeyValue : fallback;
}

function dateInput(value?: string | null): string {
  if (!value) return "";
  const text = String(value);
  if (/^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function editFromRow(row: AdminSocialMatchLogRow | null): SocialEditState {
  return {
    eventName: row?.event_name || "",
    playedOn: dateInput(row?.played_on || row?.date),
    roundNumber: row?.round_number == null ? "" : String(row.round_number),
    courtNumber: row?.court_number == null ? "" : String(row.court_number),
    miniRoundNumber: row?.mini_round_number == null ? "" : String(row.mini_round_number),
    scoreT1: row?.score_t1 == null ? "" : String(row.score_t1),
    scoreT2: row?.score_t2 == null ? "" : String(row.score_t2)
  };
}

function maybeNumber(value: string, label: string, options?: { minimum?: number }): number | null {
  const cleaned = String(value || "").trim();
  if (!cleaned) return null;
  const parsed = Number(cleaned);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) throw new Error(`${label} must be a whole number.`);
  if (options?.minimum !== undefined && parsed < options.minimum) throw new Error(`${label} must be at least ${options.minimum}.`);
  return parsed;
}

function normalizeEventName(value: unknown): string {
  return String(value ?? "").replace(/\u00a0/g, " ").trim().split(/\s+/).filter(Boolean).join(" ");
}

function buildPatch(row: AdminSocialMatchLogRow, edit: SocialEditState): Record<string, unknown> {
  const patch: Record<string, unknown> = {};

  const eventName = normalizeEventName(edit.eventName);
  const originalEventName = normalizeEventName(row.event_name);
  if (eventName !== originalEventName) {
    if (!eventName) throw new Error("Event name is required.");
    patch.event_name = eventName;
  }

  const playedOn = edit.playedOn;
  const originalPlayedOn = dateInput(row.played_on || row.date);
  if (playedOn !== originalPlayedOn) {
    if (!playedOn) throw new Error("Played on is required.");
    patch.played_on = playedOn;
  }

  const numericFields: Array<{
    key: "round_number" | "court_number" | "mini_round_number" | "score_t1" | "score_t2";
    value: string;
    original: number | null | undefined;
    label: string;
    minimum?: number;
  }> = [
    { key: "round_number", value: edit.roundNumber, original: row.round_number, label: "Round" },
    { key: "court_number", value: edit.courtNumber, original: row.court_number, label: "Court" },
    { key: "mini_round_number", value: edit.miniRoundNumber, original: row.mini_round_number, label: "Mini round" },
    { key: "score_t1", value: edit.scoreT1, original: row.score_t1, label: "Team 1 score", minimum: 0 },
    { key: "score_t2", value: edit.scoreT2, original: row.score_t2, label: "Team 2 score", minimum: 0 }
  ];
  for (const field of numericFields) {
    const nextValue = maybeNumber(field.value, field.label, { minimum: field.minimum });
    const originalValue = field.original == null ? null : Number(field.original);
    if (nextValue === originalValue) continue;
    if (nextValue === null) throw new Error(`${field.label} cannot be cleared.`);
    patch[field.key] = nextValue;
  }
  const changedFields = Object.keys(patch);
  if (changedFields.includes("event_name") && changedFields.some((field) => field !== "event_name")) {
    throw new Error("Update the Club Social event name separately from match fields.");
  }
  return patch;
}

function expectedCurrentForPatch(
  row: AdminSocialMatchLogRow,
  patch: Record<string, unknown>
): Record<string, unknown> {
  const expected: Record<string, unknown> = {};
  for (const field of Object.keys(patch)) {
    if (field === "event_name") {
      expected[field] = normalizeEventName(row.event_name);
    } else if (field === "played_on") {
      expected[field] = dateInput(row.played_on || row.date);
    } else if (
      field === "round_number" ||
      field === "court_number" ||
      field === "mini_round_number" ||
      field === "score_t1" ||
      field === "score_t2"
    ) {
      expected[field] = row[field] == null ? null : Number(row[field]);
    }
  }
  return expected;
}

function resultMessage(result: AdminMatchLogWriteResult | null): string | null {
  if (!result?.ok) return null;
  if (result.mode === "social_match_updated") return `Updated Club Social match ${result.social_match_id || "row"}.`;
  if (result.mode === "social_matches_deleted") return `Deleted ${result.deleted_count ?? 0} Club Social row(s).`;
  return "Operation completed.";
}

function playerLabel(row: AdminSocialMatchLogRow): string {
  return `${row.t1_p1 || "—"} / ${row.t1_p2 || "—"} vs ${row.t2_p1 || "—"} / ${row.t2_p2 || "—"}`;
}

function rowCountMessage(count: number): string {
  return `Loaded ${count} Club Social ${count === 1 ? "row" : "rows"}.`;
}

function refreshCountMessage(count: number): string {
  const refreshedAt = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  return `Refreshed ${count} Club Social ${count === 1 ? "row" : "rows"} at ${refreshedAt}.`;
}

function feedbackColor(feedback: Feedback): string {
  return feedback.tone === "success" ? "#166534" : "#b91c1c";
}

export default function MatchLogSocialPanel({ apiBase, clubId, enabled }: MatchLogSocialPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const sessionIdentity = String(session?.user?.id || session?.user?.email || (accessToken ? "signed-in-admin" : ""));
  const contextKey = enabled && sessionIdentity ? `${sessionIdentity}\u0000${apiBase || ""}\u0000${clubId}` : "";
  const loadGenerationRef = useRef(0);
  const accessTokenRef = useRef(accessToken);
  const contextKeyRef = useRef(contextKey);
  const effectContextKeyRef = useRef("");
  const enabledRef = useRef(enabled);
  accessTokenRef.current = accessToken;
  contextKeyRef.current = contextKey;
  enabledRef.current = enabled;
  const [rows, setRows] = useState<AdminSocialMatchLogRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [edit, setEdit] = useState<SocialEditState>(() => editFromRow(null));
  const [loadingRows, setLoadingRows] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loadFeedback, setLoadFeedback] = useState<Feedback | null>(null);
  const [mutationFeedback, setMutationFeedback] = useState<Feedback | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);
  const [writeRecovery, setWriteRecovery] = useState<SocialWriteRecovery | null>(null);
  const [checkingWriteRecovery, setCheckingWriteRecovery] = useState(false);
  const selectedRow = rows.find((row) => rowId(row) === selectedId) || null;
  const writeRecoveryStorageKey = `jupr-match-log-social-write-recovery:${clubId}`;

  useEffect(() => {
    setWriteRecovery(null);
    try {
      const raw = globalThis.sessionStorage?.getItem(writeRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<StoredSocialWriteRecovery>;
      if (
        stored.version === 1
        && typeof stored.operationKey === "string"
        && typeof stored.socialMatchId === "string"
        && Array.isArray(stored.changedFields)
        && typeof stored.status === "string"
        && typeof stored.message === "string"
      ) {
        setWriteRecovery(stored as StoredSocialWriteRecovery);
      }
    } catch {
      // The in-memory guard remains available if session storage is blocked.
    }
  }, [writeRecoveryStorageKey]);

  function retainWriteRecovery(recovery: SocialWriteRecovery) {
    setWriteRecovery(recovery);
    try {
      globalThis.sessionStorage?.setItem(
        writeRecoveryStorageKey,
        JSON.stringify({ version: 1, ...recovery } satisfies StoredSocialWriteRecovery),
      );
    } catch {
      // The in-memory state still blocks another write in this page session.
    }
  }

  function clearWriteRecovery() {
    setWriteRecovery(null);
    try {
      globalThis.sessionStorage?.removeItem(writeRecoveryStorageKey);
    } catch {
      // A conclusive server response remains authoritative if cleanup is blocked.
    }
  }

  async function loadRows(options: LoadRowsOptions = {}): Promise<boolean> {
    const requestGeneration = loadGenerationRef.current + 1;
    const requestAccessToken = accessTokenRef.current;
    const requestContextKey = contextKeyRef.current;
    loadGenerationRef.current = requestGeneration;
    setLoadFeedback(null);
    setWarnings([]);
    if (!apiBase) {
      setLoadingRows(false);
      setLoadFeedback({ tone: "error", text: "API base URL is not configured." });
      return false;
    }
    if (!enabledRef.current || !requestAccessToken || !requestContextKey) {
      setLoadingRows(false);
      setRows([]);
      setSelectedId("");
      setEdit(editFromRow(null));
      return false;
    }
    setLoadingRows(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social?limit=500`), {
        cache: "no-store",
        headers: { accept: "application/json", Authorization: `Bearer ${requestAccessToken}` }
      });
      const payload = await response.json().catch(() => null) as AdminSocialMatchLogResponse | { detail?: unknown } | null;
      if (
        requestGeneration !== loadGenerationRef.current
        || requestContextKey !== contextKeyRef.current
        || !enabledRef.current
      ) return false;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      const nextRows = Array.isArray((payload as AdminSocialMatchLogResponse).rows) ? (payload as AdminSocialMatchLogResponse).rows : [];
      setRows(nextRows);
      setWarnings((payload as AdminSocialMatchLogResponse).warnings || []);
      const preferredId = options.preferredId ?? selectedId;
      const nextSelected = nextRows.find((row) => rowId(row) === preferredId) || nextRows[0] || null;
      setSelectedId(rowId(nextSelected));
      setEdit(editFromRow(nextSelected));
      if (options.announce !== false) {
        setLoadFeedback({
          tone: "success",
          text: options.announce === "refresh" ? refreshCountMessage(nextRows.length) : rowCountMessage(nextRows.length)
        });
      }
      return true;
    } catch (error) {
      if (
        requestGeneration === loadGenerationRef.current
        && requestContextKey === contextKeyRef.current
        && enabledRef.current
      ) {
        setLoadFeedback({ tone: "error", text: error instanceof Error ? error.message : "Unable to load Club Social rows." });
      }
      return false;
    } finally {
      if (requestGeneration === loadGenerationRef.current) setLoadingRows(false);
    }
  }

  useEffect(() => {
    const contextChanged = effectContextKeyRef.current !== contextKey;
    effectContextKeyRef.current = contextKey;
    if (contextChanged || !contextKey) {
      loadGenerationRef.current += 1;
      setLoadingRows(false);
      setBusy(false);
      setRows([]);
      setSelectedId("");
      setEdit(editFromRow(null));
      setWarnings([]);
      setLoadFeedback(null);
      setMutationFeedback(null);
      setResult(null);
    }
    if (enabled && accessToken && contextKey) void loadRows();
    return () => {
      loadGenerationRef.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, accessToken, apiBase, clubId, contextKey]);

  function selectRow(nextId: string) {
    const row = rows.find((item) => rowId(item) === nextId) || null;
    setSelectedId(nextId);
    setEdit(editFromRow(row));
    setMutationFeedback(null);
    setResult(null);
  }

  function refreshRows() {
    setMutationFeedback(null);
    setResult(null);
    void loadRows({ announce: "refresh", preferredId: selectedId });
  }

  function resetFields() {
    setEdit(editFromRow(selectedRow));
    setMutationFeedback(null);
    setResult(null);
  }

  async function fetchSocialOperation<T>(path: string, init: RequestInit = {}): Promise<T> {
    const mutationAccessToken = accessTokenRef.current;
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!mutationAccessToken) throw new Error("Sign in before inspecting the retained Club Social operation.");
    const response = await fetch(apiUrl(apiBase, path), {
      ...init,
      headers: {
        ...(init.body === undefined ? {} : { "Content-Type": "application/json" }),
        Authorization: `Bearer ${mutationAccessToken}`,
        ...init.headers,
      },
    });
    const payload = await response.json().catch(() => null) as T | { detail?: unknown } | null;
    if (!response.ok) {
      const detail = (payload as { detail?: unknown } | null)?.detail;
      throw new SocialApiRequestError(socialApiErrorMessage(detail, response.status), response.status, detail);
    }
    return payload as T;
  }

  async function reconcileSocialOperation(
    retainedRecovery: SocialWriteRecovery | null = writeRecovery,
  ): Promise<ActionCompletion> {
    if (!retainedRecovery) throw new Error("No Club Social operation is waiting for recovery.");
    const path = `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social/operations/${encodeURIComponent(retainedRecovery.operationKey)}`;
    setCheckingWriteRecovery(true);
    setMutationFeedback(null);
    try {
      const operation = await fetchSocialOperation<AdminSocialMatchOperationResponse>(path);
      let operationStatus = String(operation.status || "unknown");
      let recoveryRequired = operation.recovery_required === true;
      let recovered = operation.result || null;

      if (operationStatus !== "completed" && operationStatus !== "failed") {
        recovered = await fetchSocialOperation<AdminMatchLogWriteResult>(`${path}/reconcile`, {
          method: "POST",
          body: JSON.stringify({
            confirmation_text: "RECONCILE SOCIAL MATCH",
            source: "next_match_log_social_operation_reconcile",
          }),
        });
        operationStatus = recovered.ok === false ? String(recovered.status || "failed") : "completed";
        recoveryRequired = recovered.recovery_required === true;
      }

      if (operationStatus === "completed" && recovered?.ok !== false) {
        await loadRows({ announce: false, preferredId: retainedRecovery.socialMatchId });
        const authoritativeResult = recovered || { ok: true, mode: "social_match_updated", social_match_id: retainedRecovery.socialMatchId };
        setResult(authoritativeResult);
        clearWriteRecovery();
        const successMessage = `${resultMessage(authoritativeResult) || "The Club Social edit completed."} The authoritative row was refreshed.`;
        setMutationFeedback({ tone: "success", text: successMessage });
        return actionSuccess("Club Social operation reconciled", successMessage);
      }

      if (operationStatus === "failed" && !recoveryRequired) {
        clearWriteRecovery();
        const failedMessage = `Exact Club Social operation ${retainedRecovery.operationKey} is proven failed. Review the current row before submitting a new edit.`;
        setMutationFeedback({ tone: "success", text: failedMessage });
        return actionSuccess("Club Social operation checked", failedMessage);
      }

      const pendingMessage = operation.error || retainedRecovery.message;
      const pending = { ...retainedRecovery, status: operationStatus, message: pendingMessage };
      retainWriteRecovery(pending);
      return actionUncertain(
        "Club Social operation still needs verification",
        `Operation ${retainedRecovery.operationKey} is ${operationStatus.replace(/_/g, " ")}. New Club Social writes remain blocked.`,
        retainedRecovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcileSocialOperation(pending),
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to reconcile the Club Social edit.";
      const pending = { ...retainedRecovery, status: "recovery_required", message: errorMessage };
      retainWriteRecovery(pending);
      setMutationFeedback({ tone: "error", text: `${errorMessage} Operation ${retainedRecovery.operationKey} remains retained; do not submit another edit.` });
      return actionUncertain(
        "Club Social operation still needs verification",
        `${errorMessage} The exact operation reference remains retained.`,
        retainedRecovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcileSocialOperation(pending),
      );
    } finally {
      setCheckingWriteRecovery(false);
    }
  }

  async function executeSocialSave(request: SocialSaveRequest): Promise<ActionCompletion> {
    const mutationAccessToken = accessTokenRef.current;
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!mutationAccessToken || request.contextKey !== contextKeyRef.current) {
      throw new Error("Sign in at /admin/login and reload the selected Club Social row before saving.");
    }
    if (writeRecovery) throw new Error(`Resolve exact operation ${writeRecovery.operationKey} before saving another Club Social edit.`);
    setBusy(true);
    setMutationFeedback(null);
    setResult(null);
    let requestSent = false;
    try {
      requestSent = true;
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social/${encodeURIComponent(request.socialMatchId)}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${mutationAccessToken}` },
        body: JSON.stringify(request.body)
      });
      const payload = await response.json().catch(() => null) as AdminMatchLogWriteResult | { detail?: unknown } | null;
      if (!response.ok) {
        const detail = (payload as { detail?: unknown } | null)?.detail;
        throw new SocialApiRequestError(
          socialApiErrorMessage(detail, response.status),
          response.status,
          detail
        );
      }
      const writeResult = payload as AdminMatchLogWriteResult;
      if (!writeResult.ok) throw new Error("The Club Social edit returned without authoritative success.");
      const successMessage = `${resultMessage(writeResult) || "Saved Club Social row."} Changed ${request.changedFields.length} ${request.changedFields.length === 1 ? "field" : "fields"}: ${request.changedFields.join(", ")}.`;
      if (request.contextKey === contextKeyRef.current && enabledRef.current) {
        await loadRows({ announce: false, preferredId: request.socialMatchId });
        if (request.contextKey === contextKeyRef.current && enabledRef.current) {
          setResult(writeResult);
          setMutationFeedback({ tone: "success", text: successMessage });
        }
      }
      return actionSuccess("Club Social match saved", successMessage);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to save Club Social row.";
      if (requestSent && isUncertainSocialError(error)) {
        const retainedOperationKey = socialErrorOperationKey(error, request.body.idempotency_key);
        const recoveryMessage = `${errorMessage} The exact edit is retained as ${retainedOperationKey}; check and reconcile it before making another edit.`;
        const recovery: SocialWriteRecovery = {
          operationKey: retainedOperationKey,
          socialMatchId: request.socialMatchId,
          changedFields: request.changedFields,
          status: "uncertain",
          message: errorMessage,
        };
        retainWriteRecovery(recovery);
        if (request.contextKey === contextKeyRef.current && enabledRef.current) {
          setMutationFeedback({ tone: "error", text: recoveryMessage });
        }
        return actionUncertain(
          "Club Social edit needs verification",
          recoveryMessage,
          retainedOperationKey,
          "Check and reconcile exact operation",
          () => reconcileSocialOperation(recovery)
        );
      }
      if (request.contextKey === contextKeyRef.current && enabledRef.current) {
        setMutationFeedback({ tone: "error", text: errorMessage });
      }
      throw error;
    } finally {
      if (request.contextKey === contextKeyRef.current) setBusy(false);
    }
  }

  function saveRow(confirmationText: string): Promise<ActionCompletion> {
    if (!selectedRow) throw new Error("Select a Club Social row before saving.");
    if (writeRecovery) throw new Error(`Resolve exact operation ${writeRecovery.operationKey} before saving another Club Social row.`);
    const mutationContextKey = contextKeyRef.current;
    if (!mutationContextKey) throw new Error("Sign in before saving a Club Social row.");
    const socialMatchId = rowId(selectedRow);
    const patch = buildPatch(selectedRow, edit);
    const changedFields = Object.keys(patch);
    if (!changedFields.length) throw new Error("No Club Social changes detected for the selected row.");
    return executeSocialSave({
      socialMatchId,
      contextKey: mutationContextKey,
      changedFields,
      body: {
        ...patch,
        expected_current: expectedCurrentForPatch(selectedRow, patch),
        idempotency_key: operationKey("save-social-match", socialMatchId),
        confirmation_text: confirmationText,
        source: "next_match_log_social_editor"
      }
    });
  }

  async function deleteRow(confirmationText: string): Promise<ActionCompletion> {
    if (!selectedRow) throw new Error("Select a Club Social row before deleting it.");
    if (writeRecovery) throw new Error(`Resolve exact operation ${writeRecovery.operationKey} before deleting a Club Social row.`);
    const mutationAccessToken = accessTokenRef.current;
    const mutationContextKey = contextKeyRef.current;
    setBusy(true);
    setMutationFeedback(null);
    setResult(null);
    try {
      if (!apiBase) throw new Error("API base URL is not configured.");
      if (!mutationAccessToken || !mutationContextKey) throw new Error("Sign in at /admin/login before deleting Club Social rows.");
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/social/delete`), {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${mutationAccessToken}` },
        body: JSON.stringify({ social_match_ids: [rowId(selectedRow)], confirmation_text: confirmationText, source: "next_match_log_social_editor" })
      });
      const payload = await response.json().catch(() => null) as AdminMatchLogWriteResult | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      const writeResult = payload as AdminMatchLogWriteResult;
      if (!writeResult.ok) throw new Error("The Club Social delete did not complete.");
      if (mutationContextKey !== contextKeyRef.current || !enabledRef.current) throw new Error("The admin session changed before the deleted row response was applied.");
      await loadRows({ announce: false });
      if (mutationContextKey !== contextKeyRef.current || !enabledRef.current) throw new Error("The admin session changed before Club Social rows could be refreshed.");
      setResult(writeResult);
      const successMessage = resultMessage(writeResult) || "Deleted Club Social row.";
      setMutationFeedback({ tone: "success", text: successMessage });
      return actionSuccess("Club Social row deleted", successMessage);
    } catch (error) {
      if (mutationContextKey === contextKeyRef.current && enabledRef.current) {
        setMutationFeedback({ tone: "error", text: error instanceof Error ? error.message : "Unable to delete Club Social row." });
      }
      throw error;
    } finally {
      if (mutationContextKey === contextKeyRef.current) setBusy(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Club Social Match Log editor unavailable</h2>
        <p style={{ color: "#475569" }}>Enable the Next Match Log pilot before editing Club Social rows in Next.</p>
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Club Social editor</h2>
      <p style={{ color: "#475569" }}>
        Edit or delete unrated Club Social rows from the Match Log workflow. Rated match history is not changed by this panel.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to load and edit Club Social rows." : sessionLoading ? "Checking admin session…" : "Sign in before editing Club Social rows."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={refreshRows} disabled={loadingRows || busy || !accessToken} style={secondaryButtonStyle}>{loadingRows ? "Loading…" : "Refresh Club Social rows"}</button>
      </p>
      {loadFeedback ? <p role={loadFeedback.tone === "error" ? "alert" : "status"} style={{ color: feedbackColor(loadFeedback), fontWeight: 700 }}>{loadFeedback.text}</p> : null}
      {warnings.length ? <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      {writeRecovery ? (
        <section aria-live="polite" style={{ border: "1px solid #f59e0b", borderRadius: "12px", padding: "0.75rem", background: "#fffbeb", marginBottom: "1rem" }}>
          <h3 style={{ marginTop: 0 }}>Club Social edit needs exact-operation recovery</h3>
          <p style={{ color: "#92400e" }}>Do not save or delete another Club Social row until this exact operation is reconciled.</p>
          <p><strong>Operation key:</strong> <code style={{ overflowWrap: "anywhere" }}>{writeRecovery.operationKey}</code><br /><strong>Last known status:</strong> {writeRecovery.status.replace(/_/g, " ")}</p>
          <p>{writeRecovery.message}</p>
          <button type="button" onClick={() => void reconcileSocialOperation()} disabled={checkingWriteRecovery || !accessToken} style={secondaryButtonStyle}>
            {checkingWriteRecovery ? "Checking and reconciling…" : "Check and reconcile exact operation"}
          </button>
        </section>
      ) : null}
      {rows.length ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          <label><strong>Club Social row</strong><br />
            <select value={selectedId} onChange={(event) => selectRow(event.target.value)} style={inputStyle}>
              {rows.map((row) => (
                <option key={rowId(row)} value={rowId(row)}>
                  {row.event_name || "Club Social"} · {row.played_on || row.date || "—"} · {playerLabel(row)} · {row.score_t1 ?? 0}-{row.score_t2 ?? 0}
                </option>
              ))}
            </select>
          </label>
          {selectedRow ? (
            <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
              <strong>{selectedRow.event_name || "Club Social"}</strong>
              <p style={{ margin: "0.35rem 0", color: "#475569" }}>{playerLabel(selectedRow)} · {selectedRow.score_t1 ?? 0}-{selectedRow.score_t2 ?? 0}</p>
              <p style={{ margin: 0, color: "#64748b" }}>Status: {selectedRow.status || "—"} · Submission: {selectedRow.submission_mode || "—"} · Match key: {selectedRow.match_key || "—"}</p>
            </div>
          ) : null}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
            <label><strong>Event name</strong><br /><input value={edit.eventName} onChange={(event) => setEdit((current) => ({ ...current, eventName: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Played on</strong><br /><input type="date" value={edit.playedOn} onChange={(event) => setEdit((current) => ({ ...current, playedOn: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Round</strong><br /><input type="number" step="1" value={edit.roundNumber} onChange={(event) => setEdit((current) => ({ ...current, roundNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Court</strong><br /><input type="number" step="1" value={edit.courtNumber} onChange={(event) => setEdit((current) => ({ ...current, courtNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Mini round</strong><br /><input type="number" step="1" value={edit.miniRoundNumber} onChange={(event) => setEdit((current) => ({ ...current, miniRoundNumber: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Team 1 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT1} onChange={(event) => setEdit((current) => ({ ...current, scoreT1: event.target.value }))} style={inputStyle} /></label>
            <label><strong>Team 2 score</strong><br /><input type="number" min="0" step="1" value={edit.scoreT2} onChange={(event) => setEdit((current) => ({ ...current, scoreT2: event.target.value }))} style={inputStyle} /></label>
          </div>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <ConfirmAction
              triggerLabel="Save Club Social row"
              title="Save this Club Social match?"
              description="This compares every changed field with the exact values you loaded, then records the reviewed edit through one durable operation."
              preview={selectedRow ? <p style={{ margin: 0 }}><strong>Match:</strong> {playerLabel(selectedRow)} · {selectedRow.score_t1 ?? 0}-{selectedRow.score_t2 ?? 0}</p> : undefined}
              confirmLabel="Yes, save match"
              confirmationText="SAVE SOCIAL MATCH"
              disabled={busy || !accessToken || !selectedRow || Boolean(writeRecovery)}
              busy={busy}
              onConfirm={saveRow}
            />
            <button type="button" onClick={resetFields} disabled={busy || !selectedRow} style={secondaryButtonStyle}>Reset fields</button>
          </p>
          {mutationFeedback ? <p role={mutationFeedback.tone === "error" ? "alert" : "status"} style={{ color: feedbackColor(mutationFeedback), fontWeight: 700 }}>{mutationFeedback.text}</p> : null}
          {result?.warnings?.length ? <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
          <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "0.5rem 0" }} />
          <h3>Delete Club Social row</h3>
          <p style={{ color: "#475569" }}>This deletes the selected unrated Club Social row only. It does not replay or change rated history.</p>
          <p>
            <ConfirmAction
              triggerLabel="Delete selected Club Social row"
              title="Delete this Club Social row?"
              description={<>This permanently deletes the selected unrated Club Social row. Rated match history is not changed.</>}
              confirmLabel="Yes, delete row"
              confirmationText="DELETE"
              tone="danger"
              disabled={busy || !accessToken || !selectedRow || Boolean(writeRecovery)}
              busy={busy}
              onConfirm={deleteRow}
            />
          </p>
        </div>
      ) : <p style={{ color: "#475569" }}>{loadingRows ? "Loading Club Social rows…" : "No Club Social rows loaded."}</p>}
      {rows.length ? (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", marginTop: "1rem" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "900px" }}>
            <thead><tr style={{ textAlign: "left", background: "#f8fafc" }}><th style={{ padding: "0.5rem" }}>Event</th><th style={{ padding: "0.5rem" }}>Played</th><th style={{ padding: "0.5rem" }}>Players</th><th style={{ padding: "0.5rem" }}>Score</th><th style={{ padding: "0.5rem" }}>Status</th></tr></thead>
            <tbody>{rows.slice(0, 25).map((row) => <tr key={rowId(row)}><td style={{ padding: "0.5rem" }}>{row.event_name || "—"}</td><td style={{ padding: "0.5rem" }}>{row.played_on || row.date || "—"}</td><td style={{ padding: "0.5rem" }}>{playerLabel(row)}</td><td style={{ padding: "0.5rem" }}>{row.score_t1 ?? 0}-{row.score_t2 ?? 0}</td><td style={{ padding: "0.5rem" }}>{row.status || "—"}</td></tr>)}</tbody>
          </table>
        </div>
      ) : null}
    </article>
  );
}
