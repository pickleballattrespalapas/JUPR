"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminMatchExclusionOperation } from "@/lib/adminMatchLogApi";
import { useAdminSession } from "@/lib/useAdminSession";

type MatchLogExclusionRecoveryPanelProps = {
  apiBase: string | null;
  clubId: string;
  operation: AdminMatchExclusionOperation | null;
  onOperationChange: (operation: AdminMatchExclusionOperation | null) => void;
  onMutationComplete: () => void;
};

const panelStyle = {
  borderRadius: "12px",
  padding: "0.9rem",
  marginBottom: "1rem"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function operationFromPayload(
  payload: unknown,
  fallback: AdminMatchExclusionOperation
): AdminMatchExclusionOperation {
  if (!payload || typeof payload !== "object") return fallback;
  const record = payload as Record<string, unknown>;
  const nested = record.operation;
  const source = nested && typeof nested === "object"
    ? nested as Record<string, unknown>
    : record;
  return {
    ...fallback,
    ...source,
    id: String(source.id || source.operation_id || fallback.id),
    status: String(source.status || source.operation_status || fallback.status),
    recovery_stage: source.recovery_stage == null
      ? fallback.recovery_stage
      : String(source.recovery_stage),
    replay_job_id: source.replay_job_id == null
      ? fallback.replay_job_id
      : String(source.replay_job_id),
    error_text: source.error_text == null
      ? fallback.error_text
      : String(source.error_text)
  };
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    return String(record.message || record.code || fallback);
  }
  return detail ? String(detail) : fallback;
}

export default function MatchLogExclusionRecoveryPanel({
  apiBase,
  clubId,
  operation,
  onOperationChange,
  onMutationComplete
}: MatchLogExclusionRecoveryPanelProps) {
  const { accessToken } = useAdminSession();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  if (!operation || operation.status === "succeeded") return null;
  const activeOperation = operation;
  const recoveryRequired = activeOperation.status === "recovery_required";

  async function request(path: string, init?: RequestInit) {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before inspecting or recovering this exclusion.");
    const response = await fetch(apiUrl(apiBase, path), {
      cache: "no-store",
      ...init,
      headers: {
        accept: "application/json",
        Authorization: `Bearer ${accessToken}`,
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...(init?.headers || {})
      }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(errorMessage(payload, `API error (${response.status})`));
    return operationFromPayload(payload, activeOperation);
  }

  async function refreshOperation() {
    setBusy(true);
    setMessage(null);
    try {
      const current = await request(
        `/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclusions/${encodeURIComponent(activeOperation.id)}`
      );
      onOperationChange(current);
      setMessage(current.status === "succeeded"
        ? "Recovery is complete. Refresh Match Log before starting another exclusion."
        : `Operation is still ${current.status.replace(/_/g, " ")}.`);
      if (current.status === "succeeded") onMutationComplete();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to inspect exclusion recovery.");
    } finally {
      setBusy(false);
    }
  }

  async function recoverOperation(confirmationText: string) {
    setBusy(true);
    setMessage(null);
    try {
      const current = await request(
        `/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclusions/${encodeURIComponent(activeOperation.id)}/recover`,
        {
          method: "POST",
          body: JSON.stringify({
            confirmation_text: confirmationText,
            source: "next_match_log_exclusion_recovery"
          })
        }
      );
      onOperationChange(current);
      setMessage(current.status === "succeeded"
        ? "The exact exclusion operation, Replay, and badge repair all completed."
        : `Recovery remains ${current.status.replace(/_/g, " ")}. Do not start another exclusion.`);
      if (current.status === "succeeded") onMutationComplete();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to resume exclusion recovery.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article
      role="alert"
      data-testid="match-exclusion-recovery"
      style={{
        ...panelStyle,
        border: `2px solid ${recoveryRequired ? "#dc2626" : "#d97706"}`,
        background: recoveryRequired ? "#fef2f2" : "#fffbeb"
      }}
    >
      <h2 style={{ marginTop: 0 }}>
        {recoveryRequired ? "Match exclusion recovery required" : "Match exclusion in progress"}
      </h2>
      <p>
        The soft exclusion already has durable identity <code>{activeOperation.id}</code>.
        Do not submit a new exclusion or duplicate cleanup. Refresh or continue only this exact operation.
      </p>
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.6rem" }}>
        <div><dt><strong>Status</strong></dt><dd style={{ margin: 0 }}>{activeOperation.status.replace(/_/g, " ")}</dd></div>
        <div><dt><strong>Recovery stage</strong></dt><dd style={{ margin: 0 }}>{activeOperation.recovery_stage?.replace(/_/g, " ") || "checking"}</dd></div>
        <div><dt><strong>Replay job</strong></dt><dd style={{ margin: 0 }}><code>{activeOperation.replay_job_id || "pending"}</code></dd></div>
      </dl>
      {activeOperation.error_text ? <p style={{ color: "#991b1b" }}>{activeOperation.error_text}</p> : null}
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={refreshOperation} disabled={busy || !accessToken}>
          {busy ? "Checking…" : "Refresh recovery state"}
        </button>
        <ConfirmAction
          triggerLabel={recoveryRequired ? "Resume exact recovery" : "Continue exact operation"}
          title={recoveryRequired ? "Resume this exact exclusion recovery?" : "Continue this exact exclusion operation?"}
          description={<>This resumes operation <code>{activeOperation.id}</code>. It never repeats the soft exclusion.</>}
          confirmLabel={recoveryRequired ? "Yes, resume recovery" : "Yes, continue operation"}
          confirmationText="RECOVER"
          disabled={busy || !accessToken}
          busy={busy}
          onConfirm={recoverOperation}
        />
      </p>
      {message ? <p style={{ color: message.includes("complete") ? "#166534" : "#991b1b" }}>{message}</p> : null}
    </article>
  );
}
