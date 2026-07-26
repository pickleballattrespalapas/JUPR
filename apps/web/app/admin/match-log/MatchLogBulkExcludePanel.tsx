"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminMatchExclusionOperation,
  AdminMatchExclusionTarget,
  AdminMatchLogMatch,
  AdminMatchLogWriteResult
} from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogBulkExcludePanelProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
  matches: AdminMatchLogMatch[];
  exclusionOperation: AdminMatchExclusionOperation | null;
  onExclusionOperationChange: (operation: AdminMatchExclusionOperation | null) => void;
  onMutationComplete: () => void;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function requestKey(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (token) => {
    const value = Math.floor(Math.random() * 16);
    return (token === "x" ? value : (value & 0x3) | 0x8).toString(16);
  });
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19);
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function playerNames(players: Array<{ id: number | null; name: string }>): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
}

function operationFromPayload(payload: unknown): AdminMatchExclusionOperation | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const detail = record.detail && typeof record.detail === "object"
    ? record.detail as Record<string, unknown>
    : record;
  const nested = detail.operation && typeof detail.operation === "object"
    ? detail.operation as Record<string, unknown>
    : detail;
  const id = nested.id || nested.operation_id;
  if (!id) return null;
  return {
    ...nested,
    id: String(id),
    status: String(nested.status || nested.operation_status || "recovery_required"),
    recovery_stage: nested.recovery_stage == null ? null : String(nested.recovery_stage),
    replay_job_id: nested.replay_job_id == null ? null : String(nested.replay_job_id),
    error_text: nested.error_text == null
      ? (nested.message == null ? null : String(nested.message))
      : String(nested.error_text)
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

function resultSummary(result: AdminMatchLogWriteResult | null): string | null {
  if (!result) return null;
  const excluded = result.excluded_count ?? result.deleted_count ?? 0;
  const status = result.status || result.operation_status || result.operation?.status;
  if (status && status !== "succeeded") {
    return `Soft-excluded ${excluded} match(es); durable recovery is ${status.replace(/_/g, " ")}. Resume the exact operation instead of submitting again.`;
  }
  return `Soft-excluded ${excluded} match(es), completed Replay ALL, and reconciled the affected live match badges.`;
}

export default function MatchLogBulkExcludePanel({
  apiBase,
  clubId,
  enabled,
  matches,
  exclusionOperation,
  onExclusionOperationChange,
  onMutationComplete
}: MatchLogBulkExcludePanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [note, setNote] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchLogWriteResult | null>(null);
  const [idempotencyKey, setIdempotencyKey] = useState(requestKey);

  const selectableMatches = matches.filter(
    (match) => match.id != null && Number(match.row_version || 0) > 0
  ).slice(0, 100);
  const selectedSet = new Set(selectedIds);
  const exclusionBlocked = Boolean(exclusionOperation && exclusionOperation.status !== "succeeded");

  function toggleMatch(matchId: number) {
    setSelectedIds((current) => current.includes(matchId) ? current.filter((id) => id !== matchId) : [...current, matchId].sort((a, b) => a - b));
  }

  async function submitExclude(confirmationText: string) {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before excluding rated matches.");
      return;
    }
    if (!selectedIds.length) {
      setMessage("Select at least one match to exclude.");
      return;
    }
    if (exclusionBlocked) {
      setMessage(`Recover exclusion operation ${exclusionOperation?.id} before starting another.`);
      return;
    }
    setPending(true);
    try {
      const targets: AdminMatchExclusionTarget[] = selectedIds.map((matchId) => {
        const match = selectableMatches.find((candidate) => Number(candidate.id) === matchId);
        const rowVersion = Number(match?.row_version || 0);
        if (!match || rowVersion < 1) throw new Error(`Match #${matchId} has no current row version. Refresh Match Log.`);
        return { match_id: matchId, expected_row_version: rowVersion };
      });
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclude`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          targets,
          idempotency_key: idempotencyKey,
          confirmation_text: confirmationText,
          note,
          source: "next_match_log_bulk_exclude_panel"
        })
      });
      const payload = await response.json().catch(() => null);
      const durableOperation = operationFromPayload(payload);
      if (durableOperation) onExclusionOperationChange(durableOperation);
      if (!response.ok) throw new Error(errorMessage(payload, `API error (${response.status})`));
      const typed = payload as AdminMatchLogWriteResult;
      setResult(typed);
      setMessage(resultSummary(typed));
      const status = typed.status || typed.operation_status || typed.operation?.status;
      if (typed.ok && (!status || status === "succeeded")) {
        setSelectedIds([]);
        setNote("");
        setIdempotencyKey(requestKey());
        onExclusionOperationChange(null);
        onMutationComplete();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to exclude selected matches.");
    } finally {
      setPending(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Bulk exclude is disabled</h2>
        <p style={{ color: "#475569" }}>
          Excluding rated matches requires both <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1</code> and <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=1</code> on FastAPI, plus Supabase JWT delete-match authorization.
        </p>
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Exclude rated matches</h2>
      <p style={{ color: "#475569" }}>
        This atomically soft-excludes exact reviewed row versions, creates one durable recovery operation, recomputes player activity, runs Replay ALL, and strictly reconciles only affected live match-trigger badges.
      </p>
      {exclusionBlocked ? (
        <p role="alert" style={{ color: "#991b1b" }}>
          New exclusions are disabled until operation <code>{exclusionOperation?.id}</code> succeeds in the recovery panel above.
        </p>
      ) : null}
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized exclude requests." : sessionLoading ? "Checking admin session…" : "Sign in before excluding rated matches."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>

      <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white", marginBottom: "0.75rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
          <thead>
            <tr style={{ textAlign: "left", background: "#f8fafc" }}>
              <th style={{ padding: "0.55rem" }}>Exclude</th>
              <th style={{ padding: "0.55rem" }}>ID</th>
              <th style={{ padding: "0.55rem" }}>Version</th>
              <th style={{ padding: "0.55rem" }}>Date</th>
              <th style={{ padding: "0.55rem" }}>League / Week</th>
              <th style={{ padding: "0.55rem" }}>Team 1</th>
              <th style={{ padding: "0.55rem" }}>Score</th>
              <th style={{ padding: "0.55rem" }}>Team 2</th>
            </tr>
          </thead>
          <tbody>
            {selectableMatches.map((match) => {
              const id = Number(match.id);
              return (
                <tr key={id}>
                  <td style={{ padding: "0.55rem" }}><input type="checkbox" checked={selectedSet.has(id)} onChange={() => toggleMatch(id)} /></td>
                  <td style={{ padding: "0.55rem" }}>#{id}</td>
                  <td style={{ padding: "0.55rem" }}>{match.row_version}</td>
                  <td style={{ padding: "0.55rem" }}>{dateLabel(match.date)}</td>
                  <td style={{ padding: "0.55rem" }}>{match.league || "—"}<br /><span style={{ color: "#64748b" }}>{match.week_tag || "—"}</span></td>
                  <td style={{ padding: "0.55rem" }}>{playerNames(match.team1)}</td>
                  <td style={{ padding: "0.55rem" }}><strong>{match.score?.display || "—"}</strong></td>
                  <td style={{ padding: "0.55rem" }}>{playerNames(match.team2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!selectableMatches.length ? <p style={{ color: "#475569" }}>No matches are available in the current filtered view.</p> : null}
      <p style={{ color: "#64748b" }}>Selected: {selectedIds.length} / {selectableMatches.length}. The table uses the current filtered Match Log view and caps the bulk action at 100 visible rows.</p>
      <p style={{ color: "#64748b" }}>Reviewed request UUID: <code>{idempotencyKey}</code>. It is preserved after timeouts or recovery errors.</p>
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => setSelectedIds(selectableMatches.map((match) => Number(match.id)).filter(Number.isFinite))} disabled={pending || !selectableMatches.length} style={secondaryButtonStyle}>Select all visible</button>
        <button type="button" onClick={() => setSelectedIds([])} disabled={pending || !selectedIds.length} style={secondaryButtonStyle}>Clear selection</button>
      </p>
      <label><strong>Delete/exclude note</strong><br /><input value={note} onChange={(event) => setNote(event.target.value)} style={inputStyle} placeholder="Why these rated matches are being excluded" /></label>
      <p>
        <ConfirmAction
          triggerLabel={`Exclude ${selectedIds.length || "selected"} rated match(es)`}
          title={`Exclude ${selectedIds.length || "the selected"} rated match(es)?`}
          description={<>This will soft-exclude the exact selected match IDs and row versions, then finish their durable Replay and narrow badge recovery. This changes official rating history.</>}
          confirmLabel="Yes, exclude and replay"
          confirmationText="DELETE"
          tone="danger"
          disabled={pending || !accessToken || !selectedIds.length || exclusionBlocked}
          busy={pending}
          onConfirm={submitExclude}
        />
      </p>
      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result?.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      {result?.replay_result ? (
        <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
          <div><dt style={{ fontWeight: 700 }}>Matches scanned</dt><dd style={{ margin: 0 }}>{String(result.replay_result.matches_scanned_total ?? "—")}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Matches rewritten</dt><dd style={{ margin: 0 }}>{String(result.replay_result.matches_rewritten ?? "—")}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>League ratings rows</dt><dd style={{ margin: 0 }}>{String(result.replay_result.league_ratings_rows ?? "—")}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Skipped incomplete</dt><dd style={{ margin: 0 }}>{String(result.replay_result.skipped_incomplete ?? "—")}</dd></div>
        </dl>
      ) : null}
    </article>
  );
}
