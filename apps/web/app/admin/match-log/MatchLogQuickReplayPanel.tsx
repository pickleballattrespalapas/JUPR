"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type { AdminReplayResultResponse } from "@/lib/adminReplayApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogQuickReplayPanelProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
  options: string[];
  defaultTarget: string;
  recommendedTarget?: string | null;
  statusError?: string | null;
  warnings?: string[];
  onMutationComplete: () => void;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function normalizeRecommendedTarget(value: string | null | undefined, defaultTarget: string, options: string[]): string {
  const cleaned = String(value || "").trim();
  if (!cleaned) return defaultTarget || options[0] || "ALL (Full System Reset)";
  if (cleaned.toUpperCase() === "ALL") return defaultTarget || options[0] || "ALL (Full System Reset)";
  return options.includes(cleaned) ? cleaned : defaultTarget || options[0] || cleaned;
}

function replayMessage(result: AdminReplayResultResponse | null): string | null {
  if (!result) return null;
  if (!result.ok) return `Replay job ${result.job_id || "—"} is ${result.job_status || "not complete"}. Check Replay History before retrying.`;
  return `Replay complete for ${result.target_reset}: scanned ${result.result.matches_scanned_total ?? 0}, rewrote ${result.result.matches_rewritten ?? 0} snapshot row(s), rebuilt ${result.result.league_ratings_rows ?? 0} league rating row(s). Job ${result.job_id || "—"}.`;
}

function requestKey(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `quick-replay-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function MatchLogQuickReplayPanel({
  apiBase,
  clubId,
  enabled,
  options,
  defaultTarget,
  recommendedTarget,
  statusError,
  warnings = [],
  onMutationComplete
}: MatchLogQuickReplayPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const replayOptions = useMemo(
    () => (options.length ? options : [defaultTarget || "ALL (Full System Reset)"]),
    [defaultTarget, options]
  );
  const initialTarget = useMemo(() => normalizeRecommendedTarget(recommendedTarget, defaultTarget, replayOptions), [recommendedTarget, defaultTarget, replayOptions]);
  const [targetReset, setTargetReset] = useState(initialTarget);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(statusError || null);
  const [result, setResult] = useState<AdminReplayResultResponse | null>(null);
  const [idempotencyKey, setIdempotencyKey] = useState(requestKey);

  async function onSubmit(confirmationText: string): Promise<ActionCompletion> {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      const error = new Error("API base URL is not configured.");
      setMessage(error.message);
      throw error;
    }
    if (!accessToken) {
      const error = new Error("Sign in at /admin/login before running Replay History.");
      setMessage(error.message);
      throw error;
    }
    setPending(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/replay-history`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          target_reset: targetReset,
          confirmation_text: confirmationText,
          source: "next_match_log_quick_replay",
          idempotency_key: idempotencyKey
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      const typed = payload as AdminReplayResultResponse;
      const summary = replayMessage(typed) || "Replay completed.";
      setResult(typed);
      setMessage(summary);
      if (!typed.ok || typed.job_status !== "succeeded") throw new Error(summary);
      setIdempotencyKey(requestKey());
      onMutationComplete();
      return actionSuccess("Replay complete", summary);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to run replay.");
      throw error;
    } finally {
      setPending(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Quick Replay is disabled</h2>
        <p style={{ color: "#475569" }}>
          Enable <code>JUPR_ENABLE_NEXT_ADMIN_REPLAY=1</code> on FastAPI to run Replay History from Match Log.
        </p>
        {statusError ? <p style={{ color: "#b91c1c" }}>{statusError}</p> : null}
      </article>
    );
  }

  return (
    <article style={{ ...cardStyle, display: "grid", gap: "0.75rem" }}>
      <h2 style={{ marginTop: 0 }}>Quick Replay</h2>
      <p style={{ color: "#475569", marginTop: 0 }}>
        Streamlit parity control for running Replay History directly after Match Log edits or duplicate cleanup. Prefer a league-specific replay when appropriate; use full reset when the cleanup preview recommends ALL or when ratings may be broadly stale.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized replay requests." : sessionLoading ? "Checking admin session…" : "Sign in before running Replay History."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      {recommendedTarget ? (
        <p style={{ color: "#92400e", margin: 0 }}>
          Current Match Log recommendation: <strong>{recommendedTarget}</strong>
        </p>
      ) : null}
      <label><strong>Replay scope</strong><br />
        <select value={targetReset} onChange={(event) => setTargetReset(event.target.value)} style={inputStyle}>
          {replayOptions.map((option) => <option key={option}>{option}</option>)}
        </select>
      </label>
      <ConfirmAction
        triggerLabel="Run Quick Replay"
        title="Run Replay History now?"
        description={<>This will run Replay History for <strong>{targetReset}</strong>. Ratings and derived history may be rebuilt across that scope.</>}
        confirmLabel="Yes, run replay"
        confirmationText="REPLAY"
        disabled={pending || !accessToken}
        busy={pending}
        onConfirm={onSubmit}
      />
      <p style={{ margin: 0 }}><Link href="/admin/replay-history"><strong>Open Replay History</strong></Link> to view recent jobs and their status.</p>
      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result ? (
        <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
          <div><dt style={{ fontWeight: 700 }}>Job</dt><dd style={{ margin: 0, fontFamily: "monospace" }}>{result.job_id || "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Status</dt><dd style={{ margin: 0 }}>{result.job_status || "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Players updated</dt><dd style={{ margin: 0 }}>{result.result.players_updated ? "Yes" : "No"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Skipped incomplete</dt><dd style={{ margin: 0 }}>{result.result.skipped_incomplete ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Snapshot rows updated</dt><dd style={{ margin: 0 }}>{result.result.matches_snapshots_updated_rows ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>League ratings rows</dt><dd style={{ margin: 0 }}>{result.result.league_ratings_rows ?? "—"}</dd></div>
        </dl>
      ) : null}
      {(result?.warnings?.length || warnings.length) ? (
        <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>
          {[...(warnings || []), ...(result?.warnings || [])].map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </article>
  );
}
