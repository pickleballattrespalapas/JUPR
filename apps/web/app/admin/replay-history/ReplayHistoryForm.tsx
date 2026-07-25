"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  getAdminReplayStatus,
  type AdminReplayJob,
  type AdminReplayResultResponse
} from "@/lib/adminReplayApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard
} from "@/lib/useAuthenticatedAutoLoad";

type ReplayHistoryFormProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
  options: string[];
  defaultTarget: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function resultMessage(result: AdminReplayResultResponse | null): string | null {
  if (!result) return null;
  if (result.mode === "replay_incomplete") {
    return `Replay job ${result.job_id || "—"} finished without an exact replay-managed singles recovery attestation. Treat recovery as incomplete and do not accept the result.`;
  }
  if (!result.ok) return `Replay job ${result.job_id || "—"} is ${result.job_status || "not complete"}. Refresh job history before retrying.`;
  const details = result.result;
  const singlesSummary = result.target_reset === "ALL (Full System Reset)"
    ? `, and rebuilt ${details.singles_matches_rewritten ?? 0} replay-managed singles row(s)`
    : "";
  return `Replay complete for ${result.target_reset}: scanned ${details.matches_scanned_total ?? 0}, rewrote ${details.matches_rewritten ?? 0} snapshot row(s), rebuilt ${details.league_ratings_rows ?? 0} league rating row(s)${singlesSummary}. Job ${result.job_id || "—"}${result.idempotent_replay ? " (idempotent retry)" : ""}.`;
}

function requestKey(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `replay-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function ReplayHistoryForm({ apiBase, clubId, enabled, options, defaultTarget }: ReplayHistoryFormProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [targetReset, setTargetReset] = useState(defaultTarget || options[0] || "ALL (Full System Reset)");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminReplayResultResponse | null>(null);
  const [idempotencyKey, setIdempotencyKey] = useState(requestKey);
  const [recentJobs, setRecentJobs] = useState<AdminReplayJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const replayRequest = useLatestRequestGuard(accessToken, clearProtectedReplayState);
  const jobsRequest = useLatestRequestGuard(accessToken, clearProtectedJobState);
  useAuthenticatedAutoLoad(
    enabled && apiBase ? accessToken : "",
    loadJobHistory,
    `${clubId}\u0000replay-jobs`
  );

  function clearProtectedReplayState() {
    setPending(false);
    setMessage(null);
    setResult(null);
    setIdempotencyKey(requestKey());
  }

  function clearProtectedJobState() {
    setRecentJobs([]);
    setJobsLoading(false);
    setJobsError(null);
  }

  async function loadJobHistory() {
    if (!accessToken || !apiBase || !enabled) {
      clearProtectedJobState();
      return;
    }
    const generation = jobsRequest.begin();
    setJobsLoading(true);
    setJobsError(null);
    const response = await getAdminReplayStatus(clubId, {
      accessToken,
      apiBase,
      includeJobs: true
    });
    if (!jobsRequest.isCurrent(generation)) return;
    if (response.error || !response.data) {
      setRecentJobs([]);
      setJobsError(response.error || "Replay job history is unavailable.");
    } else {
      setRecentJobs(response.data.recent_jobs || []);
    }
    if (jobsRequest.isCurrent(generation)) setJobsLoading(false);
  }

  async function onSubmit(confirmationText: string) {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before running Replay History.");
      return;
    }
    const generation = replayRequest.begin();
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
          source: "next_replay_history",
          idempotency_key: idempotencyKey
        })
      });
      const payload = await response.json().catch(() => null);
      if (!replayRequest.isCurrent(generation)) return;
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      setResult(payload as AdminReplayResultResponse);
      setMessage(resultMessage(payload as AdminReplayResultResponse));
      if ((payload as AdminReplayResultResponse).ok) setIdempotencyKey(requestKey());
      void loadJobHistory();
    } catch (error) {
      if (replayRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run replay.");
    } finally {
      if (replayRequest.isCurrent(generation)) setPending(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next replay is disabled</h2>
        <p style={{ color: "#475569" }}>
          Enable <code>JUPR_ENABLE_NEXT_ADMIN_REPLAY=1</code> on FastAPI for the closed-club pilot, then use this page with a Supabase JWT that has replay permission.
        </p>
      </article>
    );
  }

  return (
    <article style={{ ...cardStyle, display: "grid", gap: "0.75rem" }}>
      <h2 style={{ marginTop: 0 }}>Run replay</h2>
      <p style={{ color: "#475569", marginTop: 0 }}>
        Replay runs on FastAPI through the Python replay domain function. Full reset updates overall player stats and replay-managed singles; league replay rebuilds league-specific ratings and snapshots for that league.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized replay requests." : sessionLoading ? "Checking admin session…" : "Sign in before running Replay History."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      <label><strong>Replay scope</strong><br />
        <select value={targetReset} onChange={(event) => setTargetReset(event.target.value)} disabled={pending} style={inputStyle}>
          {options.map((option) => <option key={option}>{option}</option>)}
        </select>
      </label>
      <ConfirmAction
        triggerLabel="Run replay"
        title="Run Replay History now?"
        description={<>This will run Replay History for <strong>{targetReset}</strong>. Ratings and derived history may be rebuilt across that scope.</>}
        confirmLabel="Yes, run replay"
        confirmationText="REPLAY"
        disabled={pending || !accessToken}
        busy={pending}
        onConfirm={onSubmit}
      />
      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result ? (
        <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
          <div><dt style={{ fontWeight: 700 }}>Job</dt><dd style={{ margin: 0, fontFamily: "monospace" }}>{result.job_id || "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Status</dt><dd style={{ margin: 0 }}>{result.job_status || "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Players updated</dt><dd style={{ margin: 0 }}>{result.result.players_updated ? "Yes" : "No"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Skipped incomplete</dt><dd style={{ margin: 0 }}>{result.result.skipped_incomplete ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Snapshot rows updated</dt><dd style={{ margin: 0 }}>{result.result.matches_snapshots_updated_rows ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>League ratings rows</dt><dd style={{ margin: 0 }}>{result.result.league_ratings_rows ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Singles rows rebuilt</dt><dd style={{ margin: 0 }}>{result.result.singles_matches_rewritten ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Singles recovery</dt><dd style={{ margin: 0 }}>{result.target_reset !== "ALL (Full System Reset)" ? "Not run for league replay" : result.result.singles_replay_supported === true ? "Verified" : "Unavailable"}</dd></div>
        </dl>
      ) : null}
      {result?.warnings?.length ? (
        <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
      <section style={{ ...cardStyle, marginTop: "0.5rem" }} data-testid="replay-job-history">
        <div style={{ display: "flex", gap: "0.75rem", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>Recent durable replay jobs</h2>
          <button
            type="button"
            onClick={() => void loadJobHistory()}
            disabled={!accessToken || jobsLoading}
            style={{ padding: "0.5rem 0.8rem" }}
          >
            {jobsLoading ? "Refreshing…" : "Refresh history"}
          </button>
        </div>
        {!accessToken ? (
          <p style={{ color: "#475569" }}>Sign in to load protected replay job history.</p>
        ) : jobsError ? (
          <p style={{ color: "#b91c1c" }}>{jobsError}</p>
        ) : jobsLoading && !recentJobs.length ? (
          <p style={{ color: "#475569" }}>Loading replay job history…</p>
        ) : recentJobs.length ? (
          <div style={{ overflowX: "auto", marginTop: "0.75rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
              <thead><tr>{["Created", "Scope", "Status", "Actor", "Source", "Job ID"].map((label) => <th key={label} style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.55rem" }}>{label}</th>)}</tr></thead>
              <tbody>{recentJobs.map((job) => (
                <tr key={job.id} data-replay-status={job.status}>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.created_at ? new Date(job.created_at).toISOString().slice(0, 19).replace("T", " ") : "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.target_reset || "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.status}{job.error_text ? ` · ${job.error_text}` : ""}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.actor_email || "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.source || "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", fontFamily: "monospace" }}>{job.id}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: "#475569" }}>No replay jobs have been recorded for this club yet.</p>
        )}
      </section>
    </article>
  );
}
