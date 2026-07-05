"use client";

import Link from "next/link";
import { useState } from "react";
import type { FormEvent } from "react";
import type { AdminReplayResultResponse } from "@/lib/adminReplayApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type ReplayHistoryFormProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
  options: string[];
  defaultTarget: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.65rem 1rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function resultMessage(result: AdminReplayResultResponse | null): string | null {
  if (!result?.ok) return null;
  const details = result.result;
  return `Replay complete for ${result.target_reset}: scanned ${details.matches_scanned_total}, rewrote ${details.matches_rewritten} snapshot row(s), rebuilt ${details.league_ratings_rows} league rating row(s).`;
}

export default function ReplayHistoryForm({ apiBase, clubId, enabled, options, defaultTarget }: ReplayHistoryFormProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [targetReset, setTargetReset] = useState(defaultTarget || options[0] || "ALL (Full System Reset)");
  const [confirmationText, setConfirmationText] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminReplayResultResponse | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
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
          source: "next_replay_history"
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      setResult(payload as AdminReplayResultResponse);
      setMessage(resultMessage(payload as AdminReplayResultResponse));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to run replay.");
    } finally {
      setPending(false);
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
    <form onSubmit={onSubmit} style={{ ...cardStyle, display: "grid", gap: "0.75rem" }}>
      <h2 style={{ marginTop: 0 }}>Run replay</h2>
      <p style={{ color: "#475569", marginTop: 0 }}>
        Replay runs on FastAPI through the Python replay domain function. Full reset updates overall player stats; league replay rebuilds league-specific ratings and snapshots for that league.
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
        <select value={targetReset} onChange={(event) => setTargetReset(event.target.value)} style={inputStyle}>
          {options.map((option) => <option key={option}>{option}</option>)}
        </select>
      </label>
      <label><strong>Type REPLAY to confirm</strong><br /><input value={confirmationText} onChange={(event) => setConfirmationText(event.target.value)} style={inputStyle} /></label>
      <button type="submit" disabled={pending || !accessToken || confirmationText.trim().toUpperCase() !== "REPLAY"} style={buttonStyle}>
        {pending ? "Running replay…" : "Run replay"}
      </button>
      {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result ? (
        <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
          <div><dt style={{ fontWeight: 700 }}>Players updated</dt><dd style={{ margin: 0 }}>{result.result.players_updated ? "Yes" : "No"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Skipped incomplete</dt><dd style={{ margin: 0 }}>{result.result.skipped_incomplete}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Snapshot rows updated</dt><dd style={{ margin: 0 }}>{result.result.matches_snapshots_updated_rows}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>League ratings rows</dt><dd style={{ margin: 0 }}>{result.result.league_ratings_rows}</dd></div>
        </dl>
      ) : null}
      {result?.warnings?.length ? (
        <ul style={{ color: "#92400e", paddingLeft: "1.25rem" }}>
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </form>
  );
}
