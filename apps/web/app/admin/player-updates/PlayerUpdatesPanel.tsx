"use client";

import { useState } from "react";
import type { AdminPlayerUpdatesRangeResponse, AdminPlayerUpdatesStatusResponse } from "@/lib/adminPlayerUpdatesApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminPlayerUpdatesStatusResponse;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const confirmText = "SEND PLAYER UPDATES";

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function daysAgoIsoDate(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().slice(0, 10);
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function metric(result: Record<string, unknown> | undefined, key: string): string {
  const value = result?.[key];
  return value == null ? "0" : String(value);
}

export default function PlayerUpdatesPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [startDate, setStartDate] = useState(daysAgoIsoDate(7));
  const [endDate, setEndDate] = useState(todayIsoDate());
  const [onlyPlayersWithMatches, setOnlyPlayersWithMatches] = useState(true);
  const [sendNow, setSendNow] = useState(true);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminPlayerUpdatesRangeResponse | null>(null);

  async function runReport() {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before sending player update emails.");
      return;
    }
    if (!status.enabled) {
      setMessage("Next Player Updates Admin is disabled on the API.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/player-updates/send-range`), {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          only_players_with_matches: onlyPlayersWithMatches,
          send_now: sendNow,
          confirmation_text: confirmation,
          source: "next_player_updates_admin_range"
        })
      });
      const payload = await response.json().catch(() => null) as AdminPlayerUpdatesRangeResponse | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      setResult(payload as AdminPlayerUpdatesRangeResponse);
      setConfirmation("");
      const sendResult = (payload as AdminPlayerUpdatesRangeResponse).send_result || {};
      setMessage(`Generated ${metric((payload as AdminPlayerUpdatesRangeResponse).generation_result, "saved")} digest(s); sent ${metric(sendResult, "sent")} email(s), skipped ${metric(sendResult, "skipped")}, errors ${metric(sendResult, "errors")}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to run player update report.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Player Updates Admin is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Player Updates pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Player Updates admin session</h2>
        <p style={{ color: "#475569" }}>This sends selected date-range player summaries to active verified subscribers. Use dry-run or staging redirect email mode until production email delivery is intentionally enabled.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send authorized Player Updates requests." : sessionLoading ? "Checking admin session…" : "Sign in before sending player update emails."}</p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        </div>
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
        <h2 style={{ marginTop: 0 }}>Generate and send date-range report</h2>
        <p style={{ color: "#7c2d12" }}>This creates player digest records for the selected date window, queues matching active subscribers, and sends the queued rows for that exact range.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>Start date</strong><br /><input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} style={inputStyle} /></label>
          <label><strong>End date</strong><br /><input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} style={inputStyle} /></label>
          <label><strong>Player filter</strong><br /><select value={onlyPlayersWithMatches ? "matches" : "all"} onChange={(event) => setOnlyPlayersWithMatches(event.target.value === "matches")} style={inputStyle}><option value="matches">Only players with matches</option><option value="all">All active subscriptions</option></select></label>
          <label><strong>Action</strong><br /><select value={sendNow ? "send" : "queue"} onChange={(event) => setSendNow(event.target.value === "send")} style={inputStyle}><option value="send">Generate + send now</option><option value="queue">Generate + queue only</option></select></label>
          <label><strong>Type {confirmText}</strong><br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} style={inputStyle} /></label>
        </div>
        <p><button type="button" onClick={runReport} disabled={busy || !accessToken || confirmation.trim().toUpperCase() !== confirmText} style={buttonStyle}>{busy ? "Running…" : "Run player update report"}</button></p>
      </article>

      {result ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Report result</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
            <div><strong>Saved digests</strong><br />{metric(result.generation_result, "saved")}</div>
            <div><strong>Queued</strong><br />{metric(result.generation_result, "queued")}</div>
            <div><strong>Email attempted</strong><br />{metric(result.send_result, "attempted")}</div>
            <div><strong>Sent</strong><br />{metric(result.send_result, "sent")}</div>
            <div><strong>Skipped</strong><br />{metric(result.send_result, "skipped")}</div>
            <div><strong>Errors</strong><br />{metric(result.send_result, "errors")}</div>
          </div>
          {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        </article>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
