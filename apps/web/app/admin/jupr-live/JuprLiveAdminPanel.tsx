"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type LiveSession = {
  session_key: string;
  title?: string | null;
  status: string;
  source?: string | null;
  event_type?: string | null;
  created_by_email?: string | null;
  updated_at?: string | null;
  expires_at?: string | null;
};

type StatusResponse = { enabled: boolean; status: string; counts?: Record<string, number>; warnings?: string[] };
type ListResponse = { ok: boolean; sessions: LiveSession[]; count: number };
type WriteResponse = { ok: boolean; session: LiveSession };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    return String(payload.detail || text);
  } catch {
    return text.slice(0, 240);
  }
}

export default function JuprLiveAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [filter, setFilter] = useState("active");
  const [title, setTitle] = useState(`JUPR Live ${new Date().toISOString().slice(0, 10)}`);
  const [eventType, setEventType] = useState("round_robin");
  const [participantNames, setParticipantNames] = useState("");
  const [createConfirm, setCreateConfirm] = useState("");
  const [sessionConfirm, setSessionConfirm] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using JUPR Live Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function loadSessions() {
    setBusy(true); setMessage(null);
    try {
      const query = filter ? `?status=${encodeURIComponent(filter)}&limit=100` : `?limit=100`;
      const payload = await requestJson<ListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions${query}`);
      setSessions(payload.sessions || []);
      setMessage(`Loaded ${payload.count ?? payload.sessions?.length ?? 0} live session(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load live sessions."); }
    finally { setBusy(false); }
  }

  async function createSession() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions`, {
        method: "POST",
        body: JSON.stringify({ title, event_type: eventType, participant_names: participantNames.replace(/,/g, "\n").split("\n").map((x) => x.trim()).filter(Boolean), confirmation_text: createConfirm })
      });
      setCreateConfirm("");
      setMessage(`Created live session ${payload.session.session_key}.`);
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create live session."); }
    finally { setBusy(false); }
  }

  async function updateSession(row: LiveSession, nextStatus: string) {
    setBusy(true); setMessage(null);
    try {
      await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}`, {
        method: "PATCH",
        body: JSON.stringify({ status: nextStatus, title: row.title || undefined, confirmation_text: sessionConfirm[row.session_key] || "" })
      });
      setMessage(`${row.title || row.session_key} marked ${nextStatus}.`);
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update live session."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>JUPR Live Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE on FastAPI."}</p></article>;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Active: {status.counts?.active ?? "—"} · Completed: {status.counts?.completed ?? "—"} · Abandoned: {status.counts?.abandoned ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Create durable live session</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Title<br /><input value={title} onChange={(e) => setTitle(e.target.value)} style={inputStyle} /></label>
          <label>Event type<br /><select value={eventType} onChange={(e) => setEventType(e.target.value)} style={inputStyle}><option value="round_robin">Round Robin</option><option value="league_ladder">League / Ladder</option><option value="tournament">Tournament</option></select></label>
          <label>Confirmation<br /><input value={createConfirm} onChange={(e) => setCreateConfirm(e.target.value)} placeholder="CREATE LIVE SESSION" style={inputStyle} /></label>
        </div>
        <label>Participant names, optional<br /><textarea value={participantNames} onChange={(e) => setParticipantNames(e.target.value)} rows={4} style={inputStyle} /></label>
        <p><button type="button" onClick={createSession} disabled={busy} style={buttonStyle}>Create session</button></p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Manage sessions</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Status filter<br /><select value={filter} onChange={(e) => setFilter(e.target.value)} style={inputStyle}><option value="active">active</option><option value="completed">completed</option><option value="abandoned">abandoned</option><option value="archived">archived</option><option value="">all</option></select></label>
          <button type="button" onClick={loadSessions} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load sessions"}</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {sessions.map((row) => (
        <article key={row.session_key} style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>{row.title || row.session_key}</h3>
          <p style={{ color: "#475569" }}>{row.session_key} · {row.status} · {row.event_type || "event"} · updated {row.updated_at ? String(row.updated_at).slice(0, 19) : "—"}</p>
          <p><a href={`/clubs/tres-palapas/live/${encodeURIComponent(row.session_key)}`}>Public view</a></p>
          <label>Confirmation<br /><input value={sessionConfirm[row.session_key] || ""} onChange={(e) => setSessionConfirm((current) => ({ ...current, [row.session_key]: e.target.value }))} placeholder="SAVE LIVE SESSION" style={inputStyle} /></label>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => updateSession(row, "completed")} disabled={busy} style={buttonStyle}>Complete</button><button type="button" onClick={() => updateSession(row, "abandoned")} disabled={busy} style={ghostButtonStyle}>Abandon</button><button type="button" onClick={() => updateSession(row, "archived")} disabled={busy} style={ghostButtonStyle}>Archive</button></p>
        </article>
      ))}
    </div>
  );
}
