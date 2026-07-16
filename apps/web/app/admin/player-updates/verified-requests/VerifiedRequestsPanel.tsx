"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type RequestRow = {
  id: string;
  player_id: number;
  player_name: string;
  email_masked: string;
  request_status: string;
  request_note?: string | null;
  admin_note?: string | null;
  created_at?: string | null;
};

type ListResponse = { ok: boolean; requests: RequestRow[]; count: number };
type StatusResponse = { enabled: boolean; status: string; counts?: Record<string, number>; warnings?: string[] };

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

export default function VerifiedRequestsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [rows, setRows] = useState<RequestRow[]>([]);
  const [filter, setFilter] = useState("pending");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [confirmations, setConfirmations] = useState<Record<string, string>>({});

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before reviewing verified update requests.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function loadRows() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<ListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/verified-updates/requests?status=${encodeURIComponent(filter)}&limit=200`);
      setRows(payload.requests || []);
      setMessage(`Loaded ${payload.count ?? payload.requests?.length ?? 0} request(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load requests.");
    } finally {
      setBusy(false);
    }
  }

  async function applyAction(row: RequestRow, action: string) {
    setBusy(true);
    setMessage(null);
    try {
      await requestJson(`/admin/clubs/${encodeURIComponent(clubId)}/verified-updates/requests/${encodeURIComponent(row.id)}`, {
        method: "PATCH",
        body: JSON.stringify({ action, admin_note: notes[row.id] || "", confirmation_text: confirmations[row.id] || "", source: "next_verified_updates_request_review" })
      });
      setMessage(`${action} saved for ${row.player_name}.`);
      await loadRows();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update request.");
    } finally {
      setBusy(false);
    }
  }

  if (!status?.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Verified requests are disabled</h2><p>{status?.warnings?.[0] || "Enable Player Updates Admin on FastAPI."}</p></article>;
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Pending: {status.counts?.pending ?? "—"} · Active: {status.counts?.active ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Request queue</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Status<br /><select value={filter} onChange={(event) => setFilter(event.target.value)} style={inputStyle}><option value="pending">pending</option><option value="active">active</option><option value="rejected">rejected</option><option value="unsubscribed">unsubscribed</option></select></label>
          <button type="button" onClick={loadRows} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load requests"}</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {rows.map((row) => (
        <article key={row.id} style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>{row.player_name}</h3>
          <p style={{ color: "#475569" }}>#{row.player_id} · {row.email_masked || "email hidden"} · {row.request_status} · {row.created_at ? String(row.created_at).slice(0, 10) : "—"}</p>
          {row.request_note ? <p><strong>Requester note:</strong> {row.request_note}</p> : null}
          <label>Admin note<br /><textarea value={notes[row.id] ?? row.admin_note ?? ""} onChange={(event) => setNotes((current) => ({ ...current, [row.id]: event.target.value }))} rows={3} style={inputStyle} /></label>
          <label>Confirmation<br /><input value={confirmations[row.id] || ""} onChange={(event) => setConfirmations((current) => ({ ...current, [row.id]: event.target.value }))} placeholder="SAVE VERIFIED REQUEST" style={inputStyle} /></label>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" onClick={() => applyAction(row, "approve")} disabled={busy} style={buttonStyle}>Approve</button>
            <button type="button" onClick={() => applyAction(row, "reject")} disabled={busy} style={ghostButtonStyle}>Reject</button>
            <button type="button" onClick={() => applyAction(row, "unsubscribe")} disabled={busy} style={ghostButtonStyle}>Unsubscribe</button>
          </p>
        </article>
      ))}
    </div>
  );
}
