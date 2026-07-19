"use client";

import Link from "next/link";
import { useState } from "react";
import type { AdminTournament, AdminTournamentListResponse, AdminTournamentStatusResponse, AdminTournamentWriteResponse } from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function statusStyle(value?: string | null) {
  const normalized = String(value || "").toLowerCase();
  if (["published", "registration", "registration_open", "draft"].includes(normalized)) return { background: "#dcfce7", borderColor: "#bbf7d0" };
  if (["archived", "registration_closed"].includes(normalized)) return { background: "#f1f5f9", borderColor: "#cbd5e1" };
  return { background: "#fef3c7", borderColor: "#fde68a" };
}

function StatusChip({ value }: { value?: string | null }) {
  return <span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.5rem", fontSize: "0.78rem", ...statusStyle(value) }}>{value || "—"}</span>;
}

export default function TournamentStatusPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(true);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [action, setAction] = useState("archive");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const selectedTournament = tournaments.find((row) => row.id === selectedTournamentId) || null;
  const expectedConfirm = action === "archive" ? "ARCHIVE" : "UNARCHIVE";

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadTournaments() {
    setBusy(true);
    setMessage(null);
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments${suffix}`);
      setTournaments(payload.tournaments || []);
      setMessage(`Loaded ${payload.count ?? payload.tournaments?.length ?? 0} tournament(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      setBusy(false);
    }
  }

  async function submitAction() {
    if (!selectedTournamentId) {
      setMessage("Select a tournament first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/status-action`,
        {
          method: "PATCH",
          body: JSON.stringify({
            action,
            expected_updated_at: selectedTournament?.updated_at,
            confirmation_text: confirm,
            source: "next_tournament_admin_status_page"
          })
        }
      );
      setTournaments((current) => current.map((row) => row.id === selectedTournamentId && payload.tournament ? { ...row, ...payload.tournament } : row));
      setConfirm("");
      setMessage(payload.idempotent_replay ? "Tournament status response reconciled from the durable operation." : `Tournament ${payload.action || action} completed and audit-completed.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update tournament status.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Archive / unarchive tournament</h2>
        <p style={{ color: "#475569" }}>Reversible tournament shell status actions. Delete Draft remains Streamlit-only in this slice.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send guarded tournament status requests." : sessionLoading ? "Checking admin session…" : "Sign in before using status actions."}</p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}>
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />
          Include archived tournaments
        </label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load tournaments"}</button>
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Status action</h2>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(160px, 220px) minmax(180px, 260px) auto", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Tournament</strong><br />
              <select value={selectedTournamentId} onChange={(event) => setSelectedTournamentId(event.target.value)} style={inputStyle}>
                <option value="">Choose a tournament…</option>
                {tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}
              </select>
            </label>
            <label><strong>Action</strong><br />
              <select value={action} onChange={(event) => { setAction(event.target.value); setConfirm(""); }} style={inputStyle}>
                <option value="archive">Archive</option>
                <option value="unarchive">Unarchive</option>
              </select>
            </label>
            <label><strong>Type {expectedConfirm}</strong><br />
              <input value={confirm} onChange={(event) => setConfirm(event.target.value)} style={inputStyle} />
            </label>
            <button type="button" onClick={submitAction} disabled={busy || !selectedTournamentId || !selectedTournament?.updated_at || confirm.trim().toUpperCase() !== expectedConfirm} style={ghostButtonStyle}>Apply</button>
          </div>
          {selectedTournament ? <p style={{ color: selectedTournament.updated_at ? "#64748b" : "#b91c1c" }}>Selected: <strong>{selectedTournament.name}</strong> <StatusChip value={selectedTournament.status} />{selectedTournament.updated_at ? "" : " · missing version; reload"}</p> : null}
        </article>
      ) : null}

      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("reload") || message.toLowerCase().includes("changed") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
