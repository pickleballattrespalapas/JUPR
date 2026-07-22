"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminTournament, AdminTournamentDetailResponse, AdminTournamentListResponse, AdminTournamentStatusResponse, AdminTournamentWriteResponse } from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
};

const REGISTRATION_STATUS_OPTIONS = ["", "confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["", "unpaid", "paid", "refunded"];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function statusStyle(value?: string | null) {
  const normalized = String(value || "").toLowerCase();
  if (["open", "active", "confirmed", "paid"].includes(normalized)) return { background: "#dcfce7", borderColor: "#bbf7d0" };
  if (["closed", "cancelled", "refunded", "archived"].includes(normalized)) return { background: "#f1f5f9", borderColor: "#cbd5e1" };
  return { background: "#fef3c7", borderColor: "#fde68a" };
}

function StatusChip({ value }: { value?: string | null }) {
  return <span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.5rem", fontSize: "0.78rem", ...statusStyle(value) }}>{value || "—"}</span>;
}

export default function BulkRegistrationPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [registrationStatus, setRegistrationStatus] = useState("");
  const [paymentStatus, setPaymentStatus] = useState("");
  const [appendNote, setAppendNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

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
    setDetail(null);
    setSelectedIds([]);
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

  async function loadDetail(tournamentId: string) {
    setSelectedTournamentId(tournamentId);
    setDetail(null);
    setSelectedIds([]);
    if (!tournamentId) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`);
      setDetail(payload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load tournament detail.");
    } finally {
      setBusy(false);
    }
  }

  function toggleRegistration(id: string) {
    setSelectedIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id]);
  }

  async function saveBulkUpdate(confirmationText: string) {
    if (!detail) {
      setMessage("Select a tournament first.");
      return;
    }
    if (!selectedIds.length) {
      setMessage("Select at least one registration.");
      return;
    }
    if (!registrationStatus && !paymentStatus && !appendNote.trim()) {
      setMessage("Choose a status/payment change or note to append.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(detail.tournament.id)}/registrations/bulk`,
        {
          method: "PATCH",
          body: JSON.stringify({
            registration_ids: selectedIds,
            registration_status: registrationStatus,
            payment_status: paymentStatus,
            append_note: appendNote,
            expected_state_fingerprint: detail.state_fingerprint,
            expected_versions: Object.fromEntries(detail.registrations.filter((row) => selectedIds.includes(row.id)).map((row) => [row.id, row.updated_at || ""])),
            confirmation_text: confirmationText,
            source: "next_tournament_admin_bulk_registration_editor"
          })
        }
      );
      const refreshed = await requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(detail.tournament.id)}`);
      setDetail(refreshed);
      setSelectedIds([]);
      setMessage(payload.idempotent_replay ? "Bulk response reconciled from the durable operation." : `Updated ${payload.updated_count ?? payload.registration_ids?.length ?? 0} registration(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update registrations.");
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
        <h2 style={{ marginTop: 0 }}>Bulk registration actions</h2>
        <p style={{ color: "#475569" }}>Apply a status/payment change or append an admin note to selected registrations.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send guarded bulk registration updates." : sessionLoading ? "Checking admin session…" : "Sign in before loading tournaments."}</p>
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
          <h2 style={{ marginTop: 0 }}>Select tournament</h2>
          <select value={selectedTournamentId} onChange={(event) => loadDetail(event.target.value)} style={inputStyle}>
            <option value="">Choose a tournament…</option>
            {tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status} · {tournament.registration_count ?? 0} registrations</option>)}
          </select>
        </article>
      ) : null}

      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Bulk edit selected registrations</h2>
            <p style={{ color: "#64748b" }}>{selectedIds.length} selected out of {detail.registrations.length} loaded registration(s).</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
              <label><strong>Registration status</strong><br />
                <select value={registrationStatus} onChange={(event) => setRegistrationStatus(event.target.value)} style={inputStyle}>
                  {REGISTRATION_STATUS_OPTIONS.map((value) => <option key={value || "no-change"} value={value}>{value || "No change"}</option>)}
                </select>
              </label>
              <label><strong>Payment status</strong><br />
                <select value={paymentStatus} onChange={(event) => setPaymentStatus(event.target.value)} style={inputStyle}>
                  {PAYMENT_STATUS_OPTIONS.map((value) => <option key={value || "no-change"} value={value}>{value || "No change"}</option>)}
                </select>
              </label>
            </div>
            <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Append admin note</strong><br />
              <textarea value={appendNote} onChange={(event) => setAppendNote(event.target.value)} rows={3} style={inputStyle} />
            </label>
            <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <ConfirmAction triggerLabel="Apply bulk update" title="Update the selected registrations?" description={`This applies the chosen status, payment, or note changes to ${selectedIds.length} selected registration${selectedIds.length === 1 ? "" : "s"}.`} confirmLabel="Yes, update registrations" confirmationText="BULK UPDATE REGISTRATIONS" tone={registrationStatus === "cancelled" || paymentStatus === "refunded" ? "danger" : "default"} disabled={!selectedIds.length || !detail.state_fingerprint || detail.registrations.filter((row) => selectedIds.includes(row.id)).some((row) => !row.updated_at)} busy={busy} onConfirm={saveBulkUpdate} />
              <button type="button" onClick={() => setSelectedIds(detail.registrations.map((row) => row.id))} disabled={busy} style={ghostButtonStyle}>Select all loaded</button>
              <button type="button" onClick={() => setSelectedIds([])} disabled={busy} style={ghostButtonStyle}>Clear selection</button>
            </p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Registrations</h2>
            {detail.registrations.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "980px" }}>
                  <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Select</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Registrant</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Email</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Registration</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Payment</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Notes</th></tr></thead>
                  <tbody>{detail.registrations.map((row) => <tr key={row.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input type="checkbox" checked={selectedIds.includes(row.id)} onChange={() => toggleRegistration(row.id)} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.display_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.email || "—"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><StatusChip value={row.registration_status} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><StatusChip value={row.payment_status} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.notes || "—"}</td></tr>)}</tbody>
                </table>
              </div>
            ) : <p style={{ color: "#64748b" }}>No registrations are loaded for this tournament.</p>}
          </article>
        </>
      ) : null}

      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("reload") || message.toLowerCase().includes("changed") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
