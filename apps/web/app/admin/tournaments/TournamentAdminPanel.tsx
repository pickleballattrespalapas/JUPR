"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminTournament,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentRegistration,
  AdminTournamentSelection,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
};

type RegistrationEdit = {
  registrationStatus: string;
  paymentStatus: string;
  notes: string;
  confirm: string;
};

type SelectionEdit = {
  eventOptionId: string;
  partnerMode: string;
  partnerName: string;
  partnerEmail: string;
  partnerPhone: string;
  partnerNote: string;
  confirm: string;
};

const REGISTRATION_STATUS_OPTIONS = ["confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "refunded"];
const PARTNER_MODE_OPTIONS = ["NONE", "HAS_PARTNER", "NEEDS_PARTNER"];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  return String(value).slice(0, 10);
}

function kvList(value?: Record<string, number>): string {
  const entries = Object.entries(value || {});
  return entries.length ? entries.map(([key, count]) => `${key}: ${count}`).join(" · ") : "—";
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

function eventOptionLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Event");
}

function editStateFromRegistration(row: AdminTournamentRegistration | null): RegistrationEdit {
  return {
    registrationStatus: row?.registration_status || "confirmed",
    paymentStatus: row?.payment_status || "unpaid",
    notes: row?.notes || "",
    confirm: ""
  };
}

function editStateFromSelection(row: AdminTournamentSelection | null): SelectionEdit {
  return {
    eventOptionId: row?.event_option_id || "",
    partnerMode: row?.partner_mode || "NONE",
    partnerName: row?.partner_name || "",
    partnerEmail: row?.partner_email || "",
    partnerPhone: row?.partner_phone || "",
    partnerNote: row?.partner_note || "",
    confirm: ""
  };
}

export default function TournamentAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [selectedRegistrationId, setSelectedRegistrationId] = useState("");
  const [selectedSelectionId, setSelectedSelectionId] = useState("");
  const [registrationEdit, setRegistrationEdit] = useState<RegistrationEdit>(() => editStateFromRegistration(null));
  const [selectionEdit, setSelectionEdit] = useState<SelectionEdit>(() => editStateFromSelection(null));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const selectedRegistration = detail?.registrations.find((row) => row.id === selectedRegistrationId) || null;
  const registrationSelections = (detail?.selections || []).filter((row) => row.registration_id === selectedRegistrationId);
  const selectedSelection = registrationSelections.find((row) => row.id === selectedSelectionId) || null;

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

  async function refreshDetail(tournamentId: string): Promise<AdminTournamentDetailResponse> {
    return requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`);
  }

  async function loadTournaments() {
    setBusy(true);
    setMessage(null);
    setDetail(null);
    setSelectedRegistrationId("");
    setSelectedSelectionId("");
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
    setSelectedId(tournamentId);
    setDetail(null);
    setMessage(null);
    setSelectedRegistrationId("");
    setSelectedSelectionId("");
    if (!tournamentId) return;
    setBusy(true);
    try {
      const payload = await refreshDetail(tournamentId);
      setDetail(payload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load tournament detail.");
    } finally {
      setBusy(false);
    }
  }

  function selectRegistration(row: AdminTournamentRegistration) {
    setSelectedRegistrationId(row.id);
    setRegistrationEdit(editStateFromRegistration(row));
    const firstSelection = (detail?.selections || []).find((selection) => selection.registration_id === row.id) || null;
    setSelectedSelectionId(firstSelection?.id || "");
    setSelectionEdit(editStateFromSelection(firstSelection));
    setMessage(null);
  }

  function selectSelection(selectionId: string) {
    const nextSelection = registrationSelections.find((row) => row.id === selectionId) || null;
    setSelectedSelectionId(selectionId);
    setSelectionEdit(editStateFromSelection(nextSelection));
    setMessage(null);
  }

  async function saveRegistrationEdit() {
    if (!detail || !selectedRegistration) {
      setMessage("Select a registration before saving.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(detail.tournament.id)}/registrations/${encodeURIComponent(selectedRegistration.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            registration_status: registrationEdit.registrationStatus,
            payment_status: registrationEdit.paymentStatus,
            notes: registrationEdit.notes,
            confirmation_text: registrationEdit.confirm,
            source: "next_tournament_admin_registration_editor"
          })
        }
      );
      const refreshed = await refreshDetail(detail.tournament.id);
      setDetail(refreshed);
      setSelectedRegistrationId(payload.registration?.id || selectedRegistration.id);
      setRegistrationEdit(editStateFromRegistration(payload.registration || selectedRegistration));
      setMessage("Registration saved and audit-flagged for review.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save registration.");
    } finally {
      setBusy(false);
    }
  }

  async function saveSelectionEdit() {
    if (!detail || !selectedSelection) {
      setMessage("Select an event entry before saving.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(detail.tournament.id)}/selections/${encodeURIComponent(selectedSelection.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            event_option_id: selectionEdit.eventOptionId,
            partner_mode: selectionEdit.partnerMode,
            partner_name: selectionEdit.partnerName,
            partner_email: selectionEdit.partnerEmail,
            partner_phone: selectionEdit.partnerPhone,
            partner_note: selectionEdit.partnerNote,
            confirmation_text: selectionEdit.confirm,
            source: "next_tournament_admin_selection_editor"
          })
        }
      );
      const refreshed = await refreshDetail(detail.tournament.id);
      setDetail(refreshed);
      setSelectedSelectionId(payload.selection?.id || selectedSelection.id);
      setSelectionEdit(editStateFromSelection(payload.selection || selectedSelection));
      setMessage("Event entry saved and audit-flagged for review.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save event entry.");
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
        <h2 style={{ marginTop: 0 }}>Tournament Admin session</h2>
        <p style={{ color: "#475569" }}>
          This workflow loads tournament setup, registration summaries, event options, and registrants through FastAPI. Registration and event-entry edits are guarded by role checks, audit logging, and explicit confirmation text.
        </p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to load guarded tournament admin data." : sessionLoading ? "Checking admin session…" : "Sign in before loading tournaments."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}>
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />
          Include archived tournaments
        </label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load tournaments"}</button>
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Tournaments</h2>
          <select value={selectedId} onChange={(event) => loadDetail(event.target.value)} style={inputStyle}>
            <option value="">Choose a tournament…</option>
            {tournaments.map((tournament) => (
              <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status} · {tournament.registration_count ?? 0} registrations</option>
            ))}
          </select>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
            {tournaments.map((tournament) => (
              <button key={tournament.id} type="button" onClick={() => loadDetail(tournament.id)} style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}>
                <strong>{tournament.name}</strong><br />
                <StatusChip value={tournament.status} /> <StatusChip value={tournament.registration_status || "registration n/a"} />
                <p style={{ color: "#64748b", margin: "0.45rem 0 0" }}>{dateLabel(tournament.start_date)} – {dateLabel(tournament.end_date)}</p>
                <p style={{ color: "#64748b", margin: "0.2rem 0 0" }}>{tournament.registration_count ?? 0} registrations · {tournament.selection_count ?? 0} selections</p>
              </button>
            ))}
          </div>
        </article>
      ) : null}

      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{detail.tournament.name}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Tournament status</strong><br /><StatusChip value={detail.tournament.status} /></div>
              <div><strong>Registration status</strong><br /><StatusChip value={detail.tournament.registration_status} /></div>
              <div><strong>Registrations</strong><br />{detail.summary.registrations}</div>
              <div><strong>Selections</strong><br />{detail.summary.selections}</div>
            </div>
            <p style={{ color: "#64748b" }}><strong>By registration status:</strong> {kvList(detail.summary.by_registration_status)}</p>
            <p style={{ color: "#64748b" }}><strong>By payment status:</strong> {kvList(detail.summary.by_payment_status)}</p>
          </article>

          {selectedRegistration ? (
            <article style={{ ...cardStyle, background: "#f8fafc" }}>
              <h2 style={{ marginTop: 0 }}>Edit registration</h2>
              <p style={{ color: "#475569" }}>
                Editing <strong>{selectedRegistration.display_name}</strong>. Status/payment changes are audit-flagged. Type <code>SAVE REGISTRATION</code> to confirm.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
                <label><strong>Registration status</strong><br />
                  <select value={registrationEdit.registrationStatus} onChange={(event) => setRegistrationEdit((current) => ({ ...current, registrationStatus: event.target.value }))} style={inputStyle}>
                    {REGISTRATION_STATUS_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
                <label><strong>Payment status</strong><br />
                  <select value={registrationEdit.paymentStatus} onChange={(event) => setRegistrationEdit((current) => ({ ...current, paymentStatus: event.target.value }))} style={inputStyle}>
                    {PAYMENT_STATUS_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
                <label><strong>Type SAVE REGISTRATION</strong><br />
                  <input value={registrationEdit.confirm} onChange={(event) => setRegistrationEdit((current) => ({ ...current, confirm: event.target.value }))} style={inputStyle} />
                </label>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Admin notes</strong><br />
                <textarea value={registrationEdit.notes} onChange={(event) => setRegistrationEdit((current) => ({ ...current, notes: event.target.value }))} rows={3} style={inputStyle} />
              </label>
              <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button type="button" onClick={saveRegistrationEdit} disabled={busy || !accessToken || registrationEdit.confirm.trim().toUpperCase() !== "SAVE REGISTRATION"} style={buttonStyle}>{busy ? "Saving…" : "Save registration"}</button>
                <button type="button" onClick={() => selectedRegistration ? setRegistrationEdit(editStateFromRegistration(selectedRegistration)) : undefined} disabled={busy} style={ghostButtonStyle}>Reset fields</button>
              </p>
            </article>
          ) : null}

          {selectedRegistration && registrationSelections.length ? (
            <article style={{ ...cardStyle, background: "#f8fafc" }}>
              <h2 style={{ marginTop: 0 }}>Edit event entry / partner</h2>
              <p style={{ color: "#475569" }}>
                Move the selected registration entry to another division or update partner-board fields. Type <code>SAVE SELECTION</code> to confirm. Entries already imported into a draw cannot be moved.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
                <label><strong>Event entry</strong><br />
                  <select value={selectedSelectionId} onChange={(event) => selectSelection(event.target.value)} style={inputStyle}>
                    <option value="">Choose an entry…</option>
                    {registrationSelections.map((selection) => <option key={selection.id} value={selection.id}>{selection.event_label || selection.event_option_id || selection.id}</option>)}
                  </select>
                </label>
                <label><strong>Division</strong><br />
                  <select value={selectionEdit.eventOptionId} onChange={(event) => setSelectionEdit((current) => ({ ...current, eventOptionId: event.target.value }))} style={inputStyle}>
                    <option value="">Choose a division…</option>
                    {detail.event_options.map((eventOption) => <option key={String(eventOption.id)} value={String(eventOption.id)}>{eventOptionLabel(eventOption)}</option>)}
                  </select>
                </label>
                <label><strong>Partner mode</strong><br />
                  <select value={selectionEdit.partnerMode} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerMode: event.target.value }))} style={inputStyle}>
                    {PARTNER_MODE_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
                <label><strong>Type SAVE SELECTION</strong><br />
                  <input value={selectionEdit.confirm} onChange={(event) => setSelectionEdit((current) => ({ ...current, confirm: event.target.value }))} style={inputStyle} />
                </label>
                <label><strong>Partner name</strong><br /><input value={selectionEdit.partnerName} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerName: event.target.value }))} style={inputStyle} /></label>
                <label><strong>Partner email</strong><br /><input value={selectionEdit.partnerEmail} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerEmail: event.target.value }))} style={inputStyle} /></label>
                <label><strong>Partner phone</strong><br /><input value={selectionEdit.partnerPhone} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerPhone: event.target.value }))} style={inputStyle} /></label>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Partner note</strong><br />
                <textarea value={selectionEdit.partnerNote} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerNote: event.target.value }))} rows={3} style={inputStyle} />
              </label>
              <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button type="button" onClick={saveSelectionEdit} disabled={busy || !accessToken || !selectedSelection || selectionEdit.confirm.trim().toUpperCase() !== "SAVE SELECTION"} style={buttonStyle}>{busy ? "Saving…" : "Save event entry"}</button>
                <button type="button" onClick={() => selectedSelection ? setSelectionEdit(editStateFromSelection(selectedSelection)) : undefined} disabled={busy} style={ghostButtonStyle}>Reset fields</button>
              </p>
            </article>
          ) : null}

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Event options</h2>
            {detail.event_options.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                  <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Event family</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Division</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Format</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Status</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Enabled</th></tr></thead>
                  <tbody>{detail.event_options.map((row) => <tr key={String(row.id)}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{String(row.event_family_label || "—")}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{String(row.division_name || "—")}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{String(row.event_format_default || "—")}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><StatusChip value={String(row.status || "—")} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.enabled === false ? "No" : "Yes"}</td></tr>)}</tbody>
                </table>
              </div>
            ) : <p style={{ color: "#64748b" }}>No event options are configured yet.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Registrations</h2>
            {detail.registrations.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "980px" }}>
                  <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Registrant</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Email</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Registration</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Payment</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Selections</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Partner contact</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Action</th></tr></thead>
                  <tbody>{detail.registrations.map((row) => <tr key={row.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.display_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.email || "—"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><StatusChip value={row.registration_status} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><StatusChip value={row.payment_status} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.selection_count ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wants_partner_board_contact ? "Yes" : "No"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><button type="button" onClick={() => selectRegistration(row)} style={ghostButtonStyle}>Edit</button></td></tr>)}</tbody>
                </table>
              </div>
            ) : <p style={{ color: "#64748b" }}>No registrations are loaded for this tournament.</p>}
          </article>
        </>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
