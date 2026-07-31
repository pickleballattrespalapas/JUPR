"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournament,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentRegistration,
  AdminTournamentSelection,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
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
};

type SelectionEdit = {
  eventOptionId: string;
  partnerMode: string;
  partnerNote: string;
  expectedUpdatedAt: string;
};

type TournamentEdit = { name: string; startDate: string; endDate: string };

const REGISTRATION_STATUS_OPTIONS = ["confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "refunded"];
const EDITABLE_PARTNER_MODE_OPTIONS = ["NONE", "NEEDS_PARTNER"];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

class ApiRequestError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

function isRecoveryConflict(error: ApiRequestError): boolean {
  return /recovery|reconcil|partial|response-lost|completion audit|durable (?:result|completed)|stored result|operation key/i.test(error.message);
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
    notes: row?.notes || ""
  };
}

function editStateFromSelection(row: AdminTournamentSelection | null): SelectionEdit {
  return {
    eventOptionId: row?.event_option_id || "",
    partnerMode: row?.partner_mode || "NONE",
    partnerNote: row?.partner_note || "",
    expectedUpdatedAt: row?.updated_at || ""
  };
}

function editStateFromTournament(row: AdminTournament | null): TournamentEdit {
  return { name: row?.name || "", startDate: row?.start_date || "", endDate: row?.end_date || "" };
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
  const [tournamentEdit, setTournamentEdit] = useState<TournamentEdit>(() => editStateFromTournament(null));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedTournamentState);
  const detailRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);
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
    if (!response.ok) throw new ApiRequestError(String(payload?.detail || `API error (${response.status})`), response.status);
    return payload as T;
  }

  async function refreshDetail(tournamentId: string): Promise<AdminTournamentDetailResponse> {
    return requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`);
  }

  function clearProtectedTournamentState() {
    detailRequest.invalidate();
    setBusy(false); setMessage(null);
    setTournaments([]); setSelectedId(""); setDetail(null);
    setSelectedRegistrationId(""); setSelectedSelectionId("");
    setRegistrationEdit(editStateFromRegistration(null));
    setSelectionEdit(editStateFromSelection(null));
    setTournamentEdit(editStateFromTournament(null));
  }

  async function loadTournaments() {
    const selectedBeforeRefresh = selectedId;
    const selectedRegistrationBeforeRefresh = selectedRegistrationId;
    const selectedSelectionBeforeRefresh = selectedSelectionId;
    const generation = listRequest.begin();
    detailRequest.invalidate();
    setBusy(true);
    setMessage(null);
    setDetail(null);
    setSelectedRegistrationId("");
    setSelectedSelectionId("");
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments${suffix}`);
      if (!listRequest.isCurrent(generation)) return;
      const nextTournaments = payload.tournaments || [];
      const selectionStillAvailable = Boolean(selectedBeforeRefresh && nextTournaments.some((row) => row.id === selectedBeforeRefresh));
      setTournaments(nextTournaments);
      setMessage(nextTournaments.length ? `Loaded ${payload.count ?? nextTournaments.length} tournament(s).` : "No tournaments match this view.");
      if (selectionStillAvailable) await loadDetail(selectedBeforeRefresh, selectedRegistrationBeforeRefresh, selectedSelectionBeforeRefresh);
      else setSelectedId("");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadDetail(tournamentId: string, preferredRegistrationId = "", preferredSelectionId = "") {
    const generation = detailRequest.begin();
    setSelectedId(tournamentId);
    setDetail(null);
    setMessage(null);
    setSelectedRegistrationId("");
    setSelectedSelectionId("");
    if (!tournamentId) return;
    setBusy(true);
    try {
      const payload = await refreshDetail(tournamentId);
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setTournamentEdit(editStateFromTournament(payload.tournament));
      const nextRegistration = payload.registrations.find((row) => row.id === preferredRegistrationId) || null;
      const availableSelections = nextRegistration
        ? payload.selections.filter((row) => row.registration_id === nextRegistration.id)
        : [];
      const nextSelection = availableSelections.find((row) => row.id === preferredSelectionId) || availableSelections[0] || null;
      setSelectedRegistrationId(nextRegistration?.id || "");
      setRegistrationEdit(editStateFromRegistration(nextRegistration));
      setSelectedSelectionId(nextSelection?.id || "");
      setSelectionEdit(editStateFromSelection(nextSelection));
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament detail.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveTournamentEdit(confirmationText: string) {
    if (!detail?.tournament.updated_at) { setMessage("Reload: this tournament is missing its write version."); return; }
    const generation = actionRequest.begin();
    const tournamentId = detail.tournament.id;
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
        { method: "PATCH", body: JSON.stringify({ name: tournamentEdit.name, start_date: tournamentEdit.startDate || null, end_date: tournamentEdit.endDate || null, expected_updated_at: detail.tournament.updated_at, confirmation_text: confirmationText, source: "next_tournament_admin_tournament_editor" }) }
      );
      if (!actionRequest.isCurrent(generation)) return;
      const refreshed = await refreshDetail(tournamentId);
      if (!actionRequest.isCurrent(generation)) return;
      setDetail(refreshed); setTournamentEdit(editStateFromTournament(refreshed.tournament));
      setMessage(payload.idempotent_replay ? "Tournament response reconciled from the durable operation." : "Tournament details saved and audit-completed.");
    } catch (error) {
      if (!actionRequest.isCurrent(generation)) return;
      if (error instanceof ApiRequestError && error.status === 409) {
        if (isRecoveryConflict(error)) { setMessage(`${error.message} Keep these reviewed values and retry only this identical request to reconcile, or use the Streamlit fallback.`); }
        else try {
          const refreshed = await refreshDetail(tournamentId);
          if (!actionRequest.isCurrent(generation)) return;
          setDetail(refreshed); setTournamentEdit(editStateFromTournament(refreshed.tournament)); setMessage("Tournament data changed. The authoritative values were reloaded; review before submitting a new request.");
        } catch (refreshError) {
          if (actionRequest.isCurrent(generation)) setMessage(refreshError instanceof Error ? refreshError.message : "Unable to recover tournament state.");
        }
      } else setMessage(error instanceof Error ? error.message : "Unable to save tournament details.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
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

  async function saveRegistrationEdit(confirmationText: string) {
    if (!detail || !selectedRegistration) {
      setMessage("Select a registration before saving.");
      return;
    }
    if (!selectedRegistration.updated_at) {
      setMessage("Reload: this registration is missing its write version.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = detail.tournament.id;
    const registrationId = selectedRegistration.id;
    const requestedRegistration = selectedRegistration;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registrations/${encodeURIComponent(registrationId)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            registration_status: registrationEdit.registrationStatus,
            payment_status: registrationEdit.paymentStatus,
            notes: registrationEdit.notes,
            expected_updated_at: selectedRegistration.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_admin_registration_editor"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      const refreshed = await refreshDetail(tournamentId);
      if (!actionRequest.isCurrent(generation)) return;
      setDetail(refreshed);
      setSelectedRegistrationId(payload.registration?.id || registrationId);
      setRegistrationEdit(editStateFromRegistration(payload.registration || requestedRegistration));
      setMessage(payload.idempotent_replay ? "Registration response reconciled from the durable operation." : "Registration saved and audit-completed.");
    } catch (error) {
      if (!actionRequest.isCurrent(generation)) return;
      if (error instanceof ApiRequestError && error.status === 409) {
        if (isRecoveryConflict(error)) { setMessage(`${error.message} Keep these reviewed values and retry only this identical request to reconcile, or use the Streamlit fallback.`); }
        else try {
          const refreshed = await refreshDetail(tournamentId);
          if (!actionRequest.isCurrent(generation)) return;
          const latest = refreshed.registrations.find((row) => row.id === registrationId) || null;
          setDetail(refreshed); setRegistrationEdit(editStateFromRegistration(latest)); setMessage("Registration data changed. The authoritative row was reloaded; review before submitting a new request.");
        } catch (refreshError) {
          if (actionRequest.isCurrent(generation)) setMessage(refreshError instanceof Error ? refreshError.message : "Unable to recover registration state.");
        }
      } else setMessage(error instanceof Error ? error.message : "Unable to save registration.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveSelectionEdit(confirmationText: string) {
    if (!detail || !selectedSelection) {
      setMessage("Select an event entry before saving.");
      return;
    }
    if (!selectionEdit.expectedUpdatedAt) {
      setMessage("Unable to save: this event entry has no version timestamp. Reload the tournament and try again.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = detail.tournament.id;
    const selectionId = selectedSelection.id;
    const requestedSelection = selectedSelection;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/selections/${encodeURIComponent(selectionId)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            event_option_id: selectionEdit.eventOptionId,
            partner_mode: selectionEdit.partnerMode,
            partner_note: selectionEdit.partnerNote,
            expected_updated_at: selectionEdit.expectedUpdatedAt,
            confirmation_text: confirmationText,
            source: "next_tournament_admin_selection_editor"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      const refreshed = await refreshDetail(tournamentId);
      if (!actionRequest.isCurrent(generation)) return;
      setDetail(refreshed);
      setSelectedSelectionId(payload.selection?.id || selectionId);
      setSelectionEdit(editStateFromSelection(payload.selection || requestedSelection));
      setMessage(payload.idempotent_replay ? "Event-entry response reconciled from the durable operation." : "Event entry saved and audit-completed.");
    } catch (error) {
      if (!actionRequest.isCurrent(generation)) return;
      if (error instanceof ApiRequestError && error.status === 409) {
        if (isRecoveryConflict(error)) {
          setMessage(`${error.message} Keep these reviewed values and retry only this identical request to reconcile, or use the Streamlit fallback.`);
        } else try {
          const refreshed = await refreshDetail(tournamentId);
          if (!actionRequest.isCurrent(generation)) return;
          const latestSelection = refreshed.selections.find((row) => row.id === selectionId) || null;
          setDetail(refreshed);
          setSelectedSelectionId(latestSelection?.id || "");
          setSelectionEdit(editStateFromSelection(latestSelection));
          setMessage("Unable to save: this event entry changed after you loaded it. The latest values were reloaded; review them before saving again.");
        } catch (refreshError) {
          if (actionRequest.isCurrent(generation)) {
            setSelectionEdit((current) => ({ ...current, expectedUpdatedAt: "" }));
            setMessage(refreshError instanceof Error ? `Unable to reload the changed event entry: ${refreshError.message}` : "Unable to reload the changed event entry.");
          }
        }
      } else {
        setMessage(error instanceof Error ? error.message : "Unable to save event entry.");
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments, includeArchived ? "archived" : "active");

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
      <article style={{ ...cardStyle, display: detail ? "none" : "block" }}>
        <h2 style={{ marginTop: 0 }}>Create or open a tournament</h2>
        <p style={{ color: "#475569" }}>
          This workflow loads tournament setup, registration summaries, event options, and registrants through FastAPI. Registration and event-entry edits are guarded by role checks, audit logging, and an action-specific confirmation dialog.
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
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} disabled={busy} />
          Include archived tournaments
        </label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh tournaments"}</button>
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      {tournaments.length ? (
        <article style={{ ...cardStyle, display: detail ? "none" : "block" }}>
          <h2 style={{ marginTop: 0 }}>Open tournament</h2>
          <select value={selectedId} onChange={(event) => loadDetail(event.target.value)} disabled={busy} style={inputStyle}>
            <option value="">Choose a tournament…</option>
            {tournaments.map((tournament) => (
              <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status} · {tournament.registration_count ?? 0} registrations</option>
            ))}
          </select>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
            {tournaments.map((tournament) => (
              <button key={tournament.id} type="button" onClick={() => loadDetail(tournament.id)} disabled={busy} style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}>
                <strong>{tournament.name}</strong><br />
                <StatusChip value={tournament.status} /> <StatusChip value={tournament.registration_status || "registration n/a"} />
                <p style={{ color: "#64748b", margin: "0.45rem 0 0" }}>{dateLabel(tournament.start_date)} – {dateLabel(tournament.end_date)}</p>
                <p style={{ color: "#64748b", margin: "0.2rem 0 0" }}>{tournament.registration_count ?? 0} registrations · {tournament.selection_count ?? 0} selections</p>
              </button>
            ))}
          </div>
        </article>
      ) : <article style={cardStyle}><p style={{ color: "#64748b" }}>{busy ? "Loading tournaments…" : "No tournaments match this view."}</p></article>}

      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{detail.tournament.name}</h2>
            <p style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap", alignItems: "center" }}><strong>Selected tournament</strong><button type="button" onClick={() => { setDetail(null); setSelectedId(""); setSelectedRegistrationId(""); setSelectedSelectionId(""); }} style={ghostButtonStyle}>Change tournament</button></p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Tournament status</strong><br /><StatusChip value={detail.tournament.status} /></div>
              <div><strong>Registration status</strong><br /><StatusChip value={detail.tournament.registration_status} /></div>
              <div><strong>Registrations</strong><br />{detail.summary.registrations}</div>
              <div><strong>Selections</strong><br />{detail.summary.selections}</div>
            </div>
            <p style={{ color: "#64748b" }}><strong>By registration status:</strong> {kvList(detail.summary.by_registration_status)}</p>
            <p style={{ color: "#64748b" }}><strong>By payment status:</strong> {kvList(detail.summary.by_payment_status)}</p>
          </article>

          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <h2 style={{ marginTop: 0 }}>Tournament home</h2>
            <p style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap" }}>
              <Link href="/admin/tournament-setup">Setup</Link>
              <Link href="/admin/tournaments/registrations">Registrations and reports</Link>
              <Link href="/admin/tournaments/bulk">Bulk actions</Link>
              <Link href="/admin/tournaments/commerce">Extras and fulfillment</Link>
              <Link href="/admin/tournaments/team-competition">Ratings and team play</Link>
              <Link href="/admin/tournaments/ops">Operations and results</Link>
              <Link href="/admin/tournament-live">Live runner</Link>
              <Link href="/admin/tournaments/status">Status and recovery</Link>
            </p>
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Tournament settings</h2>
            <p style={{ color: "#475569" }}>Name and date edits use the loaded tournament version; a stale response reloads authoritative data before another attempt.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
              <label><strong>Name</strong><br /><input value={tournamentEdit.name} onChange={(event) => setTournamentEdit((current) => ({ ...current, name: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Start date</strong><br /><input type="date" value={dateLabel(tournamentEdit.startDate) === "—" ? "" : dateLabel(tournamentEdit.startDate)} onChange={(event) => setTournamentEdit((current) => ({ ...current, startDate: event.target.value }))} style={inputStyle} /></label>
              <label><strong>End date</strong><br /><input type="date" value={dateLabel(tournamentEdit.endDate) === "—" ? "" : dateLabel(tournamentEdit.endDate)} onChange={(event) => setTournamentEdit((current) => ({ ...current, endDate: event.target.value }))} style={inputStyle} /></label>
            </div>
            <p><ConfirmAction triggerLabel="Save tournament" title="Save tournament details?" description={`This updates the name and dates for ${detail.tournament.name}. The loaded version must still be current.`} confirmLabel="Yes, save tournament" confirmationText="SAVE TOURNAMENT" disabled={!detail.tournament.updated_at} busy={busy} onConfirm={saveTournamentEdit} /></p>
          </article>

          {selectedRegistration ? (
            <article style={{ ...cardStyle, background: "#f8fafc" }}>
              <h2 style={{ marginTop: 0 }}>Edit registration</h2>
              <p style={{ color: "#475569" }}>
                Editing <strong>{selectedRegistration.display_name}</strong>. Status and payment changes are audit-flagged.
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
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Admin notes</strong><br />
                <textarea value={registrationEdit.notes} onChange={(event) => setRegistrationEdit((current) => ({ ...current, notes: event.target.value }))} rows={3} style={inputStyle} />
              </label>
              <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <ConfirmAction triggerLabel="Save registration" title="Save this registration update?" description={`This changes ${selectedRegistration.display_name}'s registration to ${registrationEdit.registrationStatus} with payment status ${registrationEdit.paymentStatus}, and replaces the saved admin notes with the current notes field.`} confirmLabel="Yes, save registration" confirmationText="SAVE REGISTRATION" tone={registrationEdit.registrationStatus === "cancelled" || registrationEdit.paymentStatus === "refunded" ? "danger" : "default"} disabled={!accessToken || !selectedRegistration.updated_at} busy={busy} onConfirm={saveRegistrationEdit} />
                <button type="button" onClick={() => selectedRegistration ? setRegistrationEdit(editStateFromRegistration(selectedRegistration)) : undefined} disabled={busy} style={ghostButtonStyle}>Reset fields</button>
              </p>
            </article>
          ) : null}

          {selectedRegistration && registrationSelections.length ? (
            <article style={{ ...cardStyle, background: "#f8fafc" }}>
              <h2 style={{ marginTop: 0 }}>Edit event entry / partner</h2>
              <p style={{ color: "#475569" }}>
                Move the selected registration entry to another division or update its partner-board mode and note. Linked partner identity is read-only here. Entries already imported into a draw cannot be moved.
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
                    {(selectedSelection?.partner_mode === "HAS_PARTNER" ? ["HAS_PARTNER", ...EDITABLE_PARTNER_MODE_OPTIONS] : EDITABLE_PARTNER_MODE_OPTIONS).map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
                <div><strong>Partner name</strong><br /><span>{selectedSelection?.partner_name || "—"}</span></div>
                <div><strong>Partner email</strong><br /><span>{selectedSelection?.partner_email || "—"}</span></div>
                <div><strong>Partner phone</strong><br /><span>{selectedSelection?.partner_phone || "—"}</span></div>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Partner note</strong><br />
                <textarea value={selectionEdit.partnerNote} onChange={(event) => setSelectionEdit((current) => ({ ...current, partnerNote: event.target.value }))} rows={3} style={inputStyle} />
              </label>
              {!selectionEdit.expectedUpdatedAt ? <p style={{ color: "#b91c1c" }}>This entry is missing a version timestamp. Reload the tournament before attempting to save.</p> : null}
              <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <ConfirmAction triggerLabel="Save event entry" title="Save this event-entry update?" description="This changes the selected division or partner-board details for this tournament registration." confirmLabel="Yes, save event entry" confirmationText="SAVE SELECTION" disabled={!accessToken || !selectedSelection || !selectionEdit.expectedUpdatedAt} busy={busy} onConfirm={saveSelectionEdit} />
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

      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("changed") || message.toLowerCase().includes("reload") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
