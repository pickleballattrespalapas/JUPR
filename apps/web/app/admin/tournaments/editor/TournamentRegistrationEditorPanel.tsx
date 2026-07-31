"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentRegistration,
  AdminTournamentSelection,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  tournamentId: string;
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
};

const REGISTRATION_STATUS_OPTIONS = ["confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "refunded"];
const PARTNER_MODE_OPTIONS = ["NONE", "NEEDS_PARTNER"];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const tableHeaderStyle = { textAlign: "left" as const, padding: "0.55rem", borderBottom: "1px solid #cbd5e1" };
const tableCellStyle = { padding: "0.55rem", borderBottom: "1px solid #e2e8f0", verticalAlign: "top" as const };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function registrationEdit(row: AdminTournamentRegistration | null): RegistrationEdit {
  return {
    registrationStatus: row?.registration_status || "confirmed",
    paymentStatus: row?.payment_status || "unpaid",
    notes: row?.notes || ""
  };
}

function selectionEdit(row: AdminTournamentSelection | null): SelectionEdit {
  return {
    eventOptionId: row?.event_option_id || "",
    partnerMode: row?.partner_mode || "NONE",
    partnerNote: row?.partner_note || ""
  };
}

function eventLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Event");
}

export default function TournamentRegistrationEditorPanel({ apiBase, clubId, status, tournamentId }: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [search, setSearch] = useState("");
  const [selectedRegistrationId, setSelectedRegistrationId] = useState("");
  const [selectedSelectionId, setSelectedSelectionId] = useState("");
  const [registrationDraft, setRegistrationDraft] = useState<RegistrationEdit>(() => registrationEdit(null));
  const [selectionDraft, setSelectionDraft] = useState<SelectionEdit>(() => selectionEdit(null));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${tournamentId}`, clearProtectedState);
  const actionRequest = useLatestRequestGuard(accessToken);

  const selectedRegistration = detail?.registrations.find((row) => row.id === selectedRegistrationId) || null;
  const registrationSelections = (detail?.selections || []).filter((row) => row.registration_id === selectedRegistrationId);
  const selectedSelection = registrationSelections.find((row) => row.id === selectedSelectionId) || null;
  const visibleRegistrations = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return detail?.registrations || [];
    return (detail?.registrations || []).filter((row) => [row.display_name, row.email, row.phone].join(" ").toLowerCase().includes(needle));
  }, [detail, search]);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setSearch("");
    setSelectedRegistrationId("");
    setSelectedSelectionId("");
    setRegistrationDraft(registrationEdit(null));
    setSelectionDraft(selectionEdit(null));
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before editing tournament registrations.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadDetail(preferredRegistrationId = selectedRegistrationId, preferredSelectionId = selectedSelectionId) {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      const registration = payload.registrations.find((row) => row.id === preferredRegistrationId) || payload.registrations[0] || null;
      setSelectedRegistrationId(registration?.id || "");
      setRegistrationDraft(registrationEdit(registration));
      const selections = registration ? payload.selections.filter((row) => row.registration_id === registration.id) : [];
      const selection = selections.find((row) => row.id === preferredSelectionId) || selections[0] || null;
      setSelectedSelectionId(selection?.id || "");
      setSelectionDraft(selectionEdit(selection));
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament registrations.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function chooseRegistration(row: AdminTournamentRegistration) {
    setSelectedRegistrationId(row.id);
    setRegistrationDraft(registrationEdit(row));
    const firstSelection = (detail?.selections || []).find((selection) => selection.registration_id === row.id) || null;
    setSelectedSelectionId(firstSelection?.id || "");
    setSelectionDraft(selectionEdit(firstSelection));
    setMessage(null);
  }

  function chooseSelection(selectionId: string) {
    const selection = registrationSelections.find((row) => row.id === selectionId) || null;
    setSelectedSelectionId(selection?.id || "");
    setSelectionDraft(selectionEdit(selection));
    setMessage(null);
  }

  async function saveRegistration(confirmationText: string) {
    if (!selectedRegistration?.updated_at) {
      setMessage("Reload this registration before saving.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registrations/${encodeURIComponent(selectedRegistration.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            registration_status: registrationDraft.registrationStatus,
            payment_status: registrationDraft.paymentStatus,
            notes: registrationDraft.notes,
            expected_updated_at: selectedRegistration.updated_at,
            confirmation_text: confirmationText,
            source: "next_selected_tournament_registration_editor"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedRegistration.id, selectedSelectionId);
      if (actionRequest.isCurrent(generation)) setMessage("Registration saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save registration.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveSelection(confirmationText: string) {
    if (!selectedSelection?.updated_at) {
      setMessage("Reload this event entry before saving.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/selections/${encodeURIComponent(selectedSelection.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            event_option_id: selectionDraft.eventOptionId,
            partner_mode: selectionDraft.partnerMode,
            partner_note: selectionDraft.partnerNote,
            expected_updated_at: selectedSelection.updated_at,
            confirmation_text: confirmationText,
            source: "next_selected_tournament_selection_editor"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedRegistrationId, selectedSelection.id);
      if (actionRequest.isCurrent(generation)) setMessage("Event entry saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save event entry.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${tournamentId}` : "", loadDetail);

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}>Tournament registration editing is unavailable.</article>;
  if (sessionLoading) return <p role="status">Checking admin access…</p>;
  if (!accessToken) return <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}><h2 style={{ marginTop: 0 }}>Admin sign-in required</h2><p><Link href="/admin/login">Open admin login</Link></p></article>;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {message ? <p role="status" style={{ color: /unable|error|required|reload/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Registrations</h2>
            <label><strong>Search</strong><br /><input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name, email, or phone" style={inputStyle} /></label>
            <div style={{ overflowX: "auto", marginTop: "0.75rem" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                <thead><tr><th style={tableHeaderStyle}>Registrant</th><th style={tableHeaderStyle}>Status</th><th style={tableHeaderStyle}>Payment</th><th style={tableHeaderStyle}>Entries</th><th style={tableHeaderStyle}>Action</th></tr></thead>
                <tbody>{visibleRegistrations.map((row) => <tr key={row.id}><td style={tableCellStyle}>{row.display_name}<br /><small>{row.email || "—"}</small></td><td style={tableCellStyle}>{row.registration_status || "—"}</td><td style={tableCellStyle}>{row.payment_status || "—"}</td><td style={tableCellStyle}>{row.selection_count ?? detail.selections.filter((selection) => selection.registration_id === row.id).length}</td><td style={tableCellStyle}><button type="button" onClick={() => chooseRegistration(row)} disabled={busy}>{selectedRegistrationId === row.id ? "Selected" : "Edit"}</button></td></tr>)}</tbody>
              </table>
            </div>
          </article>

          {selectedRegistration ? (
            <article style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>Edit {selectedRegistration.display_name}</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
                <label><strong>Registration status</strong><br /><select value={registrationDraft.registrationStatus} onChange={(event) => setRegistrationDraft((current) => ({ ...current, registrationStatus: event.target.value }))} style={inputStyle}>{REGISTRATION_STATUS_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label><strong>Payment status</strong><br /><select value={registrationDraft.paymentStatus} onChange={(event) => setRegistrationDraft((current) => ({ ...current, paymentStatus: event.target.value }))} style={inputStyle}>{PAYMENT_STATUS_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Admin notes</strong><br /><textarea value={registrationDraft.notes} onChange={(event) => setRegistrationDraft((current) => ({ ...current, notes: event.target.value }))} rows={3} style={inputStyle} /></label>
              <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Save registration"} title="Save this registration update?" description={`Update ${selectedRegistration.display_name}'s registration and offline payment status.`} confirmLabel="Yes, save registration" confirmationText="SAVE REGISTRATION" busy={busy} onConfirm={saveRegistration} /></p>
            </article>
          ) : null}

          {selectedRegistration && registrationSelections.length ? (
            <article style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>Event entries</h2>
              <label><strong>Select entry</strong><br /><select value={selectedSelectionId} onChange={(event) => chooseSelection(event.target.value)} style={inputStyle}>{registrationSelections.map((row) => <option key={row.id} value={row.id}>{row.event_label || row.event_option_id || row.id}</option>)}</select></label>
              {selectedSelection ? (
                <div style={{ marginTop: "0.75rem" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem" }}>
                    <label><strong>Division</strong><br /><select value={selectionDraft.eventOptionId} onChange={(event) => setSelectionDraft((current) => ({ ...current, eventOptionId: event.target.value }))} style={inputStyle}><option value="">Choose division</option>{detail.event_options.map((row) => <option key={String(row.id)} value={String(row.id)}>{eventLabel(row)}</option>)}</select></label>
                    <label><strong>Partner mode</strong><br /><select value={selectionDraft.partnerMode} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerMode: event.target.value }))} style={inputStyle}>{PARTNER_MODE_OPTIONS.map((value) => <option key={value} value={value}>{value === "NONE" ? "No partner request" : "Needs partner"}</option>)}</select></label>
                  </div>
                  <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Partner note</strong><br /><textarea value={selectionDraft.partnerNote} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerNote: event.target.value }))} rows={3} style={inputStyle} /></label>
                  <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Save event entry"} title="Save this event-entry update?" description="Update the selected division and partner-board state using the loaded row version." confirmLabel="Yes, save event entry" confirmationText="SAVE EVENT ENTRY" busy={busy} onConfirm={saveSelection} /></p>
                </div>
              ) : null}
            </article>
          ) : null}
        </>
      ) : busy ? <p role="status">Loading registrations…</p> : null}
    </div>
  );
}
