"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type {
  AdminTournament,
  AdminTournamentBroadcastPreviewResponse,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentRegistration,
  AdminTournamentSelection,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentEmailDelivery from "./TournamentEmailDelivery";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  initialTournamentId: string;
  initialTournamentName: string;
  initialDrawId: string;
};
type ImportHandoff = { ok: boolean; dry_run: true; write_count: 0; state_fingerprint: string; confirmed_registration_count: number; imported_selection_count: number; direct_import_available: false; ops_path: string; required_ops_confirmation: string; integrity_notice: string };

const REGISTRATION_STATUS_OPTIONS = ["", "confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["", "unpaid", "paid", "refunded"];
const PARTNER_MODE_OPTIONS = ["", "NONE", "HAS_PARTNER", "NEEDS_PARTNER"];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const tableHeaderStyle = { textAlign: "left" as const, padding: "0.55rem", borderBottom: "1px solid #cbd5e1" };
const tableCellStyle = { padding: "0.55rem", borderBottom: "1px solid #e2e8f0", verticalAlign: "top" as const };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function eventLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Event");
}

function dayLabel(row: Record<string, unknown>): string {
  return String(row.label || row.event_date || row.date || row.id || "Day");
}

function emailExclusion(registration: AdminTournamentRegistration, includeCancelled: boolean): string {
  if (!registration.email?.trim()) return "No email address";
  if (registration.registration_status === "cancelled" && !includeCancelled) return "Cancelled registration";
  return "";
}

function downloadText(filename: string, content: string, mime = "text/csv;charset=utf-8"): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export default function RegistrationManagementPanel({ apiBase, clubId, status, initialTournamentId, initialTournamentName, initialDrawId }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState(initialTournamentId);
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [importHandoff, setImportHandoff] = useState<ImportHandoff | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const sessionScope = `${accessToken}\u0000${apiBase}\u0000${clubId}\u0000${initialTournamentId}`;
  const listRequest = useLatestRequestGuard(sessionScope, clearProtectedRegistrationState);
  const detailRequest = useLatestRequestGuard(sessionScope);
  const actionRequest = useLatestRequestGuard(sessionScope);

  const [registrationStatus, setRegistrationStatus] = useState("");
  const [paymentStatus, setPaymentStatus] = useState("");
  const [partnerMode, setPartnerMode] = useState("");
  const [registrationDayId, setRegistrationDayId] = useState("");
  const [eventOptionId, setEventOptionId] = useState("");
  const [search, setSearch] = useState("");

  const [broadcastSubject, setBroadcastSubject] = useState("");
  const [broadcastMessage, setBroadcastMessage] = useState("");
  const [includeCancelled, setIncludeCancelled] = useState(false);
  const [selectedRegistrationIds, setSelectedRegistrationIds] = useState<string[]>([]);
  const [broadcastPreview, setBroadcastPreview] = useState<AdminTournamentBroadcastPreviewResponse | null>(null);
  const [broadcastPreviewScope, setBroadcastPreviewScope] = useState("");
  const [previewBusy, setPreviewBusy] = useState(false);
  const previewScope = JSON.stringify([accessToken, apiBase, clubId, initialTournamentId, selectedTournamentId, selectedRegistrationIds, includeCancelled, broadcastSubject, broadcastMessage]);
  const previewRequest = useLatestRequestGuard(previewScope, clearBroadcastPreview);
  const currentPreview = broadcastPreviewScope === previewScope ? broadcastPreview : null;

  function clearBroadcastPreview() {
    setBroadcastPreview(null);
    setBroadcastPreviewScope("");
    setPreviewBusy(false);
    setMessage(null);
  }

  function clearParticipantSelection() {
    previewRequest.invalidate();
    clearBroadcastPreview();
    setSelectedRegistrationIds([]);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw Object.assign(new Error(typeof payload?.detail === "string" ? payload.detail : `API error (${response.status})`), { status: response.status });
    return payload as T;
  }

  function clearProtectedRegistrationState() {
    detailRequest.invalidate();
    setBusy(false); setMessage(null);
    setTournaments([]); setSelectedTournamentId(initialTournamentId); setDetail(null); setImportHandoff(null);
    setBroadcastSubject(""); setBroadcastMessage(""); setBroadcastPreview(null);
    clearParticipantSelection();
  }

  async function loadTournaments(): Promise<void> {
    const selectedBeforeRefresh = initialTournamentId;
    const generation = listRequest.begin();
    detailRequest.invalidate();
    clearParticipantSelection();
    setBusy(true);
    setMessage(null);
    setDetail(null);
    setImportHandoff(null);
    setBroadcastPreview(null);
    try {
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments`);
      if (!listRequest.isCurrent(generation)) return;
      const nextTournaments = payload.tournaments || [];
      const selectionStillAvailable = Boolean(selectedBeforeRefresh && nextTournaments.some((row) => row.id === selectedBeforeRefresh));
      setTournaments(nextTournaments);
      setMessage(nextTournaments.length ? `Loaded ${payload.count ?? nextTournaments.length} tournament(s).` : "No tournaments are available.");
      if (selectionStillAvailable) await loadDetail(selectedBeforeRefresh, true);
      else setMessage("The selected tournament is not available to this admin session.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadDetail(tournamentId: string, preserveFilters = false): Promise<void> {
    const generation = detailRequest.begin();
    clearParticipantSelection();
    setSelectedTournamentId(tournamentId);
    setDetail(null);
    setImportHandoff(null);
    setBroadcastPreview(null);
    if (!preserveFilters) {
      setRegistrationStatus("");
      setPaymentStatus("");
      setPartnerMode("");
      setRegistrationDayId("");
      setEventOptionId("");
      setSearch("");
    }
    if (!tournamentId) return;
    setBusy(true);
    setMessage(null);
    try {
      const [detailPayload, handoffPayload] = await Promise.all([
        requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`),
        requestJson<ImportHandoff>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registrations/import-handoff`)
      ]);
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(detailPayload); setImportHandoff(handoffPayload);
      if (preserveFilters) {
        setRegistrationDayId((current) => detailPayload.days.some((row) => String(row.id || "") === current) ? current : "");
        setEventOptionId((current) => detailPayload.event_options.some((row) => String(row.id || "") === current) ? current : "");
      }
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load registration reporting data.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  const selectionByRegistration = useMemo(() => {
    const grouped = new Map<string, AdminTournamentSelection[]>();
    for (const selection of detail?.selections || []) {
      grouped.set(selection.registration_id, [...(grouped.get(selection.registration_id) || []), selection]);
    }
    return grouped;
  }, [detail]);

  const filteredRegistrations = useMemo(() => {
    if (!detail) return [];
    const needle = search.trim().toLowerCase();
    return detail.registrations.filter((registration) => {
      if (registrationStatus && registration.registration_status !== registrationStatus) return false;
      if (paymentStatus && registration.payment_status !== paymentStatus) return false;
      const selections = selectionByRegistration.get(registration.id) || [];
      if (partnerMode && (selections.length ? !selections.some((selection) => (selection.partner_mode || "NONE") === partnerMode) : partnerMode !== "NONE")) return false;
      if (registrationDayId && !selections.some((selection) => selection.registration_day_id === registrationDayId)) return false;
      if (eventOptionId && !selections.some((selection) => selection.event_option_id === eventOptionId)) return false;
      if (needle) {
        const searchable = [
          registration.display_name,
          registration.email,
          registration.phone,
          ...selections.flatMap((selection) => [selection.event_label, selection.partner_name, selection.partner_email])
        ].join(" ").toLowerCase();
        if (!searchable.includes(needle)) return false;
      }
      return true;
    });
  }, [detail, eventOptionId, partnerMode, paymentStatus, registrationDayId, registrationStatus, search, selectionByRegistration]);

  const selectedIds = new Set(selectedRegistrationIds);
  const selectedRegistrations = (detail?.registrations || []).filter((row) => selectedIds.has(row.id) && !emailExclusion(row, includeCancelled));
  const selectableFiltered = filteredRegistrations.filter((row) => !emailExclusion(row, includeCancelled));
  const selectedEmails = [...new Set(selectedRegistrations.map((row) => row.email!.trim().toLowerCase()))].sort();
  const visibleIds = new Set(filteredRegistrations.map((row) => row.id));
  const hiddenSelectedCount = selectedRegistrations.filter((row) => !visibleIds.has(row.id)).length;

  function selectParticipant(id: string, checked: boolean) {
    setMessage(null);
    setSelectedRegistrationIds((current) => checked ? [...new Set([...current, id])] : current.filter((value) => value !== id));
  }

  function selectAllFiltered() {
    setMessage(null);
    setSelectedRegistrationIds((current) => [...new Set([...current, ...selectableFiltered.map((row) => row.id)])]);
  }

  function changeIncludeCancelled(checked: boolean) {
    setMessage(null);
    setIncludeCancelled(checked);
    if (!checked) {
      const cancelledIds = new Set((detail?.registrations || []).filter((row) => row.registration_status === "cancelled").map((row) => row.id));
      setSelectedRegistrationIds((current) => current.filter((id) => !cancelledIds.has(id)));
    }
  }

  function filterQuery(): URLSearchParams {
    const query = new URLSearchParams();
    if (registrationStatus) query.set("registration_status", registrationStatus);
    if (paymentStatus) query.set("payment_status", paymentStatus);
    if (partnerMode) query.set("partner_mode", partnerMode);
    if (registrationDayId) query.set("registration_day_id", registrationDayId);
    if (eventOptionId) query.set("event_option_id", eventOptionId);
    if (search.trim()) query.set("search", search.trim());
    return query;
  }

  async function exportCsv(): Promise<void> {
    if (!apiBase || !accessToken || !detail) return;
    const generation = actionRequest.begin();
    const requestedTournamentId = detail.tournament.id;
    setBusy(true);
    setMessage(null);
    try {
      const query = filterQuery().toString();
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(requestedTournamentId)}/registrations/export.csv${query ? `?${query}` : ""}`),
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      const csv = await response.text();
      if (!actionRequest.isCurrent(generation)) return;
      downloadText(`${requestedTournamentId}-registrations.csv`, csv);
      setMessage(`Downloaded ${response.headers.get("X-JUPR-Export-Row-Count") || "filtered"} registration row(s).`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to export registrations.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function previewBroadcast(): Promise<void> {
    if (!detail || !selectedRegistrations.length) return;
    const generation = previewRequest.begin();
    const requestedTournamentId = detail.tournament.id;
    const requestedIds = selectedRegistrations.map((row) => row.id).sort();
    const requestedScope = previewScope;
    setPreviewBusy(true);
    setBroadcastPreview(null);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentBroadcastPreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(requestedTournamentId)}/registrations/broadcast-preview`,
        {
          method: "POST",
          body: JSON.stringify({
            subject: broadcastSubject,
            message: broadcastMessage,
            include_cancelled: includeCancelled,
            registration_ids: requestedIds
          })
        }
      );
      if (!previewRequest.isCurrent(generation)) return;
      if (JSON.stringify(payload.selected_registration_ids?.slice().sort()) !== JSON.stringify(requestedIds)) {
        throw new Error("Participant selection could not be verified. Refresh the page and preview again.");
      }
      if (JSON.stringify(payload.recipients.map((row) => row.email.trim().toLowerCase()).sort()) !== JSON.stringify(selectedEmails)) {
        throw new Error("Participant email details changed. Refresh the participants and review your selection.");
      }
      setBroadcastPreview(payload);
      setBroadcastPreviewScope(requestedScope);
      setMessage(`Previewed ${payload.recipient_count} unique recipient(s). No email was sent.`);
    } catch (error) {
      if (previewRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to preview broadcast recipients.");
    } finally {
      if (previewRequest.isCurrent(generation)) setPreviewBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments, `${apiBase}\u0000${clubId}\u0000${initialTournamentId}`);

  if (!status.enabled) {
    return <article style={cardStyle}><h2>Tournament Admin is disabled</h2><p>{status.warnings?.[0]}</p></article>;
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Registration reporting session</h2>
        <p style={{ color: "#475569" }}>Find participants using the filters, then select who to email.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "0.75rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready for authenticated registration reporting." : sessionLoading ? "Checking admin session…" : "Sign in before loading registrations."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <Link href="/admin/login">Open admin login</Link> : null}
          {!apiBase ? <p style={{ color: "#b91c1c" }}>The Tournament Admin API base URL is not configured.</p> : null}
        </div>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken || !apiBase} style={buttonStyle}>
          {busy ? "Refreshing…" : "Refresh tournaments"}
        </button>
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Selected tournament</h2>
          <p><strong>{tournaments.find((tournament) => tournament.id === selectedTournamentId)?.name || initialTournamentName}</strong></p>
        </article>
      ) : <article style={cardStyle}><p style={{ color: "#64748b" }}>{busy ? "Loading tournaments…" : "No tournaments are available."}</p></article>}

      {detail ? (
        <>
          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <h2 style={{ marginTop: 0 }}>Operations import handoff</h2>
            <p><strong>Registration Admin cannot bypass draw integrity.</strong> This is a read-only handoff, not an import button.</p>
            <p>{importHandoff?.integrity_notice || "Load the handoff before importing registrations."}</p>
            {importHandoff ? <><p><strong>{importHandoff.confirmed_registration_count}</strong> confirmed registrations · <strong>{importHandoff.imported_selection_count}</strong> entries already represented in a registration-sourced draw.</p><p>This page performs <strong>{importHandoff.write_count} writes</strong>. Tournament Ops owns the separate <code>{importHandoff.required_ops_confirmation}</code> mutation and refuses imports after games exist.</p><Link href={tournamentRouteHref("/admin/tournaments/ops/import", { tournamentId: initialTournamentId, tournamentName: initialTournamentName, drawId: initialDrawId })}>Open guarded Tournament Ops import</Link></> : <p>Handoff unavailable; do not import from this surface.</p>}
          </article>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Filters and CSV export</h2>
            <p style={{ color: "#475569" }}>Filters narrow the participant list and registration CSV. Your email selection stays selected as you search.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
              <label><strong>Status</strong><br /><select value={registrationStatus} onChange={(event) => setRegistrationStatus(event.target.value)} style={inputStyle}>{REGISTRATION_STATUS_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Payment</strong><br /><select value={paymentStatus} onChange={(event) => setPaymentStatus(event.target.value)} style={inputStyle}>{PAYMENT_STATUS_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Partner mode</strong><br /><select value={partnerMode} onChange={(event) => setPartnerMode(event.target.value)} style={inputStyle}>{PARTNER_MODE_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Day</strong><br /><select value={registrationDayId} onChange={(event) => setRegistrationDayId(event.target.value)} style={inputStyle}><option value="">All</option>{detail.days.map((day) => <option key={String(day.id)} value={String(day.id)}>{dayLabel(day)}</option>)}</select></label>
              <label><strong>Division</strong><br /><select value={eventOptionId} onChange={(event) => setEventOptionId(event.target.value)} style={inputStyle}><option value="">All</option>{detail.event_options.map((option) => <option key={String(option.id)} value={String(option.id)}>{eventLabel(option)}</option>)}</select></label>
              <label><strong>Search</strong><br /><input type="search" placeholder="Name or email" value={search} onChange={(event) => setSearch(event.target.value)} style={inputStyle} /></label>
            </div>
            <p><button type="button" onClick={exportCsv} disabled={busy} style={buttonStyle}>Download filtered CSV</button></p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Choose participants</h2>
            <p>Select one person, several people, or everyone matching the filters above.</p>
            <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", minHeight: "44px" }}><input type="checkbox" checked={includeCancelled} onChange={(event) => changeIncludeCancelled(event.target.checked)} disabled={busy} /> Include cancelled registrations</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", margin: "0.75rem 0" }}>
              <button type="button" onClick={selectAllFiltered} disabled={busy || !selectableFiltered.some((row) => !selectedIds.has(row.id))} style={buttonStyle}>Select all filtered ({selectableFiltered.length})</button>
              <button type="button" onClick={() => { clearParticipantSelection(); setMessage(null); }} disabled={busy || !selectedRegistrationIds.length} style={ghostButtonStyle}>Clear selection</button>
            </div>
            <p role="status" aria-live="polite"><strong>{selectedRegistrations.length} participant{selectedRegistrations.length === 1 ? "" : "s"} selected · {selectedEmails.length} email recipient{selectedEmails.length === 1 ? "" : "s"}</strong>{hiddenSelectedCount ? <><br />{hiddenSelectedCount} selected participant{hiddenSelectedCount === 1 ? " is" : "s are"} outside the current filters.</> : null}</p>
            {selectedRegistrations.length ? <details style={{ marginBottom: "0.75rem" }}>
              <summary>Review selected participants ({selectedRegistrations.length})</summary>
              <ul style={{ paddingLeft: "1.25rem" }}>{selectedRegistrations.map((row) => <li key={row.id} style={{ overflowWrap: "anywhere", marginTop: "0.5rem" }}>
                {row.display_name} · {row.email} <button type="button" aria-label={`Remove ${row.display_name} from selection`} onClick={() => selectParticipant(row.id, false)} disabled={busy} style={{ ...ghostButtonStyle, minHeight: "44px" }}>Remove</button>
              </li>)}</ul>
            </details> : <p>Choose at least one participant to preview an email.</p>}
            <fieldset style={{ margin: 0, padding: "0.5rem", border: "1px solid #e2e8f0", borderRadius: "8px", minWidth: 0 }} disabled={busy}>
              <legend>Participants matching your filters ({filteredRegistrations.length})</legend>
              <div style={{ maxHeight: "360px", overflowY: "auto" }}>
                {filteredRegistrations.map((row) => {
                  const exclusion = emailExclusion(row, includeCancelled);
                  return <label key={row.id} style={{ display: "flex", alignItems: "center", gap: "0.75rem", minHeight: "56px", padding: "0.5rem", borderBottom: "1px solid #e2e8f0", background: selectedIds.has(row.id) ? "#eff6ff" : "white" }}>
                    <input type="checkbox" aria-label={`Select ${row.display_name} (${row.email || "no email address"})`} checked={selectedIds.has(row.id) && !exclusion} disabled={Boolean(exclusion)} onChange={(event) => selectParticipant(row.id, event.target.checked)} style={{ width: "20px", height: "20px", flexShrink: 0 }} />
                    <span style={{ minWidth: 0, overflowWrap: "anywhere" }}><strong>{row.display_name}</strong><br /><span style={{ color: "#475569" }}>{row.email || "No email address"}</span>{exclusion ? <><br /><small>{exclusion}</small></> : null}</span>
                  </label>;
                })}
                {!filteredRegistrations.length ? <p>No participants match these filters.</p> : null}
              </div>
            </fieldset>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Email participants</h2>
            <p style={{ color: "#475569" }}>Write your email, review the selected recipients, then send. Each recipient receives a separate email. Shared email addresses receive one copy.</p>
            <label><strong>Subject</strong><br /><input value={broadcastSubject} disabled={busy} maxLength={200} onChange={(event) => setBroadcastSubject(event.target.value)} style={inputStyle} /></label>
            <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Message</strong><br /><textarea value={broadcastMessage} disabled={busy} maxLength={10000} onChange={(event) => setBroadcastMessage(event.target.value)} rows={6} style={inputStyle} /></label>
            <p><button type="button" onClick={previewBroadcast} disabled={busy || previewBusy || !selectedRegistrations.length || !broadcastSubject.trim() || !broadcastMessage.trim()} style={buttonStyle}>{previewBusy ? "Building preview…" : "Preview recipients"}</button></p>
            {currentPreview ? (
              <div style={{ background: "#f8fafc", borderRadius: "10px", padding: "0.75rem" }}>
                <strong>{currentPreview.recipient_count} unique recipient(s)</strong>
                <p><button type="button" onClick={() => downloadText(`${detail.tournament.id}-broadcast-recipients.csv`, currentPreview.recipient_csv)} style={ghostButtonStyle}>Download recipient CSV</button></p>
                {currentPreview.recipients.length ? (
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}>
                      <thead><tr><th style={tableHeaderStyle}>Recipient</th><th style={tableHeaderStyle}>Email</th><th style={tableHeaderStyle}>Status</th><th style={tableHeaderStyle}>Payment</th></tr></thead>
                      <tbody>{currentPreview.recipients.map((recipient) => <tr key={recipient.email}><td style={tableCellStyle}>{recipient.name}</td><td style={tableCellStyle}>{recipient.email}</td><td style={tableCellStyle}>{recipient.registration_status}</td><td style={tableCellStyle}>{recipient.payment_status}</td></tr>)}</tbody>
                    </table>
                  </div>
                ) : <p>No recipients matched the current filters.</p>}
                <h3>Message preview</h3>
                <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{currentPreview.preview.text}</pre>
              </div>
            ) : null}
            <TournamentEmailDelivery clubId={clubId} tournamentId={detail.tournament.id} accessToken={accessToken}
              apiBase={apiBase} preview={currentPreview} previewScope={previewScope} subject={broadcastSubject}
              message={broadcastMessage} includeCancelled={includeCancelled} busy={busy} onBusy={setBusy} requestJson={requestJson} />
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Filtered registrations</h2>
            <p style={{ color: "#64748b" }}>{filteredRegistrations.length} of {detail.registrations.length} registration(s).</p>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "900px" }}>
                <thead><tr><th style={tableHeaderStyle}>Registrant</th><th style={tableHeaderStyle}>Email</th><th style={tableHeaderStyle}>Status</th><th style={tableHeaderStyle}>Payment</th><th style={tableHeaderStyle}>Entries</th><th style={tableHeaderStyle}>Notes</th></tr></thead>
                <tbody>
                  {filteredRegistrations.map((registration) => {
                    const entries = (selectionByRegistration.get(registration.id) || []).map((selection) => selection.event_label || selection.event_option_id).filter(Boolean);
                    return <tr key={registration.id}><td style={tableCellStyle}>{registration.display_name}</td><td style={tableCellStyle}>{registration.email || "—"}</td><td style={tableCellStyle}>{registration.registration_status || "—"}</td><td style={tableCellStyle}>{registration.payment_status || "—"}</td><td style={tableCellStyle}>{entries.join(", ") || "—"}</td><td style={tableCellStyle}>{registration.notes || "—"}</td></tr>;
                  })}
                  {!filteredRegistrations.length ? <tr><td colSpan={6} style={tableCellStyle}>No registrations match the current filters.</td></tr> : null}
                </tbody>
              </table>
            </div>
          </article>
        </>
      ) : null}

      {message ? <p role="status" style={{ color: /unable|error|sign in|not configured|reload|changed/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
