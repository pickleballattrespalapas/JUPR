"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type {
  AdminTournament,
  AdminTournamentBroadcastPreviewResponse,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentSelection,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";

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
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedRegistrationState);
  const detailRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  const [registrationStatus, setRegistrationStatus] = useState("");
  const [paymentStatus, setPaymentStatus] = useState("");
  const [partnerMode, setPartnerMode] = useState("");
  const [registrationDayId, setRegistrationDayId] = useState("");
  const [eventOptionId, setEventOptionId] = useState("");
  const [search, setSearch] = useState("");

  const [broadcastSubject, setBroadcastSubject] = useState("");
  const [broadcastMessage, setBroadcastMessage] = useState("");
  const [includeCancelled, setIncludeCancelled] = useState(false);
  const [broadcastPreview, setBroadcastPreview] = useState<AdminTournamentBroadcastPreviewResponse | null>(null);

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

  function clearProtectedRegistrationState() {
    detailRequest.invalidate();
    setBusy(false); setMessage(null);
    setTournaments([]); setSelectedTournamentId(initialTournamentId); setDetail(null); setImportHandoff(null);
    setBroadcastSubject(""); setBroadcastMessage(""); setBroadcastPreview(null);
  }

  async function loadTournaments(): Promise<void> {
    const selectedBeforeRefresh = selectedTournamentId;
    const generation = listRequest.begin();
    detailRequest.invalidate();
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
    if (!detail) return;
    const generation = actionRequest.begin();
    const requestedTournamentId = detail.tournament.id;
    setBusy(true);
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
            registration_status: registrationStatus || null,
            payment_status: paymentStatus || null,
            partner_mode: partnerMode || null,
            registration_day_id: registrationDayId || null,
            event_option_id: eventOptionId || null,
            search: search.trim() || null
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setBroadcastPreview(payload);
      setMessage(`Previewed ${payload.recipient_count} unique recipient(s). No email was sent.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to preview broadcast recipients.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments);

  if (!status.enabled) {
    return <article style={cardStyle}><h2>Tournament Admin is disabled</h2><p>{status.warnings?.[0]}</p></article>;
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Registration reporting session</h2>
        <p style={{ color: "#475569" }}>Tournament options load automatically after the admin session is ready. CSV downloads and recipient previews use the same filters shown below.</p>
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
            <p style={{ color: "#475569" }}>Filters apply to the table, authenticated CSV export, and broadcast recipient preview.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem" }}>
              <label><strong>Status</strong><br /><select value={registrationStatus} onChange={(event) => setRegistrationStatus(event.target.value)} style={inputStyle}>{REGISTRATION_STATUS_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Payment</strong><br /><select value={paymentStatus} onChange={(event) => setPaymentStatus(event.target.value)} style={inputStyle}>{PAYMENT_STATUS_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Partner mode</strong><br /><select value={partnerMode} onChange={(event) => setPartnerMode(event.target.value)} style={inputStyle}>{PARTNER_MODE_OPTIONS.map((value) => <option key={value || "all"} value={value}>{value || "All"}</option>)}</select></label>
              <label><strong>Day</strong><br /><select value={registrationDayId} onChange={(event) => setRegistrationDayId(event.target.value)} style={inputStyle}><option value="">All</option>{detail.days.map((day) => <option key={String(day.id)} value={String(day.id)}>{dayLabel(day)}</option>)}</select></label>
              <label><strong>Division</strong><br /><select value={eventOptionId} onChange={(event) => setEventOptionId(event.target.value)} style={inputStyle}><option value="">All</option>{detail.event_options.map((option) => <option key={String(option.id)} value={String(option.id)}>{eventLabel(option)}</option>)}</select></label>
              <label><strong>Search</strong><br /><input value={search} onChange={(event) => setSearch(event.target.value)} style={inputStyle} /></label>
            </div>
            <p><button type="button" onClick={exportCsv} disabled={busy} style={buttonStyle}>Download filtered CSV</button></p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Broadcast preview — no send</h2>
            <p style={{ color: "#475569" }}>Build a deduplicated recipient list, recipient CSV, and one personalized message preview. This reporting surface has no send action.</p>
            <label><strong>Subject</strong><br /><input value={broadcastSubject} onChange={(event) => setBroadcastSubject(event.target.value)} style={inputStyle} /></label>
            <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Message</strong><br /><textarea value={broadcastMessage} onChange={(event) => setBroadcastMessage(event.target.value)} rows={6} style={inputStyle} /></label>
            <label style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}><input type="checkbox" checked={includeCancelled} onChange={(event) => setIncludeCancelled(event.target.checked)} /> Include cancelled registrations</label>
            <p><button type="button" onClick={previewBroadcast} disabled={busy || !broadcastSubject.trim() || !broadcastMessage.trim()} style={buttonStyle}>Preview recipients</button></p>
            {broadcastPreview ? (
              <div style={{ background: "#f8fafc", borderRadius: "10px", padding: "0.75rem" }}>
                <strong>{broadcastPreview.recipient_count} unique recipient(s)</strong>
                <p><button type="button" onClick={() => downloadText(`${detail.tournament.id}-broadcast-recipients.csv`, broadcastPreview.recipient_csv)} style={ghostButtonStyle}>Download recipient CSV</button></p>
                {broadcastPreview.recipients.length ? (
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}>
                      <thead><tr><th style={tableHeaderStyle}>Recipient</th><th style={tableHeaderStyle}>Email</th><th style={tableHeaderStyle}>Status</th><th style={tableHeaderStyle}>Payment</th></tr></thead>
                      <tbody>{broadcastPreview.recipients.map((recipient) => <tr key={recipient.email}><td style={tableCellStyle}>{recipient.name}</td><td style={tableCellStyle}>{recipient.email}</td><td style={tableCellStyle}>{recipient.registration_status}</td><td style={tableCellStyle}>{recipient.payment_status}</td></tr>)}</tbody>
                    </table>
                  </div>
                ) : <p>No recipients matched the current filters.</p>}
                <h3>Personalized message sample</h3>
                <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{broadcastPreview.preview.text}</pre>
              </div>
            ) : null}
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
