"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournamentDetailResponse,
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
  initialName?: string | null;
};

type TournamentEdit = {
  name: string;
  startDate: string;
  endDate: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white", minWidth: 0 };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const moduleStyle = { ...cardStyle, display: "grid", gap: "0.35rem", alignContent: "start", textDecoration: "none", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function dateValue(value?: string | null): string {
  return value ? String(value).slice(0, 10) : "";
}

function selectedHref(path: string, tournamentId: string, tournamentName: string): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}

function editFromDetail(detail: AdminTournamentDetailResponse | null, fallbackName: string): TournamentEdit {
  return {
    name: detail?.tournament.name || fallbackName,
    startDate: dateValue(detail?.tournament.start_date),
    endDate: dateValue(detail?.tournament.end_date)
  };
}

function StatusChip({ value }: { value?: string | null }) {
  const normalized = String(value || "").toLowerCase();
  const background = ["open", "active", "confirmed", "paid"].includes(normalized) ? "#dcfce7" : ["closed", "cancelled", "archived", "refunded"].includes(normalized) ? "#f1f5f9" : "#fef3c7";
  const borderColor = background === "#dcfce7" ? "#bbf7d0" : background === "#f1f5f9" ? "#cbd5e1" : "#fde68a";
  return <span style={{ width: "fit-content", border: `1px solid ${borderColor}`, borderRadius: "999px", padding: "0.15rem 0.5rem", background, fontSize: "0.8rem", fontWeight: 700 }}>{value || "—"}</span>;
}

export default function TournamentHomePanel({ apiBase, clubId, status, tournamentId, initialName }: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [edit, setEdit] = useState<TournamentEdit>(() => editFromDetail(null, initialName || ""));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${tournamentId}`, clearProtectedState);
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setEdit(editFromDetail(null, initialName || ""));
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before opening this tournament.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setEdit(editFromDetail(payload, initialName || ""));
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament home.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveTournament(confirmationText: string) {
    if (!detail?.tournament.updated_at) {
      setMessage("Reload this tournament before saving changes.");
      return;
    }
    if (edit.startDate && edit.endDate && edit.endDate < edit.startDate) {
      setMessage("Tournament end date cannot be before its start date.");
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            name: edit.name.trim(),
            start_date: edit.startDate || null,
            end_date: edit.endDate || null,
            expected_updated_at: detail.tournament.updated_at,
            confirmation_text: confirmationText,
            source: "next_selected_tournament_home"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(payload.idempotent_replay ? "Tournament update safely reconciled." : "Tournament details saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save tournament details.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${tournamentId}` : "", loadDetail);

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}>Tournament Manager is currently unavailable.</article>;
  if (sessionLoading) return <p role="status">Checking admin access…</p>;
  if (!accessToken) return <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}><h2 style={{ marginTop: 0 }}>Admin sign-in required</h2><p><Link href="/admin/login">Open admin login</Link></p></article>;

  const tournamentName = detail?.tournament.name || initialName || tournamentId;
  const modules = [
    ["Setup", "/admin/tournament-setup", "Registration settings, days, divisions, impact review, and setup publishing."],
    ["Registrations", "/admin/tournaments/editor", "Edit individual registrations and event entries."],
    ["Reports", "/admin/tournaments/registrations", "Filter registrations, export CSV, and preview recipient handoff."],
    ["Bulk actions", "/admin/tournaments/bulk", "Apply reviewed registration updates to multiple players."],
    ["Extras & fulfillment", "/admin/tournaments/commerce", "Catalog, bundles, offline payment states, pickup, and recovery."],
    ["Ratings & team play", "/admin/tournaments/team-competition", "Combined ratings, four-player teams, and team competition settings."],
    ["Operations", "/admin/tournaments/ops", "Draws, teams, scoring, podiums, and recoverable tournament operations."],
    ["Results", "/admin/tournaments/ops/results", "Review and import tournament result files."],
    ["Live runner", "/admin/tournament-live", "Run draw-scoped live scoring and progression."],
    ["Official publish", "/admin/tournaments/ops/publish", "Publish official tournament matches and verify evidence."],
    ["Status & recovery", "/admin/tournaments/status", "Lifecycle state, recovery operations, and audit evidence."]
  ] as const;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {busy && !detail ? <p role="status">Loading {tournamentName}…</p> : null}
      {message ? <p role="status" style={{ color: /unable|error|required|reload|cannot/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}

      {detail ? (
        <>
          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <h2 style={{ margin: 0 }}>{tournamentName}</h2>
                <p style={{ color: "#475569", marginBottom: 0 }}>{dateValue(detail.tournament.start_date) || "Date not set"} – {dateValue(detail.tournament.end_date) || "Date not set"}</p>
              </div>
              <div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap" }}>
                <StatusChip value={detail.tournament.status} />
                <StatusChip value={detail.tournament.registration_status || "registration n/a"} />
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
              <div><strong>Registrations</strong><br />{detail.summary.registrations}</div>
              <div><strong>Event entries</strong><br />{detail.summary.selections}</div>
              <div><strong>Days</strong><br />{detail.days.length}</div>
              <div><strong>Divisions</strong><br />{detail.event_options.length}</div>
            </div>
          </article>

          <section aria-label={`${tournamentName} modules`}>
            <h2>Tournament tools</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {modules.map(([label, path, description]) => (
                <Link key={path} href={selectedHref(path, tournamentId, tournamentName)} style={moduleStyle}>
                  <strong>{label}</strong>
                  <span style={{ color: "#475569" }}>{description}</span>
                </Link>
              ))}
              {String(detail.tournament.status).toLowerCase() === "draft" ? (
                <Link href={selectedHref("/admin/tournaments/delete-draft", tournamentId, tournamentName)} style={{ ...moduleStyle, borderColor: "#fecaca", color: "#991b1b" }}>
                  <strong>Delete draft</strong>
                  <span>Delete only this unlaunched tournament draft through the guarded recovery workflow.</span>
                </Link>
              ) : null}
            </div>
          </section>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Tournament details</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem" }}>
              <label><strong>Name</strong><br /><input value={edit.name} onChange={(event) => setEdit((current) => ({ ...current, name: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Start date</strong><br /><input type="date" value={edit.startDate} onChange={(event) => setEdit((current) => ({ ...current, startDate: event.target.value }))} style={inputStyle} /></label>
              <label><strong>End date</strong><br /><input type="date" min={edit.startDate || undefined} value={edit.endDate} onChange={(event) => setEdit((current) => ({ ...current, endDate: event.target.value }))} style={inputStyle} /></label>
            </div>
            <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Save tournament details"} title="Save tournament details?" description={`Update the name and dates for ${tournamentName}.`} confirmLabel="Yes, save tournament" confirmationText="SAVE TOURNAMENT" disabled={!edit.name.trim() || !detail.tournament.updated_at} busy={busy} onConfirm={saveTournament} /></p>
          </article>
        </>
      ) : null}
    </div>
  );
}
