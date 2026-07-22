"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerLeague,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerRosterRow,
  AdminLeagueManagerSchedulePreviewResponse,
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { GuidedLeagueSettingsEditor } from "./GuidedLeagueSettingsEditor";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };
type LifecycleAction = "start" | "pause" | "resume" | "end" | "archive";

const lifecycleConfirmations: Record<LifecycleAction, string> = {
  start: "START LEAGUE",
  pause: "PAUSE LEAGUE",
  resume: "RESUME LEAGUE",
  end: "END LEAGUE",
  archive: "ARCHIVE LEAGUE"
};
const lifecycleLabels: Record<LifecycleAction, string> = {
  start: "Start league",
  pause: "Pause league",
  resume: "Resume league",
  end: "End league",
  archive: "Archive league"
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function juprLabel(value?: number | null): string { return value == null ? "—" : Number(value).toFixed(2); }
function statusChipStyle(status: string) { if (status === "active") return { background: "#dcfce7", borderColor: "#bbf7d0" }; if (status === "ended" || status === "archived") return { background: "#f1f5f9", borderColor: "#cbd5e1" }; return { background: "#fef3c7", borderColor: "#fde68a" }; }
function compactJson(value: unknown): string { if (!value || (typeof value === "object" && Object.keys(value as Record<string, unknown>).length === 0)) return "—"; return JSON.stringify(value, null, 2); }
function downloadTextFile(filename: string, content: string, mediaType: string) { const blob = new Blob([content], { type: mediaType }); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url); }
function rosterActionFor(row?: AdminLeagueManagerRosterRow | null): "activate" | "deactivate" { return row?.in_league ? "deactivate" : "activate"; }
function startingJuprFor(row?: AdminLeagueManagerRosterRow | null): string { return row?.rating_jupr == null ? "3.5" : Number(row.rating_jupr).toFixed(2); }
function lifecycleActionsFor(status: string): LifecycleAction[] {
  if (status === "draft") return ["start"];
  if (status === "active") return ["pause", "end"];
  if (status === "paused") return ["resume", "end"];
  if (status === "ended") return ["archive"];
  return [];
}

export default function LeagueManagerPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<AdminLeagueManagerLeague[]>([]);
  const [selectedLeague, setSelectedLeague] = useState("");
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const [createName, setCreateName] = useState("");
  const [createDescription, setCreateDescription] = useState("");
  const [createMinGames, setCreateMinGames] = useState("6");
  const [createKFactor, setCreateKFactor] = useState("32");

  const [duplicateName, setDuplicateName] = useState("");

  const [rosterPlayerId, setRosterPlayerId] = useState("");
  const [rosterAction, setRosterAction] = useState<"activate" | "deactivate">("activate");
  const [rosterStartingJupr, setRosterStartingJupr] = useState("3.5");

  const selectedRosterRow = detail?.roster?.find((row) => String(row.player_id) === rosterPlayerId) || null;

  function requireReady(): boolean {
    if (!apiBase) { setMessage("API base URL is not configured."); return false; }
    if (!accessToken) { setMessage("Sign in at /admin/login before using League Manager."); return false; }
    if (!status.enabled) { setMessage("Next League Manager is disabled on the API."); return false; }
    return true;
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Manager.");
    const headers = new Headers(options?.headers); headers.set("Authorization", `Bearer ${accessToken}`); if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function hydrateRoster(payload: AdminLeagueManagerDetailResponse) {
    const current = payload.roster?.find((row) => String(row.player_id) === rosterPlayerId) || payload.roster?.[0] || null;
    if (current) {
      setRosterPlayerId(String(current.player_id));
      setRosterAction(rosterActionFor(current));
      setRosterStartingJupr(startingJuprFor(current));
    } else {
      setRosterPlayerId("");
      setRosterAction("activate");
      setRosterStartingJupr("3.5");
    }
  }

  function hydrateAll(payload: AdminLeagueManagerDetailResponse) { hydrateRoster(payload); }

  async function loadLeagues() {
    setMessage(null); setDetail(null);
    if (!requireReady()) return;
    setSaving(true);
    try { const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`); setLeagues(payload.leagues || []); setMessage(`Loaded ${payload.count ?? payload.leagues?.length ?? 0} league(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load leagues."); }
    finally { setSaving(false); }
  }

  async function loadDetail(leagueName: string) {
    setSelectedLeague(leagueName); setDetail(null); setMessage(null);
    if (!leagueName || !requireReady()) return;
    setSaving(true);
    try { const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`); setDetail(payload); hydrateAll(payload); setDuplicateName(`${leagueName} Copy`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load league detail."); }
    finally { setSaving(false); }
  }

  async function createLeagueDraft(confirmationText: string) {
    if (!requireReady()) return;
    const name = createName.trim();
    const minGames = Number(createMinGames);
    const kFactor = Number(createKFactor);
    if (!name) { setMessage("League name is required."); return; }
    if (!Number.isInteger(minGames) || minGames < 0 || minGames > 1000) { setMessage("Minimum games must be a whole number from 0 to 1000."); return; }
    if (!Number.isInteger(kFactor) || kFactor < 1 || kFactor > 128) { setMessage("K-factor must be a whole number from 1 to 128."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`, { method: "POST", body: JSON.stringify({ league_name: name, description: createDescription, min_games: minGames, k_factor: kFactor, confirmation_text: confirmationText, source: "next_league_manager_create_form" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      setLeagues(listing.leagues || []);
      setSelectedLeague(payload.league?.league_name || name);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setDuplicateName(`${payload.league?.league_name || name} Copy`);
      setCreateName(""); setCreateDescription(""); setCreateMinGames("6"); setCreateKFactor("32");
      setMessage(`Created draft league ${payload.league?.league_name || name}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create league draft."); }
    finally { setSaving(false); }
  }

  async function duplicateLeagueDraft(confirmationText: string) {
    if (!selectedLeague || !detail) { setMessage("Select a league before duplicating it."); return; }
    if (!requireReady()) return;
    const targetName = duplicateName.trim();
    if (!targetName) { setMessage("New draft name is required."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/duplicate`, { method: "POST", body: JSON.stringify({ target_league_name: targetName, confirmation_text: confirmationText, source: "next_league_manager_duplicate_form" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      const createdName = payload.league?.league_name || payload.league_name || targetName;
      setLeagues(listing.leagues || []);
      setSelectedLeague(createdName);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setDuplicateName(`${createdName} Copy`);
      setMessage(`Duplicated ${payload.source_league_name || selectedLeague} as draft ${createdName}. Roster and results were not copied.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to duplicate league draft."); }
    finally { setSaving(false); }
  }

  async function transitionLeagueLifecycle(action: LifecycleAction, confirmationText: string) {
    if (!selectedLeague || !detail) { setMessage("Select a league before changing its lifecycle."); return; }
    if (!requireReady()) return;
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/lifecycle`, { method: "POST", body: JSON.stringify({ action, confirmation_text: confirmationText, source: "next_league_manager_lifecycle_controls" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      setLeagues(listing.leagues || []);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setMessage(`${lifecycleLabels[action]} completed: ${payload.previous_status || detail.league.status} → ${payload.new_status || payload.league?.status || "updated"}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to change league lifecycle."); }
    finally { setSaving(false); }
  }

  async function saveLeagueSettings(settingsPatch: Record<string, unknown>, confirmationText: string): Promise<boolean> {
    if (!selectedLeague || !detail) { setMessage("Select a league before saving settings."); return false; }
    if (!requireReady()) return false;
    if (detail.league.status === "ended" || detail.league.status === "archived") { setMessage(`League settings are read-only after a league is ${detail.league.status}.`); return false; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`, { method: "PATCH", body: JSON.stringify({ ...settingsPatch, confirmation_text: confirmationText, source: "next_league_manager_guided_settings" }) });
      if (payload.detail) { setDetail(payload.detail); } else { await loadDetail(selectedLeague); }
      setMessage(`Saved settings for ${payload.league?.league_name || selectedLeague}.`);
      return true;
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save league settings."); return false; }
    finally { setSaving(false); }
  }

  async function previewLeagueSchedule(scheduleConfig: Record<string, unknown>): Promise<AdminLeagueManagerSchedulePreviewResponse | null> {
    if (!selectedLeague || !detail) { setMessage("Select a league before previewing its schedule."); return null; }
    if (!requireReady()) return null;
    setSaving(true); setMessage(null);
    try {
      return await requestJson<AdminLeagueManagerSchedulePreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/schedule/preview`, { method: "POST", body: JSON.stringify({ schedule_config: scheduleConfig }) });
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to preview league schedule."); return null; }
    finally { setSaving(false); }
  }

  async function saveRosterMembership(confirmationText: string) {
    if (!selectedLeague || !rosterPlayerId) { setMessage("Select a league and player before saving roster membership."); return; }
    if (!requireReady()) return;
    if (detail?.capabilities?.roster_mutable === false) { setMessage("This league roster is read-only in its current lifecycle state."); return; }
    const rating = Number(rosterStartingJupr);
    if (!Number.isFinite(rating) || rating < 1 || rating > 2800) { setMessage("Starting rating must be a JUPR value from 1.0-7.0 or Elo from 400-2800."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/roster/${encodeURIComponent(rosterPlayerId)}`, { method: "PATCH", body: JSON.stringify({ action: rosterAction, starting_rating: rating, confirmation_text: confirmationText, source: "next_league_manager_roster_editor" }) });
      if (payload.detail) { setDetail(payload.detail); hydrateRoster(payload.detail); } else { await loadDetail(selectedLeague); }
      setMessage(`${rosterAction === "activate" ? "Activated" : "Deactivated"} player ${payload.player_id ?? rosterPlayerId} for ${selectedLeague}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save roster membership."); }
    finally { setSaving(false); }
  }

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next League Manager is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the League Manager pilot flag on FastAPI."}</p></article>;

  const availableLifecycleActions = detail?.capabilities?.lifecycle_actions || (detail ? lifecycleActionsFor(detail.league.status) : []);
  const rosterMutable = detail?.capabilities?.roster_mutable !== false;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>League Manager admin session</h2>
        <p style={{ color: "#475569" }}>Create league drafts, manage settings and rosters, run persisted live rounds, and load Python-authoritative leader printouts through guarded FastAPI workflows.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}><strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong><p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send authorized League Manager requests." : sessionLoading ? "Checking admin session…" : "Sign in before using League Manager."}</p>{sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}{!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}</div>
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}><button type="button" onClick={loadLeagues} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Working…" : "Load leagues"}</button><Link href="/admin/league-manager/print">League night printout</Link><Link href="/admin/top-players-printable">Previous-month Top 50</Link></p>{status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Create league draft</h2>
        <p style={{ color: "#475569" }}>Matches the Streamlit starting workflow without activating the league. Names are unique within this club, and every creation is audit-attributed.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>League name</strong><br /><input value={createName} onChange={(event) => setCreateName(event.target.value)} maxLength={120} style={inputStyle} /></label>
          <label><strong>Minimum games</strong><br /><input type="number" value={createMinGames} onChange={(event) => setCreateMinGames(event.target.value)} min={0} max={1000} style={inputStyle} /></label>
          <label><strong>K-factor</strong><br /><input type="number" value={createKFactor} onChange={(event) => setCreateKFactor(event.target.value)} min={1} max={128} style={inputStyle} /></label>
        </div>
        <label><strong>Description</strong><br /><textarea value={createDescription} onChange={(event) => setCreateDescription(event.target.value)} maxLength={2000} rows={3} style={inputStyle} /></label>
        <p><ConfirmAction triggerLabel={saving ? "Working…" : "Create draft"} title="Create this league draft?" description={`Create ${createName.trim() || "this league"} as an inactive draft with the reviewed description and rating settings.`} confirmLabel="Yes, create draft" confirmationText="CREATE LEAGUE" disabled={!accessToken || !createName.trim()} busy={saving} onConfirm={createLeagueDraft} /></p>
      </article>

      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Select league</h2><select value={selectedLeague} onChange={(event) => loadDetail(event.target.value)} style={inputStyle} disabled={!accessToken}><option value="">Choose a league</option>{leagues.map((league) => <option key={league.league_name} value={league.league_name}>{league.league_name} · {league.status}</option>)}</select>{leagues.length ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>{leagues.map((league) => <button key={league.league_name} type="button" onClick={() => loadDetail(league.league_name)} disabled={!accessToken} style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}><strong>{league.league_name}</strong><br /><span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem", ...statusChipStyle(league.status) }}>{league.status}</span></button>)}</div> : null}</article>

      {detail ? <>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>{detail.league.league_name}</h2>{detail.league.description ? <p style={{ color: "#475569" }}>{detail.league.description}</p> : null}<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}><div><strong>Status</strong><br />{detail.league.status}</div><div><strong>K-factor</strong><br />{detail.league.k_factor ?? "—"}</div><div><strong>Min games</strong><br />{detail.league.min_games ?? "—"}</div><div><strong>Started</strong><br />{detail.league.started_at ? String(detail.league.started_at).slice(0, 10) : "—"}</div><div><strong>Ended</strong><br />{detail.league.ended_at ? String(detail.league.ended_at).slice(0, 10) : "—"}</div><div><strong>Standings rows</strong><br />{detail.standings_count}</div><div><strong>League roster</strong><br />{detail.league_roster_count ?? 0} / {detail.roster_count ?? detail.roster?.length ?? 0}</div></div>{detail.validation ? <div style={{ marginTop: "0.9rem", padding: "0.75rem", borderRadius: "10px", border: `1px solid ${detail.validation.valid ? "#bbf7d0" : "#fecaca"}`, background: detail.validation.valid ? "#f0fdf4" : "#fef2f2" }}><strong>{detail.validation.valid ? "Server validation passed" : "Server validation requires attention"}</strong>{detail.validation.errors.length ? <ul style={{ color: "#b91c1c" }}>{detail.validation.errors.map((item) => <li key={item}>{item}</li>)}</ul> : null}{detail.validation.warnings.length ? <ul style={{ color: "#92400e" }}>{detail.validation.warnings.map((item) => <li key={item}>{item}</li>)}</ul> : null}</div> : null}</article>

        <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
          <h2 style={{ marginTop: 0 }}>League lifecycle</h2>
          <p style={{ color: "#7c2d12" }}>Lifecycle changes are separate from settings. Only legal transitions are shown. Ending freezes the league; award calculation and badge minting remain in the Awards workflow.</p>
          {availableLifecycleActions.length ? <>
            <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{availableLifecycleActions.map((action) => <ConfirmAction key={action} triggerLabel={lifecycleLabels[action]} title={`${lifecycleLabels[action]}?`} description={`${lifecycleLabels[action]} changes ${detail.league.league_name} from its current ${detail.league.status} state. ${action === "end" ? "Ending freezes the league before its separate awards workflow." : action === "archive" ? "Archiving closes further lifecycle changes." : "Only the selected legal transition will be applied."}`} confirmLabel={`Yes, ${lifecycleLabels[action].toLowerCase()}`} confirmationText={lifecycleConfirmations[action]} tone={action === "end" || action === "archive" ? "danger" : "default"} disabled={!accessToken} busy={saving} onConfirm={(confirmationText) => transitionLeagueLifecycle(action, confirmationText)} />)}</p>
          </> : <p style={{ color: "#64748b" }}>Archived leagues have no further lifecycle actions.</p>}
        </article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Duplicate this league as a draft</h2>
          <p style={{ color: "#475569" }}>Copies the description, schedule, court, rules, awards, ratings, and event-tag configuration. It does not copy roster membership, standings, results, start/end dates, or awards already issued.</p>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>New draft name</strong><br /><input value={duplicateName} onChange={(event) => setDuplicateName(event.target.value)} maxLength={120} style={inputStyle} /></label>
            <ConfirmAction triggerLabel={saving ? "Duplicating…" : "Duplicate draft"} title="Duplicate this league as a new draft?" description={`Copy ${selectedLeague} settings into ${duplicateName.trim() || "the named draft"}. Rosters, standings, results, dates, and issued awards will not be copied.`} confirmLabel="Yes, duplicate draft" confirmationText="DUPLICATE LEAGUE" disabled={!accessToken || !duplicateName.trim()} busy={saving} onConfirm={duplicateLeagueDraft} />
          </div>
        </article>

        <GuidedLeagueSettingsEditor detail={detail} saving={saving} canWrite={Boolean(accessToken)} onSave={saveLeagueSettings} onPreview={previewLeagueSchedule} />

        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Roster membership editor</h2>
          <p style={{ color: "#475569" }}>Activate a player into this league with a starting JUPR/Elo seed, or deactivate an existing league row without deleting history. Reactivation preserves prior ratings and record.</p>
          {!rosterMutable ? <p style={{ color: "#92400e" }}><strong>Roster is read-only after league close.</strong></p> : null}
          {detail.roster?.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(160px, 220px) minmax(140px, 180px) auto", gap: "0.75rem", alignItems: "end" }}>
              <label><strong>Player</strong><br /><select value={rosterPlayerId} disabled={!rosterMutable} onChange={(event) => { const next = detail.roster?.find((row) => String(row.player_id) === event.target.value) || null; setRosterPlayerId(event.target.value); setRosterAction(rosterActionFor(next)); setRosterStartingJupr(startingJuprFor(next)); }} style={inputStyle}><option value="">Choose player…</option>{detail.roster.map((row) => <option key={row.player_id} value={String(row.player_id)}>{row.player_name} · {row.in_league ? "in league" : "not in league"}</option>)}</select></label>
              <label><strong>Action</strong><br /><select value={rosterAction} disabled={!rosterMutable} onChange={(event) => setRosterAction(event.target.value as "activate" | "deactivate")} style={inputStyle}><option value="activate">Activate/add to league</option><option value="deactivate">Deactivate from league</option></select></label>
              <label><strong>Starting JUPR/Elo</strong><br /><input value={rosterStartingJupr} onChange={(event) => setRosterStartingJupr(event.target.value)} disabled={!rosterMutable || rosterAction === "deactivate"} style={inputStyle} /></label>
              <ConfirmAction triggerLabel={saving ? "Saving…" : "Save roster"} title={`${rosterAction === "activate" ? "Activate" : "Deactivate"} this league player?`} description={`${selectedRosterRow?.player_name || "The selected player"} will be ${rosterAction === "activate" ? `activated with starting rating ${rosterStartingJupr}` : "deactivated without deleting prior league history"}.`} confirmLabel={`Yes, ${rosterAction === "activate" ? "activate player" : "deactivate player"}`} confirmationText="SAVE ROSTER" tone={rosterAction === "deactivate" ? "danger" : "default"} disabled={!accessToken || !rosterMutable || !rosterPlayerId} busy={saving} onConfirm={saveRosterMembership} />
            </div>
          ) : <p style={{ color: "#64748b" }}>Load a roster snapshot before editing membership.</p>}
          {selectedRosterRow ? <p style={{ color: "#475569" }}>Selected: <strong>{selectedRosterRow.player_name}</strong> · {selectedRosterRow.in_league ? "currently in league" : "not yet in league"} · current league JUPR {juprLabel(selectedRosterRow.rating_jupr)}</p> : null}
        </article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Schedule preview</h2>{detail.schedule_preview.length ? <><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Session</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Date</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Start</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>End</th></tr></thead><tbody>{detail.schedule_preview.map((row) => <tr key={`${row.session}-${row.date}`}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.session}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.date}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.start || "—"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.end || "—"}</td></tr>)}</tbody></table></div>{detail.schedule_ics ? <p><button type="button" onClick={() => downloadTextFile(detail.schedule_ics_filename || "league-schedule.ics", detail.schedule_ics || "", "text/calendar;charset=utf-8")} style={ghostButtonStyle}>Download ICS calendar</button></p> : null}</> : <p style={{ color: "#64748b" }}>No schedule preview is configured for this league yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Standings snapshot</h2>{detail.standings.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Rank</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>MP</th></tr></thead><tbody>{detail.standings.map((row) => <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No league standings rows are available yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Roster snapshot</h2>{detail.roster?.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>In league</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>MP</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Last game</th></tr></thead><tbody>{detail.roster.map((row) => <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.in_league ? "Yes" : "No"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.last_game_at ? String(row.last_game_at).slice(0, 10) : "—"}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No roster rows are available yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Configuration snapshot</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}><div><strong>Schedule config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.schedule_config)}</pre></div><div><strong>Court board defaults</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.court_board_defaults)}</pre></div><div><strong>Rules config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.rules_config)}</pre></div><div><strong>Awards config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.awards_config)}</pre></div></div></article>
      </> : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("required") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("json") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
