"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerLeague,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerRosterRow,
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

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
function jsonText(value: unknown): string { return JSON.stringify(value && typeof value === "object" ? value : {}, null, 2); }
function parseJsonObject(label: string, value: string): Record<string, unknown> { const text = value.trim(); if (!text) return {}; const parsed = JSON.parse(text) as unknown; if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error(`${label} must be a JSON object.`); return parsed as Record<string, unknown>; }
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
  const [createConfirm, setCreateConfirm] = useState("");

  const [duplicateName, setDuplicateName] = useState("");
  const [duplicateConfirm, setDuplicateConfirm] = useState("");
  const [lifecycleConfirm, setLifecycleConfirm] = useState("");

  const [settingsDescription, setSettingsDescription] = useState("");
  const [settingsKFactor, setSettingsKFactor] = useState("32");
  const [settingsMinGames, setSettingsMinGames] = useState("3");
  const [scheduleConfigText, setScheduleConfigText] = useState("{}");
  const [courtDefaultsText, setCourtDefaultsText] = useState("{}");
  const [rulesConfigText, setRulesConfigText] = useState("{}");
  const [awardsConfigText, setAwardsConfigText] = useState("{}");
  const [eventTagsText, setEventTagsText] = useState("{}");
  const [settingsConfirm, setSettingsConfirm] = useState("");

  const [rosterPlayerId, setRosterPlayerId] = useState("");
  const [rosterAction, setRosterAction] = useState<"activate" | "deactivate">("activate");
  const [rosterStartingJupr, setRosterStartingJupr] = useState("3.5");
  const [rosterConfirm, setRosterConfirm] = useState("");

  const selectedRosterRow = detail?.roster?.find((row) => String(row.player_id) === rosterPlayerId) || null;
  const settingsStatus = detail?.league.status || "";
  const settingsDraft = settingsStatus === "draft";
  const settingsClosed = settingsStatus === "ended" || settingsStatus === "archived";

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

  function hydrateSettings(payload: AdminLeagueManagerDetailResponse) {
    const league = payload.league;
    setSettingsDescription(String(league.description || ""));
    setSettingsKFactor(league.k_factor == null ? "32" : String(league.k_factor));
    setSettingsMinGames(league.min_games == null ? "3" : String(league.min_games));
    setScheduleConfigText(jsonText(league.schedule_config));
    setCourtDefaultsText(jsonText(league.court_board_defaults));
    setRulesConfigText(jsonText(league.rules_config));
    setAwardsConfigText(jsonText(league.awards_config));
    setEventTagsText(jsonText(league.event_tags));
    setSettingsConfirm("");
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
    setRosterConfirm("");
  }

  function hydrateAll(payload: AdminLeagueManagerDetailResponse) { hydrateSettings(payload); hydrateRoster(payload); }

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
    try { const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`); setDetail(payload); hydrateAll(payload); setDuplicateName(`${leagueName} Copy`); setDuplicateConfirm(""); setLifecycleConfirm(""); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load league detail."); }
    finally { setSaving(false); }
  }

  async function createLeagueDraft() {
    if (!requireReady()) return;
    const name = createName.trim();
    const minGames = Number(createMinGames);
    const kFactor = Number(createKFactor);
    if (!name) { setMessage("League name is required."); return; }
    if (!Number.isInteger(minGames) || minGames < 0 || minGames > 1000) { setMessage("Minimum games must be a whole number from 0 to 1000."); return; }
    if (!Number.isInteger(kFactor) || kFactor < 1 || kFactor > 128) { setMessage("K-factor must be a whole number from 1 to 128."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`, { method: "POST", body: JSON.stringify({ league_name: name, description: createDescription, min_games: minGames, k_factor: kFactor, confirmation_text: createConfirm, source: "next_league_manager_create_form" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      setLeagues(listing.leagues || []);
      setSelectedLeague(payload.league?.league_name || name);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setDuplicateName(`${payload.league?.league_name || name} Copy`); setDuplicateConfirm("");
      setCreateName(""); setCreateDescription(""); setCreateMinGames("6"); setCreateKFactor("32"); setCreateConfirm("");
      setMessage(`Created draft league ${payload.league?.league_name || name}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create league draft."); }
    finally { setSaving(false); }
  }

  async function duplicateLeagueDraft() {
    if (!selectedLeague || !detail) { setMessage("Select a league before duplicating it."); return; }
    if (!requireReady()) return;
    const targetName = duplicateName.trim();
    if (!targetName) { setMessage("New draft name is required."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/duplicate`, { method: "POST", body: JSON.stringify({ target_league_name: targetName, confirmation_text: duplicateConfirm, source: "next_league_manager_duplicate_form" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      const createdName = payload.league?.league_name || payload.league_name || targetName;
      setLeagues(listing.leagues || []);
      setSelectedLeague(createdName);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setDuplicateName(`${createdName} Copy`); setDuplicateConfirm("");
      setMessage(`Duplicated ${payload.source_league_name || selectedLeague} as draft ${createdName}. Roster and results were not copied.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to duplicate league draft."); }
    finally { setSaving(false); }
  }

  async function transitionLeagueLifecycle(action: LifecycleAction) {
    if (!selectedLeague || !detail) { setMessage("Select a league before changing its lifecycle."); return; }
    if (!requireReady()) return;
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/lifecycle`, { method: "POST", body: JSON.stringify({ action, confirmation_text: lifecycleConfirm, source: "next_league_manager_lifecycle_controls" }) });
      const listing = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      setLeagues(listing.leagues || []);
      if (payload.detail) { setDetail(payload.detail); hydrateAll(payload.detail); }
      setLifecycleConfirm("");
      setMessage(`${lifecycleLabels[action]} completed: ${payload.previous_status || detail.league.status} → ${payload.new_status || payload.league?.status || "updated"}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to change league lifecycle."); }
    finally { setSaving(false); }
  }

  async function saveLeagueSettings() {
    if (!selectedLeague || !detail) { setMessage("Select a league before saving settings."); return; }
    if (!requireReady()) return;
    if (settingsClosed) { setMessage(`League settings are read-only after a league is ${settingsStatus}.`); return; }
    const settingsPatch: Record<string, unknown> = { description: settingsDescription };
    if (settingsDraft) {
      const kFactor = Number(settingsKFactor); const minGames = Number(settingsMinGames);
      if (!Number.isInteger(kFactor) || kFactor < 1 || kFactor > 128) { setMessage("K-factor must be a whole number from 1 to 128."); return; }
      if (!Number.isInteger(minGames) || minGames < 0 || minGames > 1000) { setMessage("Minimum games must be a whole number from 0 to 1000."); return; }
      try {
        Object.assign(settingsPatch, {
          k_factor: kFactor,
          min_games: minGames,
          schedule_config: parseJsonObject("Schedule config", scheduleConfigText),
          court_board_defaults: parseJsonObject("Court board defaults", courtDefaultsText),
          rules_config: parseJsonObject("Rules config", rulesConfigText),
          awards_config: parseJsonObject("Awards config", awardsConfigText),
          event_tags: parseJsonObject("Event tags", eventTagsText)
        });
      } catch (error) { setMessage(error instanceof Error ? error.message : "Invalid JSON settings."); return; }
    }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`, { method: "PATCH", body: JSON.stringify({ ...settingsPatch, confirmation_text: settingsConfirm, source: "next_league_manager_settings_editor" }) });
      if (payload.detail) { setDetail(payload.detail); hydrateSettings(payload.detail); } else { await loadDetail(selectedLeague); }
      setMessage(`Saved settings for ${payload.league?.league_name || selectedLeague}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save league settings."); }
    finally { setSaving(false); }
  }

  async function saveRosterMembership() {
    if (!selectedLeague || !rosterPlayerId) { setMessage("Select a league and player before saving roster membership."); return; }
    if (!requireReady()) return;
    const rating = Number(rosterStartingJupr);
    if (!Number.isFinite(rating) || rating < 1 || rating > 2800) { setMessage("Starting rating must be a JUPR value from 1.0-7.0 or Elo from 400-2800."); return; }
    setSaving(true); setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/roster/${encodeURIComponent(rosterPlayerId)}`, { method: "PATCH", body: JSON.stringify({ action: rosterAction, starting_rating: rating, confirmation_text: rosterConfirm, source: "next_league_manager_roster_editor" }) });
      if (payload.detail) { setDetail(payload.detail); hydrateRoster(payload.detail); } else { await loadDetail(selectedLeague); }
      setMessage(`${rosterAction === "activate" ? "Activated" : "Deactivated"} player ${payload.player_id ?? rosterPlayerId} for ${selectedLeague}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save roster membership."); }
    finally { setSaving(false); }
  }

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next League Manager is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the League Manager pilot flag on FastAPI."}</p></article>;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>League Manager admin session</h2>
        <p style={{ color: "#475569" }}>Create league drafts, manage settings and rosters, run persisted live rounds, print operations sheets, and close league awards through guarded FastAPI workflows.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}><strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong><p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send authorized League Manager requests." : sessionLoading ? "Checking admin session…" : "Sign in before using League Manager."}</p>{sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}{!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}</div>
        <button type="button" onClick={loadLeagues} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Working…" : "Load leagues"}</button>{status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Create league draft</h2>
        <p style={{ color: "#475569" }}>Matches the Streamlit starting workflow without activating the league. Names are unique within this club, and every creation is audit-attributed.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>League name</strong><br /><input value={createName} onChange={(event) => setCreateName(event.target.value)} maxLength={120} style={inputStyle} /></label>
          <label><strong>Minimum games</strong><br /><input type="number" value={createMinGames} onChange={(event) => setCreateMinGames(event.target.value)} min={0} max={1000} style={inputStyle} /></label>
          <label><strong>K-factor</strong><br /><input type="number" value={createKFactor} onChange={(event) => setCreateKFactor(event.target.value)} min={1} max={128} style={inputStyle} /></label>
          <label><strong>Type CREATE LEAGUE</strong><br /><input value={createConfirm} onChange={(event) => setCreateConfirm(event.target.value)} placeholder="CREATE LEAGUE" style={inputStyle} /></label>
        </div>
        <label><strong>Description</strong><br /><textarea value={createDescription} onChange={(event) => setCreateDescription(event.target.value)} maxLength={2000} rows={3} style={inputStyle} /></label>
        <p><button type="button" onClick={createLeagueDraft} disabled={saving || !accessToken || !createName.trim() || createConfirm.trim().toUpperCase() !== "CREATE LEAGUE"} style={buttonStyle}>{saving ? "Working…" : "Create draft"}</button></p>
      </article>

      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Select league</h2><select value={selectedLeague} onChange={(event) => loadDetail(event.target.value)} style={inputStyle} disabled={!accessToken}><option value="">Choose a league</option>{leagues.map((league) => <option key={league.league_name} value={league.league_name}>{league.league_name} · {league.status}</option>)}</select>{leagues.length ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>{leagues.map((league) => <button key={league.league_name} type="button" onClick={() => loadDetail(league.league_name)} disabled={!accessToken} style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}><strong>{league.league_name}</strong><br /><span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem", ...statusChipStyle(league.status) }}>{league.status}</span></button>)}</div> : null}</article>

      {detail ? <>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>{detail.league.league_name}</h2>{detail.league.description ? <p style={{ color: "#475569" }}>{detail.league.description}</p> : null}<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}><div><strong>Status</strong><br />{detail.league.status}</div><div><strong>K-factor</strong><br />{detail.league.k_factor ?? "—"}</div><div><strong>Min games</strong><br />{detail.league.min_games ?? "—"}</div><div><strong>Started</strong><br />{detail.league.started_at ? String(detail.league.started_at).slice(0, 10) : "—"}</div><div><strong>Ended</strong><br />{detail.league.ended_at ? String(detail.league.ended_at).slice(0, 10) : "—"}</div><div><strong>Standings rows</strong><br />{detail.standings_count}</div><div><strong>League roster</strong><br />{detail.league_roster_count ?? 0} / {detail.roster_count ?? detail.roster?.length ?? 0}</div></div></article>

        <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
          <h2 style={{ marginTop: 0 }}>League lifecycle</h2>
          <p style={{ color: "#7c2d12" }}>Lifecycle changes are separate from settings. Only legal transitions are shown, and each action requires its exact phrase. Ending freezes the league; award calculation and badge minting remain in the Awards workflow.</p>
          {lifecycleActionsFor(detail.league.status).length ? <>
            <label><strong>Confirmation phrase</strong><br /><input value={lifecycleConfirm} onChange={(event) => setLifecycleConfirm(event.target.value)} placeholder={lifecycleActionsFor(detail.league.status).map((action) => lifecycleConfirmations[action]).join(" or ")} style={inputStyle} /></label>
            <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{lifecycleActionsFor(detail.league.status).map((action) => <button key={action} type="button" onClick={() => transitionLeagueLifecycle(action)} disabled={saving || !accessToken || lifecycleConfirm.trim().toUpperCase() !== lifecycleConfirmations[action]} style={action === "end" || action === "archive" ? { ...buttonStyle, background: "#9a3412", borderColor: "#9a3412" } : buttonStyle}>{lifecycleLabels[action]} · type {lifecycleConfirmations[action]}</button>)}</p>
          </> : <p style={{ color: "#64748b" }}>Archived leagues have no further lifecycle actions.</p>}
        </article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Duplicate this league as a draft</h2>
          <p style={{ color: "#475569" }}>Copies the description, schedule, court, rules, awards, ratings, and event-tag configuration. It does not copy roster membership, standings, results, start/end dates, or awards already issued.</p>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>New draft name</strong><br /><input value={duplicateName} onChange={(event) => setDuplicateName(event.target.value)} maxLength={120} style={inputStyle} /></label>
            <label><strong>Type DUPLICATE LEAGUE</strong><br /><input value={duplicateConfirm} onChange={(event) => setDuplicateConfirm(event.target.value)} placeholder="DUPLICATE LEAGUE" style={inputStyle} /></label>
            <button type="button" onClick={duplicateLeagueDraft} disabled={saving || !accessToken || !duplicateName.trim() || duplicateConfirm.trim().toUpperCase() !== "DUPLICATE LEAGUE"} style={buttonStyle}>{saving ? "Duplicating…" : "Duplicate draft"}</button>
          </div>
        </article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Guided settings editor</h2>
          <p style={{ color: settingsClosed ? "#92400e" : "#475569" }}>{settingsDraft ? <>Draft leagues allow description, ratings, schedule, court, rules, awards, and event-tag configuration.</> : settingsClosed ? <>This league is {settingsStatus}; settings are read-only. Use the separate Awards workflow for award review.</> : <>This league is {settingsStatus}; only its description is a safe edit. Pause/end/resume actions remain in the lifecycle controls.</>} {!settingsClosed ? <>Type <code>SAVE LEAGUE</code> before saving.</> : null}</p>
          <label><strong>Description</strong><br /><textarea value={settingsDescription} onChange={(event) => setSettingsDescription(event.target.value)} disabled={settingsClosed} maxLength={2000} rows={3} style={inputStyle} /></label>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>K-factor</strong><br /><input type="number" value={settingsKFactor} onChange={(event) => setSettingsKFactor(event.target.value)} disabled={!settingsDraft} min={1} max={128} style={inputStyle} /></label>
            <label><strong>Min games</strong><br /><input type="number" value={settingsMinGames} onChange={(event) => setSettingsMinGames(event.target.value)} disabled={!settingsDraft} min={0} max={1000} style={inputStyle} /></label>
            <label><strong>Type SAVE LEAGUE</strong><br /><input value={settingsConfirm} onChange={(event) => setSettingsConfirm(event.target.value)} disabled={settingsClosed} style={inputStyle} /></label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}><label><strong>Schedule config JSON</strong><br /><textarea value={scheduleConfigText} onChange={(event) => setScheduleConfigText(event.target.value)} disabled={!settingsDraft} rows={8} style={{ ...inputStyle, fontFamily: "monospace" }} /></label><label><strong>Court board defaults JSON</strong><br /><textarea value={courtDefaultsText} onChange={(event) => setCourtDefaultsText(event.target.value)} disabled={!settingsDraft} rows={8} style={{ ...inputStyle, fontFamily: "monospace" }} /></label><label><strong>Rules config JSON</strong><br /><textarea value={rulesConfigText} onChange={(event) => setRulesConfigText(event.target.value)} disabled={!settingsDraft} rows={8} style={{ ...inputStyle, fontFamily: "monospace" }} /></label><label><strong>Awards config JSON</strong><br /><textarea value={awardsConfigText} onChange={(event) => setAwardsConfigText(event.target.value)} disabled={!settingsDraft} rows={8} style={{ ...inputStyle, fontFamily: "monospace" }} /></label><label><strong>Event tags JSON</strong><br /><textarea value={eventTagsText} onChange={(event) => setEventTagsText(event.target.value)} disabled={!settingsDraft} rows={8} style={{ ...inputStyle, fontFamily: "monospace" }} /></label></div>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={saveLeagueSettings} disabled={saving || settingsClosed || !accessToken || settingsConfirm.trim().toUpperCase() !== "SAVE LEAGUE"} style={buttonStyle}>{saving ? "Saving…" : settingsDraft ? "Save draft settings" : "Save description"}</button><button type="button" onClick={() => hydrateSettings(detail)} disabled={saving} style={ghostButtonStyle}>Reset from loaded league</button></p>
        </article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Roster membership editor</h2><p style={{ color: "#475569" }}>Activate a player into this league with a starting JUPR/Elo seed, or deactivate an existing league row without deleting history. Type <code>SAVE ROSTER</code> before saving.</p>{detail.roster?.length ? <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(160px, 220px) minmax(140px, 180px) minmax(180px, 220px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Player</strong><br /><select value={rosterPlayerId} onChange={(event) => { const next = detail.roster?.find((row) => String(row.player_id) === event.target.value) || null; setRosterPlayerId(event.target.value); setRosterAction(rosterActionFor(next)); setRosterStartingJupr(startingJuprFor(next)); }} style={inputStyle}><option value="">Choose player…</option>{detail.roster.map((row) => <option key={row.player_id} value={String(row.player_id)}>{row.player_name} · {row.in_league ? "in league" : "not in league"}</option>)}</select></label><label><strong>Action</strong><br /><select value={rosterAction} onChange={(event) => setRosterAction(event.target.value as "activate" | "deactivate")} style={inputStyle}><option value="activate">Activate/add to league</option><option value="deactivate">Deactivate from league</option></select></label><label><strong>Starting JUPR/Elo</strong><br /><input value={rosterStartingJupr} onChange={(event) => setRosterStartingJupr(event.target.value)} disabled={rosterAction === "deactivate"} style={inputStyle} /></label><label><strong>Type SAVE ROSTER</strong><br /><input value={rosterConfirm} onChange={(event) => setRosterConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={saveRosterMembership} disabled={saving || !accessToken || !rosterPlayerId || rosterConfirm.trim().toUpperCase() !== "SAVE ROSTER"} style={buttonStyle}>{saving ? "Saving…" : "Save roster"}</button></div> : <p style={{ color: "#64748b" }}>Load a roster snapshot before editing membership.</p>}{selectedRosterRow ? <p style={{ color: "#475569" }}>Selected: <strong>{selectedRosterRow.player_name}</strong> · {selectedRosterRow.in_league ? "currently in league" : "not yet in league"} · current league JUPR {juprLabel(selectedRosterRow.rating_jupr)}</p> : null}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Schedule preview</h2>{detail.schedule_preview.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Session</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Date</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Start</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>End</th></tr></thead><tbody>{detail.schedule_preview.map((row) => <tr key={`${row.session}-${row.date}`}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.session}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.date}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.start || "—"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.end || "—"}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No schedule preview is configured for this league yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Standings snapshot</h2>{detail.standings.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Rank</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>MP</th></tr></thead><tbody>{detail.standings.map((row) => <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No league standings rows are available yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Roster snapshot</h2>{detail.roster?.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>In league</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>MP</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Last game</th></tr></thead><tbody>{detail.roster.map((row) => <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.in_league ? "Yes" : "No"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.last_game_at ? String(row.last_game_at).slice(0, 10) : "—"}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No roster rows are available yet.</p>}</article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Configuration snapshot</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}><div><strong>Schedule config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.schedule_config)}</pre></div><div><strong>Court board defaults</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.court_board_defaults)}</pre></div><div><strong>Rules config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.rules_config)}</pre></div><div><strong>Awards config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.awards_config)}</pre></div></div></article>
      </> : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("required") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("json") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
