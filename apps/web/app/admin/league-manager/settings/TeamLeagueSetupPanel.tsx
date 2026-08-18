"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type { AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";

type Props = {
  apiBase: string | null;
  clubId: string;
  leagueName: string;
  leagueStatus: string;
  status: AdminLeagueManagerStatusResponse;
};

type TeamSettings = {
  league_name: string;
  registration_open?: boolean;
  team_size?: 2 | 3 | 4;
  team_category?: "open" | "mens" | "womens" | "mixed";
  max_alternates?: number;
  substitute_pool_enabled?: boolean;
  mixed_required_men?: number;
  mixed_required_women?: number;
  allow_substitutes?: boolean;
  playoff_format?: string;
  playoff_team_count?: number | null;
  start_date?: string | null;
  weekday?: number;
  start_time?: string;
  timezone?: string;
  venue?: string | null;
  registration_closes_at?: string | null;
  settings_version?: number;
};

type SettingsDraft = {
  registrationOpen: boolean;
  teamSize: string;
  teamCategory: string;
  maxAlternates: string;
  substitutePoolEnabled: boolean;
  mixedRequiredMen: string;
  mixedRequiredWomen: string;
  allowSubstitutes: boolean;
  playoffFormat: string;
  playoffTeamCount: string;
  startDate: string;
  weekday: string;
  startTime: string;
  timezone: string;
  venue: string;
  registrationClosesAt: string;
};

type TeamLeagueListResponse = { leagues?: TeamSettings[] };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" };
const TIMEZONE_OPTIONS = ["America/Mazatlan", "America/Phoenix", "America/Los_Angeles", "America/Denver", "America/Chicago", "America/New_York", "America/Mexico_City", "UTC"];
const confirmedWriteRefreshWarning = " The setup was saved, but the latest view could not be refreshed. Reload this page before making another change; do not repeat the completed action.";

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function operationKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return `team-settings:${crypto.randomUUID()}`;
  return `team-settings:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

function draftFrom(settings?: TeamSettings | null): SettingsDraft {
  return {
    registrationOpen: Boolean(settings?.registration_open),
    teamSize: String(settings?.team_size || 2),
    teamCategory: String(settings?.team_category || "open"),
    maxAlternates: String(settings?.max_alternates || 0),
    substitutePoolEnabled: Boolean(settings?.substitute_pool_enabled),
    mixedRequiredMen: String(settings?.mixed_required_men || 1),
    mixedRequiredWomen: String(settings?.mixed_required_women || 1),
    allowSubstitutes: Boolean(settings?.allow_substitutes),
    playoffFormat: String(settings?.playoff_format || "none"),
    playoffTeamCount: settings?.playoff_team_count == null ? "" : String(settings.playoff_team_count),
    startDate: String(settings?.start_date || ""),
    weekday: String(settings?.weekday ?? 0),
    startTime: String(settings?.start_time || "18:00").slice(0, 5),
    timezone: String(settings?.timezone || "America/Mazatlan"),
    venue: String(settings?.venue || ""),
    registrationClosesAt: settings?.registration_closes_at ? String(settings.registration_closes_at).slice(0, 16) : ""
  };
}

function categoryLabel(category: string): string {
  return ({ mens: "Men's", womens: "Women's", mixed: "Mixed", open: "Open" } as Record<string, string>)[category] || "Open";
}

async function refreshAfterConfirmedWrite(refresh: () => Promise<void>): Promise<string> {
  try {
    await refresh();
    return "";
  } catch {
    return confirmedWriteRefreshWarning;
  }
}

export default function TeamLeagueSetupPanel({ apiBase, clubId, leagueName, leagueStatus, status }: Props) {
  const { accessToken } = useAdminSession();
  const [settings, setSettings] = useState<TeamSettings | null>(null);
  const [draft, setDraft] = useState<SettingsDraft>(() => draftFrom());
  const [loadedDraft, setLoadedDraft] = useState<SettingsDraft>(() => draftFrom());
  const [idempotencyKey, setIdempotencyKey] = useState(operationKey);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const requestGuard = useLatestRequestGuard(`${accessToken}\u0000${leagueName}`, clearProtectedState);

  function clearProtectedState() {
    setSettings(null);
    const empty = draftFrom();
    setDraft(empty);
    setLoadedDraft(empty);
    setIdempotencyKey(operationKey());
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before loading team league setup.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function refreshSettings() {
    const payload = await requestJson<TeamLeagueListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues`);
    const loaded = (payload.leagues || []).find((row) => row.league_name === leagueName) || null;
    const nextDraft = draftFrom(loaded);
    setSettings(loaded);
    setDraft(nextDraft);
    setLoadedDraft(nextDraft);
  }

  async function load() {
    const generation = requestGuard.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<TeamLeagueListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues`);
      if (!requestGuard.isCurrent(generation)) return;
      const loaded = (payload.leagues || []).find((row) => row.league_name === leagueName) || null;
      const nextDraft = draftFrom(loaded);
      setSettings(loaded);
      setDraft(nextDraft);
      setLoadedDraft(nextDraft);
    } catch (error) {
      if (requestGuard.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load team league setup.");
      }
    } finally {
      if (requestGuard.isCurrent(generation)) setBusy(false);
    }
  }

  function update<K extends keyof SettingsDraft>(key: K, value: SettingsDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
    setIdempotencyKey(operationKey());
  }

  function rejectSave(messageText: string): never {
    setMessage(messageText);
    throw new Error(messageText);
  }

  async function save(confirmationText: string): Promise<ActionCompletion> {
    const teamSize = Number(draft.teamSize);
    const maxAlternates = Number(draft.maxAlternates);
    const mixedRequiredMen = Number(draft.mixedRequiredMen);
    const mixedRequiredWomen = Number(draft.mixedRequiredWomen);
    const playoffCount = draft.playoffTeamCount ? Number(draft.playoffTeamCount) : null;
    if (![2, 3, 4].includes(teamSize)) rejectSave("Primary roster size must be 2, 3, or 4 players.");
    if (!Number.isInteger(maxAlternates) || maxAlternates < 0 || maxAlternates > 4) rejectSave("Maximum alternates must be a whole number from 0 to 4.");
    if (draft.teamCategory === "mixed" && (!Number.isInteger(mixedRequiredMen) || !Number.isInteger(mixedRequiredWomen) || mixedRequiredMen < 1 || mixedRequiredWomen < 1 || mixedRequiredMen + mixedRequiredWomen !== teamSize)) rejectSave("Mixed roster counts must each be at least one and total the primary roster size.");
    if (draft.substitutePoolEnabled && !draft.allowSubstitutes) rejectSave("Enable substitutes before enabling the shared substitute pool.");
    if (draft.playoffFormat === "all_team_single_elimination" && (!Number.isInteger(playoffCount) || Number(playoffCount) < 2 || Number(playoffCount) > 128)) rejectSave("Playoff team count must be a whole number from 2 to 128.");

    const generation = requestGuard.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues/${encodeURIComponent(leagueName)}/settings`, {
        method: "PUT",
        body: JSON.stringify({
          settings: {
            registration_open: draft.registrationOpen,
            team_size: teamSize,
            team_category: draft.teamCategory,
            max_alternates: maxAlternates,
            substitute_pool_enabled: draft.substitutePoolEnabled,
            mixed_required_men: mixedRequiredMen,
            mixed_required_women: mixedRequiredWomen,
            allow_substitutes: draft.allowSubstitutes,
            playoff_format: draft.playoffFormat,
            playoff_team_count: playoffCount,
            start_date: draft.startDate || null,
            weekday: Number(draft.weekday),
            start_time: draft.startTime,
            timezone: draft.timezone,
            venue: draft.venue || null,
            registration_closes_at: draft.registrationClosesAt ? new Date(draft.registrationClosesAt).toISOString() : null
          },
          expected_settings_version: Number(settings?.settings_version || 0),
          idempotency_key: idempotencyKey,
          confirmation_text: confirmationText,
          source: "next_selected_league_settings_team_setup"
        })
      });
      if (!requestGuard.isCurrent(generation)) throw new Error("The admin session changed before the team setup response was applied.");
      setIdempotencyKey(operationKey());
      setLoadedDraft(draft);
      const refreshWarning = await refreshAfterConfirmedWrite(refreshSettings);
      const successMessage = `Team eligibility, registration, roster, substitute, schedule, and playoff settings were saved together.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Team league setup saved", successMessage);
    } catch (error) {
      if (requestGuard.isCurrent(generation)) {
        setMessage(`${error instanceof Error ? error.message : "Unable to save team league setup."} The same request key is retained for a safe retry.`);
      }
      throw error;
    } finally {
      if (requestGuard.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(accessToken ? `${accessToken}\u0000${leagueName}` : "", load);

  const isDraft = leagueStatus === "draft";
  const writeReady = status.league_manager_writes_enabled !== false;
  const hasChanges = JSON.stringify(draft) !== JSON.stringify(loadedDraft);

  return (
    <article style={cardStyle} data-testid="team-league-setup">
      <h2 style={{ marginTop: 0 }}>Team league setup</h2>
      <p style={{ color: "#475569" }}>Set team eligibility, roster size, alternates, substitute policy, registration, weekly schedule, and playoffs before league operations begin.</p>
      {busy && !settings ? <p role="status">Loading team league setup…</p> : null}
      {isDraft ? (
        <>
          <div style={gridStyle}>
            <label><input type="checkbox" checked={draft.registrationOpen} onChange={(event) => update("registrationOpen", event.target.checked)} /> <strong>Registration open</strong></label>
            <label><strong>Primary roster size</strong><br /><select value={draft.teamSize} onChange={(event) => { const size = Number(event.target.value); update("teamSize", event.target.value); update("mixedRequiredMen", String(Math.floor(size / 2))); update("mixedRequiredWomen", String(size - Math.floor(size / 2))); }} style={inputStyle}><option value="2">2 players</option><option value="3">3 players</option><option value="4">4 players</option></select></label>
            <label><strong>Maximum alternates</strong><br /><select value={draft.maxAlternates} onChange={(event) => update("maxAlternates", event.target.value)} style={inputStyle}>{[0, 1, 2, 3, 4].map((count) => <option key={count} value={String(count)}>{count}</option>)}</select></label>
            <label><strong>Team eligibility</strong><br /><select value={draft.teamCategory} onChange={(event) => { update("teamCategory", event.target.value); if (event.target.value === "mixed") { const size = Number(draft.teamSize); update("mixedRequiredMen", String(Math.floor(size / 2))); update("mixedRequiredWomen", String(size - Math.floor(size / 2))); } }} style={inputStyle}><option value="open">Open</option><option value="mens">Men&apos;s</option><option value="womens">Women&apos;s</option><option value="mixed">Mixed</option></select></label>
            <label><input type="checkbox" checked={draft.allowSubstitutes} onChange={(event) => { update("allowSubstitutes", event.target.checked); if (!event.target.checked) update("substitutePoolEnabled", false); }} /> <strong>Allow substitutes</strong></label>
            <label><input type="checkbox" checked={draft.substitutePoolEnabled} disabled={!draft.allowSubstitutes} onChange={(event) => update("substitutePoolEnabled", event.target.checked)} /> <strong>Shared substitute pool</strong></label>
            {draft.teamCategory === "mixed" ? <><label><strong>Required men on primary roster</strong><br /><input type="number" min={1} max={3} value={draft.mixedRequiredMen} onChange={(event) => update("mixedRequiredMen", event.target.value)} style={inputStyle} /></label><label><strong>Required women on primary roster</strong><br /><input type="number" min={1} max={3} value={draft.mixedRequiredWomen} onChange={(event) => update("mixedRequiredWomen", event.target.value)} style={inputStyle} /></label></> : null}
            <label><strong>Playoffs</strong><br /><select value={draft.playoffFormat} onChange={(event) => update("playoffFormat", event.target.value)} style={inputStyle}><option value="none">No playoffs</option><option value="top_2_final">Top 2 final</option><option value="top_4_single_elimination">Top 4 single elimination</option><option value="all_team_single_elimination">All-team single elimination</option></select></label>
            {draft.playoffFormat === "all_team_single_elimination" ? <label><strong>Playoff team count</strong><br /><input type="number" min={2} max={128} value={draft.playoffTeamCount} onChange={(event) => update("playoffTeamCount", event.target.value)} style={inputStyle} /></label> : null}
            <label><strong>Season start</strong><br /><input type="date" value={draft.startDate} onChange={(event) => update("startDate", event.target.value)} style={inputStyle} /></label>
            <label><strong>League night</strong><br /><select value={draft.weekday} onChange={(event) => update("weekday", event.target.value)} style={inputStyle}><option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option><option value="3">Thursday</option><option value="4">Friday</option><option value="5">Saturday</option><option value="6">Sunday</option></select></label>
            <label><strong>Start time</strong><br /><input type="time" value={draft.startTime} onChange={(event) => update("startTime", event.target.value)} style={inputStyle} /></label>
            <label><strong>Timezone</strong><br /><select value={draft.timezone} onChange={(event) => update("timezone", event.target.value)} style={inputStyle}>{!TIMEZONE_OPTIONS.includes(draft.timezone) ? <option value={draft.timezone}>{draft.timezone}</option> : null}{TIMEZONE_OPTIONS.map((zone) => <option key={zone} value={zone}>{zone}</option>)}</select></label>
            <label><strong>Venue</strong><br /><input value={draft.venue} onChange={(event) => update("venue", event.target.value)} maxLength={240} style={inputStyle} /></label>
            <label><strong>Registration closes</strong><br /><input type="datetime-local" value={draft.registrationClosesAt} onChange={(event) => update("registrationClosesAt", event.target.value)} style={inputStyle} /></label>
          </div>
          {writeReady ? <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Save team league setup"} title="Save this team league setup?" description="This saves roster size, eligibility, alternates, substitute policy, weekly schedule, and playoff choices together." confirmLabel="Yes, save setup" confirmationText="SAVE TEAM LEAGUE" disabled={busy || !hasChanges} busy={busy} onConfirm={save} /></p> : <p style={{ color: "#92400e" }}>Team league setup changes are currently unavailable.</p>}
        </>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Eligibility</strong><br />{categoryLabel(draft.teamCategory)}</div>
          <div><strong>Roster</strong><br />{draft.teamSize} primary · {draft.maxAlternates} alternates</div>
          <div><strong>Substitutes</strong><br />{draft.allowSubstitutes ? (draft.substitutePoolEnabled ? "Shared pool enabled" : "Allowed") : "Not allowed"}</div>
          <div><strong>Registration</strong><br />{draft.registrationOpen ? "Open" : "Closed"}</div>
          <div><strong>Season</strong><br />{draft.startDate || "Start not set"} · {draft.startTime} {draft.timezone}</div>
          <div><strong>Playoffs</strong><br />{draft.playoffFormat.replace(/_/g, " ")}</div>
        </div>
      )}
      {message ? <p role="status" style={{ color: /unable|error|must|retry|required/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
  );
}
