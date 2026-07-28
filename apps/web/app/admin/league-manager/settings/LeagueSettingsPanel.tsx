"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerSchedulePreviewResponse,
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { GuidedLeagueSettingsEditor } from "../GuidedLeagueSettingsEditor";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function LeagueSettingsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedState);
  const detailRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before editing league settings.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function clearProtectedState() {
    detailRequest.invalidate();
    actionRequest.invalidate();
    setLeagues([]);
    setLeagueName("");
    setDetail(null);
    setBusy(false);
    setMessage(null);
  }

  async function loadLeagues() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`
      );
      if (!listRequest.isCurrent(generation)) return;
      const names = (payload.leagues || []).map((league) => league.league_name);
      setLeagues(names);
      if (leagueName && names.includes(leagueName)) await loadDetail(leagueName);
      else if (leagueName) {
        setLeagueName("");
        setDetail(null);
      }
      setMessage(names.length ? `Loaded ${names.length} league(s).` : "No leagues are available.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadDetail(selectedLeague: string) {
    const generation = detailRequest.begin();
    setLeagueName(selectedLeague);
    setDetail(null);
    setMessage(null);
    if (!selectedLeague) return;
    setBusy(true);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load league settings.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveSettings(patch: Record<string, unknown>, confirmationText: string): Promise<boolean> {
    if (!leagueName || !detail) return false;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`,
        {
          method: "PATCH",
          body: JSON.stringify({ ...patch, confirmation_text: confirmationText, source: "next_league_manager_settings_page" })
        }
      );
      if (!actionRequest.isCurrent(generation)) return false;
      if (payload.detail) setDetail(payload.detail);
      else await loadDetail(leagueName);
      setMessage(`Saved settings for ${leagueName}.`);
      return true;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save league settings.");
      return false;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function previewSchedule(scheduleConfig: Record<string, unknown>): Promise<AdminLeagueManagerSchedulePreviewResponse | null> {
    if (!leagueName) return null;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerSchedulePreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/schedule/preview`,
        { method: "POST", body: JSON.stringify({ schedule_config: scheduleConfig }) }
      );
      return actionRequest.isCurrent(generation) ? payload : null;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to preview the schedule.");
      return null;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Choose a league</h2>
        <p style={{ color: "#475569" }}>
          Signed in as {adminSessionLabel(session)}. Drafts expose the complete setup; active leagues limit edits to safe fields.
        </p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p><Link href="/admin/login">Open admin login</Link></p> : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>
            <strong>League</strong><br />
            <select value={leagueName} onChange={(event) => void loadDetail(event.target.value)} disabled={busy || !accessToken} style={inputStyle}>
              <option value="">Select a league</option>
              {leagues.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <button type="button" onClick={() => void loadLeagues()} disabled={busy || !accessToken} style={buttonStyle}>
            {busy ? "Working…" : "Refresh leagues"}
          </button>
        </div>
        {message ? <p role="status" style={{ color: /unable|error|sign in/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {detail ? (
        <GuidedLeagueSettingsEditor
          detail={detail}
          saving={busy}
          canWrite={Boolean(accessToken)}
          onSave={saveSettings}
          onPreview={previewSchedule}
        />
      ) : null}
    </div>
  );
}
