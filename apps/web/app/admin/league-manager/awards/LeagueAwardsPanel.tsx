"use client";

import { useState } from "react";
import type { AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };
type AwardRow = { category_key: string; category_label: string; player_id: number; player_name?: string; metric_display?: string; rank?: number; min_games?: number };
type AwardsResponse = {
  ok: boolean;
  mode?: string;
  league_name: string;
  league?: Record<string, unknown>;
  awards: AwardRow[];
  award_count: number;
  badge_candidate_count?: number;
  warnings?: string[];
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

export default function LeagueAwardsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [preview, setPreview] = useState<AwardsResponse | null>(null);
  const [awardBadges, setAwardBadges] = useState(true);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Awards.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadLeagues() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      if (!leagueName && names.length) setLeagueName(names[0]);
      setMessage(`Loaded ${names.length} league(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      setBusy(false);
    }
  }

  async function previewAwards() {
    if (!leagueName) {
      setMessage("Select a league before previewing awards.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AwardsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/awards/preview`);
      setPreview(payload);
      setMessage(`Previewed ${payload.award_count || 0} award row(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to preview league awards.");
    } finally {
      setBusy(false);
    }
  }

  async function closeLeague() {
    if (!leagueName) {
      setMessage("Select a league before closing awards.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AwardsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/awards/close`, {
        method: "POST",
        body: JSON.stringify({ award_badges: awardBadges, confirmation_text: confirmation, source: "next_league_manager_awards_close" })
      });
      setPreview(payload);
      setConfirmation("");
      setMessage(`League closed with ${payload.award_count || 0} award row(s) and ${payload.badge_candidate_count || 0} badge candidate(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to close league awards.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Manager is disabled</h2>
        <p style={{ color: "#475569" }}>Enable the guarded League Manager flag before using awards.</p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Select league</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => { setLeagueName(event.target.value); setPreview(null); }} style={inputStyle}>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load leagues"}</button>
          <button type="button" onClick={previewAwards} disabled={busy || !leagueName} style={ghostButtonStyle}>Preview awards</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {preview ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>2. Award preview</h2>
          <p style={{ color: "#475569" }}>League status: <strong>{shortValue(preview.league?.status)}</strong> · Active: <strong>{shortValue(preview.league?.is_active)}</strong> · Min games: <strong>{shortValue(preview.league?.min_games)}</strong></p>
          {preview.awards.length ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                <thead><tr><th align="left">Category</th><th align="left">Rank</th><th align="left">Player</th><th align="right">Metric</th><th align="right">Min games</th></tr></thead>
                <tbody>{preview.awards.map((award, index) => (
                  <tr key={`${award.category_key}-${award.player_id}-${index}`}>
                    <td style={{ padding: "0.45rem", borderBottom: "1px solid #e2e8f0" }}>{award.category_label || award.category_key}</td>
                    <td style={{ padding: "0.45rem", borderBottom: "1px solid #e2e8f0" }}>{award.rank ?? "—"}</td>
                    <td style={{ padding: "0.45rem", borderBottom: "1px solid #e2e8f0" }}>{award.player_name || `Player ${award.player_id}`}</td>
                    <td align="right" style={{ padding: "0.45rem", borderBottom: "1px solid #e2e8f0" }}>{award.metric_display || "—"}</td>
                    <td align="right" style={{ padding: "0.45rem", borderBottom: "1px solid #e2e8f0" }}>{award.min_games ?? "—"}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <p style={{ color: "#92400e" }}>No qualifying award rows. Check league min games and standings.</p>}
          {preview.warnings?.length ? <p style={{ color: "#92400e" }}>{preview.warnings.join(" · ")}</p> : null}
        </article>
      ) : null}

      {preview ? (
        <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
          <h2 style={{ marginTop: 0 }}>3. Close league and award</h2>
          <p style={{ color: "#7f1d1d" }}>This closes the league, writes end-award metadata, audit-flags the action, and optionally awards top performer badges. Corrections after close should go through Player Editor, Match Log, and Replay History.</p>
          <label style={{ display: "block", marginBottom: "0.75rem" }}><input type="checkbox" checked={awardBadges} onChange={(event) => setAwardBadges(event.target.checked)} /> Award top performer badges when closing</label>
          <label>Confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="CLOSE LEAGUE" style={inputStyle} /></label>
          <button type="button" onClick={closeLeague} disabled={busy || !leagueName} style={{ ...buttonStyle, marginTop: "0.75rem", background: "#991b1b", borderColor: "#991b1b" }}>{busy ? "Closing…" : "Close league and award"}</button>
        </article>
      ) : null}
    </div>
  );
}
