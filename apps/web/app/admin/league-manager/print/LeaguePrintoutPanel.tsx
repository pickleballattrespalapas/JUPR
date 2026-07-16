"use client";

import { useState } from "react";
import type { AdminLeagueManagerDetailResponse, AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function ratingLabel(value?: number | null): string {
  if (value == null) return "—";
  return Number(value).toFixed(3);
}

export default function LeaguePrintoutPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before loading league printouts.");
    const response = await fetch(apiUrl(apiBase, path), { headers: { Authorization: `Bearer ${accessToken}` } });
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

  async function loadDetail() {
    if (!leagueName) {
      setMessage("Select a league first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`);
      setDetail(payload);
      setMessage("Printout loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load league printout.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Manager is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the League Manager flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } body { background: white !important; } article { break-inside: avoid; } table { font-size: 12px; } }`}</style>
      <article className="no-print" style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article className="no-print" style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Load printout</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => setLeagueName(event.target.value)} style={inputStyle}>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load leagues"}</button>
          <button type="button" onClick={loadDetail} disabled={busy || !leagueName} style={buttonStyle}>{busy ? "Loading…" : "Load selected"}</button>
          <button type="button" onClick={() => window.print()} disabled={!detail} style={buttonStyle}>Print</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {detail ? (
        <section>
          <h1 style={{ marginBottom: "0.25rem" }}>{detail.league.league_name} league night printout</h1>
          <p style={{ color: "#475569" }}>Status: {detail.league.status} · K-factor: {detail.league.k_factor ?? "—"} · Min games: {detail.league.min_games ?? "—"}</p>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Schedule</h2>
            {detail.schedule_preview?.length ? (
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th align="left">Session</th><th align="left">Date</th><th align="left">Start</th><th align="left">End</th></tr></thead>
                <tbody>{detail.schedule_preview.map((row) => <tr key={`${row.session}-${row.date}`}><td>{row.session}</td><td>{row.date}</td><td>{row.start || "—"}</td><td>{row.end || "—"}</td></tr>)}</tbody>
              </table>
            ) : <p style={{ color: "#64748b" }}>No schedule preview configured.</p>}
          </article>

          <article style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Standings</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Rank</th><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="right">Matches</th></tr></thead>
              <tbody>{detail.standings.map((row) => <tr key={row.player_id}><td>{row.rank}</td><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td align="right">{row.matches_played ?? 0}</td></tr>)}</tbody>
            </table>
          </article>

          <article style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Roster checklist</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="left">Present</th><th align="left">Notes</th></tr></thead>
              <tbody>{(detail.roster || []).filter((row) => row.in_league).map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td>□</td><td>________________</td></tr>)}</tbody>
            </table>
          </article>
        </section>
      ) : null}
    </div>
  );
}
