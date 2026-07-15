"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerLeague,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerStatusResponse
} from "@/lib/adminLeagueManagerApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function juprLabel(value?: number | null): string {
  return value == null ? "—" : Number(value).toFixed(2);
}

function statusChipStyle(status: string) {
  if (status === "active") return { background: "#dcfce7", borderColor: "#bbf7d0" };
  if (status === "ended" || status === "archived") return { background: "#f1f5f9", borderColor: "#cbd5e1" };
  return { background: "#fef3c7", borderColor: "#fde68a" };
}

function compactJson(value: unknown): string {
  if (!value || (typeof value === "object" && Object.keys(value as Record<string, unknown>).length === 0)) return "—";
  return JSON.stringify(value, null, 2);
}

export default function LeagueManagerPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<AdminLeagueManagerLeague[]>([]);
  const [selectedLeague, setSelectedLeague] = useState("");
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  function requireReady(): boolean {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return false;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before using League Manager.");
      return false;
    }
    if (!status.enabled) {
      setMessage("Next League Manager is disabled on the API.");
      return false;
    }
    return true;
  }

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Manager.");
    const response = await fetch(apiUrl(apiBase, path), {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadLeagues() {
    setMessage(null);
    setDetail(null);
    if (!requireReady()) return;
    setSaving(true);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      setLeagues(payload.leagues || []);
      setMessage(`Loaded ${payload.count ?? payload.leagues?.length ?? 0} league(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      setSaving(false);
    }
  }

  async function loadDetail(leagueName: string) {
    setSelectedLeague(leagueName);
    setDetail(null);
    setMessage(null);
    if (!leagueName || !requireReady()) return;
    setSaving(true);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`);
      setDetail(payload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load league detail.");
    } finally {
      setSaving(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next League Manager is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the League Manager pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>League Manager admin session</h2>
        <p style={{ color: "#475569" }}>This foundation route is read-only: league list, schedule preview, config visibility, and standings. Setup, court movement, scoring, and awards stay on Streamlit for now.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to send authorized League Manager requests." : sessionLoading ? "Checking admin session…" : "Sign in before using League Manager."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <button type="button" onClick={loadLeagues} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Working…" : "Load leagues"}</button>
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Select league</h2>
        <select value={selectedLeague} onChange={(event) => loadDetail(event.target.value)} style={inputStyle} disabled={!accessToken}>
          <option value="">Choose a league</option>
          {leagues.map((league) => <option key={league.league_name} value={league.league_name}>{league.league_name} · {league.status}</option>)}
        </select>
        {leagues.length ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
            {leagues.map((league) => (
              <button key={league.league_name} type="button" onClick={() => loadDetail(league.league_name)} disabled={!accessToken} style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}>
                <strong>{league.league_name}</strong><br />
                <span style={{ border: "1px solid", borderRadius: "999px", padding: "0.12rem 0.45rem", fontSize: "0.78rem", ...statusChipStyle(league.status) }}>{league.status}</span>
              </button>
            ))}
          </div>
        ) : null}
      </article>

      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{detail.league.league_name}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Status</strong><br />{detail.league.status}</div>
              <div><strong>K-factor</strong><br />{detail.league.k_factor ?? "—"}</div>
              <div><strong>Min games</strong><br />{detail.league.min_games ?? "—"}</div>
              <div><strong>Standings rows</strong><br />{detail.standings_count}</div>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Schedule preview</h2>
            {detail.schedule_preview.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Session</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Date</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Start</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>End</th></tr></thead>
                  <tbody>
                    {detail.schedule_preview.map((row) => (
                      <tr key={`${row.session}-${row.date}`}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.session}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.date}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.start || "—"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.end || "—"}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p style={{ color: "#64748b" }}>No schedule preview is configured for this league yet.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Standings snapshot</h2>
            {detail.standings.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Rank</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>MP</th></tr></thead>
                  <tbody>
                    {detail.standings.map((row) => (
                      <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p style={{ color: "#64748b" }}>No league standings rows are available yet.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Configuration snapshot</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Schedule config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.schedule_config)}</pre></div>
              <div><strong>Court board defaults</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.court_board_defaults)}</pre></div>
              <div><strong>Rules config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.rules_config)}</pre></div>
              <div><strong>Awards config</strong><pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{compactJson(detail.league.awards_config)}</pre></div>
            </div>
          </article>
        </>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("required") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
