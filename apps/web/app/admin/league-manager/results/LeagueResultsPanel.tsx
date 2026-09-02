"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type {
  LeagueResultsResponse,
  LeagueResultsStanding,
  LeagueResultsStatRow
} from "@/lib/api";
import type { AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import { leagueRouteHref } from "@/lib/leagueRouteContext";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard
} from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type AdminLeagueResultsResponse = Omit<LeagueResultsResponse, "club"> & {
  ok: boolean;
  mode: "league_manager_results" | string;
  league_id: string;
  league_name: string;
  league_type: string;
  league_status: string;
  publicly_visible: boolean;
};

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
  initialLeagueId: string;
  initialLeague: string;
  initialLeagueType?: string | null;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};
const buttonStyle = {
  padding: "0.5rem 0.75rem",
  borderRadius: "999px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function rating(value?: number | null): string {
  return value == null || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(3);
}

function percentage(value?: number | null): string {
  return value == null || Number.isNaN(Number(value))
    ? "—"
    : `${Number(value).toFixed(1)}%`;
}

function StandingsTable({ rows }: { rows: LeagueResultsStanding[] }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No standings are recorded for this league.</p>;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
        <thead>
          <tr>
            {["Rank", "Player", "Rating", "Matches", "Wins", "Losses", "Win %", "Rating change"].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.player_id)}>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_jupr)}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{percentage(row.win_pct)}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WeeklyTable({ rows }: { rows: LeagueResultsStatRow[] }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No scored results are recorded for this week.</p>;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}>
        <thead>
          <tr>
            {["Player", "Games", "Wins", "Losses", "Win %", "Rating change"].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.week_num}-${row.player_id}`}>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.games ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{percentage(row.win_pct)}</td>
              <td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function LeagueResultsPanel({
  apiBase,
  clubId,
  status,
  initialLeagueId,
  initialLeague,
  initialLeagueType
}: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [results, setResults] = useState<AdminLeagueResultsResponse | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const resultsRequest = useLatestRequestGuard(
    `${accessToken}\u0000${initialLeague}`,
    clearProtectedState
  );

  function clearProtectedState() {
    setResults(null);
    setSelectedWeek(null);
    setBusy(false);
    setMessage(null);
  }

  async function requestResults(): Promise<AdminLeagueResultsResponse> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before loading league results.");
    const path = `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/results`;
    const response = await fetch(apiUrl(apiBase, path), {
      cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as AdminLeagueResultsResponse;
  }

  async function loadResults() {
    const generation = resultsRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestResults();
      if (!resultsRequest.isCurrent(generation)) return;
      if (String(payload.selected_league || "").trim() !== initialLeague.trim()) {
        throw new Error("The selected league did not match the returned results.");
      }
      if (String(payload.league_type || "").trim().toLowerCase() === "team") {
        router.replace(
          leagueRouteHref("/admin/league-manager/teams", {
            leagueId: payload.league_id || initialLeagueId,
            leagueName: payload.league_name || initialLeague,
            leagueType: payload.league_type || initialLeagueType || "Team"
          })
        );
        return;
      }
      setResults(payload);
      setSelectedWeek(payload.selected_week ?? null);
      setMessage("League results loaded.");
    } catch (error) {
      if (resultsRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load league results.");
      }
    } finally {
      if (resultsRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status.enabled ? `${accessToken}\u0000${initialLeague}` : "",
    loadResults
  );

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}>League Manager is currently unavailable.</article>;
  }
  if (sessionLoading) return <p role="status">Checking admin access…</p>;
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p>Historical league results are available only to authorized league managers.</p>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  const weeklyRows = (results?.weekly_results || []).filter(
    (row) => selectedWeek == null || row.week_num === selectedWeek
  );
  const leagueName = results?.league_name || initialLeague;
  const leagueId = results?.league_id || initialLeagueId;
  const leagueType = results?.league_type || initialLeagueType || "Individual";
  const publicHref = leagueRouteHref("/clubs/tres-palapas/league-results", {
    leagueId,
    leagueName,
    leagueType
  });

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <p style={{ margin: 0 }}>
        <button type="button" onClick={() => void loadResults()} disabled={busy} style={buttonStyle}>
          {busy ? "Loading…" : "Reload results"}
        </button>
      </p>
      {message ? <p role="status" style={{ color: /unable|error|match|sign in/i.test(message) ? "#b91c1c" : "#166534", margin: 0 }}>{message}</p> : null}

      {results ? (
        <>
          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <p style={{ display: "flex", gap: "1rem", flexWrap: "wrap", margin: 0 }}>
              <span><strong>Status:</strong> {results.league_status}</span>
              <span><strong>Players:</strong> {results.standings.length}</span>
              <span><strong>Latest week:</strong> {results.selected_week ? `Week ${results.selected_week}` : "No weekly results"}</span>
              {results.publicly_visible ? <Link href={publicHref}>Open public league results</Link> : <span>Historical results are admin-only.</span>}
            </p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Standings</h2>
            <StandingsTable rows={results.standings} />
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{selectedWeek ? `Week ${selectedWeek} results` : "Weekly results"}</h2>
            {results.weeks.length ? (
              <div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                {results.weeks.map((week) => {
                  const active = week.week_num === selectedWeek;
                  return (
                    <button
                      key={week.week_num}
                      type="button"
                      onClick={() => setSelectedWeek(week.week_num)}
                      aria-pressed={active}
                      style={{
                        ...buttonStyle,
                        background: active ? "#2563eb" : "white",
                        color: active ? "white" : "#0f172a",
                        borderColor: active ? "#2563eb" : "#cbd5e1"
                      }}
                    >
                      {week.week_label}{week.has_results === false ? " · no results" : ""}
                    </button>
                  );
                })}
              </div>
            ) : null}
            <WeeklyTable rows={weeklyRows} />
          </article>
        </>
      ) : busy ? <p role="status">Loading {initialLeague} results…</p> : null}
    </div>
  );
}
