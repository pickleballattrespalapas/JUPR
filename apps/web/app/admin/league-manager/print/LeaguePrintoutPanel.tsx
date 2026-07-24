"use client";

import { useState } from "react";
import type { AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse, AdminLeaguePrintoutResponse } from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
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

function signedRatingDelta(value?: number | null): string {
  if (value == null) return "—";
  const numeric = Number(value);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(3)}`;
}

export default function LeaguePrintoutPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [printout, setPrintout] = useState<AdminLeaguePrintoutResponse | null>(null);
  const [weekNum, setWeekNum] = useState("");
  const [loadingLeagues, setLoadingLeagues] = useState(false);
  const [loadingPrintout, setLoadingPrintout] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  function resetWorkspace() {
    setLeagues([]);
    setLeagueName("");
    setPrintout(null);
    setWeekNum("");
    setLoadingLeagues(false);
    setLoadingPrintout(false);
    setMessage(null);
  }

  const listRequest = useLatestRequestGuard(accessToken, resetWorkspace);
  const detailRequest = useLatestRequestGuard(accessToken);
  const busy = loadingLeagues || loadingPrintout;

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before loading league printouts.");
    const response = await fetch(apiUrl(apiBase, path), { headers: { Authorization: `Bearer ${accessToken}` } });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadLeagues() {
    const selectedLeagueBeforeRefresh = leagueName;
    const selectedWeekBeforeRefresh = weekNum;
    const generation = listRequest.begin();
    detailRequest.invalidate();
    setLoadingLeagues(true);
    setMessage(null);
    setPrintout(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      if (!listRequest.isCurrent(generation)) return;
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      const selectedLeague = names.includes(selectedLeagueBeforeRefresh) ? selectedLeagueBeforeRefresh : (names[0] || "");
      setLeagueName(selectedLeague);
      if (selectedLeague) {
        const selectedWeek = selectedLeague === selectedLeagueBeforeRefresh ? selectedWeekBeforeRefresh : "";
        if (selectedLeague !== selectedLeagueBeforeRefresh) setWeekNum("");
        await loadDetail(selectedLeague, selectedWeek);
      } else {
        setWeekNum("");
        setMessage("No leagues are available.");
      }
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      if (listRequest.isCurrent(generation)) setLoadingLeagues(false);
    }
  }

  async function loadDetail(selectedLeague = leagueName, selectedWeek = weekNum) {
    const generation = detailRequest.begin();
    if (!selectedLeague) {
      setMessage("Select a league first.");
      return;
    }
    setLoadingPrintout(true);
    setMessage(null);
    setPrintout(null);
    try {
      const weekQuery = selectedWeek ? `?week_num=${encodeURIComponent(selectedWeek)}` : "";
      const payload = await requestJson<AdminLeaguePrintoutResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/printout${weekQuery}`);
      if (!detailRequest.isCurrent(generation)) return;
      setPrintout(payload);
      setWeekNum(payload.selected_week == null ? "" : String(payload.selected_week));
      setMessage(`Printout loaded${payload.selected_week ? ` for Week ${payload.selected_week}` : ""}.`);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load league printout.");
    } finally {
      if (detailRequest.isCurrent(generation)) setLoadingPrintout(false);
    }
  }

  function selectLeague(selectedLeague: string) {
    setLeagueName(selectedLeague);
    setWeekNum("");
    void loadDetail(selectedLeague, "");
  }

  function selectWeek(selectedWeek: string) {
    setWeekNum(selectedWeek);
    if (leagueName) void loadDetail(leagueName, selectedWeek);
    else detailRequest.invalidate();
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

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
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } body { background: white !important; } [data-print-surface] { display: block !important; } .print-section { break-inside: avoid; page-break-inside: avoid; } .print-break-before { break-before: page; page-break-before: always; } table { font-size: 11px; } thead { display: table-header-group; } tr { break-inside: avoid; page-break-inside: avoid; } @page { size: auto; margin: 10mm; } }`}</style>
      <article className="no-print" style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article className="no-print" style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Select printout</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => selectLeague(event.target.value)} disabled={busy || !accessToken} aria-busy={loadingLeagues} style={inputStyle}><option value="" disabled>{loadingLeagues ? "Loading leagues…" : "Choose a league"}</option>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <label>Scored week<br /><select value={weekNum} onChange={(event) => selectWeek(event.target.value)} disabled={busy || !printout?.available_weeks.length} style={inputStyle}><option value="">Latest scored week</option>{(printout?.available_weeks || []).map((week) => <option key={week} value={String(week)}>Week {week}</option>)}</select></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{loadingLeagues ? "Refreshing leagues…" : "Refresh leagues"}</button>
          <button type="button" onClick={() => void loadDetail()} disabled={busy || !leagueName} style={buttonStyle}>{loadingPrintout ? "Loading printout…" : "Reload printout"}</button>
          <button type="button" onClick={() => window.print()} disabled={busy || !printout} style={buttonStyle}>Print or save PDF</button>
        </div>
        {loadingPrintout ? <p role="status" style={{ color: "#475569" }}>Loading {leagueName || "selected league"}…</p> : null}
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {printout ? (
        <section data-print-surface="league-night">
          <h1 style={{ marginBottom: "0.25rem" }}>{printout.detail.league.league_name} league night printout</h1>
          <p style={{ color: "#475569" }}>Status: {printout.detail.league.status} · {printout.selected_week ? `Week ${printout.selected_week}` : "No scored week"} · K-factor: {printout.detail.league.k_factor ?? "—"} · Min games: {printout.detail.league.min_games ?? "—"}</p>
          {printout.warnings.length ? <ul className="no-print" style={{ color: "#92400e" }}>{printout.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}

          <article className="print-section" style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Schedule</h2>
            {printout.detail.schedule_preview?.length ? (
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th align="left">Session</th><th align="left">Date</th><th align="left">Start</th><th align="left">End</th></tr></thead>
                <tbody>{printout.detail.schedule_preview.map((row) => <tr key={`${row.session}-${row.date}`}><td>{row.session}</td><td>{row.date}</td><td>{row.start || "—"}</td><td>{row.end || "—"}</td></tr>)}</tbody>
              </table>
            ) : <p style={{ color: "#64748b" }}>No schedule preview configured.</p>}
          </article>

          <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Weekly leaders</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
              <div><h3>Highest rating gained</h3>{printout.weekly_rating_leaders.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Δ JUPR</th><th align="right">Games</th></tr></thead><tbody>{printout.weekly_rating_leaders.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{signedRatingDelta(row.rating_delta_jupr)}</td><td align="right">{row.games}</td></tr>)}</tbody></table> : <p>No rating leaders for this week.</p>}</div>
              <div><h3>Most wins</h3>{printout.weekly_win_leaders.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Wins</th><th align="right">Games</th></tr></thead><tbody>{printout.weekly_win_leaders.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{row.wins}</td><td align="right">{row.games}</td></tr>)}</tbody></table> : <p>No win leaders for this week.</p>}</div>
            </div>
          </article>

          <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Season leaders (Top Performers)</h2>
            {printout.season_top_performers.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Category</th><th align="right">Place</th><th align="left">Player</th><th align="right">Metric</th><th align="right">Min games</th></tr></thead><tbody>{printout.season_top_performers.map((row) => <tr key={`${row.category_key}-${row.rank}-${row.player_id}`}><td>{row.category_label}</td><td align="right">{row.rank}</td><td>{row.player_name}</td><td align="right">{row.metric_display}</td><td align="right">{row.min_games}</td></tr>)}</tbody></table> : <p>No configured Top Performer winners are eligible yet.</p>}
          </article>

          <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Standings</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Rank</th><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="right">Matches</th></tr></thead>
              <tbody>{printout.detail.standings.map((row) => <tr key={row.player_id}><td>{row.rank}</td><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td align="right">{row.matches_played ?? 0}</td></tr>)}</tbody>
            </table>
          </article>

          <article className="print-section print-break-before" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Roster checklist</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="left">Present</th><th align="left">Notes</th></tr></thead>
              <tbody>{(printout.detail.roster || []).filter((row) => row.in_league).map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td>□</td><td>________________</td></tr>)}</tbody>
            </table>
          </article>
        </section>
      ) : null}
    </div>
  );
}
