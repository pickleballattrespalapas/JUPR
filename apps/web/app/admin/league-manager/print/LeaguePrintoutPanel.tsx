"use client";

import Link from "next/link";
import { useState } from "react";
import type { AdminLeagueManagerStatusResponse, AdminLeaguePrintoutResponse } from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
  initialLeague: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Number(value).toFixed(3);
}

function signedRatingDelta(value?: number | null): string {
  if (value == null) return "—";
  const numeric = Number(value);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(3)}`;
}

export default function LeaguePrintoutPanel({ apiBase, clubId, status, initialLeague }: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [printout, setPrintout] = useState<AdminLeaguePrintoutResponse | null>(null);
  const [weekNum, setWeekNum] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${initialLeague}`, clearProtectedState);

  function clearProtectedState() {
    setPrintout(null);
    setWeekNum("");
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before loading the league printout.");
    const response = await fetch(apiUrl(apiBase, path), {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadDetail(selectedWeek = weekNum) {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const weekQuery = selectedWeek ? `?week_num=${encodeURIComponent(selectedWeek)}` : "";
      const payload = await requestJson<AdminLeaguePrintoutResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/printout${weekQuery}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setPrintout(payload);
      setWeekNum(payload.selected_week == null ? "" : String(payload.selected_week));
      setMessage(
        payload.has_printable_data
          ? `Printout loaded${payload.selected_week ? ` for Week ${payload.selected_week}` : ""}.`
          : "Nothing to print yet. Add a schedule, league roster, or scored results before printing."
      );
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load league printout.");
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectWeek(selectedWeek: string) {
    setWeekNum(selectedWeek);
    void loadDetail(selectedWeek);
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${initialLeague}` : "", () => loadDetail(""));

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}>League Manager is currently unavailable.</article>;
  }

  if (sessionLoading) return <p role="status">Checking admin access…</p>;

  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  const hasPrintableData = Boolean(printout?.has_printable_data);

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <style>{`@media print { nav, header, footer, aside, .no-print { display: none !important; } body { background: white !important; font-size: 10pt; } main, [data-print-surface] { display: block !important; width: 100% !important; max-width: none !important; margin: 0 !important; padding: 0 !important; } [data-print-surface] h1 { font-size: 18pt; margin: 0 0 2mm !important; } [data-print-surface] h2 { font-size: 13pt; margin-bottom: 2mm !important; } [data-print-surface] h3 { font-size: 11pt; margin: 2mm 0 !important; } .print-section { padding: 3mm !important; margin-top: 3mm !important; } table { font-size: 9pt; } th, td { padding: 1mm 1.5mm; } thead { display: table-header-group; } tr, h1, h2, h3 { break-inside: avoid; page-break-inside: avoid; } @page { size: auto; margin: 8mm; } }`}</style>

      <article className="no-print" style={cardStyle}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>Scored week</strong><br /><select value={weekNum} onChange={(event) => selectWeek(event.target.value)} disabled={busy || !printout?.available_weeks.length} style={inputStyle}><option value="">Latest scored week</option>{(printout?.available_weeks || []).map((week) => <option key={week} value={String(week)}>Week {week}</option>)}</select></label>
          <button type="button" onClick={() => void loadDetail()} disabled={busy} style={buttonStyle}>{busy ? "Loading…" : "Reload printout"}</button>
          <button type="button" onClick={() => window.print()} disabled={busy || !hasPrintableData} style={buttonStyle}>Print or save PDF</button>
        </div>
        {message ? <p role="status" style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : /^Printout loaded/.test(message) ? "#166534" : "#475569" }}>{message}</p> : null}
      </article>

      {printout && !hasPrintableData ? (
        <article className="no-print" style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>No league-night printout available yet</h2>
          <p style={{ marginBottom: 0, color: "#475569" }}>
            {printout.detail.league.league_name} has no schedule, league roster, or scored results to print.
          </p>
        </article>
      ) : null}

      {printout && hasPrintableData ? (
        <section data-print-surface="league-night">
          <h1 style={{ marginBottom: "0.25rem" }}>{printout.detail.league.league_name} league night printout</h1>
          <p style={{ color: "#475569" }}>Status: {printout.detail.league.status} · {printout.selected_week ? `Week ${printout.selected_week}` : "No scored week"} · K-factor: {printout.detail.league.k_factor ?? "—"} · Min games: {printout.detail.league.min_games ?? "—"}</p>
          {printout.warnings.length ? <ul className="no-print" style={{ color: "#92400e" }}>{printout.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}

          {printout.printable_sections.schedule ? <article className="print-section" style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Schedule</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Session</th><th align="left">Date</th><th align="left">Start</th><th align="left">End</th></tr></thead><tbody>{printout.detail.schedule_preview.map((row) => <tr key={`${row.session}-${row.date}`}><td>{row.session}</td><td>{row.date}</td><td>{row.start || "—"}</td><td>{row.end || "—"}</td></tr>)}</tbody></table>
          </article> : null}

          {printout.printable_sections.weekly_leaders ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Weekly leaders</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
              <div><h3>Highest rating gained</h3>{printout.weekly_rating_leaders.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Δ JUPR</th><th align="right">Games</th></tr></thead><tbody>{printout.weekly_rating_leaders.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{signedRatingDelta(row.rating_delta_jupr)}</td><td align="right">{row.games}</td></tr>)}</tbody></table> : <p>No rating leaders for this week.</p>}</div>
              <div><h3>Most wins</h3>{printout.weekly_win_leaders.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Wins</th><th align="right">Games</th></tr></thead><tbody>{printout.weekly_win_leaders.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{row.wins}</td><td align="right">{row.games}</td></tr>)}</tbody></table> : <p>No win leaders for this week.</p>}</div>
            </div>
          </article> : null}

          {printout.printable_sections.season_leaders ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Season leaders</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Category</th><th align="right">Place</th><th align="left">Recipient</th><th align="right">Metric</th><th align="right">Min games</th></tr></thead><tbody>{printout.season_top_performers.map((row) => <tr key={`${row.category_key}-${row.rank}-${row.recipient_type || "player"}-${row.team_id || row.player_id}`}><td>{row.category_label}</td><td align="right">{row.rank}</td><td>{row.recipient_name || row.team_name || row.player_name || "—"}</td><td align="right">{row.metric_display}</td><td align="right">{row.min_games}</td></tr>)}</tbody></table>
          </article> : null}

          {printout.printable_sections.standings ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Standings</h2>
              <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Rank</th><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="right">Matches</th></tr></thead><tbody>{printout.detail.standings.map((row) => <tr key={row.player_id}><td>{row.rank}</td><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td align="right">{row.matches_played ?? 0}</td></tr>)}</tbody></table>
          </article> : null}

          {printout.printable_sections.team_standings ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Team standings</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Rank</th><th align="left">Team</th><th align="right">Record</th><th align="right">Played</th><th align="right">Points</th><th align="right">Diff.</th></tr></thead><tbody>{printout.team_print.standings.map((row) => <tr key={row.team_id}><td>{row.rank}</td><td>{row.team_name}</td><td align="right">{row.wins}-{row.losses}</td><td align="right">{row.games_played}</td><td align="right">{row.points_for}-{row.points_against}</td><td align="right">{row.point_differential >= 0 ? "+" : ""}{row.point_differential}</td></tr>)}</tbody></table>
          </article> : null}

          {printout.printable_sections.roster ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Roster checklist</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="left">Present</th><th align="left">Notes</th></tr></thead><tbody>{(printout.detail.roster || []).filter((row) => row.in_league).map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{ratingLabel(row.rating_jupr)}</td><td align="right">{row.wins ?? 0}-{row.losses ?? 0}</td><td>□</td><td>________________</td></tr>)}</tbody></table>
          </article> : null}

          {printout.printable_sections.team_rosters ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Team rosters and availability</h2>
            {printout.team_print.teams.map((team) => <section key={team.team_id} style={{ marginTop: "0.75rem" }}><h3>{team.team_name} · {team.roster_complete ? "Ready" : "Forming"}</h3><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="left">Role</th><th align="left">Status</th><th align="left">Present</th><th align="left">Notes</th></tr></thead><tbody>{team.members.map((member) => <tr key={`${team.team_id}-${member.player_id}`}><td>{member.player_name}</td><td>{member.role}</td><td>{member.status}</td><td>□</td><td>________________</td></tr>)}</tbody></table></section>)}
          </article> : null}

          {printout.printable_sections.substitute_pool ? <article className="print-section" style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Shared substitute pool</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="left">Availability</th><th align="left">Contacted</th><th align="left">Notes</th></tr></thead><tbody>{printout.team_print.substitute_pool.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td>{row.status}</td><td>□</td><td>{row.note || "________________"}</td></tr>)}</tbody></table>
          </article> : null}
        </section>
      ) : null}
    </div>
  );
}
