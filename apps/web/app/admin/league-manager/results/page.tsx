import Link from "next/link";
import { redirect } from "next/navigation";
import { getClubLeagueResults } from "@/lib/api";
import LeagueManagerNav from "../LeagueManagerNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

function rating(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
}

export default async function AdminLeagueResultsPage({ searchParams }: Props) {
  const requestedLeague = first(searchParams?.league).trim();
  const leagueType = first(searchParams?.mode).trim();
  if (!requestedLeague) redirect("/admin/league-manager");

  const { data, error } = await getClubLeagueResults("tres-palapas", requestedLeague);
  const selectedLeague = data?.selected_league || requestedLeague;
  const selectedWeek = data?.selected_week || null;
  const weeklyRows = (data?.weekly_results || []).filter((row) => selectedWeek == null || row.week_num === selectedWeek);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>{selectedLeague} results</h1>
      <LeagueManagerNav leagueName={selectedLeague} leagueType={leagueType || null} />

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}

      {data ? (
        <>
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <p style={{ display: "flex", gap: "1rem", flexWrap: "wrap", margin: 0 }}>
              <span><strong>Players:</strong> {data.standings.length}</span>
              <span><strong>Latest week:</strong> {selectedWeek ? `Week ${selectedWeek}` : "No weekly results"}</span>
              <Link href={`/clubs/tres-palapas/league-results?league=${encodeURIComponent(selectedLeague)}`}>Open public league results</Link>
            </p>
          </article>

          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Current standings</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
                <thead><tr>{["Rank", "Player", "Rating", "Matches", "Wins", "Losses", "Win %", "Rating change"].map((heading) => <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>)}</tr></thead>
                <tbody>{data.standings.map((row) => <tr key={String(row.player_id)}><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_jupr)}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.win_pct == null ? "—" : `${Number(row.win_pct).toFixed(1)}%`}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td></tr>)}</tbody>
              </table>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{selectedWeek ? `Week ${selectedWeek} results` : "Weekly results"}</h2>
            {weeklyRows.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}><thead><tr>{["Player", "Games", "Wins", "Losses", "Win %", "Rating change"].map((heading) => <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>)}</tr></thead><tbody>{weeklyRows.map((row) => <tr key={`${row.week_num}-${row.player_id}`}><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.games ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.win_pct == null ? "—" : `${Number(row.win_pct).toFixed(1)}%`}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No weekly results are available yet.</p>}
          </article>
        </>
      ) : null}
    </section>
  );
}
