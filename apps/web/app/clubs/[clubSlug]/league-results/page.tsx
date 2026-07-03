import Link from "next/link";
import { getClubLeagueResults } from "@/lib/api";
import type { LeagueResultsStatRow, LeagueResultsStanding } from "@/lib/api";

type LeagueResultsPageProps = {
  params: { clubSlug: string };
  searchParams?: { league?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function juprLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function percentLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return `${n >= 0 ? "+" : ""}${n.toFixed(3)}`;
}

function playerHref(clubSlug: string, playerId: string | number): string {
  return `/clubs/${clubSlug}/players/${playerId}`;
}

function HighlightList({ title, rows, clubSlug }: { title: string; rows: LeagueResultsStatRow[]; clubSlug: string }) {
  return (
    <article style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {rows.length === 0 ? <p style={{ color: "#64748b" }}>No data yet.</p> : null}
      <ol style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
        {rows.map((row) => (
          <li key={`${row.week_num ?? "all"}-${row.player_id}-${title}`}>
            <Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link>
            <span style={{ color: "#475569" }}> — {row.wins ?? 0}-{row.losses ?? 0}, {row.games ?? 0} games</span>
            {row.rating_delta_jupr != null ? <span style={{ color: "#475569" }}> · Δ {deltaLabel(row.rating_delta_jupr)}</span> : null}
          </li>
        ))}
      </ol>
    </article>
  );
}

function StandingsTable({ standings, clubSlug }: { standings: LeagueResultsStanding[]; clubSlug: string }) {
  if (!standings.length) return <p>No public standings are available for this league yet.</p>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
        <thead>
          <tr>
            {[
              "Rank",
              "Player",
              "JUPR",
              "Games",
              "Wins",
              "Losses",
              "Win %",
              "Rating Δ"
            ].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {standings.map((row) => (
            <tr key={String(row.player_id)}>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}><Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link></td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{percentLabel(row.win_pct)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatTable({ rows, clubSlug, title }: { rows: LeagueResultsStatRow[]; clubSlug: string; title: string }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No {title.toLowerCase()} data yet.</p>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "640px" }}>
        <thead>
          <tr>
            {[
              "Player",
              "Week",
              "Games",
              "Wins",
              "Losses",
              "Win %",
              "Rating Δ"
            ].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.week_num ?? "all"}-${row.player_id}`}>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}><Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link></td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.week_num ? `Week ${row.week_num}` : "Season"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.games ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{percentLabel(row.win_pct)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default async function LeagueResultsPage({ params, searchParams }: LeagueResultsPageProps) {
  const { clubSlug } = params;
  const leagueName = searchParams?.league ? decodeURIComponent(searchParams.league) : null;
  const { data, error } = await getClubLeagueResults(clubSlug, leagueName);
  const selectedLeague = data?.selected_league ?? null;
  const recentWeek = data?.weeks?.length ? data.weeks[data.weeks.length - 1]?.week_num : null;
  const recentWeeklyRows = recentWeek ? (data?.weekly_results ?? []).filter((row) => row.week_num === recentWeek) : [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        League Results
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} league results</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Public league standings, weekly results, and season performance. This page is read-only and uses public-safe FastAPI summaries.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>League Results are temporarily unavailable. {error}</p> : null}
      {!error && !data?.leagues?.length ? <p>No public leagues are available yet.</p> : null}

      {data?.leagues?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.leagues.map((league) => {
            const active = league.name === selectedLeague;
            return (
              <Link key={league.name} href={`/clubs/${clubSlug}/league-results?league=${encodeURIComponent(league.name)}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {league.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {data && selectedLeague ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>League</strong><br />{selectedLeague}</article>
            <article style={cardStyle}><strong>Minimum games</strong><br />{data.league?.min_games ?? 0}</article>
            <article style={cardStyle}><strong>K-factor</strong><br />{data.league?.k_factor ?? "Default"}</article>
            <article style={cardStyle}><strong>Weeks with results</strong><br />{data.weeks.length}</article>
          </div>

          <h2>Current standings</h2>
          <StandingsTable standings={data.standings} clubSlug={clubSlug} />

          <h2>Latest weekly highlights{recentWeek ? ` — Week ${recentWeek}` : ""}</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
            <HighlightList title="Biggest climbers" rows={data.highlights.biggest_climbers} clubSlug={clubSlug} />
            <HighlightList title="Best win %" rows={data.highlights.best_win_pct} clubSlug={clubSlug} />
            <HighlightList title="Most active" rows={data.highlights.most_active} clubSlug={clubSlug} />
          </div>

          <h2>Latest weekly results{recentWeek ? ` — Week ${recentWeek}` : ""}</h2>
          <StatTable title="Weekly" rows={recentWeeklyRows.slice(0, 25)} clubSlug={clubSlug} />

          <h2>Season cumulative performance</h2>
          <StatTable title="Season" rows={data.cumulative.slice(0, 25)} clubSlug={clubSlug} />
        </>
      ) : null}
    </section>
  );
}
