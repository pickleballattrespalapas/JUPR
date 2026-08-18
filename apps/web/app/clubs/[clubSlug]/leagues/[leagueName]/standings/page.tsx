import Link from "next/link";
import PublicLeagueNav from "@/components/PublicLeagueNav";
import { LeagueAwardRaceGrid } from "@/components/LeagueAwardRace";
import LeaguePlayerRoster from "@/components/LeaguePlayerRoster";
import { getClubLeagueResults, type LeagueResultsStatRow } from "@/lib/api";
import PrintButton from "../../../league-results/PrintButton";

type Props = {
  params: { clubSlug: string; leagueName: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

function decodeLeagueName(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function highlightNames(rows: LeagueResultsStatRow[]): string {
  if (!rows.length) return "No qualifying players yet";
  return rows
    .slice(0, 3)
    .map((row) => row.player_name)
    .join(", ");
}

export default async function PublicLeagueStandingsPage({ params }: Props) {
  const leagueName = decodeLeagueName(params.leagueName);
  const { data, error } = await getClubLeagueResults(
    params.clubSlug,
    leagueName
  );
  const found = data?.selected_league === leagueName;

  if (error || !data || !found) {
    return (
      <section>
        <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          League Standings
        </p>
        <h1>{leagueName}</h1>
        <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
          <h2 style={{ marginTop: 0 }}>Standings unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>
            {error || "This league is not currently available as an active public league."}
          </p>
          <Link href={`/clubs/${params.clubSlug}/leagues`}>Return to all leagues</Link>
        </article>
      </section>
    );
  }

  const ratedPlayers = data.standings.filter((row) => row.rating_jupr != null).length;
  const resultWeeks = data.weeks.filter((week) => week.has_results !== false).length;

  return (
    <section>
      <style>{`
        @media print {
          @page { margin: 8mm; }
          .no-print { display: none !important; }
          body { background: white !important; font-size: 10pt; }
          a { color: inherit !important; text-decoration: none !important; }
          main { max-width: none !important; margin: 0 !important; padding: 0 !important; }
          article { padding: 3mm !important; break-inside: avoid; page-break-inside: avoid; }
          h1 { font-size: 18pt; margin-bottom: 2mm !important; }
          h2 { font-size: 13pt; margin: 3mm 0 2mm !important; }
        }
      `}</style>

      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        League Awards Race
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} awards race</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Awards-race placement and qualification come first. An unranked player roster remains available below as a reference.
      </p>

      <PublicLeagueNav clubSlug={params.clubSlug} leagueName={leagueName} active="overall" />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Players with a league rating</strong><br />{ratedPlayers}</article>
        <article style={cardStyle}><strong>Players shown</strong><br />{data.standings.length}</article>
        <article style={cardStyle}><strong>Minimum games</strong><br />{data.league?.min_games ?? 0}</article>
        <article style={cardStyle}><strong>Weeks with results</strong><br />{resultWeeks}</article>
      </div>

      <div className="no-print" style={{ marginBottom: "1rem" }}>
        <PrintButton />
      </div>

      <section style={{ marginBottom: "1.25rem" }} data-testid="league-awards">
        <h2>Awards race</h2>
        {data.award_progress.awards.length ? (
          <>
            <p style={{ color: "#64748b" }}>Top five qualified players are shown for each award. Expand a race to see every eligible player.</p>
            <LeagueAwardRaceGrid progress={data.award_progress} clubSlug={params.clubSlug} />
          </>
        ) : <article style={{ ...cardStyle, background: "#f8fafc" }}>No player has met the current award qualification criteria yet.</article>}
      </section>

      <section>
        <h2>Player roster</h2>
        <p style={{ color: "#64748b" }}>
          This is an unranked reference roster. Sort it by the measure you need; award placement is shown above.
        </p>
        <LeaguePlayerRoster standings={data.standings} clubSlug={params.clubSlug} />
      </section>

      <section style={{ marginTop: "1.25rem" }}>
        <h2>Season highlights</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Biggest climbers</h3>
            <p style={{ marginBottom: 0 }}>{highlightNames(data.season_highlights.biggest_climbers)}</p>
          </article>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Best win percentage</h3>
            <p style={{ marginBottom: 0 }}>{highlightNames(data.season_highlights.best_win_pct)}</p>
          </article>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Most active</h3>
            <p style={{ marginBottom: 0 }}>{highlightNames(data.season_highlights.most_active)}</p>
          </article>
        </div>
      </section>
    </section>
  );
}
