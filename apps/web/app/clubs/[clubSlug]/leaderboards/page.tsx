import { getClubLeaderboard } from "@/lib/api";

type LeaderboardPageProps = {
  params: { clubSlug: string };
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };

export default async function ClubLeaderboardPage({ params }: LeaderboardPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubLeaderboard(clubSlug);
  const entries = data?.leaderboard ?? [];
  const clubName = data?.club?.name ?? clubSlug;

  if (error) {
    return (
      <section>
        <h1>{clubName} Leaderboards</h1>
        <p style={{ color: "#b91c1c" }}>Leaderboard data is temporarily unavailable. {error}</p>
      </section>
    );
  }

  return (
    <section>
      <h1>{clubName} Leaderboards</h1>
      {entries.length === 0 ? <p>No leaderboard data is currently available.</p> : null}
      {entries.length > 0 ? (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "8px", background: "white" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem" }}>
            <thead>
              <tr>
                <th style={thStyle}>Rank</th>
                <th style={thStyle}>Player</th>
                <th style={thStyle}>JUPR</th>
                <th style={thStyle}>Matches</th>
                <th style={thStyle}>W/L</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry, index) => {
                const wins = entry.wins;
                const losses = entry.losses;
                const wl = wins == null && losses == null ? "—" : `${wins ?? 0}/${losses ?? 0}`;
                return (
                  <tr key={`${entry.player_name}-${entry.player_id ?? index}`}>
                    <td style={tdStyle}>{entry.rank ?? entry.rank_position ?? index + 1}</td>
                    <td style={tdStyle}>{entry.player_name}</td>
                    <td style={tdStyle}>{entry.rating_jupr ?? entry.rating ?? "—"}</td>
                    <td style={tdStyle}>{entry.matches_played ?? "—"}</td>
                    <td style={tdStyle}>{wl}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
