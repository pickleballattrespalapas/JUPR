import { getClubLeaderboard } from "@/lib/api";

type LeaderboardPageProps = {
  params: { clubSlug: string };
};

export default async function ClubLeaderboardPage({ params }: LeaderboardPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubLeaderboard(clubSlug);

  return (
    <section>
      <h1>{clubSlug} Leaderboards</h1>
      {error ? <p style={{ color: "#b91c1c" }}>Could not load leaderboard data. {error}</p> : null}
      {!error && (!data || data.length === 0) ? <p>No leaderboard data is currently available.</p> : null}
      {data && data.length > 0 ? (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", background: "white" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Rank</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Player</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Rating</th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Matches</th>
              </tr>
            </thead>
            <tbody>
              {data.map((entry) => (
                <tr key={`${entry.player_name}-${entry.rank}`}>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{entry.rank}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{entry.player_name}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{entry.rating ?? "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{entry.matches_played ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
