import Link from "next/link";
import { getClubPlayers } from "@/lib/api";

type PlayersPageProps = {
  params: { clubSlug: string };
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

export default async function ClubPlayersPage({ params }: PlayersPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubPlayers(clubSlug);
  const clubName = data?.club?.name ?? clubSlug;
  const players = data?.players ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Player profiles
      </p>
      <h1 style={{ marginTop: 0 }}>{clubName} players</h1>
      <p style={{ color: "#334155" }}>
        Player profiles connect the core JUPR loop: ratings, match history, and league-specific performance.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Players are temporarily unavailable. {error}</p> : null}
      {!error && players.length === 0 ? <p>No public players are available yet.</p> : null}

      {players.length > 0 ? (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem" }}>
            <thead>
              <tr>
                <th style={thStyle}>Player</th>
                <th style={thStyle}>JUPR</th>
                <th style={thStyle}>Matches</th>
                <th style={thStyle}>W/L</th>
                <th style={thStyle}>Status</th>
              </tr>
            </thead>
            <tbody>
              {players.map((player) => {
                const wins = player.wins ?? 0;
                const losses = player.losses ?? 0;
                return (
                  <tr key={String(player.id)}>
                    <td style={tdStyle}><Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link></td>
                    <td style={tdStyle}>{ratingLabel(player.rating)}</td>
                    <td style={tdStyle}>{player.matches_played ?? wins + losses}</td>
                    <td style={tdStyle}>{wins}/{losses}</td>
                    <td style={tdStyle}>{player.is_active === false ? "Inactive" : "Active"}</td>
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
