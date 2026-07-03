import Link from "next/link";
import { getClubPlayerProfile, type PublicMatch } from "@/lib/api";

type PlayerProfilePageProps = {
  params: { clubSlug: string; playerId: string };
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

function formatMatchDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function teamLabel(players: Array<{ name: string }>): string {
  return players.map((p) => p.name).filter(Boolean).join(" / ") || "—";
}

function matchLabel(match: PublicMatch): string {
  const scoreA = match.score_t1 ?? null;
  const scoreB = match.score_t2 ?? null;
  return scoreA == null && scoreB == null ? "—" : `${scoreA ?? 0}–${scoreB ?? 0}`;
}

export default async function PlayerProfilePage({ params }: PlayerProfilePageProps) {
  const { clubSlug, playerId } = params;
  const { data, error } = await getClubPlayerProfile(clubSlug, playerId);
  const player = data?.player;

  if (error || !player) {
    return (
      <section>
        <h1>Player unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this player profile. {error}</p>
        <p><Link href={`/clubs/${clubSlug}/players`}>Back to players</Link></p>
      </section>
    );
  }

  const wins = player.wins ?? 0;
  const losses = player.losses ?? 0;
  const leagueRatings = data?.league_ratings ?? [];
  const matches = data?.recent_matches ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {data?.club?.name ?? clubSlug} · Player profile
      </p>
      <h1 style={{ marginTop: 0 }}>{player.name}</h1>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Current JUPR</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{ratingLabel(player.rating)}</div></article>
        <article style={cardStyle}><strong>Matches</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{player.matches_played ?? wins + losses}</div></article>
        <article style={cardStyle}><strong>Record</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{wins}/{losses}</div></article>
        <article style={cardStyle}><strong>Status</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{player.is_active === false ? "Inactive" : "Active"}</div></article>
      </div>

      <section style={{ ...cardStyle, marginBottom: "1rem" }}>
        <h2 style={{ marginTop: 0 }}>League ratings</h2>
        {leagueRatings.length === 0 ? <p style={{ color: "#475569" }}>No league-specific ratings yet.</p> : null}
        {leagueRatings.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={thStyle}>League</th><th style={thStyle}>Rating</th><th style={thStyle}>Matches</th><th style={thStyle}>W/L</th></tr></thead>
              <tbody>
                {leagueRatings.map((row, index) => (
                  <tr key={`${row.league_name ?? "league"}-${index}`}>
                    <td style={tdStyle}>{row.league_name ?? "Overall"}</td>
                    <td style={tdStyle}>{ratingLabel(row.rating)}</td>
                    <td style={tdStyle}>{row.matches_played ?? (row.wins ?? 0) + (row.losses ?? 0)}</td>
                    <td style={tdStyle}>{row.wins ?? 0}/{row.losses ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Recent matches</h2>
        {matches.length === 0 ? <p style={{ color: "#475569" }}>No recent public matches yet.</p> : null}
        {matches.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={thStyle}>Date</th><th style={thStyle}>Team 1</th><th style={thStyle}>Score</th><th style={thStyle}>Team 2</th><th style={thStyle}>League</th></tr></thead>
              <tbody>
                {matches.map((match, index) => {
                  const detailHref = match.id ? `/clubs/${clubSlug}/matches/${match.id}` : `/clubs/${clubSlug}/matches`;
                  return (
                    <tr key={`${match.id ?? index}`}>
                      <td style={tdStyle}>{match.id ? <Link href={detailHref}>{formatMatchDate(match.date)}</Link> : formatMatchDate(match.date)}</td>
                      <td style={tdStyle}>{teamLabel(match.team_1)}</td>
                      <td style={tdStyle}>{match.id ? <Link href={detailHref}>{matchLabel(match)}</Link> : matchLabel(match)}</td>
                      <td style={tdStyle}>{teamLabel(match.team_2)}</td>
                      <td style={tdStyle}>{match.league ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${clubSlug}/players`}>Back to players</Link></p>
    </section>
  );
}
