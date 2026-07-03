import Link from "next/link";
import { getClubMatches, type PublicMatch } from "@/lib/api";

type MatchesPageProps = {
  params: { clubSlug: string };
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };

function formatMatchDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function teamLabel(clubSlug: string, players: PublicMatch["team_1"]): JSX.Element {
  return (
    <>
      {players.map((player, index) => (
        <span key={String(player.id)}>
          {index > 0 ? " / " : ""}
          <Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link>
        </span>
      ))}
    </>
  );
}

function scoreLabel(match: PublicMatch): string {
  const scoreA = match.score_t1 ?? null;
  const scoreB = match.score_t2 ?? null;
  return scoreA == null && scoreB == null ? "—" : `${scoreA ?? 0}–${scoreB ?? 0}`;
}

export default async function MatchesPage({ params }: MatchesPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubMatches(clubSlug);
  const clubName = data?.club?.name ?? clubSlug;
  const matches = data?.matches ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Match history
      </p>
      <h1 style={{ marginTop: 0 }}>{clubName} matches</h1>
      <p style={{ color: "#475569" }}>Public match history is the connective tissue between score entry, ratings, leaderboards, and player profiles.</p>

      {error ? <p style={{ color: "#b91c1c" }}>Match history is temporarily unavailable. {error}</p> : null}
      {!error && matches.length === 0 ? <p>No public matches are available yet.</p> : null}

      {matches.length > 0 ? (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem" }}>
            <thead>
              <tr>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Team 1</th>
                <th style={thStyle}>Score</th>
                <th style={thStyle}>Team 2</th>
                <th style={thStyle}>League</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((match, index) => {
                const detailHref = match.id ? `/clubs/${clubSlug}/matches/${match.id}` : `/clubs/${clubSlug}/matches`;
                return (
                  <tr key={`${match.id ?? index}`}>
                    <td style={tdStyle}>{match.id ? <Link href={detailHref}>{formatMatchDate(match.date)}</Link> : formatMatchDate(match.date)}</td>
                    <td style={tdStyle}>{teamLabel(clubSlug, match.team_1)}</td>
                    <td style={tdStyle}>{match.id ? <Link href={detailHref}>{scoreLabel(match)}</Link> : scoreLabel(match)}</td>
                    <td style={tdStyle}>{teamLabel(clubSlug, match.team_2)}</td>
                    <td style={tdStyle}>{match.league ?? "—"}</td>
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
