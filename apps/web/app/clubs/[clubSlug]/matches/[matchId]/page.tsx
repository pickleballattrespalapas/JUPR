import Link from "next/link";
import { getClubMatch, type PublicMatchPlayer, type PublicRatingSnapshotEntry } from "@/lib/api";

type MatchDetailPageProps = {
  params: { clubSlug: string; matchId: string };
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };

function formatDateTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function scoreLabel(scoreA?: number | null, scoreB?: number | null): string {
  if (scoreA == null && scoreB == null) return "—";
  return `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function teamLabel(clubSlug: string, players: PublicMatchPlayer[]): JSX.Element {
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

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

function snapshotRows(players: PublicMatchPlayer[], snapshot: PublicRatingSnapshotEntry[] | undefined) {
  return players.map((player) => {
    const snap = (snapshot || []).find((entry) => String(entry.player_id) === String(player.id));
    return { player, snap };
  });
}

export default async function MatchDetailPage({ params }: MatchDetailPageProps) {
  const { clubSlug, matchId } = params;
  const { data, error } = await getClubMatch(clubSlug, matchId);
  const match = data?.match;

  if (error || !match) {
    return (
      <section>
        <h1>Match unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this match. {error}</p>
        <p><Link href={`/clubs/${clubSlug}/matches`}>Back to matches</Link></p>
      </section>
    );
  }

  const team1Won = match.winner === "team_1";
  const team2Won = match.winner === "team_2";

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {data?.club?.name ?? clubSlug} · Match detail
      </p>
      <h1 style={{ marginTop: 0 }}>{scoreLabel(match.score_t1, match.score_t2)}</h1>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Date</strong><div>{formatDateTime(match.date)}</div></article>
        <article style={cardStyle}><strong>League</strong><div>{match.league ?? "—"}</div></article>
        <article style={cardStyle}><strong>Type</strong><div>{match.match_type ?? "—"}</div></article>
        <article style={cardStyle}><strong>Rating scope</strong><div>{match.rating_scope ?? "—"}</div></article>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        <article style={{ ...cardStyle, borderColor: team1Won ? "#86efac" : "#e2e8f0" }}>
          <h2 style={{ marginTop: 0 }}>Team 1 {team1Won ? "· Winner" : ""}</h2>
          <p>{teamLabel(clubSlug, match.team_1)}</p>
          <div style={{ fontSize: "2rem", fontWeight: 800 }}>{match.score_t1 ?? "—"}</div>
        </article>
        <article style={{ ...cardStyle, borderColor: team2Won ? "#86efac" : "#e2e8f0" }}>
          <h2 style={{ marginTop: 0 }}>Team 2 {team2Won ? "· Winner" : ""}</h2>
          <p>{teamLabel(clubSlug, match.team_2)}</p>
          <div style={{ fontSize: "2rem", fontWeight: 800 }}>{match.score_t2 ?? "—"}</div>
        </article>
      </div>

      {match.rating_snapshot ? (
        <section style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Rating snapshot</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={thStyle}>Player</th><th style={thStyle}>Team</th><th style={thStyle}>Start</th><th style={thStyle}>End</th><th style={thStyle}>Change</th></tr></thead>
              <tbody>
                {[...snapshotRows(match.team_1, match.rating_snapshot.team_1).map((row) => ({ ...row, team: "Team 1" })), ...snapshotRows(match.team_2, match.rating_snapshot.team_2).map((row) => ({ ...row, team: "Team 2" }))].map(({ player, snap, team }) => {
                  const start = snap?.start_rating ?? null;
                  const end = snap?.end_rating ?? null;
                  const delta = start == null || end == null ? null : end - start;
                  return (
                    <tr key={`${team}-${player.id}`}>
                      <td style={tdStyle}><Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link></td>
                      <td style={tdStyle}>{team}</td>
                      <td style={tdStyle}>{ratingLabel(start)}</td>
                      <td style={tdStyle}>{ratingLabel(end)}</td>
                      <td style={tdStyle}>{delta == null ? "—" : `${delta >= 0 ? "+" : ""}${Math.round(delta)}`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${clubSlug}/matches`}>Back to matches</Link></p>
    </section>
  );
}
