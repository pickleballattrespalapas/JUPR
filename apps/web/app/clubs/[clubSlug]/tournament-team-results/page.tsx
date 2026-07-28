import Link from "next/link";
import { getPublicTeamTournamentIndex } from "@/lib/tournamentTeamCompetitionApi";

type Props = {
  params: { clubSlug: string };
};

const card = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default async function TournamentTeamResultsIndex({ params }: Props) {
  const { data, error } = await getPublicTeamTournamentIndex(params.clubSlug);

  return (
    <section>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 800,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        Team tournaments
      </p>
      <h1 style={{ marginTop: 0 }}>Follow four-player team standings</h1>
      <p style={{ color: "#475569", maxWidth: "52rem" }}>
        Published round-robin standings, playoff brackets, rosters, and podiums.
      </p>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Team tournament results are temporarily unavailable. {error}
        </p>
      ) : null}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "0.85rem"
        }}
      >
        {(data?.draws || []).map((draw) => (
          <article key={draw.id} style={card}>
            <p style={{ color: "#2563eb", fontWeight: 750, marginTop: 0 }}>
              {draw.tournament_name}
            </p>
            <h2>{draw.name}</h2>
            <p style={{ color: "#475569" }}>
              {[draw.event_family_label, draw.division_name]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <p>
              <strong>{draw.team_count}</strong> active teams
            </p>
            <Link
              href={`/clubs/${params.clubSlug}/tournament-team-results/${encodeURIComponent(
                draw.tournament_id
              )}/${encodeURIComponent(draw.id)}`}
              style={{ fontWeight: 800 }}
            >
              View standings and bracket
            </Link>
          </article>
        ))}
      </div>
      {!error && !data?.draws?.length ? (
        <p>No four-player team results have been published yet.</p>
      ) : null}
    </section>
  );
}
