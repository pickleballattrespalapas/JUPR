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

function divisionLabel(eventFamily?: string | null, divisionName?: string | null): string {
  const family = String(eventFamily || "").trim();
  const division = String(divisionName || "").trim();
  if (!family) return division;
  if (!division) return family;
  if (division.toLocaleLowerCase().startsWith(family.toLocaleLowerCase())) return division;
  return `${family} · ${division}`;
}

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
      <h1 style={{ marginTop: 0 }}>Four-player team results</h1>
      <p style={{ color: "#475569", maxWidth: "52rem" }}>
        Follow standings, brackets, rosters, and medal winners.
      </p>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Team tournament results are unavailable right now. Please try again shortly.
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
              {divisionLabel(draw.event_family_label, draw.division_name)}
            </p>
            <p>
              <strong>{draw.team_count}</strong> {draw.team_count === 1 ? "team competing" : "teams competing"}
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
        <p>No team results yet.</p>
      ) : null}
    </section>
  );
}
