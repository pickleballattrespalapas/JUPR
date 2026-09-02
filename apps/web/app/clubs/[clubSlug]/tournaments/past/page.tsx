import Link from "next/link";
import { getPublicTournamentResultsIndex } from "@/lib/tournamentResultsApi";

type Props = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function dateLabel(value?: string | null): string {
  if (!value) return "TBD";
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value.slice(0, 10);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  }).format(date);
}

export default async function PastTournamentsPage({ params }: Props) {
  const { data, error } = await getPublicTournamentResultsIndex(
    params.clubSlug,
    "past"
  );

  return (
    <section>
      <p style={{ margin: "0 0 0.75rem" }}>
        <Link href={`/clubs/${params.clubSlug}/tournaments`}>
          ← Current tournaments
        </Link>
      </p>
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
        Tournament archive
      </p>
      <h1 style={{ marginTop: 0 }}>Past tournaments</h1>
      <p style={{ color: "#475569", maxWidth: "52rem" }}>
        Officially completed tournaments with published scores, brackets,
        standings, and podiums.
      </p>

      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Past tournaments are temporarily unavailable. {error}
        </p>
      ) : null}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "0.75rem"
        }}
      >
        {(data?.tournaments || []).map((tournament) => (
          <article key={tournament.id} style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{tournament.name}</h2>
            <p style={{ color: "#475569" }}>
              {dateLabel(tournament.start_date)}
              {tournament.end_date
                ? ` – ${dateLabel(tournament.end_date)}`
                : ""}
            </p>
            <Link
              href={`/clubs/${params.clubSlug}/tournament-results?tournament_id=${encodeURIComponent(tournament.id)}&view=past`}
              style={{ fontWeight: 800 }}
            >
              View final results
            </Link>
          </article>
        ))}
      </div>

      {!error && !data?.tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No published past tournaments</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Completed tournaments will appear here after their results are
            officially published.
          </p>
        </article>
      ) : null}
    </section>
  );
}
