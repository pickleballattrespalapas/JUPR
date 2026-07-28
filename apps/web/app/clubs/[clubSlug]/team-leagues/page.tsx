import Link from "next/link";
import { getPublicTeamLeagues } from "@/lib/teamLeagueApi";

type Props = { params: { clubSlug: string } };
const card = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default async function TeamLeaguesPage({ params }: Props) {
  const { data, error } = await getPublicTeamLeagues(params.clubSlug);
  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Fixed-partner leagues
      </p>
      <h1>Team leagues</h1>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        Register with the same partner for the season, play one scheduled match
        each week, and face every other team before optional playoffs.
      </p>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        {(data?.leagues || []).map((league) => (
          <article key={league.league_name} style={card}>
            <h2 style={{ marginTop: 0 }}>{league.league_name}</h2>
            <p>{league.venue || "Venue to be announced"}</p>
            <p>
              {league.registration_open
                ? "Registration open"
                : league.status.replaceAll("_", " ")}
              {" · "}
              {league.allow_substitutes ? "Substitutes allowed" : "Fixed partners only"}
            </p>
            <Link href={`/clubs/${params.clubSlug}/team-leagues/${encodeURIComponent(league.league_name)}`}>
              Open league
            </Link>
          </article>
        ))}
      </div>
      {!error && !data?.leagues?.length ? <p>No team leagues are published yet.</p> : null}
      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${params.clubSlug}`}>Club home</Link>
      </p>
    </section>
  );
}
