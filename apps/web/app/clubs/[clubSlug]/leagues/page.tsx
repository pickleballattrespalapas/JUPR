import Link from "next/link";
import { publicLeagueHomeHref } from "@/components/PublicLeagueNav";
import { getClubLeagueResults } from "@/lib/api";

type Props = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

export default async function PublicLeaguesPage({ params }: Props) {
  const { data, error } = await getClubLeagueResults(params.clubSlug);
  const leagues = data?.leagues || [];

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
        Leagues
      </p>
      <h1 style={{ marginTop: 0 }}>Choose a league</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Open an active league to enter its League Home, standings, weekly
        history, and player summaries.
      </p>

      {error ? (
        <article
          role="alert"
          style={{
            ...cardStyle,
            borderColor: "#fecaca",
            background: "#fef2f2",
            marginBottom: "1rem"
          }}
        >
          <h2 style={{ marginTop: 0 }}>Leagues are temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>{error}</p>
        </article>
      ) : null}

      {leagues.length ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
            gap: "0.85rem"
          }}
        >
          {leagues.map((league) => (
            <Link
              key={league.name}
              href={publicLeagueHomeHref(params.clubSlug, league.name)}
              style={{
                ...cardStyle,
                color: "#0f172a",
                textDecoration: "none"
              }}
            >
              <h2 style={{ margin: "0 0 0.45rem", fontSize: "1.12rem" }}>
                {league.name}
              </h2>
              <p style={{ margin: 0, color: "#475569" }}>
                {league.num_weeks
                  ? `${league.num_weeks} scheduled weeks`
                  : "Season schedule available inside"}
              </p>
              <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>
                Minimum games: {league.min_games ?? 0}
              </p>
              <p style={{ margin: "0.75rem 0 0", fontWeight: 800 }}>
                Open League Home →
              </p>
            </Link>
          ))}
        </div>
      ) : !error ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No active leagues</h2>
          <p style={{ color: "#475569" }}>
            There are no public active leagues available for this club right now.
          </p>
        </article>
      ) : null}
    </section>
  );
}
