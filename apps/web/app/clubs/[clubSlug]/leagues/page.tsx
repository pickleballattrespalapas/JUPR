import Link from "next/link";

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

export default function PublicLeaguesPage({ params }: Props) {
  const base = `/clubs/${params.clubSlug}`;
  const modules = [
    {
      title: "League Results",
      description:
        "Review standings, weekly results, player summaries, and public-safe rating movement.",
      href: `${base}/league-results`
    },
    {
      title: "Team Leagues",
      description:
        "Open fixed-partner weekly leagues, registration, schedules, standings, and playoffs.",
      href: `${base}/team-leagues`
    },
    {
      title: "Challenge Ladder",
      description:
        "Follow ladder tiers, player status, active challenge groups, and completed results.",
      href: `${base}/challenge-ladder`
    },
    {
      title: "Club Leaderboards",
      description:
        "See current club ratings and player rankings across the broader club community.",
      href: `${base}/leaderboards`
    }
  ];

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
      <h1 style={{ marginTop: 0 }}>Club leagues and ladders</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Open league standings, team leagues, and challenge formats from one
        public starting point.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
          gap: "0.85rem"
        }}
      >
        {modules.map((module) => (
          <article key={module.href} style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>
              {module.title}
            </h2>
            <p style={{ color: "#475569" }}>{module.description}</p>
            <Link href={module.href} style={{ fontWeight: 800 }}>
              Open {module.title}
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}
