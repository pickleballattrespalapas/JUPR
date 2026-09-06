import Link from "next/link";
import { getClub } from "@/lib/api";

type ClubPageProps = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default async function ClubPage({ params }: ClubPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClub(clubSlug);
  const clubName = data?.name ?? clubSlug;
  const base = `/clubs/${clubSlug}`;

  if (error) {
    return (
      <section>
        <h1>Club: {clubSlug}</h1>
        <p style={{ color: "#b91c1c" }}>
          We couldn&apos;t load this club right now. Please try again shortly.
        </p>
        <p>
          <Link href={`${base}/leaderboards`}>View the leaderboards instead</Link>
        </p>
      </section>
    );
  }

  const modules = [
    {
      title: "Leagues",
      description:
        "Open league results, team leagues, challenge ladders, and related standings from one place.",
      href: `${base}/leagues`,
      label: "Open leagues"
    },
    {
      title: "Tournaments",
      description:
        "Choose a tournament, then open its registration, roster, the Players Needing Partners page, and published results.",
      href: `${base}/tournaments`,
      label: "Open tournaments"
    },
    {
      title: "Play tools",
      description:
        "Create schedules and score live games.",
      href: `${base}/play`,
      label: "Open play tools"
    },
    {
      title: "Leaderboards",
      description: "See active JUPR rankings for this club.",
      href: `${base}/leaderboards`,
      label: "View leaderboards"
    },
    {
      title: "Match Explorer",
      description:
        "See how different doubles matchups could affect each player’s rating.",
      href: `${base}/match-explorer`,
      label: "Open Match Explorer"
    },
    {
      title: "Players",
      description:
        "Browse player profiles with ratings, records, awards, and recent matches.",
      href: `${base}/players`,
      label: "Open players"
    },
    {
      title: "Match history",
      description: "Browse scores and see how ratings changed after each match.",
      href: `${base}/matches`,
      label: "View matches"
    },
    {
      title: "Weekly Recap",
      description:
        "Catch up on club highlights, standout players, tournament finishes, and what’s next.",
      href: `${base}/weekly-recap`,
      label: "Open Weekly Recap"
    },
    {
      title: "Badges & Trophies",
      description:
        "Explore club badges, learn how to earn them, and see recent earners.",
      href: `${base}/badge-codex`,
      label: "Open Badges & Trophies"
    },
    {
      title: "Ratings",
      description:
        "Understand how club, league, and tournament matches feed JUPR ratings.",
      href: "/how-ratings-work",
      label: "How ratings work"
    }
  ];

  return (
    <section>
      <div style={{ marginBottom: "1.5rem" }}>
        <p
          style={{
            margin: "0 0 0.35rem",
            color: "#2563eb",
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            fontSize: "0.78rem"
          }}
        >
          Club home
        </p>
        <h1
          style={{
            margin: "0 0 0.5rem",
            fontSize: "2.4rem",
            lineHeight: 1.1
          }}
        >
          {clubName}
        </h1>
        {data?.tagline ? (
          <p style={{ margin: "0.25rem 0", color: "#334155" }}>
            {data.tagline}
          </p>
        ) : null}
        {data?.support_email ? (
          <p style={{ margin: "0.25rem 0" }}>
            <strong>Support:</strong>{" "}
            <a href={`mailto:${data.support_email}`}>{data.support_email}</a>
          </p>
        ) : null}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
          gap: "1rem"
        }}
      >
        {modules.map((module) => (
          <article key={module.href} style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>
              {module.title}
            </h2>
            <p style={{ color: "#475569" }}>{module.description}</p>
            <Link href={module.href} style={{ fontWeight: 800 }}>
              {module.label}
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}
