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

  if (error) {
    return (
      <section>
        <h1>Club: {clubSlug}</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this club right now. {error}</p>
        <p>
          <Link href={`/clubs/${clubSlug}/leaderboards`}>Try the public leaderboards</Link>
        </p>
      </section>
    );
  }

  return (
    <section>
      <div style={{ marginBottom: "1.5rem" }}>
        <p style={{ margin: "0 0 0.35rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Club home
        </p>
        <h1 style={{ margin: "0 0 0.5rem", fontSize: "2.4rem", lineHeight: 1.1 }}>{clubName}</h1>
        <p style={{ marginBottom: "0.25rem" }}><strong>Slug:</strong> {data?.slug ?? clubSlug}</p>
        {data?.tagline ? <p style={{ margin: "0.25rem 0", color: "#334155" }}>{data.tagline}</p> : null}
        {data?.support_email ? (
          <p style={{ margin: "0.25rem 0" }}>
            <strong>Support:</strong> <a href={`mailto:${data.support_email}`}>{data.support_email}</a>
          </p>
        ) : null}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Live events</h2>
          <p style={{ color: "#475569" }}>Open public JUPR Live sessions with rounds, scores, standings, and brackets when durable sessions are available.</p>
          <Link href={`/clubs/${clubSlug}/live`}>Open JUPR Live</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Leaderboards</h2>
          <p style={{ color: "#475569" }}>See active JUPR rankings for this club.</p>
          <Link href={`/clubs/${clubSlug}/leaderboards`}>View leaderboards</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>League results</h2>
          <p style={{ color: "#475569" }}>Review league standings, weekly results, player summaries, and public-safe rating movement.</p>
          <Link href={`/clubs/${clubSlug}/league-results`}>Open league results</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Badge Codex</h2>
          <p style={{ color: "#475569" }}>Browse JUPR badges, unlock paths, prestige, and recent badge earners.</p>
          <Link href={`/clubs/${clubSlug}/badge-codex`}>Open badge codex</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Challenge Ladder</h2>
          <p style={{ color: "#475569" }}>View public ladder tiers, player status, active challenge buckets, and quick rules.</p>
          <Link href={`/clubs/${clubSlug}/challenge-ladder`}>Open challenge ladder</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Weekly Recap</h2>
          <p style={{ color: "#475569" }}>Read published club highlights, spotlight reels, tournament podiums, and looking-ahead notes.</p>
          <Link href={`/clubs/${clubSlug}/weekly-recap`}>Open weekly recap</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Players</h2>
          <p style={{ color: "#475569" }}>Browse the public player directory and open profiles with ratings, records, and recent matches.</p>
          <Link href={`/clubs/${clubSlug}/players`}>Open players</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Match history</h2>
          <p style={{ color: "#475569" }}>Review recorded matches and drill into match detail pages with public-safe rating snapshots.</p>
          <Link href={`/clubs/${clubSlug}/matches`}>View matches</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Ratings</h2>
          <p style={{ color: "#475569" }}>Understand how JUPR events feed ratings and club standings.</p>
          <Link href="/how-ratings-work">How ratings work</Link>
        </article>
      </div>
    </section>
  );
}
