import Link from "next/link";

type PlayersPageProps = {
  params: { clubSlug: string };
};

export default function ClubPlayersPage({ params }: PlayersPageProps) {
  const { clubSlug } = params;

  return (
    <section style={{ maxWidth: "760px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Player profiles
      </p>
      <h1 style={{ marginTop: 0 }}>Players are next</h1>
      <p style={{ color: "#334155" }}>
        Public player profiles will make the website feel like the full JUPR product: current rating, recent matches, league ratings, badges, and movement over time.
      </p>
      <p>
        Until the player directory API is ready, start with the <Link href={`/clubs/${clubSlug}/leaderboards`}>public leaderboards</Link>.
      </p>
    </section>
  );
}
