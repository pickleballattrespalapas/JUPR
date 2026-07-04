import Link from "next/link";

export default function HowRatingsWorkPage() {
  return (
    <section style={{ maxWidth: "820px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Rating system
      </p>
      <h1 style={{ marginTop: 0 }}>How club ratings work</h1>
      <p style={{ color: "#334155" }}>
        Pickleball Club Sandwich turns club match results into durable ratings and leaderboard experiences. The public web app exposes the pieces players care about most: current rating, match history, event results, and movement over time.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
        <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Events feed results</h2>
          <p style={{ color: "#475569" }}>Round robins, ladders, tournaments, and uploaded matches become the record of play.</p>
        </article>
        <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Ratings update</h2>
          <p style={{ color: "#475569" }}>Submitted official matches update player ratings and club-specific leaderboards.</p>
        </article>
        <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Players can follow progress</h2>
          <p style={{ color: "#475569" }}>Player profiles collect current rating, recent matches, movement, and badges.</p>
        </article>
      </div>
      <p style={{ marginTop: "1.25rem" }}>
        <Link href="/clubs/tres-palapas/leaderboards">View Tres Palapas leaderboards</Link>
      </p>
    </section>
  );
}
