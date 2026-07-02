import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function HomePage() {
  return (
    <section>
      <div style={{ maxWidth: "760px", marginBottom: "1.5rem" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Live ratings for pickleball clubs
        </p>
        <h1 style={{ margin: "0 0 0.75rem", fontSize: "clamp(2rem, 5vw, 3.5rem)", lineHeight: 1.05 }}>
          The full JUPR experience is moving to the web.
        </h1>
        <p style={{ marginTop: 0, fontSize: "1.1rem", color: "#334155" }}>
          JUPR combines live event scoring, durable club leaderboards, player profiles, and rating history so organizers and players can follow a season from one public home.
        </p>
        <p>
          <Link href="/clubs/tres-palapas" style={{ fontWeight: 700 }}>Open Tres Palapas</Link>
          <span style={{ color: "#64748b" }}> · </span>
          <Link href="/clubs/tres-palapas/leaderboards" style={{ fontWeight: 700 }}>View leaderboards</Link>
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>JUPR Live</h2>
          <p style={{ color: "#475569" }}>Follow active events, scores, standings, and brackets from the public website as the Streamlit admin workflow is ported.</p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Leaderboards</h2>
          <p style={{ color: "#475569" }}>Public club rankings powered by the same ratings data organizers use to run leagues and events.</p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Player profiles</h2>
          <p style={{ color: "#475569" }}>A durable home for each player’s JUPR, match history, badges, and rating movement.</p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Organizer tools</h2>
          <p style={{ color: "#475569" }}>The website becomes the main entry point while legacy Streamlit admin remains available during migration.</p>
        </article>
      </div>
    </section>
  );
}
