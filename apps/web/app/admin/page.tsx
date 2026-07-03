import Link from "next/link";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default function AdminEntryPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Organizer tools
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR organizer dashboard</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Core workflows are moving into the web app. Use these tools for Next/Vercel testing and keep the current Streamlit console as the fallback while workflows mature.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Score entry</h2>
          <p style={{ color: "#475569" }}>Enter rated matches through the FastAPI match-processing path.</p>
          <Link href="/clubs/tres-palapas/admin/score-entry">Open score entry</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Players</h2>
          <p style={{ color: "#475569" }}>Review player ratings, records, league ratings, and profile pages.</p>
          <Link href="/clubs/tres-palapas/players">Open players</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Match history</h2>
          <p style={{ color: "#475569" }}>Inspect match records and rating snapshots for debugging.</p>
          <Link href="/clubs/tres-palapas/matches">Open matches</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Leaderboards</h2>
          <p style={{ color: "#475569" }}>Confirm that saved scores are reflected in public rankings.</p>
          <Link href="/clubs/tres-palapas/leaderboards">Open leaderboards</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Streamlit fallback</h2>
          <p style={{ color: "#475569" }}>Keep using the existing console for tournament registration and any workflow not yet migrated.</p>
          <Link href="mailto:hello@jupr.app?subject=Streamlit%20console%20link">Request fallback link</Link>
        </article>
      </div>
    </section>
  );
}
