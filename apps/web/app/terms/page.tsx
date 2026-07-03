import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function TermsPage() {
  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Terms
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR Leagues terms of use</h1>
      <p style={{ color: "#334155" }}>
        This first-party route gives the public web app a stable terms destination while final legal language is reviewed.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Ratings are club operations data</h2>
          <p style={{ color: "#475569" }}>
            JUPR ratings, rankings, badges, and standings are maintained to support club events, fair play, league organization, and player progress tracking.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Public pages are informational</h2>
          <p style={{ color: "#475569" }}>
            Public pages are read-only and informational. They do not create official match records, edit scores, change ratings, or replace staff-administered workflows.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Errors and disputes</h2>
          <p style={{ color: "#475569" }}>
            Score or player-record disputes should go through the organizer or support. Staff review is required before any correction or recomputation.
          </p>
          <Link href="/data-corrections">Request a correction</Link>
        </article>
      </div>

      <p style={{ color: "#64748b", marginTop: "1rem" }}>
        This page is an operational placeholder and should be replaced or approved by counsel before broad production launch.
      </p>
    </section>
  );
}
