import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function PrivacyPage() {
  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Privacy
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR Leagues privacy notice</h1>
      <p style={{ color: "#334155" }}>
        This first-party route gives players a stable privacy destination on juprleagues.com while the final legal policy is reviewed.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>What JUPR publishes</h2>
          <p style={{ color: "#475569" }}>
            Public JUPR pages may show player names, ratings, match history, badges, league results, tournament podiums, challenge-ladder status, and published club recaps.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>What stays private</h2>
          <p style={{ color: "#475569" }}>
            The public web app is designed not to expose service keys, admin notes, draft recap payloads, badge evaluator internals, private contact details, or admin-only operational tools.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Correction and privacy requests</h2>
          <p style={{ color: "#475569" }}>
            Players can request data corrections or privacy review through support. Public routes do not directly edit ratings or player records.
          </p>
          <Link href="/support">Contact support</Link>
        </article>
      </div>

      <p style={{ color: "#64748b", marginTop: "1rem" }}>
        This page is an operational placeholder and should be replaced or approved by counsel before broad production launch.
      </p>
    </section>
  );
}
