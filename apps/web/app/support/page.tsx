import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const supportEmail = "hello@jupr.app";

export default function SupportPage() {
  const subject = encodeURIComponent("JUPR Leagues support request");
  const body = encodeURIComponent("Name:\nClub:\nPage or issue:\nDetails:\n");
  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Support
      </p>
      <h1 style={{ marginTop: 0 }}>Contact JUPR support</h1>
      <p style={{ color: "#334155" }}>
        Use this route for support, player-profile questions, correction routing, and operational help with the public JUPR site.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>General support</h2>
          <p style={{ color: "#475569" }}>Questions about ratings, pages, access, or club setup can start here.</p>
          <a href={`mailto:${supportEmail}?subject=${subject}&body=${body}`}>Email support</a>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Data corrections</h2>
          <p style={{ color: "#475569" }}>Wrong score, teammate, opponent, league, or player profile details should use the correction checklist.</p>
          <Link href="/data-corrections">Open correction checklist</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>FAQ</h2>
          <p style={{ color: "#475569" }}>Review how JUPR ratings, doubles movement, and eligible match types work.</p>
          <Link href="/faq">Read FAQ</Link>
        </article>
      </div>
    </section>
  );
}
