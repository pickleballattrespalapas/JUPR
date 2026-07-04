import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function NotFound() {
  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Page not found
      </p>
      <h1 style={{ marginTop: 0 }}>We could not find that page.</h1>
      <p style={{ color: "#334155" }}>
        The route may have moved while the Next/Vercel site is being assembled. Use one of the main public links below.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}><strong><Link href="/">Home</Link></strong><p style={{ color: "#475569" }}>Return to the Pickleball Club Sandwich homepage.</p></article>
        <article style={cardStyle}><strong><Link href="/site-map">Site map</Link></strong><p style={{ color: "#475569" }}>Open the public route map.</p></article>
        <article style={cardStyle}><strong><Link href="/clubs/tres-palapas">Tres Palapas</Link></strong><p style={{ color: "#475569" }}>Open the club home.</p></article>
        <article style={cardStyle}><strong><Link href="/support">Support</Link></strong><p style={{ color: "#475569" }}>Get help or report a broken link.</p></article>
      </div>
    </section>
  );
}
