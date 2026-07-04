import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const routes = [
  ["Home", "/"],
  ["Club", "/clubs/tres-palapas"],
  ["Live", "/clubs/tres-palapas/live"],
  ["Leaderboards", "/clubs/tres-palapas/leaderboards"],
  ["League Results", "/clubs/tres-palapas/league-results"],
  ["Weekly Recap", "/clubs/tres-palapas/weekly-recap"],
  ["Registration", "/clubs/tres-palapas/tournament-registration"],
  ["Roster", "/clubs/tres-palapas/tournament-roster"],
  ["Directory", "/clubs/tres-palapas/players"],
  ["Match history", "/clubs/tres-palapas/matches"],
  ["Ratings explainer", "/how-ratings-work"],
  ["FAQ", "/faq"],
  ["Support", "/support"],
  ["Contact", "/contact"],
  ["Privacy", "/privacy"],
  ["Terms", "/terms"],
  ["Data corrections", "/data-corrections"],
  ["Admin", "/admin"]
];

export default function SiteMapPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Public site map
      </p>
      <h1 style={{ marginTop: 0 }}>Pickleball Club Sandwich route map</h1>
      <p style={{ color: "#334155", maxWidth: "780px" }}>
        A simple click-through map for the current Next/Vercel draft.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        {routes.map(([label, href]) => (
          <article key={href} style={cardStyle}>
            <strong><Link href={href}>{label}</Link></strong>
          </article>
        ))}
      </div>
    </section>
  );
}
