import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const primaryLinks = [
  ["Open Tres Palapas", "/clubs/tres-palapas"],
  ["View leaderboards", "/clubs/tres-palapas/leaderboards"],
  ["Explore matches", "/clubs/tres-palapas/matches"],
  ["Open site map", "/site-map"]
];

const featureGroups = [
  {
    title: "Ratings and results",
    description: "Leaderboards, league results, player profiles, match history, badges, and Match Explorer live on the public site.",
    links: [
      ["Leaderboards", "/clubs/tres-palapas/leaderboards"],
      ["League Results", "/clubs/tres-palapas/league-results"],
      ["Players", "/clubs/tres-palapas/players"],
      ["Matches", "/clubs/tres-palapas/matches"],
      ["Match Explorer", "/clubs/tres-palapas/match-explorer"],
      ["Badge Codex", "/clubs/tres-palapas/badge-codex"]
    ]
  },
  {
    title: "Events and tournaments",
    description: "Registration, roster, partner board, live event pages, weekly recaps, and the challenge ladder are ready for public navigation.",
    links: [
      ["Register", "/clubs/tres-palapas/tournament-registration"],
      ["Roster", "/clubs/tres-palapas/tournament-roster"],
      ["Partner Board", "/clubs/tres-palapas/tournament-partner-board"],
      ["Live", "/clubs/tres-palapas/live"],
      ["Weekly Recap", "/clubs/tres-palapas/weekly-recap"],
      ["Challenge Ladder", "/clubs/tres-palapas/challenge-ladder"]
    ]
  },
  {
    title: "Help and operations",
    description: "Public help pages and a separate staff sign-in keep member support easy to find while administrative tools remain protected.",
    links: [
      ["Ratings explainer", "/how-ratings-work"],
      ["FAQ", "/faq"],
      ["Support", "/support"],
      ["Data corrections", "/data-corrections"],
      ["Staff sign-in", "/admin/login"],
      ["Site Map", "/site-map"]
    ]
  }
];

export default function HomePage() {
  return (
    <section>
      <div style={{ maxWidth: "820px", marginBottom: "1.5rem" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Live ratings for pickleball clubs
        </p>
        <h1 style={{ margin: "0 0 0.75rem", fontSize: "clamp(2rem, 5vw, 3.5rem)", lineHeight: 1.05 }}>
          Pickleball Club Sandwich is the new home for club leagues, events, and ratings.
        </h1>
        <p style={{ marginTop: 0, fontSize: "1.1rem", color: "#334155" }}>
          Follow live event scoring, club leaderboards, player profiles, tournament registration, public rosters, partner-board discovery, weekly recaps, and rating history from one public website.
        </p>
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          {primaryLinks.map(([label, href]) => <Link key={href} href={href} style={{ fontWeight: 800 }}>{label}</Link>)}
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
        {featureGroups.map((group) => (
          <article key={group.title} style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>{group.title}</h2>
            <p style={{ color: "#475569" }}>{group.description}</p>
            <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
              {group.links.map(([label, href]) => <Link key={href} href={href}>{label}</Link>)}
            </div>
          </article>
        ))}
      </div>

      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Migration status</h2>
        <p style={{ color: "#475569", marginBottom: 0 }}>
          The public Next/Vercel site is the front door for club members. Admin write workflows remain guarded and are being ported one workflow at a time behind FastAPI and audit controls.
        </p>
      </article>
    </section>
  );
}
