import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const routeGroups = [
  {
    heading: "Club public pages",
    routes: [
      ["Club home", "/clubs/tres-palapas"],
      ["Leaderboards", "/clubs/tres-palapas/leaderboards"],
      ["Players", "/clubs/tres-palapas/players"],
      ["Match history", "/clubs/tres-palapas/matches"],
      ["Match Explorer", "/clubs/tres-palapas/match-explorer"],
      ["League Results", "/clubs/tres-palapas/league-results"],
      ["Badge Codex", "/clubs/tres-palapas/badge-codex"],
      ["Challenge Ladder", "/clubs/tres-palapas/challenge-ladder"],
      ["Weekly Recap", "/clubs/tres-palapas/weekly-recap"],
      ["Live", "/clubs/tres-palapas/live"]
    ]
  },
  {
    heading: "Tournament public pages",
    routes: [
      ["Registration", "/clubs/tres-palapas/tournament-registration"],
      ["Roster", "/clubs/tres-palapas/tournament-roster"],
      ["Partner Board", "/clubs/tres-palapas/tournament-partner-board"]
    ]
  },
  {
    heading: "Help and policy pages",
    routes: [
      ["Ratings explainer", "/how-ratings-work"],
      ["FAQ", "/faq"],
      ["Support", "/support"],
      ["Contact", "/contact"],
      ["Data corrections", "/data-corrections"],
      ["Privacy", "/privacy"],
      ["Terms", "/terms"]
    ]
  },
  {
    heading: "Admin migration shell",
    routes: [
      ["Admin cockpit", "/admin"],
      ["Match Log", "/admin/match-log"],
      ["Replay History", "/admin/replay-history"],
      ["Match Uploader", "/admin/match-uploader"],
      ["Players", "/admin/players"],
      ["League Manager", "/admin/league-manager"]
    ]
  }
];

export default function SiteMapPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Public site map
      </p>
      <h1 style={{ marginTop: 0 }}>Pickleball Club Sandwich route map</h1>
      <p style={{ color: "#334155", maxWidth: "780px" }}>
        A click-through map for public pages, tournament pages, support routes, and the read-only admin migration shell.
      </p>
      <div style={{ display: "grid", gap: "1.25rem" }}>
        {routeGroups.map((group) => (
          <section key={group.heading}>
            <h2>{group.heading}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {group.routes.map(([label, href]) => (
                <article key={href} style={cardStyle}>
                  <strong><Link href={href}>{label}</Link></strong>
                </article>
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
