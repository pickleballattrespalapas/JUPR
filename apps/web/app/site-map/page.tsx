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
      ["JUPR Live", "/clubs/tres-palapas/live"]
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
    heading: "Support, privacy, and account links",
    routes: [
      ["Ratings explainer", "/how-ratings-work"],
      ["FAQ", "/faq"],
      ["Support", "/support"],
      ["Contact", "/contact"],
      ["Data corrections", "/data-corrections"],
      ["Profile privacy request", "/profile-privacy"],
      ["Verified updates request", "/clubs/tres-palapas/verified-updates"],
      ["Email preferences", "/email-preferences"],
      ["Privacy", "/privacy"],
      ["Terms", "/terms"]
    ]
  },
  {
    heading: "Staff access",
    routes: [
      ["Staff sign-in", "/admin/login"],
      ["Reset password", "/admin/reset-password"]
    ]
  }
];

export default function SiteMapPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Site map
      </p>
      <h1 style={{ marginTop: 0 }}>Pickleball Club Sandwich route map</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        A click-through map for public pages, tournament pages, support routes,
        and staff sign-in. Protected administrative tools are intentionally not
        listed on the public route map.
      </p>
      <div style={{ display: "grid", gap: "1.25rem" }}>
        {routeGroups.map((group) => (
          <section key={group.heading}>
            <h2>{group.heading}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {group.routes.map(([label, href]) => (
                <article key={href} style={cardStyle}>
                  <strong><Link href={href}>{label}</Link></strong>
                  <p style={{ margin: "0.35rem 0 0", color: "#64748b", fontSize: "0.85rem", overflowWrap: "anywhere" }}>{href}</p>
                </article>
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
