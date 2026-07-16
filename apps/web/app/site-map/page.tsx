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
      ["Verified updates request", "/verified-updates"],
      ["Email preferences", "/email-preferences"],
      ["Privacy", "/privacy"],
      ["Terms", "/terms"]
    ]
  },
  {
    heading: "Staff operations shell",
    routes: [
      ["Operations cockpit", "/admin"],
      ["Operations guide", "/admin/guide"],
      ["Theme QA", "/admin/theme-qa"],
      ["Staff sign-in", "/admin/login"],
      ["Reset password", "/admin/reset-password"],
      ["Match Log", "/admin/match-log"],
      ["Replay History", "/admin/replay-history"],
      ["Match Uploader", "/admin/match-uploader"],
      ["Player Editor", "/admin/players"],
      ["Player Updates", "/admin/player-updates"],
      ["Verified Requests", "/admin/player-updates/verified-requests"],
      ["Support Requests", "/admin/support-requests"],
      ["League Manager", "/admin/league-manager"],
      ["League Live", "/admin/league-manager/live"],
      ["League Awards", "/admin/league-manager/awards"],
      ["League Print", "/admin/league-manager/print"],
      ["Top Players Printable", "/admin/top-players-printable"],
      ["Tournament Setup", "/admin/tournament-setup"],
      ["Tournament Admin", "/admin/tournaments"],
      ["Tournament Bulk Actions", "/admin/tournaments/bulk"],
      ["Tournament Ops", "/admin/tournaments/ops"],
      ["Tournament Status", "/admin/tournaments/status"],
      ["Delete Draft Tournament", "/admin/tournaments/delete-draft"],
      ["Tournament Live", "/admin/tournament-live"],
      ["Weekly Recap Admin", "/admin/weekly-recap"],
      ["Badge Diagnostics", "/admin/badges"],
      ["Moneyball", "/admin/moneyball"],
      ["JUPR Live Admin", "/admin/jupr-live"],
      ["Challenge Ladder Admin", "/admin/challenge-ladder"],
      ["Match Canonical Audit", "/admin/match-canonical-audit"],
      ["Admin Tools", "/admin/tools"]
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
        A click-through map for public pages, tournament pages, support routes, and the full staff staging surface. Admin routes require staff sign-in and workflow-specific FastAPI authorization before writes are accepted.
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
