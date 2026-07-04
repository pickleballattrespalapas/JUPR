import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const sections = [
  {
    title: "Club experience",
    routes: [
      ["Club home", "/clubs/tres-palapas", "Public hub for Tres Palapas."],
      ["Live", "/clubs/tres-palapas/live", "Public live sessions and scoring views."],
      ["Leaderboards", "/clubs/tres-palapas/leaderboards", "Club rankings and rating tables."],
      ["Match Explorer", "/clubs/tres-palapas/match-explorer", "Rating-impact previews for doubles combinations."],
      ["League Results", "/clubs/tres-palapas/league-results", "League standings, weekly summaries, and player movement."],
      ["Weekly Recap", "/clubs/tres-palapas/weekly-recap", "Published club recap stories and PDF output."]
    ]
  },
  {
    title: "Players and matches",
    routes: [
      ["Players", "/clubs/tres-palapas/players", "Public player directory."],
      ["Matches", "/clubs/tres-palapas/matches", "Recorded public match history."],
      ["Badge Codex", "/clubs/tres-palapas/badge-codex", "Badge catalog and public earners."],
      ["Challenge Ladder", "/clubs/tres-palapas/challenge-ladder", "Public ladder and challenge status."],
      ["How ratings work", "/how-ratings-work", "Plain-language rating explainer."],
      ["FAQ", "/faq", "Common player and organizer questions."]
    ]
  },
  {
    title: "Tournament flows",
    routes: [
      ["Registration", "/clubs/tres-palapas/tournament-registration", "Public tournament intake and edit-link request."],
      ["Roster", "/clubs/tres-palapas/tournament-roster", "Public-safe tournament roster."],
      ["Registration confirmation", "/clubs/tres-palapas/tournament-registration/confirmation", "Confirmation route after registration submit."],
      ["Registration edit", "/clubs/tres-palapas/tournament-registration/edit", "Secure edit-link route; requires token query parameter."],
      ["Support", "/support", "Support and help handoff."]
    ]
  },
  {
    title: "Policy and support",
    routes: [
      ["Privacy", "/privacy", "Draft privacy destination pending final legal approval."],
      ["Terms", "/terms", "Draft terms destination pending final legal approval."],
      ["Data corrections", "/data-corrections", "Public correction request checklist."],
      ["Admin cockpit", "/admin", "Staff migration cockpit and guarded admin routes."]
    ]
  }
];

export default function SiteMapPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Public site map
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR website first-draft route map</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        This page is a click-through map for the first full public Next/Vercel website draft. It intentionally includes both polished player-facing routes and draft support/legal/admin handoff routes so staging review can cover the whole surface.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        {sections.map((section) => (
          <article key={section.title} style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{section.title}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}>
              {section.routes.map(([label, href, description]) => (
                <div key={href} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                  <strong><Link href={href}>{label}</Link></strong>
                  <p style={{ color: "#475569", marginBottom: 0 }}>{description}</p>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
