import Link from "next/link";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const swatchStyle = (background: string) => ({ width: "100%", minHeight: "72px", borderRadius: "12px", background, border: "1px solid #e2e8f0" });
const buttonStyle = { display: "inline-block", padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, textDecoration: "none" };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

const routeGroups = [
  ["Public", ["/", "/clubs/tres-palapas", "/clubs/tres-palapas/leaderboards", "/clubs/tres-palapas/match-explorer", "/clubs/tres-palapas/players"]],
  ["Tournament", ["/clubs/tres-palapas/tournament-registration", "/clubs/tres-palapas/tournament-roster", "/clubs/tres-palapas/tournament-partner-board", "/admin/tournaments", "/admin/tournament-live"]],
  ["Operations", ["/admin", "/admin/match-log", "/admin/match-uploader", "/admin/league-manager", "/admin/tools", "/admin/badges"]]
] as const;

export default function ThemeQaPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Theme QA
      </p>
      <h1 style={{ marginTop: 0 }}>Next design-system QA</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Static staging page for checking typography, spacing, links, cards, alerts, form controls, table density, print behavior, and dark-on-light contrast across the Next/Vercel product shell. This replaces the old Streamlit-only Theme QA placeholder without introducing database writes.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Color and status tokens</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            <div><div style={swatchStyle("#0f172a")} /><strong>Ink</strong><br /><code>#0f172a</code></div>
            <div><div style={swatchStyle("#2563eb")} /><strong>Action</strong><br /><code>#2563eb</code></div>
            <div><div style={swatchStyle("#f8fafc")} /><strong>Page</strong><br /><code>#f8fafc</code></div>
            <div><div style={swatchStyle("#dcfce7")} /><strong>Success</strong><br /><code>#dcfce7</code></div>
            <div><div style={swatchStyle("#fef3c7")} /><strong>Warning</strong><br /><code>#fef3c7</code></div>
            <div><div style={swatchStyle("#fee2e2")} /><strong>Danger</strong><br /><code>#fee2e2</code></div>
          </div>
        </article>

        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Controls</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>Text input<br /><input defaultValue="Sample player name" style={{ width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }} /></label>
            <label>Select<br /><select defaultValue="staging" style={{ width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}><option value="staging">Staging</option><option value="production">Production</option></select></label>
            <a href="#" style={buttonStyle}>Primary action</a>
            <a href="#" style={ghostButtonStyle}>Secondary action</a>
          </div>
        </article>

        <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#f59e0b" }}>
          <h2 style={{ marginTop: 0 }}>Warning state</h2>
          <p style={{ color: "#92400e" }}>Use this treatment for staging-only warnings, high-risk admin writes, missing environment flags, and workflows that require replay or audit follow-up.</p>
        </article>

        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Table density</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
              <thead><tr><th align="left">Workflow</th><th align="left">State</th><th align="left">Risk</th><th align="left">Owner action</th></tr></thead>
              <tbody>
                {["Tournament Ops", "League Live", "Match Log", "Badge Repair"].map((row, idx) => (
                  <tr key={row}><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.45rem" }}>{row}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.45rem" }}>Ready for staging</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.45rem" }}>{idx === 3 ? "critical" : "high"}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.45rem" }}>Validate</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Route smoke groups</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}>
            {routeGroups.map(([heading, routes]) => (
              <section key={heading}>
                <h3>{heading}</h3>
                <ul>
                  {routes.map((href) => <li key={href}><Link href={href}>{href}</Link></li>)}
                </ul>
              </section>
            ))}
          </div>
        </article>

        <p><Link href="/admin">Operations cockpit</Link> · <Link href="/site-map">Site map</Link></p>
      </div>
    </section>
  );
}
