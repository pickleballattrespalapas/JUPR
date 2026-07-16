import Link from "next/link";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueAwardsPanel from "./LeagueAwardsPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminLeagueAwardsPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>League awards</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Preview top performer awards, close a league, and award top performer badges through the guarded FastAPI League Manager workflow.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.status.replace(/_/g, " ")}</article>
          <article style={cardStyle}><strong>Leagues</strong><br />{status.league_count ?? "—"}</article>
          <article style={cardStyle}><strong>Active</strong><br />{status.active_count ?? "—"}</article>
          <article style={cardStyle}><strong>Gate</strong><br /><code>manage_matches</code></article>
        </div>
      ) : null}

      {status ? <LeagueAwardsPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/league-manager">League Manager</Link> · <Link href="/admin/league-manager/live">League Live</Link> · <Link href="/admin/league-manager/print">Printouts</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
