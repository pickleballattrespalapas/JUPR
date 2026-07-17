import Link from "next/link";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerPanel from "./LeagueManagerPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminLeagueManagerPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>League Manager</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Guarded Next/FastAPI league operations: create drafts, edit settings and roster membership, run persisted live rounds with court movement, print operations sheets, and close awards. Streamlit remains the fallback while staging validation continues.
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

      {status ? <LeagueManagerPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin/players">Player Editor</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
