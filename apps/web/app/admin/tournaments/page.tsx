import Link from "next/link";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import TournamentAdminPanel from "./TournamentAdminPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminTournamentsPage() {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Admin
      </p>
      <h1 style={{ marginTop: 0 }}>Tournament registration management</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Read-foundation for moving Streamlit tournament registration management to Next/Vercel. This view is guarded by the stored admin session and FastAPI tournament permissions.
      </p>
      <p style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <Link href="/admin/tournaments/registrations">Open registration reporting</Link>
        <Link href="/admin/tournaments/bulk">Open bulk registration actions</Link>
        <Link href="/admin/tournaments/ops">Open Tournament Ops snapshot</Link>
        <Link href="/admin/tournaments/status">Open status actions</Link>
        <Link href="/admin/tournaments/delete-draft">Delete empty draft</Link>
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament Admin status is temporarily unavailable. {error}</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Status</strong><br />{data.enabled ? data.status : "Disabled"}</article>
            <article style={cardStyle}><strong>Tournaments</strong><br />{data.tournament_count ?? "—"}</article>
            <article style={cardStyle}><strong>API</strong><br />{data.tournaments_endpoint ? "Configured" : "Guarded off"}</article>
          </div>
          <TournamentAdminPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} />
        </>
      ) : null}

      <p style={{ marginTop: "1rem" }}><Link href="/admin">Back to operations cockpit</Link></p>
      {data?.streamlit_fallback_url ? <p><a href={data.streamlit_fallback_url} target="_blank" rel="noreferrer">Open Streamlit tournament recovery fallback</a></p> : null}
    </section>
  );
}
