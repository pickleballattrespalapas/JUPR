import Link from "next/link";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import BulkRegistrationPanel from "./BulkRegistrationPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminTournamentBulkRegistrationPage() {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Admin
      </p>
      <h1 style={{ marginTop: 0 }}>Bulk registration actions</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Streamlit-parity workflow for applying registration status, payment status, or admin-note updates to multiple tournament registrations through guarded FastAPI writes.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament Admin status is temporarily unavailable. {error}</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Status</strong><br />{data.enabled ? data.status : "Disabled"}</article>
            <article style={cardStyle}><strong>Tournaments</strong><br />{data.tournament_count ?? "—"}</article>
            <article style={cardStyle}><strong>API</strong><br />{data.tournaments_endpoint ? "Configured" : "Guarded off"}</article>
          </div>
          <BulkRegistrationPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} />
        </>
      ) : null}

      <p style={{ marginTop: "1rem", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
