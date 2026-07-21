import Link from "next/link";
import { getAdminSupportRequestsApiBaseUrl, getAdminSupportRequestsStatus } from "@/lib/adminSupportRequestsApi";
import SupportRequestsPanel from "./SupportRequestsPanel";

export const dynamic = "force-dynamic";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminSupportRequestsPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminSupportRequestsStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Support Requests
      </p>
      <h1 style={{ marginTop: 0 }}>Correction and privacy request queue</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Staff-review queue for public data correction, profile privacy, and support intake. This page tracks review state only; actual fixes still happen through the appropriate audited admin workflow.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Support request status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.status.replace(/_/g, " ")}</article>
          <article style={cardStyle}><strong>Requests</strong><br />{status.request_count ?? "—"}</article>
          <article style={cardStyle}><strong>Public forms</strong><br /><Link href="/data-corrections">Corrections</Link> · <Link href="/profile-privacy">Privacy</Link></article>
        </div>
      ) : null}

      {status ? <SupportRequestsPanel apiBase={getAdminSupportRequestsApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin">Operations cockpit</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin/players">Player Editor</Link>
      </p>
    </section>
  );
}
