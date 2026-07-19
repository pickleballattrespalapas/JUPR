import Link from "next/link";
import { getAdminPlayerUpdatesApiBaseUrl, getAdminPlayerUpdatesStatus } from "@/lib/adminPlayerUpdatesApi";
import PlayerUpdatesPanel from "./PlayerUpdatesPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function statusText(value?: string | null): string {
  return String(value || "unknown").replace(/_/g, " ");
}

export default async function AdminPlayerUpdatesPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminPlayerUpdatesStatus(clubId);
  const smtpConfigured = Boolean(status?.smtp_configured);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Player Updates
      </p>
      <h1 style={{ marginTop: 0 }}>Player update email reports</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Generate and send player update summaries for a selected date range. This mirrors the Streamlit Player Updates Admin workflow, using the existing SMTP-backed email sender and verified player subscriptions.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Player Updates status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{statusText(status.status)}</article>
          <article style={cardStyle}><strong>Email mode</strong><br />{status.email_mode || "unknown"}</article>
          <article style={cardStyle}><strong>SMTP</strong><br />{smtpConfigured ? "Configured" : "Not configured"}</article>
          <article style={cardStyle}><strong>Auto post-batch send</strong><br />{status.auto_send_enabled ? "Enabled" : "Disabled"}</article>
          <article style={cardStyle}><strong>Subscription data</strong><br />Sign in to load</article>
        </div>
      ) : null}

      {status ? (
        <PlayerUpdatesPanel
          apiBase={getAdminPlayerUpdatesApiBaseUrl()}
          clubId={clubId}
          status={status}
        />
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin/tournaments/ops">Tournament Ops</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
