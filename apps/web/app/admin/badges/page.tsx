import Link from "next/link";
import { getAdminBadgeDiagnosticsApiBaseUrl, getAdminBadgeDiagnosticsStatus } from "@/lib/adminBadgeDiagnosticsApi";
import BadgeDiagnosticsPanel from "./BadgeDiagnosticsPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminBadgeDiagnosticsPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminBadgeDiagnosticsStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Badge Diagnostics
      </p>
      <h1 style={{ marginTop: 0 }}>Badge Debug & Audit</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Badge diagnostics for expected-versus-actual rows, duplicate/stale/missing audit, player-specific evaluator debugging, and guarded badge-definition lifecycle controls.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Badge Diagnostics status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.status.replace(/_/g, " ")}</article>
          <article style={cardStyle}><strong>Badges</strong><br />{status.badge_count ?? "—"}</article>
          <article style={cardStyle}><strong>Player badge rows</strong><br />{status.player_badge_count ?? "—"}</article>
          <article style={cardStyle}><strong>Gate</strong><br /><code>view_audit_log</code></article>
        </div>
      ) : null}

      {status ? <BadgeDiagnosticsPanel apiBase={getAdminBadgeDiagnosticsApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/clubs/tres-palapas/badge-codex">Public Badge Codex</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
