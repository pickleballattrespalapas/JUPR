import Link from "next/link";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeaguePrintoutPanel from "./LeaguePrintoutPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function LeagueManagerPrintPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p className="no-print" style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        League Manager Printout
      </p>
      <h1 className="no-print" style={{ marginTop: 0 }}>League night printout</h1>
      <p className="no-print" style={{ color: "#334155", maxWidth: "860px" }}>
        Browser-printable League Manager roster, schedule, and standings sheet. This is export-only and does not mutate league, match, or rating data.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {error}</p> : null}
      {status ? <LeaguePrintoutPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <article className="no-print" style={{ ...cardStyle, marginTop: "1rem" }}>
        <Link href="/admin/league-manager">Back to League Manager</Link> · <Link href="/admin">Operations cockpit</Link>
      </article>
    </section>
  );
}
