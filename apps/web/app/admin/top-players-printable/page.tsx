import Link from "next/link";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import TopPlayersPrintablePanel from "./TopPlayersPrintablePanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function TopPlayersPrintablePage({ searchParams }: { searchParams?: { limit?: string } }) {
  const clubId = "tres_palapas";
  const limit = Math.max(5, Math.min(Number(searchParams?.limit || 50) || 50, 200));
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } @page { size: landscape; margin: 10mm; } }`}</style>
      <p className="no-print" style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Top Active Players PDF
      </p>
      <h1 className="no-print" style={{ marginTop: 0 }}>Top active players</h1>
      <p className="no-print" style={{ color: "#334155" }}>
        Authenticated, browser-printable previous-calendar-month ranking. Ranking and eligibility are calculated in Python/FastAPI and this route never mutates data.
      </p>
      {error ? <p style={{ color: "#b91c1c" }}>Ranking status is unavailable. {error}</p> : null}

      <article className="no-print" style={{ ...cardStyle, marginBottom: "1rem" }}>
        <p style={{ marginTop: 0, color: "#475569" }}>Use the browser print dialog to save as PDF. The optional <code>?limit=100</code> query only changes the maximum row count; active and minimum-game policies cannot be bypassed from the browser.</p>
        <Link href="/admin">Operations cockpit</Link> · <Link href="/admin/league-manager">League Manager</Link> · <Link href="/clubs/tres-palapas/leaderboards">Public leaderboard</Link>
      </article>
      {status ? <TopPlayersPrintablePanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} limit={limit} status={status} /> : null}
    </section>
  );
}
