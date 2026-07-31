import { redirect } from "next/navigation";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import LeaguePrintoutPanel from "./LeaguePrintoutPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default async function LeagueManagerPrintPage({ searchParams }: Props) {
  const leagueName = first(searchParams?.league).trim();
  const leagueType = first(searchParams?.mode).trim();
  if (!leagueName) redirect("/admin/league-manager");

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } @page { margin: 10mm; } }`}</style>
      <p className="no-print" style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 className="no-print" style={{ marginTop: 0 }}>{leagueName} league night printout</h1>
      <div className="no-print"><LeagueManagerNav leagueName={leagueName} leagueType={leagueType || null} /></div>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? <LeaguePrintoutPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} initialLeague={leagueName} /> : null}
    </section>
  );
}
