import { redirect } from "next/navigation";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import { readLeagueRouteContext } from "@/lib/leagueRouteContext";
import LeagueManagerNav from "../LeagueManagerNav";
import LeaguePrintoutPanel from "./LeaguePrintoutPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function LeagueManagerPrintPage({ searchParams }: Props) {
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } @page { margin: 10mm; } }`}</style>
      <p className="no-print" style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 className="no-print" style={{ marginTop: 0 }}>{leagueName} league night printout</h1>
      <div className="no-print"><LeagueManagerNav leagueId={context.leagueId} leagueName={leagueName} leagueType={context.leagueType || null} /></div>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? <LeaguePrintoutPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} initialLeague={leagueName} /> : null}
    </section>
  );
}
