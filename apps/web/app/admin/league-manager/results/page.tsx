import { redirect } from "next/navigation";
import {
  getAdminLeagueManagerApiBaseUrl,
  getAdminLeagueManagerStatus
} from "@/lib/adminLeagueManagerApi";
import {
  isTeamLeagueType,
  leagueRouteHref,
  readLeagueRouteContext
} from "@/lib/leagueRouteContext";
import LeagueManagerNav from "../LeagueManagerNav";
import LeagueResultsPanel from "./LeagueResultsPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function AdminLeagueResultsPage({ searchParams }: Props) {
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;
  if (isTeamLeagueType(context.leagueType)) {
    redirect(leagueRouteHref("/admin/league-manager/teams", context));
  }

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} results</h1>
      <LeagueManagerNav
        leagueId={context.leagueId}
        leagueName={leagueName}
        leagueType={context.leagueType || null}
      />

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? (
        <LeagueResultsPanel
          apiBase={getAdminLeagueManagerApiBaseUrl()}
          clubId={clubId}
          status={status}
          initialLeagueId={context.leagueId}
          initialLeague={leagueName}
          initialLeagueType={context.leagueType || null}
        />
      ) : null}
    </section>
  );
}
