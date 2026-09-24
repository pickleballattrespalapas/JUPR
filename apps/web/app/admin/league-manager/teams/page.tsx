import LeagueManagerLoadError from "../LeagueManagerLoadError";
import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import { redirect } from "next/navigation";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import { isTeamLeagueType, leagueRouteHref, readLeagueRouteContext } from "@/lib/leagueRouteContext";
import LeagueManagerNav from "../LeagueManagerNav";
import SelectedLeaguePanelScope from "../SelectedLeaguePanelScope";
import TeamLeaguesPanel from "./TeamLeaguesPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function AdminTeamLeaguesPage({ searchParams }: Props) {
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;
  if (context.leagueType && !isTeamLeagueType(context.leagueType)) {
    redirect(leagueRouteHref("/admin/league-manager/league", context));
  }

  const { clubId } = requireAdminWorkspace();
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>{leagueName} team league</h1>
      <LeagueManagerNav leagueId={context.leagueId} leagueName={leagueName} leagueType="Team" />
      {error ? <LeagueManagerLoadError error={error} /> : null}
      {status ? (
        <SelectedLeaguePanelScope leagueName={leagueName}>
          <TeamLeaguesPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} />
        </SelectedLeaguePanelScope>
      ) : null}
    </section>
  );
}
