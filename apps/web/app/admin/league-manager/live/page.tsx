import LeagueManagerLoadError from "../LeagueManagerLoadError";
import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import { redirect } from "next/navigation";
import { getClubPlayerOptions } from "@/lib/api";
import { getAdminLeagueLiveStatus, getAdminLeagueManagerStatus, getAdminLeagueManagerApiBaseUrl } from "@/lib/adminLeagueManagerApi";
import { readLeagueRouteContext } from "@/lib/leagueRouteContext";
import LeagueLiveRoundPanel from "./LeagueLiveRoundPanel";
import LeagueManagerNav from "../LeagueManagerNav";
import SelectedLeaguePanelScope from "../SelectedLeaguePanelScope";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function LeagueManagerLivePage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;

  const { clubId, clubSlug } = await requireAdminWorkspace();
  const [{ data: leagueStatus, error: leagueError }, { data: liveDomainStatus, error: liveDomainError }, { data: playersData, error: playersError }] = await Promise.all([
    getAdminLeagueManagerStatus(clubId),
    getAdminLeagueLiveStatus(clubId),
    getClubPlayerOptions(clubSlug, { status: "all", noStore: true })
  ]);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} live rounds</h1>
      <LeagueManagerNav leagueId={context.leagueId} leagueName={leagueName} leagueType={context.leagueType || null} />

      {leagueError ? <LeagueManagerLoadError error={leagueError} /> : null}
      {liveDomainError ? <LeagueManagerLoadError error={liveDomainError} service="League Live" /> : null}
      {playersError ? <LeagueManagerLoadError error={playersError} service="Player lookup" /> : null}

      {leagueStatus && liveDomainStatus ? (
        <SelectedLeaguePanelScope leagueName={leagueName}>
          <LeagueLiveRoundPanel
            key={leagueName}
            apiBase={getAdminLeagueManagerApiBaseUrl()}
            clubId={clubId}
            selectedLeagueName={leagueName}
            leagueStatus={leagueStatus}
            liveDomainStatus={liveDomainStatus}
            players={playersData?.players || []}
          />
        </SelectedLeaguePanelScope>
      ) : null}
    </section>
  );
}
