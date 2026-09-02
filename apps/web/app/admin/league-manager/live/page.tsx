import { redirect } from "next/navigation";
import { getClubPlayers } from "@/lib/api";
import { getAdminLeagueLiveStatus, getAdminLeagueManagerStatus, getAdminLeagueManagerApiBaseUrl } from "@/lib/adminLeagueManagerApi";
import { getAdminMatchUploaderStatus } from "@/lib/adminMatchUploaderApi";
import { readLeagueRouteContext } from "@/lib/leagueRouteContext";
import LeagueLiveRoundPanel from "./LeagueLiveRoundPanel";
import LeagueManagerNav from "../LeagueManagerNav";
import SelectedLeaguePanelScope from "../SelectedLeaguePanelScope";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function LeagueManagerLivePage({ searchParams }: Props) {
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;

  const clubSlug = "tres-palapas";
  const clubId = "tres_palapas";
  const [{ data: leagueStatus, error: leagueError }, { data: liveDomainStatus, error: liveDomainError }, { data: uploaderStatus, error: uploaderError }, { data: playersData, error: playersError }] = await Promise.all([
    getAdminLeagueManagerStatus(clubId),
    getAdminLeagueLiveStatus(clubId),
    getAdminMatchUploaderStatus(clubId),
    getClubPlayers(clubSlug, { status: "all", limit: 1000, sort: "name", noStore: true })
  ]);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} live rounds</h1>
      <LeagueManagerNav leagueId={context.leagueId} leagueName={leagueName} leagueType={context.leagueType || null} />

      {leagueError ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {leagueError}</p> : null}
      {liveDomainError ? <p role="alert" style={{ color: "#b91c1c" }}>League Live is unavailable. {liveDomainError}</p> : null}
      {uploaderError ? <p role="alert" style={{ color: "#b91c1c" }}>Match Uploader is unavailable. {uploaderError}</p> : null}
      {playersError ? <p role="alert" style={{ color: "#b91c1c" }}>Player lookup is unavailable. {playersError}</p> : null}

      {leagueStatus && liveDomainStatus && uploaderStatus ? (
        <SelectedLeaguePanelScope leagueName={leagueName}>
          <LeagueLiveRoundPanel
            key={leagueName}
            apiBase={getAdminLeagueManagerApiBaseUrl()}
            clubId={clubId}
            selectedLeagueName={leagueName}
            leagueStatus={leagueStatus}
            liveDomainStatus={liveDomainStatus}
            uploaderStatus={uploaderStatus}
            players={playersData?.players || []}
          />
        </SelectedLeaguePanelScope>
      ) : null}
    </section>
  );
}
